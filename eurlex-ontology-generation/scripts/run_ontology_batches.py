from pathlib import Path
import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from filelock import FileLock


MANIFEST_PATH = Path(
    "data/03_primary/chunk_batch_manifest.csv"
)

# Protects the manifest when several batches run at the same time
manifest_lock = threading.Lock()

# One row per chunk and step that did not produce a clean result (cut-off
# answer, invalid JSON, failed review or revision), appended as the run goes.
# A chunk with no row here came through every step cleanly.
ISSUES_PATH = Path("logs/chunk_issues.csv")
ISSUE_COLUMNS = ["logged_at", "batch_id", "round", "step", "chunk_id", "CELEX",
                 "status", "done_reason", "output_tokens", "detail"]
issues_lock = threading.Lock()


def log_chunk_issues(rows: list[dict]) -> None:
    if not rows:
        return
    stamp = pd.Timestamp.now().isoformat(timespec="seconds")
    frame = pd.DataFrame([{**{c: "" for c in ISSUE_COLUMNS}, **r, "logged_at": stamp} for r in rows],
                         columns=ISSUE_COLUMNS)
    with issues_lock:
        ISSUES_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(ISSUES_PATH, mode="a", header=not ISSUES_PATH.exists(), index=False, encoding="utf-8")


def _batch_rows(directory: str, batch_id: int) -> pd.DataFrame:
    path = Path(directory) / f"part_{batch_id // 1000:03d}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8")
    return frame[frame["batch_id"] == batch_id]


def _value(row, column: str, default=""):
    value = row.get(column, default)
    return default if pd.isna(value) else value


# Remove this batch's reviews from `from_round` on (reviews are keyed by
# chunk and round, so they would otherwise outlive a resumed run). Revisions
# are keyed by chunk only and are rewritten by the next revision round.
def purge_review_rounds(batch_id: int, from_round: int) -> None:
    path = Path("data/04_feature/ontology_reviews") / f"part_{batch_id // 1000:03d}.csv"
    if not path.exists():
        return
    with FileLock(str(path) + ".lock"):
        reviews = pd.read_csv(path, encoding="utf-8")
        rounds = reviews["round"].fillna(0).astype(int) if "round" in reviews.columns else 0
        stale = (reviews["batch_id"] == batch_id) & (rounds >= from_round)
        if stale.any():
            reviews[~stale].to_csv(path, index=False, encoding="utf-8")
            print(f"[batch {batch_id}] removed {int(stale.sum())} review rows from round {from_round} on",
                  flush=True)


# Generation never fails a batch on a bad chunk, but a cut-off or invalid
# answer must still be visible
def check_generation_output(batch_id: int) -> list[str]:
    rows = []
    for _, r in _batch_rows("data/04_feature/ontology_candidates", batch_id).iterrows():
        cut = _value(r, "done_reason") == "length"
        invalid = str(_value(r, "ontology")).lstrip().startswith('{"error"')
        if cut or invalid:
            rows.append({"batch_id": batch_id, "round": 0, "step": "generation",
                         "chunk_id": r["chunk_id"], "CELEX": r["CELEX"],
                         "status": "truncated_output" if cut else "invalid_json",
                         "done_reason": _value(r, "done_reason"),
                         "output_tokens": _value(r, "output_tokens")})
    log_chunk_issues(rows)
    return [r["chunk_id"] for r in rows]


# Load the persistent batch manifest
def load_manifest() -> pd.DataFrame:

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Batch manifest not found: {MANIFEST_PATH}"
        )

    return pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8",
    )


# Atomic write, so the file is never left half-written. Must be called with manifest_lock held
def save_manifest(manifest: pd.DataFrame) -> None:

    tmp_path = MANIFEST_PATH.with_suffix(".csv.tmp")
    manifest.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, MANIFEST_PATH)


# Takes the lock before touching the manifest
def update_status(
    manifest: pd.DataFrame,
    batch_id: int,
    status: str,
) -> None:

    with manifest_lock:
        manifest.loc[
            manifest["batch_id"] == batch_id,
            "status",
        ] = status

        save_manifest(manifest)


# Atomically pick the next batch and mark it as running, so two workers never take the same batch
def claim_next_batch(
    manifest: pd.DataFrame,
    state: dict,
    start_batch: int | None,
    end_batch: int | None,
    statuses: tuple[str, ...] = ("pending", "failed"),
):

    with manifest_lock:
        if state["left"] <= 0:
            return None

        mask = (
            manifest["status"].isin(list(statuses))
            & ~manifest["batch_id"].isin(state["attempted"])
        )
        if start_batch is not None:
            mask &= manifest["batch_id"] >= start_batch
        if end_batch is not None:
            mask &= manifest["batch_id"] <= end_batch

        candidates = manifest[mask]
        if candidates.empty:
            return None

        row = candidates.iloc[0]
        batch_id = int(row["batch_id"])

        state["left"] -= 1
        # a batch that failed in this run is not retried in the same run
        state["attempted"].add(batch_id)

        manifest.loc[
            manifest["batch_id"] == batch_id,
            "status",
        ] = "running"
        save_manifest(manifest)

        return batch_id, int(row["chunk_count"])


# Execute one Kedro pipeline. The output goes to a log file per batch/pipeline
def run_pipeline(
    pipeline_name: str,
    parameter_namespace: str,
    batch_id: int,
    extra_params: dict | None = None,
    log_suffix: str = "",
) -> int:

    params = {"batch_id": batch_id, **(extra_params or {})}
    command = [
        "uv",
        "run",
        "--no-sync",
        "kedro",
        "run",
        "--pipelines",
        pipeline_name,
        "--params",
        ",".join(f"{parameter_namespace}.{k}={v}" for k, v in params.items()),
    ]

    log_path = (
        Path("logs/batches")
        / f"batch_{batch_id:05d}_{pipeline_name}{log_suffix}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[batch {batch_id}] start {pipeline_name}",
        flush=True,
    )

    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    print(
        f"[batch {batch_id}] end {pipeline_name} "
        f"(exit {result.returncode})",
        flush=True,
    )

    return result.returncode


"""
Check that ontology review produced a valid result for the batch.

The review pipeline can finish with exit code 0 even when the LLM
review itself failed. Therefore the persistent output must also
be inspected.
"""
def validate_review_output(
    batch_id: int,
    expected_count: int,
    review_round: int = 0,
) -> tuple[bool, int]:
    """Returns (valid, number_of_approved_chunks) for this batch and round."""

    group_id = batch_id // 1000

    output_path = (
        Path("data/04_feature/ontology_reviews")
        / f"part_{group_id:03d}.csv"
    )

    if not output_path.exists():
        print(
            f"Review output not found: {output_path}"
        )
        return False, 0

    reviews = pd.read_csv(
        output_path,
        encoding="utf-8",
    )
    if "round" not in reviews.columns:
        reviews["round"] = 0

    batch_reviews = reviews[
        (reviews["batch_id"] == batch_id)
        & (reviews["round"].fillna(0).astype(int) == review_round)
    ]

    if len(batch_reviews) != expected_count:
        print(
            f"Review output (round {review_round}) contains {len(batch_reviews)} rows, "
            f"expected {expected_count}."
        )
        return False, 0

    failed_reviews = batch_reviews[
        batch_reviews["status"].isin(
            ["error"]
        )
    ]

    # 'invalid_ontology' does not stop the batch (the revision repairs it), but is logged
    log_chunk_issues([
        {"batch_id": batch_id, "round": review_round, "step": "review",
         "chunk_id": r["chunk_id"], "CELEX": r["CELEX"], "status": r["status"],
         "detail": _value(r, "error_reason")}
        for _, r in batch_reviews[batch_reviews["status"].isin(["error", "invalid_ontology"])].iterrows()
    ])

    if not failed_reviews.empty:
        print(
            f"Review failed for "
            f"{len(failed_reviews)} chunk(s)."
        )
        return False, 0

    approved = int(
        batch_reviews["approved"].astype(str).str.strip().str.lower().eq("true").sum()
    )
    return True, approved


"""
Check that ontology revision produced a result for every chunk of the batch.

A revision fallback (cut-off or invalid answer: the previous ontology is
kept) is a failure of that chunk only. It is logged to ISSUES_PATH and the
batch carries on, so the other chunks still get their review rounds. Only
missing output fails the batch.
"""
def validate_revision_output(
    batch_id: int,
    expected_count: int,
    revision_round: int = 1,
) -> bool:

    group_id = batch_id // 1000

    output_path = (
        Path("data/04_feature/ontology_revisions")
        / f"part_{group_id:03d}.csv"
    )

    if not output_path.exists():
        print(
            f"Revision output not found: {output_path}"
        )
        return False

    revisions = pd.read_csv(
        output_path,
        encoding="utf-8",
    )

    batch_revisions = revisions[
        revisions["batch_id"] == batch_id
    ]

    if len(batch_revisions) != expected_count:
        print(
            f"Revision output contains "
            f"{len(batch_revisions)} rows, "
            f"expected {expected_count}."
        )
        return False

    failed_revisions = batch_revisions[
        batch_revisions["revision_status"].astype(str).str.startswith(
            "failed_"
        )
    ]

    if not failed_revisions.empty:
        print(
            f"[batch {batch_id}] round {revision_round}: revision failed for "
            f"{len(failed_revisions)} chunk(s), previous version kept: "
            f"{', '.join(failed_revisions['chunk_id'].astype(str))}",
            flush=True,
        )
        log_chunk_issues([
            {"batch_id": batch_id, "round": revision_round, "step": "revision",
             "chunk_id": r["chunk_id"], "CELEX": r["CELEX"], "status": r["revision_status"],
             "done_reason": _value(r, "done_reason"), "output_tokens": _value(r, "output_tokens")}
            for _, r in failed_revisions.iterrows()
        ])

    return True


# Execute the complete ontology-processing pipeline for one batch:
#   generate -> review (round 0) -> revise (round 1) -> review (round 1)
#   -> [revise (round 2) -> review (round 2)] ... up to `review_rounds`.
# A revision is only trusted after it has been reviewed, so the loop always
# ends on a review. It stops early once every chunk of the batch is approved.
def run_batch(
    batch_id: int,
    expected_count: int,
    review_rounds: int = 2,
    from_round: int = 0,
) -> bool:

    if from_round == 0:
        # Step 1: Generate candidate ontologies.
        if run_pipeline("ontology_generation", "ontology_generation", batch_id) != 0:
            print(f"Ontology generation failed for batch {batch_id}.")
            return False
        bad = check_generation_output(batch_id)
        if bad:
            print(f"[batch {batch_id}] generation: cut-off or invalid output for {', '.join(bad)}", flush=True)

        # Step 2: Review the generator's candidates (round 0).
        if run_pipeline("ontology_review", "ontology_review", batch_id,
                        {"round": 0, "source": "candidates"}, "_r0") != 0:
            print(f"Ontology review pipeline failed for batch {batch_id}.")
            return False
    else:
        # Resume: keep the candidates and round-0 reviews already on disk,
        # drop every later review round so none of them is left stale
        purge_review_rounds(batch_id, from_round)
        print(f"[batch {batch_id}] resuming at revision round {from_round}", flush=True)
        check_generation_output(batch_id)   # log the kept candidates' issues in this run too

    # The review pipeline may exit successfully even if the LLM judge
    # returned an error for one or more chunks.
    valid, approved = validate_review_output(batch_id, expected_count, 0)
    if not valid:
        print(f"Ontology review validation failed for batch {batch_id}.")
        return False
    print(f"[batch {batch_id}] round 0: {approved}/{expected_count} approved", flush=True)

    # Steps 3+: revise, then re-review the revision. Even when everything is
    # approved in round 0 one revision pass runs, so that every chunk goes
    # through the deterministic normalisation before the merge.
    for review_round in range(max(from_round, 1), max(review_rounds, 1) + 1):
        if run_pipeline("ontology_revision", "ontology_revision", batch_id,
                        {"round": review_round}, f"_r{review_round}") != 0:
            print(f"Ontology revision pipeline failed for batch {batch_id}.")
            return False

        # The revision pipeline can keep the original ontology when the LLM
        # output is truncated or invalid; that is logged per chunk, and only
        # missing output stops the batch.
        if not validate_revision_output(batch_id=batch_id, expected_count=expected_count,
                                        revision_round=review_round):
            print(f"Ontology revision validation failed for batch {batch_id}.")
            return False

        if run_pipeline("ontology_review", "ontology_review", batch_id,
                        {"round": review_round, "source": "revisions"},
                        f"_r{review_round}") != 0:
            print(f"Ontology re-review failed for batch {batch_id} (round {review_round}).")
            return False

        valid, approved = validate_review_output(batch_id, expected_count, review_round)
        if not valid:
            print(f"Ontology re-review validation failed for batch {batch_id}.")
            return False
        print(f"[batch {batch_id}] round {review_round}: {approved}/{expected_count} approved",
              flush=True)
        if approved == expected_count:
            break

    return True


"""
Convert interrupted 'running' batches back to 'pending'.

This allows the script to resume after an unexpected interruption.
"""
def recover_interrupted_batches(
    manifest: pd.DataFrame,
) -> pd.DataFrame:    

    running_mask = manifest["status"] == "running"

    if running_mask.any():
        recovered_count = int(running_mask.sum())

        print(
            f"Recovering {recovered_count} interrupted batch(es)."
        )

        manifest.loc[
            running_mask,
            "status",
        ] = "pending"

        save_manifest(manifest)

    return manifest


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run pending ontology-processing batches "
            "with checkpoint support."
        )
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Maximum number of batches to process in this execution.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of batches processed at the same time.",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=None,
        help="Only process batches with batch_id >= this value.",
    )
    parser.add_argument(
        "--review-rounds",
        type=int,
        default=2,
        help="Maximum revise -> re-review rounds per batch (the loop stops early "
             "when every chunk is approved).",
    )
    parser.add_argument(
        "--end-batch",
        type=int,
        default=None,
        help="Only process batches with batch_id <= this value.",
    )
    parser.add_argument(
        "--from-round",
        type=int,
        default=0,
        help="Resume at this revision round (>= 1), keeping the candidates and round-0 "
             "reviews on disk; later review rounds are discarded and redone. Completed "
             "batches in the range are processed again.",
    )

    return parser.parse_args()


# Process pending ontology batches sequentially
def main() -> None:

    args = parse_arguments()

    if args.max_batches <= 0 or args.parallel <= 0:
        raise ValueError(
            "--max-batches and --parallel must be greater than zero."
        )

    manifest = load_manifest()
    manifest = recover_interrupted_batches(manifest)

    state = {
        "left": args.max_batches,
        "attempted": set(),
        "done": [],
        "failed": [],
    }

    def worker() -> None:
        while True:
            claimed = claim_next_batch(
                manifest, state, args.start_batch, args.end_batch,
                ("pending", "failed", "completed") if args.from_round else ("pending", "failed"),
            )
            if claimed is None:
                return

            batch_id, expected_count = claimed

            try:
                ok = run_batch(
                    batch_id=batch_id,
                    expected_count=expected_count,
                    review_rounds=args.review_rounds,
                    from_round=args.from_round,
                )
            except Exception as exc:
                print(f"[batch {batch_id}] exception: {exc}", flush=True)
                ok = False

            update_status(
                manifest=manifest,
                batch_id=batch_id,
                status="completed" if ok else "failed",
            )

            with manifest_lock:
                state["done" if ok else "failed"].append(batch_id)

            print(
                f"[batch {batch_id}] "
                f"{'completed' if ok else 'FAILED'}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(worker) for _ in range(args.parallel)]
        for future in futures:
            future.result()

    print()
    print(f"Completed: {sorted(state['done'])}")
    print(f"Failed:    {sorted(state['failed'])}")

    sys.exit(1 if state["failed"] else 0)


if __name__ == "__main__":
    main()