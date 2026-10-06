from pathlib import Path
import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


MANIFEST_PATH = Path(
    "data/03_primary/chunk_batch_manifest.csv"
)

# Protects the manifest when several batches run at the same time
manifest_lock = threading.Lock()


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
):

    with manifest_lock:
        if state["left"] <= 0:
            return None

        mask = (
            manifest["status"].isin(["pending", "failed"])
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
Check that ontology revision produced a valid result for the batch.

A revision fallback means that the LLM revision failed, so the
batch must not be marked as completed.
"""
def validate_revision_output(
    batch_id: int,
    expected_count: int,
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
            f"Revision failed for "
            f"{len(failed_revisions)} chunk(s)."
        )
        return False

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
) -> bool:

    # Step 1: Generate candidate ontologies.
    if run_pipeline("ontology_generation", "ontology_generation", batch_id) != 0:
        print(f"Ontology generation failed for batch {batch_id}.")
        return False

    # Step 2: Review the generator's candidates (round 0).
    if run_pipeline("ontology_review", "ontology_review", batch_id,
                    {"round": 0, "source": "candidates"}, "_r0") != 0:
        print(f"Ontology review pipeline failed for batch {batch_id}.")
        return False

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
    for review_round in range(1, max(review_rounds, 1) + 1):
        if run_pipeline("ontology_revision", "ontology_revision", batch_id,
                        {"round": review_round}, f"_r{review_round}") != 0:
            print(f"Ontology revision pipeline failed for batch {batch_id}.")
            return False

        # The revision pipeline can keep the original ontology when
        # the LLM output is truncated or invalid. Detect that explicitly.
        if not validate_revision_output(batch_id=batch_id, expected_count=expected_count):
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
                manifest, state, args.start_batch, args.end_batch
            )
            if claimed is None:
                return

            batch_id, expected_count = claimed

            try:
                ok = run_batch(
                    batch_id=batch_id,
                    expected_count=expected_count,
                    review_rounds=args.review_rounds,
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