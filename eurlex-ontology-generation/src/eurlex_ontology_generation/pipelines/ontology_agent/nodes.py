"""Agent mode for one batch: the extraction agent replaces generate -> review -> revise.

Writes the same three datasets the pipeline writes, so the merge
(ontology_merge_partition reads revisions + latest review) and the frozen T3
scorer (candidates, revisions if revision_status == "revised", reviews.text)
work unchanged:

- ontology_candidates  the ontology as the agent submitted it (schema-valid),
                       plus agent columns (status, steps, submits, cache hits, ...)
- ontology_revisions   round 1, normalized ontology, revision_status
                       "agent_final" (so the scorer reads the candidate)
- ontology_reviews     round 1, chunk text, the last check findings,
                       approved = the agent's submission was accepted
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ...agent.extraction import AgentConfig, run_act
from ...agent.extraction.tools import Act
from ...utils.persist import append_dedup_csv

ACCEPTED = ("accepted", "accepted_with_warnings")


def _partition(partition_manifest: pd.DataFrame, part_id: int) -> pd.DataFrame:
    rows = partition_manifest[partition_manifest["part_id"] == part_id]
    if rows.empty or not Path(rows.iloc[0]["file_path"]).exists():
        return pd.DataFrame()
    return pd.read_csv(rows.iloc[0]["file_path"], encoding="utf-8")


def load_acts(batch: pd.DataFrame, partition_manifest: pd.DataFrame, batch_id: int,
              batch_group_size: int) -> dict[str, Act]:
    """Every chunk of every act in the batch (the agent may read the whole act),
    also when an act crosses into the neighbouring partition file."""
    part_id = batch_id // batch_group_size
    part = _partition(partition_manifest, part_id)
    acts = {}
    for celex in dict.fromkeys(batch["CELEX"]):
        rows = part[part["CELEX"] == celex] if len(part) else batch[batch["CELEX"] == celex]
        if len(part) and len(rows):
            if rows.index.min() == part.index.min():
                prev = _partition(partition_manifest, part_id - 1)
                rows = pd.concat([prev[prev["CELEX"] == celex], rows]) if len(prev) else rows
            if rows.index.max() == part.index.max():
                nxt = _partition(partition_manifest, part_id + 1)
                rows = pd.concat([rows, nxt[nxt["CELEX"] == celex]]) if len(nxt) else rows
        rows = rows.drop_duplicates("chunk_id").sort_values("chunk_number")
        first = rows.iloc[0]
        acts[celex] = Act(celex=celex, act_name=str(first.get("Act_name", "") or ""),
                          act_type=str(first.get("Act_type", "") or ""),
                          date=str(first.get("Date_document", "") or ""),
                          chunks=[{"chunk_id": r["chunk_id"], "chunk_number": int(r["chunk_number"]),
                                   "chunk_text": str(r["chunk_text"])} for _, r in rows.iterrows()])
    return acts


def agent_extract_batch(
    batch: pd.DataFrame,
    partition_manifest: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    agent: dict,
    system_prompt: str,
    user_prompt_template: str,
    vocabulary_path: str | None = None,
    llm=None,                       # tests inject a scripted model
) -> dict:
    cfg = AgentConfig(
        backend=agent["llm_backend"], model=agent["model"], system_prompt=system_prompt,
        user_prompt_template=user_prompt_template, temperature=agent.get("temperature", 0.1),
        seed=agent.get("seed"), max_steps=agent.get("max_steps", 12), max_tokens=agent.get("max_tokens", 8192),
        max_submits=agent.get("max_submits", 3), vocabulary_path=vocabulary_path,
        skills_dir=agent.get("skills_dir"), memory_dir=agent.get("memory_dir"),
        traces_dir=agent.get("traces_dir"), llm=llm)

    acts = load_acts(batch, partition_manifest, batch_id, batch_group_size)
    texts = dict(zip(batch["chunk_id"], batch["chunk_text"]))
    results = []
    for celex, act in acts.items():
        ids = [c for c in batch.loc[batch["CELEX"] == celex, "chunk_id"]]
        results += run_act(act, ids, cfg)

    cand, rev, revw = [], [], []
    for r in results:
        ok = r.status in ACCEPTED
        error = json.dumps({"error": f"agent {r.status}", "detail": r.detail}, ensure_ascii=False)
        submitted = json.dumps(r.ontology, ensure_ascii=False) if r.ontology is not None else error
        normalized = json.dumps(r.normalized, ensure_ascii=False) if r.normalized is not None else error
        common = {"batch_id": batch_id, "chunk_id": r.chunk_id, "CELEX": r.CELEX}
        cand.append({**common, "ontology": submitted, "done_reason": r.done_reason,
                     "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
                     "cache_hit_tokens": r.cache_hit_tokens, "agent_status": r.status, "agent_steps": r.steps,
                     "agent_submits": r.submits, "agent_tool_calls": r.tool_calls,
                     "agent_unparsable_calls": r.unparsable_calls, "agent_repaired_calls": r.repaired_calls,
                     "skills_loaded": ";".join(r.skills_loaded), "library_hash": r.library_hash,
                     "seconds": r.seconds, "detail": r.detail})
        rev.append({**common, "round": 1, "text": texts[r.chunk_id], "done_reason": r.done_reason,
                    "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens, "ontology": normalized,
                    "revised": False, "revision_status": "agent_final" if r.ontology is not None else "agent_failed",
                    "approved_in_round": 1 if ok else "",
                    "normalization_changes": json.dumps(r.normalization_changes, ensure_ascii=False)})
        revw.append({**common, "round": 1, "text": texts[r.chunk_id], "ontology": normalized,
                     "review": json.dumps({"agent_status": r.status, "findings": r.findings}, ensure_ascii=False),
                     "status": "agent", "error_reason": r.detail,
                     "automatic_checks": json.dumps(r.findings, ensure_ascii=False),
                     "approved": ok, "revision_required": False})

    part = f"part_{batch_id // batch_group_size:03d}.csv"
    append_dedup_csv(Path(agent["candidates_directory"]) / part, pd.DataFrame(cand))
    append_dedup_csv(Path(agent["revisions_directory"]) / part, pd.DataFrame(rev))
    append_dedup_csv(Path(agent["reviews_directory"]) / part, pd.DataFrame(revw), key=["chunk_id", "round"])
    statuses = pd.Series([r.status for r in results]).value_counts().to_dict()
    return {"batch_id": batch_id, "chunks": len(results), "statuses": statuses,
            "prompt_tokens": sum(r.prompt_tokens for r in results),
            "cache_hit_tokens": sum(r.cache_hit_tokens for r in results),
            "output_tokens": sum(r.output_tokens for r in results)}
