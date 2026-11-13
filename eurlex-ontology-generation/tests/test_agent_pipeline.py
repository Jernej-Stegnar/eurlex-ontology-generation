"""Agent mode end to end with a scripted model: Kedro node -> the three datasets ->
merge loader, runner output check, and the columns the frozen T3 scorer reads."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

from eurlex_ontology_generation.pipelines.ontology_agent.nodes import agent_extract_batch, load_acts
from eurlex_ontology_generation.pipelines.ontology_generation.nodes import select_chunk_batch
from eurlex_ontology_generation.pipelines.ontology_merge_partition.nodes import load_partition_revisions
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import extract_entities

from .test_agent_extraction import CHUNKS, FakeLLM, onto, submit

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path):
    """Two partitions of 3 rows; the act 32099L0001 spans both (rows 2-4); batch 1 = rows 3-4."""
    other = {"CELEX": "32099R0002", "Act_name": "Other", "Act_type": "Regulation", "Subject_matter": "",
             "Date_document": "2099-02-02", "chunk_id": "32099R0002_1", "chunk_number": 1, "chunk_text": "Other act."}
    act = [{"CELEX": "32099L0001", "Act_name": "Test Directive", "Act_type": "Directive", "Subject_matter": "",
            "Date_document": "2099-01-01", **c} for c in CHUNKS]
    tail = {**other, "chunk_id": "32099R0002_2", "chunk_number": 2}
    parts = [[other, other | {"chunk_id": "32099R0002_0", "chunk_number": 0}, act[0]], [act[1], act[2], tail]]
    parts[0] = [parts[0][1], parts[0][0], parts[0][2]]
    manifest = []
    for i, rows in enumerate(parts):
        p = tmp_path / f"chunks_{i}.csv"
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
        manifest.append({"part_id": i, "file_path": str(p), "start_row": 3 * i, "end_row": 3 * i + 2})
    batches = pd.DataFrame([{"batch_id": 1, "start_row": 3, "end_row": 4, "chunk_count": 2}])
    return pd.DataFrame(manifest), batches


def _agent_params(tmp_path):
    d = tmp_path / "data"
    return {"model": "fake", "llm_backend": {}, "temperature": 0.1, "seed": 42, "max_steps": 6,
            "max_tokens": 1000, "max_submits": 3, "skills_dir": str(ROOT / "agent_skills"),
            "memory_dir": str(d / "agent_memory"), "traces_dir": str(d / "agent_traces"),
            "candidates_directory": str(d / "ontology_candidates"),
            "revisions_directory": str(d / "ontology_revisions"),
            "reviews_directory": str(d / "ontology_reviews")}


def _run(tmp_path, llm):
    parts, batches = _project(tmp_path)
    batch = select_chunk_batch(batches, 1, 1, parts)
    params = _agent_params(tmp_path)
    summary = agent_extract_batch(batch, parts, 1, 1, params, "SYSTEM {core_types}", "TEXT:\n{text}", None, llm=llm)
    return summary, params, batch


def test_load_acts_reads_the_whole_act_across_partitions(tmp_path):
    parts, batches = _project(tmp_path)
    batch = select_chunk_batch(batches, 1, 1, parts)
    assert list(batch["chunk_id"]) == ["32099L0001_2", "32099L0001_3"]
    acts = load_acts(batch, parts, 1, 1)
    assert [c["chunk_id"] for c in acts["32099L0001"].chunks] == ["32099L0001_1", "32099L0001_2", "32099L0001_3"]
    assert acts["32099L0001"].act_type == "Directive"


def test_agent_batch_writes_what_merge_runner_and_scorer_read(tmp_path, monkeypatch):
    def answer(messages, tool_choice):
        if "chunk 3 of 3" in messages[1]["content"]:
            return submit({"broken": True})            # never valid -> failed chunk
        return submit(onto())
    summary, params, batch = _run(tmp_path, FakeLLM([], default=answer))
    assert summary["chunks"] == 2 and summary["statuses"] == {"accepted": 1, "failed_no_valid_submission": 1}

    d = Path(params["candidates_directory"]).parent
    cand = pd.read_csv(d / "ontology_candidates" / "part_001.csv")
    revs = pd.read_csv(d / "ontology_revisions" / "part_001.csv")
    revw = pd.read_csv(d / "ontology_reviews" / "part_001.csv")
    assert len(cand) == len(revs) == len(revw) == 2

    # scorer (eval_gold/extract.py, locate.py): candidate ontology is JSON; revisions are not "revised",
    # so the scorer reads the candidate; reviews carry the chunk text it locates gold quotes in
    ok = cand.set_index("chunk_id").loc["32099L0001_2"]
    assert json.loads(ok["ontology"])["rights_model"]["duties"][0]["action_label"] == "notify_measures"
    assert ok["agent_status"] == "accepted" and ok["library_hash"]
    assert not (revs["revision_status"] == "revised").any()
    assert set(revw["text"]) == set(batch["chunk_text"])
    failed = cand.set_index("chunk_id").loc["32099L0001_3"]
    assert json.loads(failed["ontology"])["error"].startswith("agent failed")

    # merge: approved only for the accepted chunk; entity extraction reads the normalized revisions
    loaded = load_partition_revisions(1, str(d / "ontology_revisions"), str(d / "ontology_reviews"))
    assert dict(zip(loaded["chunk_id"], loaded["approved"])) == {"32099L0001_2": True, "32099L0001_3": False}
    assert len(extract_entities(loaded)) > 0

    # runner: output complete, the failed chunk logged in chunk_issues.csv
    spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts" / "run_ontology_batches.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "04_feature").mkdir(parents=True)
    for name in ("ontology_candidates", "ontology_revisions", "ontology_reviews"):
        (d / name).rename(tmp_path / "data" / "04_feature" / name)
        # the runner assumes 1000 batches per partition (batch 1 -> part_000)
        (tmp_path / "data" / "04_feature" / name / "part_001.csv").rename(
            tmp_path / "data" / "04_feature" / name / "part_000.csv")
    assert runner.check_agent_output(1, 2) is True
    issues = pd.read_csv(tmp_path / "logs" / "chunk_issues.csv")
    assert list(issues["chunk_id"]) == ["32099L0001_3"] and issues["step"][0] == "agent"
    assert runner.check_agent_output(1, 3) is False            # a missing chunk fails the batch


def test_rerun_of_a_batch_replaces_rows(tmp_path):
    _run(tmp_path, FakeLLM([], default=submit(onto())))
    _run(tmp_path, FakeLLM([], default=submit(onto())))
    d = tmp_path / "data"
    for name in ("ontology_candidates", "ontology_revisions", "ontology_reviews"):
        assert len(pd.read_csv(d / name / "part_001.csv")) == 2
