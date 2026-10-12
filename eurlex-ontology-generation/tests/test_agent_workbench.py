"""Tests for the curation workbench, the MCP server wrapper and scripts/curation.py."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eurlex_ontology_generation.agent.workbench import Decision, Workbench, load_parameters

MERGED = {
    "ontology_name": "M", "description": "",
    "classes": [
        {"name": "European_Commission", "type": "Party", "description": "", "sources": ["32018D1859"],
         "support": {"chunks": 5, "documents": 4}},
        {"name": "Application", "type": "Legal_extension", "description": "an application",
         "sources": ["32018D1859", "32019D0001"], "support": {"chunks": 2, "documents": 2}},
        {"name": "Latvia", "type": "Legal_extension", "description": "", "sources": ["32018D1859"],
         "support": {"chunks": 1, "documents": 1}},
    ],
    "relationships": [],
    "rights_model": {
        "permissions": [],
        "duties": [{"action": "submit_application", "party": "Latvia", "asset": "Application",
                    "constraints": [], "sources": ["32018D1859"], "support": {"chunks": 1}}],
        "prohibitions": [],
    },
    "legal_patterns": [], "example_instances": [],
}


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    base, local = tmp_path / "conf/base", tmp_path / "conf/local"
    base.mkdir(parents=True)
    local.mkdir(parents=True)
    (base / "globals.yml").write_text(yaml.safe_dump({"llm_backends": {
        "local": {"provider": "ollama"},
        "remote": {"provider": "openai_compatible", "base_url": "http://x/v1", "api_key_env": "K"}}}))
    (base / "parameters_ontology_refinement.yml").write_text(yaml.safe_dump({"ontology_refinement": {
        "input_path": "data/06_models/merged.json",
        "curation_path": "data/05_model_input/curation/decisions.jsonl",
        "min_documents_for_class": 3}}))
    (base / "parameters_ontology_generation.yml").write_text(yaml.safe_dump({"ontology_generation": {
        "model": "m", "llm_backend": "${globals:llm_backends.local}"}}))
    (base / "parameters_ontology_export.yml").write_text(yaml.safe_dump({"ontology_export": {
        "base_iri": "https://w3id.org/eurlex-odrl/"}}))
    (local / "parameters_ontology_generation.yml").write_text(yaml.safe_dump({"ontology_generation": {
        "llm_backend": "${globals:llm_backends.remote}"}}))
    d = tmp_path / "data/06_models"
    d.mkdir(parents=True)
    (d / "merged.json").write_text(json.dumps(MERGED))
    return tmp_path


def test_load_parameters_resolves_globals_and_local_overrides(root):
    p = load_parameters(root)
    assert p["ontology_generation"]["llm_backend"]["provider"] == "openai_compatible"
    assert p["ontology_generation"]["model"] == "m"                # base key kept
    assert load_parameters(root, env="none")["ontology_generation"]["llm_backend"] == {"provider": "ollama"}


def test_decision_requires_fields_for_its_type():
    with pytest.raises(ValidationError, match="needs"):
        Decision(type="merge_entities", entities=["A"], rationale="because the text says so")
    with pytest.raises(ValidationError):
        Decision(type="remove_rule", rule_id="duty_x", rationale="short")     # rationale too short
    with pytest.raises(ValidationError, match="at least one"):
        Decision(type="set_typing", entity="A", rationale="because the text says so")


def test_unknown_references_are_reported(root):
    wb = Workbench(root)
    r = wb.validate_decision({"type": "set_typing", "entity": "Nope", "is_a": "NotAClass",
                              "rationale": "made up for the test"})
    assert not r["ok"] and any("unknown entity" in p for p in r["problems"]) \
        and any("is_a must be" in p for p in r["problems"])


def test_propose_validates_and_stores_pending(root):
    wb = Workbench(root)
    rid = wb.get_entity("Application")["rules"][0]["id"]
    out = wb.propose_decision({"type": "remove_rule", "rule_id": rid, "item_id": "q1",
                               "rationale": "The duty comes from a recital, not an article.",
                               "evidence": ["32018D1859"]}, proposer="test", model="m")
    assert out["status"] == "pending" and out["validation"]["ok"]
    assert out["validation"]["counts_after"] != out["validation"]["counts_before"]
    [row] = wb.decisions()
    assert row["proposer"] == "test" and row["rule_id"] == rid
    # Pending proposals change nothing until a person accepts them.
    assert wb.decisions("accepted") == []
    accepted = wb.set_status(row["id"], "accepted", reviewer="person")
    assert accepted["status"] == "accepted" and accepted["reviewer"] == "person"
    assert len(wb.decisions("accepted")) == 1


def test_failed_validation_is_stored_and_cannot_be_accepted(root, monkeypatch):
    wb = Workbench(root)
    real = wb._evaluate
    calls = []

    def fake(curation, reasoner):
        out = real(curation, reasoner)
        if calls:                                  # the "after" evaluation adds a violation
            out["shacl"] = {**out["shacl"], "Violation": out["shacl"].get("Violation", 0) + 1}
        calls.append(1)
        return out

    monkeypatch.setattr(wb, "_evaluate", fake)
    out = wb.propose_decision({"type": "set_typing", "entity": "Latvia", "is_a": "Country", "kind": "individual",
                               "rationale": "Latvia is a Member State named in the act."}, proposer="test")
    assert out["status"] == "rejected_by_validation" and out["validation"]["new_violations"] == 1
    with pytest.raises(ValueError, match="cannot be accepted"):
        wb.set_status(out["id"], "accepted", reviewer="person")


def test_review_queue_hides_decided_items(root):
    wb = Workbench(root)
    q = wb.review_queue()
    assert q["total"] > 0
    item = q["items"][0]
    wb.curation_path.parent.mkdir(parents=True, exist_ok=True)
    wb.curation_path.write_text(json.dumps({"id": "dec_1", "type": "promote_class", "entity": "Application",
                                            "status": "pending", "item_id": item["id"]}) + "\n")
    assert item["id"] not in {i["id"] for i in wb.review_queue(limit=1000)["items"]}
    assert item["id"] in {i["id"] for i in wb.review_queue(limit=1000, include_decided=True)["items"]}


def test_mcp_server_exposes_proposals_but_not_acceptance(root):
    pytest.importorskip("mcp")
    from eurlex_ontology_generation.agent.mcp_server import build_server

    server = build_server(root, proposer="test-agent")
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"review_queue", "get_entity", "get_rule", "propose_decision", "validate_decision",
            "list_decisions", "extract_chunk"} <= names
    assert not any("accept" in n or "status" in n for n in names)


def test_curation_script_accepts(root, capsys):
    spec = importlib.util.spec_from_file_location("curation", Path(__file__).parents[1] / "scripts/curation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wb = Workbench(root)
    out = wb.propose_decision({"type": "promote_class", "entity": "Application",
                               "rationale": "Applications are a recurring procedure in EGF acts."}, proposer="t")
    assert mod.main(["--root", str(root), "list"]) == 0
    assert out["id"] in capsys.readouterr().out
    assert mod.main(["--root", str(root), "accept", out["id"], "--reviewer", "jane"]) == 0
    assert Workbench(root).decisions("accepted")[0]["reviewer"] == "jane"
