"""T13 F3: LLM pair judge in the partition merge (LLM faked)."""
import json

import pandas as pd

from eurlex_ontology_generation.utils.llm_client import LLMResult
from eurlex_ontology_generation.utils.ontology_merge_common import pair_judge as pj
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    canonicalize_clusters,
    extract_entities,
)

TEXT = ("Food business operators shall ensure that the water is potable. "
        "Operators shall keep records of the checks. Croatia and Spain shall report.")


def _revisions():
    onto = {"classes": [
        {"name": "FoodBusinessOperator", "type": "Party", "description": "Food business operators running the supply."},
        {"name": "Operator", "type": "Party", "description": "Food business operators that must keep records."},
        {"name": "Croatia", "type": "Party", "description": "Member State Croatia."},
        {"name": "Spain", "type": "Party", "description": "Member State Spain."},
    ], "rights_model": {"duties": [{"party": "Operator", "action": "keep", "asset": "Record"}]}}
    return pd.DataFrame([{"chunk_id": "32015R1474_0", "CELEX": "32015R1474", "text": TEXT,
                          "ontology": json.dumps(onto)}])


def _clusters(revisions):
    ents = extract_entities(revisions)
    ents = ents.reset_index(drop=True)
    ents["key"] = ents["name"].str.lower()
    ents["cluster_id"] = range(len(ents))
    return ents


def _n_classes(ents):
    return ents.loc[ents["entity_type"] == "class", "cluster_id"].nunique()


def _fake_llm(calls):
    def fake(backend, model, system, user, temperature, **kw):
        calls.append(json.loads(user))
        out = []
        for p in json.loads(user)["pairs"]:
            names = {p["A"]["name"], p["B"]["name"]}
            if names == {"FoodBusinessOperator", "Operator"}:
                out.append({"id": p["id"], "label": "SAME", "confidence": "high", "basis": "redundant_modifier",
                            "evidence": "Food business operators shall ensure that the water is potable"})
            else:
                out.append({"id": p["id"], "label": "SIBLING", "confidence": "high"})
        return LLMResult(json.dumps({"verdicts": out}), "stop", 100, 20)
    return fake


def _cfg(tmp_path):
    return {"enabled": True, "llm_backend": {"provider": "openai_compatible", "max_workers": 1},
            "cache_path": str(tmp_path / "cache.jsonl"), "desc_sim": 0.3}


def test_disabled_is_a_no_op():
    rev = _revisions()
    ents = _clusters(rev)
    out, report = pj.judge_and_join(ents, rev, {"enabled": False})
    assert out.equals(ents) and report == {"enabled": False}


def test_same_pairs_are_joined_and_siblings_kept_apart(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pj, "llm_chat", _fake_llm(calls))
    rev = _revisions()
    out, report = pj.judge_and_join(_clusters(rev), rev, _cfg(tmp_path))
    cmap, _ = canonicalize_clusters(out)
    assert cmap["class"]["Operator"] == cmap["class"]["FoodBusinessOperator"]
    assert cmap["class"]["Croatia"] != cmap["class"]["Spain"]
    assert report["joined_pairs"] == 1 and report["labels"]["SAME"] == 1
    # the judge saw the act sentence and the rule role of Operator
    pair = next(p for c in calls for p in c["pairs"] if {p["A"]["name"], p["B"]["name"]} == {"FoodBusinessOperator", "Operator"})
    assert "Operators shall keep records" in json.dumps(pair) or "Food business operators" in json.dumps(pair)
    assert any("keep" in f for side in ("A", "B") for f in pair[side]["facts"])


def test_cache_makes_reruns_free(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pj, "llm_chat", _fake_llm(calls))
    rev = _revisions()
    pj.judge_and_join(_clusters(rev), rev, _cfg(tmp_path))
    n = len(calls)
    _, report = pj.judge_and_join(_clusters(rev), rev, _cfg(tmp_path))
    assert len(calls) == n and report["usage"]["calls"] == 0 and report["joined_pairs"] == 1


def test_low_confidence_same_is_not_joined(tmp_path, monkeypatch):
    def fake(backend, model, system, user, temperature, **kw):
        ps = json.loads(user)["pairs"]
        return LLMResult(json.dumps({"verdicts": [{"id": p["id"], "label": "SAME", "confidence": "low"} for p in ps]}),
                         "stop", 1, 1)
    monkeypatch.setattr(pj, "llm_chat", fake)
    rev = _revisions()
    out, report = pj.judge_and_join(_clusters(rev), rev, _cfg(tmp_path))
    assert _n_classes(out) == 4 and report["joined_pairs"] == 0


def test_same_without_evidence_is_joined_only_for_form_variants():
    shown = {"A": "Food business operators shall ensure that the water is potable."}
    same = {"label": "SAME", "confidence": "high"}
    assert not pj.accepted({**same, "basis": "synonym", "evidence": ""}, shown, 1)
    assert not pj.accepted({**same, "basis": "synonym", "evidence": "operators must register every vessel"}, shown, 1)
    assert pj.accepted({**same, "basis": "synonym", "evidence": "Food business operators shall ensure"}, shown, 1)
    assert pj.accepted({**same, "basis": "acronym", "evidence": ""}, shown, 1)
    assert not pj.accepted({**same, "basis": "acronym", "confidence": "low"}, shown, 1)


def test_failed_call_leaves_pairs_apart(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("API down")
    monkeypatch.setattr(pj, "llm_chat", boom)
    rev = _revisions()
    out, report = pj.judge_and_join(_clusters(rev), rev, _cfg(tmp_path))
    assert _n_classes(out) == 4 and report["usage"]["unparsed"] == report["pairs"]


def test_blocking_finds_acronyms_and_word_variants():
    ents = [{"name": "CAA", "description": ""}, {"name": "CryptoApprovalAuthority", "description": ""},
            {"name": "TradingVenueRules", "description": ""}, {"name": "RulesOfTradingVenue", "description": ""},
            {"name": "Workplace", "description": "Place of work."}]
    found = {(ents[i]["name"], ents[j]["name"]) for i, j, _ in pj.candidate_pairs(ents, 0.5, 0.4)}
    assert ("CAA", "CryptoApprovalAuthority") in found
    assert ("TradingVenueRules", "RulesOfTradingVenue") in found
    assert not any("Workplace" in p for p in found)
