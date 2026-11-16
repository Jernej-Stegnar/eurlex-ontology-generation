"""Recital rules are dropped after generation and revision (T4 fix 3)."""
import json
from types import SimpleNamespace

import pandas as pd

from eurlex_ontology_generation.pipelines.ontology_generation import nodes as gen
from eurlex_ontology_generation.utils.ontology_normalize import drop_recital_rules


def _onto():
    return {"rights_model": {
        "permissions": [{"action": "use", "party": "A", "provision": "article"}],
        "prohibitions": [{"action": "sell", "party": "B", "provision": "Recital "}],
        "duties": [{"action": "notify", "party": "C", "provision": "recital"},
                   {"action": "report", "party": "D"}],
    }}


def test_drops_only_recital_rules():
    onto = _onto()
    out, n = drop_recital_rules(onto)
    assert n == 2
    assert out["rights_model"]["prohibitions"] == []
    assert [r["action"] for r in out["rights_model"]["duties"]] == ["report"]
    assert out["rights_model"]["permissions"] == onto["rights_model"]["permissions"]
    assert len(onto["rights_model"]["duties"]) == 2  # input not mutated


def test_no_recitals_or_bad_input_unchanged():
    onto = {"rights_model": {"duties": [{"action": "x", "party": "y", "provision": "annex"}]}}
    assert drop_recital_rules(onto) == (onto, 0)
    assert drop_recital_rules(None) == (None, 0)
    assert drop_recital_rules({"rights_model": "oops"}) == ({"rights_model": "oops"}, 0)


def test_generation_node_applies_flag(monkeypatch):
    reply = SimpleNamespace(content=json.dumps(_onto()), done_reason="stop", prompt_tokens=1, output_tokens=1)
    monkeypatch.setattr(gen, "llm_chat", lambda *a, **k: reply)
    monkeypatch.setattr(gen, "render_prompt", lambda tpl, *a, **k: tpl)
    chunks = pd.DataFrame([{"chunk_id": "c1", "CELEX": "X", "chunk_text": "t"}])
    args = (chunks, 0, "m", 0.0, "sys", "user {text}", True, {"provider": "ollama", "max_workers": 1})
    on = gen.generate_partial_ontologies(*args, drop_recitals=True).iloc[0]
    off = gen.generate_partial_ontologies(*args, drop_recitals=False).iloc[0]
    assert on.recital_rules_dropped == 2 and len(json.loads(on.ontology)["rights_model"]["duties"]) == 1
    assert off.recital_rules_dropped == 0 and len(json.loads(off.ontology)["rights_model"]["duties"]) == 2


def test_normalizer_keeps_model_action_object():
    from eurlex_ontology_generation.utils.ontology_normalize import normalize_chunk_ontology
    onto = {"classes": [{"name": "Vehicle", "type": "Asset", "is_a": "Resource", "kind": "class"},
                        {"name": "MemberState", "type": "Party", "is_a": "PublicAuthority", "kind": "class"}],
            "rights_model": {"duties": [
                {"action": "register", "action_label": "register_vehicles", "party": "MemberState", "asset": "Vehicle",
                 "action_object": "vehicles registered before 1 January 2010 with a maximum mass above 3.5 tonnes"},
                {"action": "register", "action_label": "register_vehicles", "party": "MemberState", "asset": "Vehicle"}]}}
    out, _ = normalize_chunk_ontology(onto)
    kept, derived = out["rights_model"]["duties"]
    assert kept["action_object"].startswith("vehicles registered before 1 January 2010")
    from eurlex_ontology_generation.utils.vocabulary import load_vocabulary
    assert derived["action_object"] == load_vocabulary().normalize_action("register", "register_vehicles")["action_object"]


def test_sentence_objects_are_dropped():
    from eurlex_ontology_generation.utils.ontology_normalize import (
        clean_action_objects, is_sentence_object, normalize_chunk_ontology)
    assert is_sentence_object("This Decision shall be brought to the attention of staff")
    assert is_sentence_object("operators MUST keep records")
    assert not is_sentence_object("vehicles registered before 1 January 2010")
    onto = {"rights_model": {"duties": [
        {"action": "notify", "party": "MemberState", "action_object": "the Member State shall notify the measures"},
        {"action": "notify", "party": "MemberState", "action_object": "the measures adopted"}]}}
    out, n = clean_action_objects(onto)
    assert n == 1 and "action_object" not in out["rights_model"]["duties"][0]
    assert out["rights_model"]["duties"][1]["action_object"] == "the measures adopted"
    norm, changes = normalize_chunk_ontology(onto)
    assert any(c["code"] == "action_object_sentence_dropped" for c in changes)
    assert not is_sentence_object(norm["rights_model"]["duties"][0]["action_object"])
