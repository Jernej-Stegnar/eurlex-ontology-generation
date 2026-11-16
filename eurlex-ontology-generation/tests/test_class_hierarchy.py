"""T13 step 4: narrower/broader class links from the pair judge, through merge, refinement and export."""
import json

import pandas as pd
from rdflib.namespace import RDFS

from eurlex_ontology_generation.pipelines.ontology_refinement.nodes import refine_ontology
from eurlex_ontology_generation.utils.llm_client import LLMResult
from eurlex_ontology_generation.utils.ontology_merge_common import pair_judge as pj
from eurlex_ontology_generation.utils.ontology_merge_common.assembly import assemble_merged_ontology
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    canonicalize_clusters,
    extract_entities,
)
from eurlex_ontology_generation.utils.rdf_export import export_rdf, namespaces
from eurlex_ontology_generation.utils.rdf_validate import validate_graph

ENT = namespaces()["ent"]
TEXT = ("Member States shall reduce the consumption of lightweight plastic carrier bags. "
        "Very lightweight plastic carrier bags may be excluded.")


def _revisions():
    onto = {"classes": [
        {"name": "LightweightPlasticCarrierBag", "type": "Asset", "description": "Bags below 50 microns."},
        {"name": "VeryLightweightPlasticCarrierBag", "type": "Asset", "description": "Bags below 15 microns."},
        {"name": "PlasticCarrierBag", "type": "Asset", "description": "Carrier bags made of plastic."},
    ], "rights_model": {"duties": [{"party": "MemberState", "action": "reduce",
                                    "asset": "LightweightPlasticCarrierBag"}]}}
    return pd.DataFrame([{"chunk_id": "32015L0720_0", "CELEX": "32015L0720", "text": TEXT,
                          "ontology": json.dumps(onto)}])


def _clusters(rev):
    ents = extract_entities(rev).reset_index(drop=True)
    ents["key"] = ents["name"].str.lower()
    ents["cluster_id"] = range(len(ents))
    return ents


def _fake(backend, model, system, user, temperature, **kw):
    out = []
    for p in json.loads(user)["pairs"]:
        a, b = p["A"]["name"], p["B"]["name"]
        broad = {"PlasticCarrierBag": 0, "LightweightPlasticCarrierBag": 1, "VeryLightweightPlasticCarrierBag": 2}
        if a in broad and b in broad:
            label = "A_BROADER" if broad[a] < broad[b] else "B_BROADER"
        else:
            label = "DIFFERENT"
        out.append({"id": p["id"], "label": label, "confidence": "high"})
    return LLMResult(json.dumps({"verdicts": out}), "stop", 10, 10)


def _cfg(tmp_path, **kw):
    return {"enabled": True, "join_same": False, "link_broader": True, "name_sim": 0.3,
            "llm_backend": {"provider": "openai_compatible", "max_workers": 1},
            "cache_path": str(tmp_path / "cache.jsonl"), **kw}


def test_broader_verdicts_become_links_not_joins(tmp_path, monkeypatch):
    monkeypatch.setattr(pj, "llm_chat", _fake)
    rev = _revisions()
    out, report = pj.judge_and_join(_clusters(rev), rev, _cfg(tmp_path))
    cmap, _ = canonicalize_clusters(out)
    assert len(set(cmap["class"].values())) == 3        # nothing joined
    h = pj.hierarchy_from_report(report, cmap)
    assert "LightweightPlasticCarrierBag" in h["VeryLightweightPlasticCarrierBag"]
    assert "PlasticCarrierBag" in h["LightweightPlasticCarrierBag"]
    merged, _ = assemble_merged_ontology(rev, cmap, "partition_000", hierarchy=h)
    by = {c["name"]: c for c in merged["classes"]}
    assert by["VeryLightweightPlasticCarrierBag"]["broader"] == ["LightweightPlasticCarrierBag", "PlasticCarrierBag"]
    assert "broader" not in by["PlasticCarrierBag"]


def test_link_broader_off_gives_no_links(tmp_path, monkeypatch):
    monkeypatch.setattr(pj, "llm_chat", _fake)
    rev = _revisions()
    _, report = pj.judge_and_join(_clusters(rev), rev, _cfg(tmp_path, link_broader=False))
    assert report["broader_links"] == []


def test_global_merge_carries_links_and_drops_merged_ends():
    part = {"classes": [
        {"name": "Bag", "type": "Asset", "description": "a bag"},
        {"name": "PlasticBag", "type": "Asset", "description": "a plastic bag", "broader": ["Bag", "PlasticBags"]},
    ]}
    rows = pd.DataFrame([{"chunk_id": "part_000", "CELEX": "part_000", "ontology": json.dumps(part)}])
    # the global merge joined PlasticBags into PlasticBag: that link becomes a self-loop and is dropped
    cmap = {"class": {"Bag": "Bag", "PlasticBag": "PlasticBag", "PlasticBags": "PlasticBag"}}
    merged, prov = assemble_merged_ontology(rows, cmap, "global")
    by = {c["name"]: c for c in merged["classes"]}
    assert by["PlasticBag"]["broader"] == ["Bag"]
    assert prov["dropped"]["broader_self_or_undefined"] == 1


def _merged(classes):
    return {"ontology_name": "M", "description": "", "classes": classes, "relationships": [],
            "rights_model": {"permissions": [], "duties": [], "prohibitions": []},
            "legal_patterns": [], "example_instances": []}


def test_refinement_guards_and_export():
    classes = [
        {"name": "CarrierBag", "type": "Asset", "description": "a carrier bag"},
        {"name": "LightCarrierBag", "type": "Asset", "description": "a light carrier bag", "broader": ["CarrierBag"]},
        {"name": "VeryLightCarrierBag", "type": "Asset", "description": "a very light bag",
         "broader": ["LightCarrierBag"]},
        # contradictory verdicts: X < Y and Y < X -> both dropped
        {"name": "Packaging", "type": "Asset", "description": "packaging", "broader": ["PackagingWaste"]},
        {"name": "PackagingWaste", "type": "Asset", "description": "packaging waste", "broader": ["Packaging"]},
        {"name": "BagManufacturer", "type": "Party", "description": "a producer of bags", "broader": ["CarrierBag"]},
    ]
    refined, report = refine_ontology(_merged(classes))
    by = {c["name"]: c for c in refined["classes"] + refined["individuals"]}
    assert by["LightCarrierBag"].get("broader") == ["CarrierBag"]
    assert by["VeryLightCarrierBag"].get("broader") == ["LightCarrierBag"]
    assert "broader" not in by["Packaging"] and "broader" not in by["PackagingWaste"]
    assert report["changes"]["hierarchy_dropped_cycle"] == 2
    # a party is never a kind of an asset
    assert "broader" not in by["BagManufacturer"]
    assert report["changes"]["hierarchy_dropped_type_mismatch"] == 1

    g, stats = export_rdf(refined)
    assert (ENT["VeryLightCarrierBag"], RDFS.subClassOf, ENT["LightCarrierBag"]) in g
    assert (ENT["LightCarrierBag"], RDFS.subClassOf, ENT["CarrierBag"]) in g
    assert stats["class_links"] == 2
    ok, _, summary = validate_graph(g)
    assert ok, summary["results"]
