"""Quality metrics for an ontology produced by the pipeline.

Works on both the merged ontology (06_models) and the refined one
(07_model_output), so a change can be measured as before/after. Every metric
is computed in code from the JSON; nothing calls an LLM.

Metric groups
  size          classes, individuals, relationships, rules per kind
  long_tail     share of entities supported by a single document
  typing        gazetteer agreement, type conflicts between known countries
  kind          share of known individuals modelled as individuals
  leakage       ODRL meta-model terms used as entity names
  actions       distinct actions, share in the controlled vocabulary, singletons
  constraints   share structured as left_operand/operator/right_operand
  relations     share in the controlled relation set
  consistency   rule-slot typing, undefined references, isolated entities
  linking       share of entities with an external identifier
  review        approval rate per round (from the review CSVs), optional
  gold          accuracy against an expert-annotated gold set, optional
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .entity_linking import entity_key, load_gazetteer
from .vocabulary import load_vocabulary

RULE_KINDS = ("permissions", "prohibitions", "duties")


def _dicts(value: Any) -> list[dict]:
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


def _pct(n: float, d: float) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _entities(onto: dict) -> list[dict]:
    """Classes and individuals with a uniform `kind` field."""
    out = []
    for c in _dicts(onto.get("classes")):
        out.append({**c, "kind": c.get("kind", "class" if "individuals" in onto else None)})
    for i in _dicts(onto.get("individuals")):
        out.append({**i, "kind": "individual", "is_a": i.get("instance_of")})
    return out


def _rules(onto: dict):
    rm = onto.get("rights_model") or {}
    for kind in RULE_KINDS:
        for r in _dicts(rm.get(kind)):
            yield kind, r


def compute_metrics(onto: dict) -> dict:
    vocab = load_vocabulary()
    gaz = load_gazetteer()
    ents = _entities(onto)
    names = {e["name"] for e in ents}
    types = {e["name"]: e.get("type") for e in ents}
    m: dict[str, Any] = {}

    rules = list(_rules(onto))
    m["size"] = {
        "classes": sum(1 for e in ents if e["kind"] != "individual"),
        "individuals": sum(1 for e in ents if e["kind"] == "individual"),
        "relationships": len(_dicts(onto.get("relationships"))),
        **{k: sum(1 for kk, _ in rules if kk == k) for k in RULE_KINDS},
    }

    docs = [((e.get("support") or {}).get("documents")) for e in ents]
    docs = [d for d in docs if isinstance(d, (int, float))]
    m["long_tail"] = {"pct_entities_single_document": _pct(sum(1 for d in docs if d <= 1), len(docs))}

    known = [(e, gaz.lookup(e["name"])) for e in ents]
    known = [(e, g) for e, g in known if g]
    agree = sum(1 for e, g in known if e.get("type") == g.core_type)
    country_types = Counter(e.get("type") for e, g in known if g.is_a == "Country")
    m["typing"] = {
        "type_values": dict(Counter(e.get("type") for e in ents)),
        "known_entities": len(known),
        "pct_known_entities_correct_type": _pct(agree, len(known)),
        "country_type_values": dict(country_types),
        "pct_entities_with_is_a": _pct(sum(1 for e in ents if e.get("is_a") in vocab.upper), len(ents)),
    }
    m["kind"] = {
        "pct_known_individuals_as_individuals": _pct(
            sum(1 for e, _ in known if e["kind"] == "individual"), len(known)),
        "pct_entities_with_kind": _pct(sum(1 for e in ents if e["kind"] in ("class", "individual")), len(ents)),
    }
    leaks = sorted(e["name"] for e in ents if vocab.metamodel_leakage(e["name"]))
    m["leakage"] = {"count": len(leaks), "names": leaks}

    actions = [r.get("action") for _, r in rules if r.get("action")]
    counts = Counter(actions)
    m["actions"] = {
        "rules": len(actions),
        "distinct": len(counts),
        "pct_rules_controlled": _pct(sum(1 for a in actions if a in vocab.actions), len(actions)),
        "pct_distinct_used_once": _pct(sum(1 for c in counts.values() if c == 1), len(counts)),
    }

    cons = [c for _, r in rules for c in _dicts(r.get("constraints"))]
    structured = [c for c in cons if c.get("left_operand") and c.get("operator")]
    m["constraints"] = {
        "total": len(cons),
        "pct_structured": _pct(len(structured), len(cons)),
        "pct_controlled_left_operand": _pct(
            sum(1 for c in structured if c["left_operand"] in vocab.left_operands
                and c.get("controlled", True)), len(cons)),
        "distinct_names": len({c.get("left_operand") or c.get("name") for c in cons}),
    }

    rels = _dicts(onto.get("relationships"))
    m["relations"] = {
        "distinct": len({r.get("relation") for r in rels}),
        "pct_controlled": _pct(sum(1 for r in rels if r.get("relation") in vocab.relations), len(rels)),
    }

    used: set[str] = set()
    party_bad = asset_bad = undefined = 0
    for _, r in rules:
        used.update(x for x in (r.get("party"), r.get("asset")) if x)
        if r.get("party") in types and types[r["party"]] not in ("Party", "RightsHolder"):
            party_bad += 1
        if r.get("asset") and types.get(r["asset"]) == "Party":
            asset_bad += 1
        undefined += sum(1 for x in (r.get("party"), r.get("asset")) if x and x not in names)
    for r in rels:
        used.update([r.get("source"), r.get("target")])
    for p in _dicts(onto.get("legal_patterns")):
        used.update(p.get("entities_involved") or [])
    used.update(e.get("entity") for e in _dicts(onto.get("example_instances")))
    m["consistency"] = {
        "rules_party_not_party": party_bad,
        "rules_asset_is_party": asset_bad,
        "undefined_references": undefined,
        "isolated_entities": len(names - used),
    }
    m["linking"] = {
        "pct_entities_with_identifier": _pct(sum(1 for e in ents if e.get("iri") or e.get("wikidata")), len(ents)),
    }
    return m


def review_metrics(reviews_directory: str) -> dict:
    """Approval per round from data/04_feature/ontology_reviews/part_*.csv."""
    frames = [pd.read_csv(p, encoding="utf-8") for p in sorted(Path(reviews_directory).glob("part_*.csv"))]
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    if "round" not in df.columns:
        df["round"] = 0
    df["round"] = df["round"].fillna(0).astype(int)
    df["approved_b"] = df["approved"].astype(str).str.lower().eq("true")
    per_round = {}
    for rnd, g in df.groupby("round"):
        scores = [json.loads(r).get("overall_score") for r in g["review"] if isinstance(r, str)]
        scores = [s for s in scores if isinstance(s, (int, float))]
        per_round[int(rnd)] = {"chunks": len(g), "approved": int(g["approved_b"].sum()),
                               "pct_approved": _pct(g["approved_b"].sum(), len(g)),
                               "mean_overall_score": round(sum(scores) / len(scores), 2) if scores else None}
    latest = df.sort_values("round").drop_duplicates("chunk_id", keep="last")
    final_reviewed_revision = latest[latest["round"] >= 1]
    return {
        "per_round": per_round,
        "final": {"chunks": len(latest),
                  "pct_approved_latest_round": _pct(latest["approved_b"].sum(), len(latest)),
                  "chunks_with_reviewed_revision": len(final_reviewed_revision)},
    }


def gold_metrics(onto: dict, gold_path: str) -> dict:
    """Accuracy of type / is_a / kind for gold entities found in the ontology."""
    gold = yaml.safe_load(Path(gold_path).read_text(encoding="utf-8")).get("entities", [])
    index = {entity_key(e["name"]): e for e in _entities(onto)}
    for e in _entities(onto):
        for v in e.get("merged_from", []) or []:
            index.setdefault(entity_key(v), e)
    found = []
    for g in gold:
        hit = next((index[entity_key(n)] for n in [g["name"], *g.get("aliases", [])]
                    if entity_key(n) in index), None)
        if hit:
            found.append((g, hit))
    def acc(field, gfield):
        pairs = [(g[gfield], h.get(field)) for g, h in found if gfield in g]
        return _pct(sum(1 for a, b in pairs if a == b), len(pairs))
    return {
        "gold_entities": len(gold),
        "found_in_ontology": len(found),
        "type_accuracy": acc("type", "type"),
        "is_a_accuracy": acc("is_a", "is_a"),
        "kind_accuracy": acc("kind", "kind"),
        "errors": [{"name": g["name"], "expected": {k: g[k] for k in ("type", "is_a", "kind") if k in g},
                    "got": {k: h.get(k) for k in ("type", "is_a", "kind")}}
                   for g, h in found
                   if any(k in g and g[k] != h.get(k) for k in ("type", "is_a", "kind"))],
    }


HEADLINE = [
    ("size", "classes"), ("size", "individuals"), ("size", "prohibitions"),
    ("long_tail", "pct_entities_single_document"),
    ("typing", "pct_known_entities_correct_type"), ("typing", "pct_entities_with_is_a"),
    ("kind", "pct_known_individuals_as_individuals"),
    ("leakage", "count"),
    ("actions", "distinct"), ("actions", "pct_rules_controlled"), ("actions", "pct_distinct_used_once"),
    ("constraints", "pct_structured"), ("constraints", "pct_controlled_left_operand"),
    ("relations", "pct_controlled"),
    ("consistency", "rules_party_not_party"), ("consistency", "rules_asset_is_party"),
    ("consistency", "undefined_references"), ("consistency", "isolated_entities"),
    ("linking", "pct_entities_with_identifier"),
]


def headline_table(results: dict[str, dict]) -> str:
    labels = list(results)
    lines = ["metric".ljust(48) + "".join(lbl[:22].rjust(24) for lbl in labels)]
    for group, key in HEADLINE:
        row = f"{group}.{key}".ljust(48)
        for lbl in labels:
            v = results[lbl].get(group, {}).get(key)
            row += ("-" if v is None else str(v)).rjust(24)
        lines.append(row)
    return "\n".join(lines)
