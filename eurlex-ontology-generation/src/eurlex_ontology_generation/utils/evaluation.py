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

    # Agreement with the gazetteer is only evidence when the gazetteer did not
    # produce the label. Entities typed BY the gazetteer (type_source ==
    # "gazetteer", i.e. the refined ontology) are excluded: counting them would
    # score the lookup against itself. On raw generator output nothing is
    # excluded, so the number measures the LLM.
    known = [(e, gaz.lookup(e["name"])) for e in ents]
    known = [(e, g) for e, g in known if g]
    independent = [(e, g) for e, g in known if e.get("type_source") != "gazetteer"]
    agree = sum(1 for e, g in independent if e.get("type") == g.core_type)
    country_types = Counter(e.get("type") for e, g in known if g.is_a == "Country")
    m["typing"] = {
        "type_values": dict(Counter(e.get("type") for e in ents)),
        "known_entities": len(known),
        "known_entities_typed_by_gazetteer": len(known) - len(independent),
        "pct_known_entities_correct_type": _pct(agree, len(independent)),
        "country_type_values": dict(country_types),
        "pct_entities_with_is_a": _pct(sum(1 for e in ents if e.get("is_a") in vocab.upper), len(ents)),
    }
    m["kind"] = {
        "pct_known_individuals_as_individuals": _pct(
            sum(1 for e, _ in independent if e["kind"] == "individual"), len(independent)),
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


INDEPENDENT_SOURCES = ("expert", "annotator", "adjudicated")
LABEL_FIELDS = ("type", "is_a", "kind")


def load_gold(path: str) -> list[dict]:
    """Gold entities from YAML (`entities:` list) or from a completed annotation sheet (CSV)."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, encoding="utf-8").fillna("")
        rows = []
        for _, r in df.iterrows():
            labels = {f: str(r.get(f, "")).strip() for f in LABEL_FIELDS}
            if not any(labels.values()):
                continue                                  # not annotated yet
            rows.append({"name": r["name"], **{k: v for k, v in labels.items() if v},
                         "source": str(r.get("source") or "annotator"),
                         "annotator": str(r.get("annotator") or "")})
        return rows
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    default_source = raw.get("source", "seed")
    return [{**e, "source": e.get("source", default_source)} for e in raw.get("entities", [])]


def _match(onto: dict, gold: list[dict]) -> list[tuple[dict, dict]]:
    ents = _entities(onto)
    index = {entity_key(e["name"]): e for e in ents}
    for e in ents:
        for v in e.get("merged_from", []) or []:
            index.setdefault(entity_key(v), e)
    found = []
    for g in gold:
        hit = next((index[entity_key(n)] for n in [g["name"], *g.get("aliases", [])]
                    if entity_key(n) in index), None)
        if hit:
            found.append((g, hit))
    return found


def _accuracy(pairs: list[tuple[dict, dict]]) -> dict:
    out = {"n": len(pairs)}
    for f in LABEL_FIELDS:
        scored = [(g[f], h.get(f)) for g, h in pairs if f in g]
        out[f"{f}_accuracy"] = _pct(sum(1 for a, b in scored if a == b), len(scored))
    return out


def gold_metrics(onto: dict, gold_path: str) -> dict:
    """Accuracy against gold labels, split so that nothing is scored against itself.

    held_out           independent labels (expert / annotator / adjudicated) on
                       entities whose prediction did NOT come from the gazetteer.
                       This is the number to report.
    gazetteer_lookups  independent labels on entities typed by the gazetteer:
                       checks the gazetteer's content, not the typer.
    seed_smoke_test    labels written together with the harness (source: seed).
                       A regression test only, never evidence of quality.
    """
    gold = load_gold(gold_path)
    found = _match(onto, gold)
    independent = [(g, h) for g, h in found if g.get("source") in INDEPENDENT_SOURCES]
    held_out = [(g, h) for g, h in independent if h.get("type_source") != "gazetteer"]
    lookups = [(g, h) for g, h in independent if h.get("type_source") == "gazetteer"]
    seed = [(g, h) for g, h in found if g.get("source") not in INDEPENDENT_SOURCES]
    return {
        "gold_entities": len(gold),
        "found_in_ontology": len(found),
        "independent_labels": len(independent),
        "held_out": _accuracy(held_out),
        "gazetteer_lookups": _accuracy(lookups),
        "seed_smoke_test": _accuracy(seed),
        "errors_held_out": [
            {"name": g["name"], "expected": {k: g[k] for k in LABEL_FIELDS if k in g},
             "got": {k: h.get(k) for k in LABEL_FIELDS}}
            for g, h in held_out if any(k in g and g[k] != h.get(k) for k in LABEL_FIELDS)],
    }


def typer_ablation(vocabulary_path: str | None = None) -> dict:
    """Non-circular test of the deterministic typer.

    The gazetteer's labels (Latvia is a Country, the EGF is a Fund, ...) are
    facts about the world, not outputs of the typer. Here every gazetteer name
    and alias is typed with the gazetteer switched OFF, i.e. by the head-noun
    keywords and name heuristics alone, the path used for every entity the
    gazetteer does not know. Agreement with the gazetteer is therefore an
    independent estimate of how well the fallback typer works.
    """
    from .entity_linking import Gazetteer, detect_kind
    from .ontology_normalize import type_entity

    vocab = load_vocabulary(vocabulary_path)
    gaz = load_gazetteer(None, vocabulary_path)
    empty = Gazetteer([], {})
    rows = []
    for canonical, entry in gaz.entries.items():
        for name in dict.fromkeys([canonical, *gaz.surface_forms.get(canonical, [])]):
            typed = type_entity(name, "", vocab, empty)
            kind = detect_kind(name, "", empty, vocab)["kind"]
            rows.append({"name": name, "gold_type": entry.core_type, "gold_is_a": entry.is_a,
                         "type": typed["type"], "is_a": typed["is_a"], "kind": kind})
    canon = [r for r in rows if r["name"] in gaz.entries]

    def acc(rs):
        return {"n": len(rs),
                "type_accuracy": _pct(sum(r["type"] == r["gold_type"] for r in rs), len(rs)),
                "is_a_accuracy": _pct(sum(r["is_a"] == r["gold_is_a"] for r in rs), len(rs)),
                "kind_accuracy": _pct(sum(r["kind"] == "individual" for r in rs), len(rs))}

    by_class: dict[str, list] = {}
    for r in canon:
        by_class.setdefault(r["gold_is_a"], []).append(r)
    return {
        "canonical_names": acc(canon),
        "all_surface_forms": acc(rows),
        "by_gold_is_a": {k: acc(v) for k, v in sorted(by_class.items())},
        "errors": [r for r in canon if r["type"] != r["gold_type"]][:40],
    }


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement between two annotators beyond chance."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def annotator_agreement(sheet_a: str, sheet_b: str) -> dict:
    """Cohen's kappa per label field on entities both annotators labelled."""
    a = {g["name"]: g for g in load_gold(sheet_a)}
    b = {g["name"]: g for g in load_gold(sheet_b)}
    common = sorted(set(a) & set(b))
    out = {"common_entities": len(common)}
    for f in LABEL_FIELDS:
        pairs = [(a[n].get(f, ""), b[n].get(f, "")) for n in common if a[n].get(f) and b[n].get(f)]
        out[f"{f}_kappa"] = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])
        out[f"{f}_raw_agreement"] = _pct(sum(x == y for x, y in pairs), len(pairs))
    return out


HEADLINE = [
    ("size", "classes"), ("size", "individuals"), ("size", "prohibitions"),
    ("long_tail", "pct_entities_single_document"),
    ("typing", "pct_known_entities_correct_type"), ("typing", "known_entities_typed_by_gazetteer"),
    ("typing", "pct_entities_with_is_a"),
    ("kind", "pct_known_individuals_as_individuals"),
    ("leakage", "count"),
    ("actions", "distinct"), ("actions", "pct_rules_controlled"), ("actions", "pct_distinct_used_once"),
    ("constraints", "pct_structured"), ("constraints", "pct_controlled_left_operand"),
    ("relations", "pct_controlled"),
    ("consistency", "rules_party_not_party"), ("consistency", "rules_asset_is_party"),
    ("consistency", "undefined_references"), ("consistency", "isolated_entities"),
    ("linking", "pct_entities_with_identifier"),
    ("shacl", "violations"), ("shacl", "warnings"), ("shacl", "infos"),
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
