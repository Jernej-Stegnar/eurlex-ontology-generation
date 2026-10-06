"""Deterministic refinement of the merged ontology.

Input: the global merged ontology (data/06_models/eurlex_odrl_ontology_merged.json).
Output: a refined ontology with a TBox/ABox split, plus a report.

Steps (no LLM involved; Wikidata is optional and off by default):
  1. remove ODRL meta-model leakage (AuditDuty, ...PermissionTarget)
  2. split composite parties (EuropeanParliamentAndCouncil)
  3. rename known entities to their gazetteer name and merge duplicates
     (SolidarityFund / EU Solidarity Fund -> European_Union_Solidarity_Fund)
  4. resolve type, is_a and kind for every entity (gazetteer > declared is_a >
     head-noun keyword > declared type); optionally confirm low-confidence
     kinds against Wikidata (P31 vs P279)
  5. split entities into `classes` (kind=class) and `individuals`
  6. canonical actions (+ action_object), structured constraints, canonical
     relations; merge rules/relationships that became identical
  7. consistency report: rule-slot typing, undefined references, isolated
     entities, low-confidence decisions for human review
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ...utils.entity_linking import WikidataLinker, load_gazetteer
from ...utils.ontology_normalize import CORE_ENTITIES, split_composite, type_entity
from ...utils.ontology_schema import RULE_KINDS
from ...utils.vocabulary import load_vocabulary

MAX_DESCRIPTIONS = 3


def _dicts(value: Any) -> list[dict]:
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


def _support(obj: dict) -> dict:
    s = obj.get("support")
    if isinstance(s, dict):
        return {k: int(v) for k, v in s.items() if isinstance(v, (int, float))}
    if isinstance(s, (int, float)):
        return {"chunks": int(s)}
    return {"chunks": 1}


def _add_support(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def load_merged_ontology(input_path: str) -> dict:
    return json.loads(Path(input_path).read_text(encoding="utf-8"))


def refine_ontology(merged: dict, wikidata: dict | None = None,
                    vocabulary_path: str | None = None) -> tuple[dict, dict]:
    vocab = load_vocabulary(vocabulary_path)
    gazetteer = load_gazetteer(None, vocabulary_path)
    wikidata = wikidata or {}
    linker = None
    if wikidata.get("enabled"):
        linker = WikidataLinker(wikidata["cache_path"], wikidata["user_agent"],
                                int(wikidata.get("max_lookups", 200)))

    changes: Counter = Counter()
    log: dict[str, list] = defaultdict(list)

    # ---------------------------------------------------- 1-3: names
    rename: dict[str, str | None] = {}           # old name -> new name (None = removed)
    composites: dict[str, list[str]] = {}
    for cls in _dicts(merged.get("classes")):
        name = cls.get("name")
        if not isinstance(name, str):
            continue
        leak = vocab.metamodel_leakage(name)
        if leak:
            rename[name] = None
            changes["leakage_removed"] += 1
            log["leakage_removed"].append(leak)
            continue
        parts = split_composite(name, gazetteer)
        if parts:
            composites[name] = parts
            rename[name] = None
            changes["composite_split"] += 1
            log["composite_split"].append(f"{name} -> {parts}")
            continue
        hit = gazetteer.lookup(name)
        if hit and hit.name != name:
            rename[name] = hit.name
            changes["renamed_to_gazetteer"] += 1
            log["renamed_to_gazetteer"].append(f"{name} -> {hit.name}")

    def resolve(name: Any) -> str | None:
        if not isinstance(name, str) or not name.strip():
            return None
        if name in rename:
            return rename[name]
        if vocab.metamodel_leakage(name):
            return None
        hit = gazetteer.lookup(name)
        return hit.name if hit else name

    def expand(name: Any) -> list[str]:
        if isinstance(name, str) and name in composites:
            return composites[name]
        r = resolve(name)
        return [r] if r else []

    # ---------------------------------------------------- 4: entities
    entities: dict[str, dict] = {}
    for cls in _dicts(merged.get("classes")):
        target = resolve(cls.get("name"))
        if not target:
            continue
        e = entities.setdefault(target, {"name": target, "descriptions": [], "support": {},
                                         "type_votes": Counter(), "is_a_votes": Counter(),
                                         "kind_votes": Counter(), "variants": []})
        e["variants"].append(cls["name"])
        sup = _support(cls)
        e["support"] = _add_support(e["support"], sup)
        weight = sup.get("chunks", 1)
        if cls.get("type"):
            e["type_votes"][cls["type"]] += weight
        if cls.get("is_a"):
            e["is_a_votes"][cls["is_a"]] += weight
        if cls.get("kind"):
            e["kind_votes"][cls["kind"]] += weight
        for piece in str(cls.get("description") or "").split(" | "):
            if piece and piece not in e["descriptions"] and len(e["descriptions"]) < MAX_DESCRIPTIONS:
                e["descriptions"].append(piece)
    # composite parts must exist as entities
    for parts in composites.values():
        for p in parts:
            entities.setdefault(p, {"name": p, "descriptions": [], "support": {}, "type_votes": Counter(),
                                    "is_a_votes": Counter(), "kind_votes": Counter(), "variants": []})

    review_queue = []
    typed: dict[str, dict] = {}
    for name, e in entities.items():
        declared_type = e["type_votes"].most_common(1)[0][0] if e["type_votes"] else None
        declared_is_a = e["is_a_votes"].most_common(1)[0][0] if e["is_a_votes"] else None
        declared_kind = e["kind_votes"].most_common(1)[0][0] if e["kind_votes"] else None
        if len(e["type_votes"]) > 1:
            log["type_vote_conflict"].append(f"{name}: {dict(e['type_votes'])}")
        t = type_entity(name, " ".join(e["descriptions"]), vocab, gazetteer,
                        declared_type, declared_is_a, declared_kind)
        if linker and t.get("type_source") != "gazetteer" and t["kind_confidence"] == "low":
            kind, info = linker.kind_of(name)
            if kind:
                t["kind"], t["kind_confidence"] = kind, "medium"
                t["wikidata"] = info["qid"]
                changes["kind_from_wikidata"] += 1
        if declared_type and declared_type != t["type"]:
            changes["retyped"] += 1
            log["retyped"].append(f"{name}: {declared_type} -> {t['type']} ({t['type_source']})")
        if declared_kind != t["kind"]:
            changes[f"kind_set_{t['kind']}"] += 1
        if t["kind_confidence"] == "low" and t["kind"] == "individual":
            review_queue.append({"name": name, "question": "individual or class?",
                                 "current": t["kind"], "evidence": t.get("type_source")})
        typed[name] = t

    # ---------------------------------------------------- 6: rules
    rules: dict[str, dict[tuple, dict]] = {k: {} for k in RULE_KINDS}
    rm = merged.get("rights_model") or {}
    for kind in RULE_KINDS:
        for rule in _dicts(rm.get(kind)):
            parties = expand(rule.get("party"))
            if not parties:
                changes["rule_dropped_party"] += 1
                continue
            asset = resolve(rule.get("asset")) if rule.get("asset") else None
            if rule.get("asset") and not asset:
                changes["rule_asset_cleared"] += 1
            norm = vocab.normalize_action(rule.get("action"), rule.get("action_label"))
            obj = rule.get("action_object") or norm["action_object"] or None
            if norm["action"] != rule.get("action"):
                changes["action_normalized"] += 1
            constraints = []
            for c in _dicts(rule.get("constraints")):
                sc = vocab.structure_constraint(c)
                if sc:
                    constraints.append(sc)
                    if "left_operand" not in c:
                        changes["constraint_structured"] += 1
            for party in parties:
                key = (norm["action"], obj, party, asset)
                entry = rules[kind].setdefault(key, {
                    "action": norm["action"], "action_object": obj,
                    "action_label": norm["action_label"], "action_controlled": norm["controlled"],
                    "odrl_parent_action": norm["included_in"],
                    "party": party, "asset": asset, "constraints": [], "support": {}})
                if entry["support"]:
                    changes["rule_merged"] += 1
                entry["support"] = _add_support(entry["support"], _support(rule))
                if rule.get("provision"):
                    entry.setdefault("provision", rule["provision"])
                seen = {(c["left_operand"], c["operator"], c["right_operand"]) for c in entry["constraints"]}
                for c in constraints:
                    k = (c["left_operand"], c["operator"], c["right_operand"])
                    if k not in seen:
                        entry["constraints"].append(c)
                        seen.add(k)

    # ---------------------------------------------------- relations
    relationships: dict[tuple, dict] = {}
    for rel in _dicts(merged.get("relationships")):
        for src in expand(rel.get("source")):
            for tgt in expand(rel.get("target")):
                relation, s_, t_, controlled = vocab.normalize_relation(rel.get("relation"), src, tgt)
                if not relation or s_ == t_:
                    changes["relationship_dropped"] += 1
                    continue
                if relation != rel.get("relation") or s_ != src:
                    changes["relation_normalized"] += 1
                key = (s_, relation, t_)
                entry = relationships.setdefault(key, {"source": s_, "relation": relation, "target": t_,
                                                       "description": rel.get("description", ""),
                                                       "relation_controlled": controlled, "support": {}})
                entry["support"] = _add_support(entry["support"], _support(rel))
    # derived links: "EGF_Mobilization concerns EGF"
    for name, t in typed.items():
        rel_ind = t.get("related_individual")
        if rel_ind and rel_ind in typed and rel_ind != name:
            relationships.setdefault((name, "concerns", rel_ind), {
                "source": name, "relation": "concerns", "target": rel_ind,
                "description": "derived from the entity name", "relation_controlled": True,
                "support": {}, "derived": True})
            changes["derived_relationship"] += 1

    # ---------------------------------------------------- patterns, examples
    patterns = []
    for p in _dicts(merged.get("legal_patterns")):
        ents = []
        for e in p.get("entities_involved") or []:
            ents += expand(e)
        ents = list(dict.fromkeys(ents))
        if len(ents) >= 2:
            patterns.append({**p, "entities_involved": ents})
        else:
            changes["pattern_dropped"] += 1
    examples = []
    for ex in _dicts(merged.get("example_instances")):
        target = resolve(ex.get("entity"))
        if target:
            examples.append({**ex, "entity": target})

    # ---------------------------------------------------- 5: TBox / ABox
    classes, individuals = [], []
    for name, e in entities.items():
        t = typed[name]
        record = {"name": name, "type": t["type"], "is_a": t["is_a"],
                  "description": " | ".join(e["descriptions"]),
                  "support": e["support"], "type_source": t.get("type_source"),
                  "kind_confidence": t.get("kind_confidence")}
        for k in ("iri", "member_state", "related_individual", "wikidata"):
            if t.get(k) is not None:
                record[k] = t[k]
        if len(e["variants"]) > 1:
            record["merged_from"] = sorted(set(e["variants"]))
        if t["kind"] == "individual":
            record["instance_of"] = record.pop("is_a")
            individuals.append(record)
        else:
            classes.append(record)

    rights_model = {k: [{**r, "support": r["support"]} for r in rules[k].values()] for k in RULE_KINDS}

    refined = {
        "ontology_name": merged.get("ontology_name", "EurLex ODRL Ontology") + " (refined)",
        "description": merged.get("description", ""),
        "odrl_alignment": {
            "core_entities": CORE_ENTITIES,
            "extensions": sorted(c["name"] for c in classes if c["type"] == "Legal_extension"),
        },
        "upper_ontology": [{"name": n, "core_type": s["core_type"], "description": s["description"]}
                           for n, s in vocab.upper.items()],
        "classes": sorted(classes, key=lambda c: c["name"]),
        "individuals": sorted(individuals, key=lambda c: c["name"]),
        "relationships": list(relationships.values()),
        "rights_model": rights_model,
        "legal_patterns": patterns,
        "example_instances": examples,
    }

    consistency = check_consistency(refined)
    if linker:
        linker.save()
    report = {
        "changes": dict(changes),
        "change_log": {k: v for k, v in log.items()},
        "consistency_issues": consistency,
        "review_queue": review_queue,
        "counts": {"classes": len(classes), "individuals": len(individuals),
                   "relationships": len(relationships),
                   **{k: len(v) for k, v in rights_model.items()}},
    }
    return refined, report


def check_consistency(onto: dict) -> dict:
    """Rule-slot typing, undefined references and isolated entities."""
    types = {e["name"]: e["type"] for e in onto.get("classes", []) + onto.get("individuals", [])}
    issues: dict[str, list] = defaultdict(list)
    used: set[str] = set()
    for kind, rules in (onto.get("rights_model") or {}).items():
        for r in rules:
            used.update(x for x in (r.get("party"), r.get("asset")) if x)
            if types.get(r["party"]) not in ("Party", None):
                issues["party_not_party"].append(f"{kind}: {r['action']} party={r['party']} ({types.get(r['party'])})")
            if r.get("asset") and types.get(r["asset"]) == "Party":
                issues["asset_is_party"].append(f"{kind}: {r['action']} asset={r['asset']}")
            for x in (r.get("party"), r.get("asset")):
                if x and x not in types:
                    issues["undefined_reference"].append(x)
    for rel in onto.get("relationships", []):
        used.update([rel["source"], rel["target"]])
        for x in (rel["source"], rel["target"]):
            if x not in types:
                issues["undefined_reference"].append(x)
    for p in onto.get("legal_patterns", []):
        used.update(p.get("entities_involved") or [])
    used.update(e.get("entity") for e in onto.get("example_instances", []))
    issues["isolated_entities"] = sorted(set(types) - used)
    issues["undefined_reference"] = sorted(set(issues["undefined_reference"]))
    return {k: v for k, v in issues.items()}


def persist_refined_ontology(refined: dict, report: dict, output_path: str, report_path: str) -> str:
    for path, payload in ((output_path, refined), (report_path, report)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
