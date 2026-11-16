"""LLM pair judge for entity resolution (T13 F3).

Name rules cannot tell "FoodBusinessOperator" = "Operator" (an act about food business operators)
from "Treatment" > "HeatTreatment" (a broader concept) or "Croatia" / "Spain" (siblings in one
enumeration). This step lets an LLM decide, per act, for the pairs that look alike:

1. blocking: within one act, pairs of clusters whose names (character n-grams) or descriptions
   (content words, TF-IDF) are similar, or one name is the acronym of the other;
2. judging: the LLM sees both names, types, descriptions, a few facts from the chunk graphs
   (relations, rule roles) and the act sentences that mention them, and labels each pair
   SAME / A_BROADER / B_BROADER / PART_OR_ROLE / SIBLING / DIFFERENT;
3. joining: SAME at or above `min_confidence` joins the two clusters; everything else is kept
   apart. Every verdict is returned for the provenance report (broader pairs are the input for a
   later class hierarchy).

Verdicts are cached (one JSON line per pair, keyed by model, prompt version and pair content), so
a rerun on the same outputs costs nothing and gives the same result.
Graph neighbours are context only, never the deciding signal: siblings in legal enumerations share
more neighbours than aliases do (T13 scan: Jaccard 0.22 vs 0.06).
"""
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..llm_client import llm_chat, run_parallel

logger = logging.getLogger(__name__)

PROMPT_VERSION = "pj2"
LABELS = ("SAME", "A_BROADER", "B_BROADER", "PART_OR_ROLE", "SIBLING", "DIFFERENT")
# SAME on these bases is joined on the judge's word; any other basis needs evidence from the act text
FORM_BASES = {"acronym", "word_order", "spelling_number", "defined_term"}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
DEFAULTS = {
    "enabled": False,
    "model": "deepseek-v4-pro",
    "llm_backend": None,
    "temperature": 0.0,
    "seed": 42,
    "batch_size": 12,
    "max_tokens": 3000,
    "min_confidence": "medium",
    "name_sim": 0.5,
    "desc_sim": 0.4,
    "max_pairs_per_act": 400,
    "cache_path": "data/05_model_input/pair_judge_cache.jsonl",
}

SYSTEM_PROMPT = """You decide whether two entity names extracted from one EU legal act denote the same entity.
The names come from an automatic extraction that named the same thing differently in different parts of the act.
For every pair, choose exactly one label:
- SAME: within this act, A and B denote the same real-world entity or concept and must be one node (acronym and full name, word order, paraphrase or synonym, singular/plural, spelling, a modifier the act makes redundant, a role description the act uses for one specific body).
- A_BROADER: A is a genuine generalisation of B (B is a kind of A). B_BROADER: the reverse.
- PART_OR_ROLE: closely tied but distinct: part and whole, a body and its head or member, an entity and its rules, decision or document, an activity and its result.
- SIBLING: distinct members of the same category or enumeration (two countries, two kinds of venue, two institutions).
- DIFFERENT: anything else.
Be strict: SAME only when merging the two nodes loses no distinction the act makes. Opposites (X vs non-X), different degrees (lightweight vs very lightweight) and different sizes or kinds are never SAME.
When one name adds words to the other (InternalNetwork vs CommissionInternalNetwork, Incident vs ITSecurityIncident), the default is A_BROADER or B_BROADER; choose SAME only if the act text shows that the shorter name always means the longer one in this act. A name that lists several things (StandardsAndGuidelines) is not SAME as one of them (PART_OR_ROLE). A generic word (Decision, Rule, Measure, Network) is not SAME as a specific instrument or body unless the act text says so.
For SAME, give the basis: acronym | word_order | spelling_number | defined_term (the act defines one name as the other) | synonym | redundant_modifier | other; and the evidence: a quote of at most 25 words copied from act_text that shows the two are the same (empty if there is none).
Confidence: high, medium or low.
Answer with JSON only: {"verdicts": [{"id": <pair id>, "label": "<label>", "confidence": "<high|medium|low>", "basis": "<basis or empty>", "evidence": "<quote or empty>"}]} with one verdict per pair."""

_STOP = {"the", "of", "and", "or", "a", "an", "to", "in", "for", "on", "by", "with", "as", "is", "are", "be",
         "that", "which", "this", "its", "their", "from", "at", "under", "any", "such", "other", "shall", "may"}


def _split(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def _ngrams(name: str) -> Counter:
    c: Counter = Counter()
    for w in _split(name):
        w = f" {w} "
        for n in (3, 4):
            for i in range(len(w) - n + 1):
                c[w[i:i + n]] += 1
    return c


def _words(text: str) -> Counter:
    return Counter(w for w in re.findall(r"[a-z][a-z\-]+", (text or "").lower()) if w not in _STOP and len(w) > 2)


def _tfidf(docs: list[Counter]) -> list[dict]:
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    out = []
    for d in docs:
        v = {t: (1 + math.log(c)) * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / norm for t, x in v.items()})
    return out


def _cos(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _is_acronym(a: str, b: str) -> bool:
    if not re.fullmatch(r"[A-Z]{2,8}s?", a):
        return False
    toks = _split(b)
    return a.rstrip("s").lower() in ("".join(t[0] for t in toks if t not in _STOP), "".join(t[0] for t in toks))


def candidate_pairs(entities: list[dict], name_sim: float, desc_sim: float) -> list[tuple[int, int, dict]]:
    """entities: [{name, description, ...}] of one act. Returns (i, j, signals) for pairs that look alike."""
    names = _tfidf([_ngrams(e["name"]) for e in entities])
    descs = _tfidf([_words(e.get("description") or " ".join(_split(e["name"]))) for e in entities])
    out = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            sn, sd = _cos(names[i], names[j]), _cos(descs[i], descs[j])
            ac = _is_acronym(entities[i]["name"], entities[j]["name"]) or _is_acronym(entities[j]["name"], entities[i]["name"])
            if sn >= name_sim or sd >= desc_sim or ac:
                out.append((i, j, {"name_sim": round(sn, 3), "desc_sim": round(sd, 3), "acronym": ac}))
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+", text or "") if 20 <= len(s.strip()) <= 1200]


def _snippet(sentences: list[str], *names: str, width: int = 280) -> str:
    words = {w[:5] for n in names for w in _split(n) if w not in _STOP and len(w) > 2}
    if not words:
        return ""
    best, score = "", 0
    for s in sentences:
        low = s.lower()
        sc = sum(1 for w in words if w in low)
        if sc > score:
            best, score = s, sc
    if score < max(1, len(words) // 2):
        return ""
    return best if len(best) <= width else best[:width] + "…"


def _facts(onto_rows: list[dict], name: str, limit: int = 4) -> list[str]:
    out = []
    for onto in onto_rows:
        for r in onto.get("relationships") or []:
            if isinstance(r, dict) and name in (r.get("source"), r.get("target")):
                out.append(f"{r.get('source')} {r.get('relation')} {r.get('target')}")
        rm = onto.get("rights_model") or {}
        for kind in ("permissions", "duties", "prohibitions"):
            for r in rm.get(kind) or []:
                if isinstance(r, dict) and name in (r.get("party"), r.get("asset")):
                    out.append(f"{kind[:-1]}: {r.get('party')} {r.get('action')} {r.get('asset')}")
        if len(out) >= limit:
            break
    return list(dict.fromkeys(out))[:limit]


def _pair_payload(pid: int, a: dict, b: dict, snippets: dict) -> dict:
    return {"id": pid,
            "A": {"name": a["name"], "type": a.get("type") or "", "description": a.get("description") or "",
                  "facts": a.get("facts") or []},
            "B": {"name": b["name"], "type": b.get("type") or "", "description": b.get("description") or "",
                  "facts": b.get("facts") or []},
            "act_text": snippets}


def _cache_key(cfg: dict, payload: dict) -> str:
    body = json.dumps({k: v for k, v in payload.items() if k != "id"}, sort_keys=True, ensure_ascii=False)
    extra = json.dumps((cfg.get("llm_backend") or {}).get("extra_body") or {}, sort_keys=True)  # e.g. thinking on/off
    return hashlib.sha1(f"{PROMPT_VERSION}|{cfg['model']}|{extra}|{body}".encode("utf-8")).hexdigest()


def _load_cache(path: str | None) -> dict[str, dict]:
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec["verdict"]
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _append_cache(path: str | None, records: list[dict]) -> None:
    if not path or not records:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse(content: str, ids: list[int]) -> dict[int, dict]:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        m = re.search(r"\{.*\}", content or "", re.S)
        data = json.loads(m.group(0)) if m else {}
    out = {}
    for v in (data.get("verdicts") if isinstance(data, dict) else None) or []:
        try:
            pid = int(v.get("id"))
        except (TypeError, ValueError):
            continue
        label = str(v.get("label", "")).upper()
        conf = str(v.get("confidence", "low")).lower()
        if pid in ids and label in LABELS:
            out[pid] = {"label": label, "confidence": conf if conf in CONFIDENCE_RANK else "low",
                        "basis": str(v.get("basis") or "").lower(), "evidence": str(v.get("evidence") or "")[:400]}
    return out


def _evidence_found(evidence: str, act_text: dict) -> bool:
    """The quote must come from the act sentences the judge was given (at least 80% of its words)."""
    words = re.findall(r"[a-z0-9]+", evidence.lower())
    if len(words) < 3:
        return False
    pool = set(re.findall(r"[a-z0-9]+", " ".join(act_text.values()).lower()))
    return sum(w in pool for w in words) / len(words) >= 0.8


def accepted(verdict: dict | None, act_text: dict, min_rank: int) -> bool:
    """SAME is joined only at or above the confidence floor, and only if its basis is a form
    variant (acronym, word order, spelling, a definition) or it quotes act text that was shown."""
    if not verdict or verdict.get("label") != "SAME":
        return False
    if CONFIDENCE_RANK.get(verdict.get("confidence") or "low", 0) < min_rank:
        return False
    return verdict.get("basis") in FORM_BASES or _evidence_found(verdict.get("evidence") or "", act_text)


def judge(payloads: list[dict], celex: str, cfg: dict) -> tuple[dict[int, dict], dict]:
    """Verdicts {pair id: {label, confidence}} for one act, using and filling the cache."""
    cache = _load_cache(cfg["cache_path"])
    verdicts, todo = {}, []
    for p in payloads:
        k = _cache_key(cfg, p)
        if k in cache:
            verdicts[p["id"]] = cache[k]
        else:
            todo.append((k, p))
    usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cached": len(verdicts), "unparsed": 0}
    if not todo:
        return verdicts, usage
    bs = int(cfg["batch_size"])
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]

    def call(batch):
        user = json.dumps({"act": celex, "pairs": [p for _, p in batch]}, ensure_ascii=False)
        try:
            res = llm_chat(cfg["llm_backend"], cfg["model"], SYSTEM_PROMPT, user, float(cfg["temperature"]),
                           json_mode=True, seed=cfg.get("seed"), max_tokens=int(cfg["max_tokens"]))
        except Exception as exc:  # one failed batch must not stop the merge: its pairs stay apart
            logger.warning("Pair judge call failed for %s (%s): %d pairs left unjudged.", celex, exc, len(batch))
            return batch, {}, None
        return batch, _parse(res.content, [p["id"] for _, p in batch]), res

    workers = int((cfg["llm_backend"] or {}).get("max_workers", 4))
    new_records = []
    for batch, parsed, res in run_parallel(call, batches, workers):
        if res is not None:
            usage["calls"] += 1
            usage["prompt_tokens"] += res.prompt_tokens or 0
            usage["output_tokens"] += res.output_tokens or 0
        for k, p in batch:
            v = parsed.get(p["id"])
            if v is None:
                usage["unparsed"] += 1
                continue
            verdicts[p["id"]] = v
            new_records.append({"key": k, "celex": celex, "a": p["A"]["name"], "b": p["B"]["name"], "verdict": v})
    _append_cache(cfg["cache_path"], new_records)
    return verdicts, usage


def judge_and_join(entities_with_clusters: pd.DataFrame, revisions: pd.DataFrame,
                   config: dict | None) -> tuple[pd.DataFrame, dict]:
    """Join class clusters the judge calls SAME (per act). Returns the updated frame and a report."""
    cfg = {**DEFAULTS, **(config or {})}
    report: dict[str, Any] = {"enabled": bool(cfg["enabled"])}
    if not cfg["enabled"] or entities_with_clusters.empty or not cfg.get("llm_backend"):
        return entities_with_clusters, report

    ents = entities_with_clusters.copy()
    cls = ents[ents["entity_type"] == "class"]
    # chunk ontologies and texts per act
    ontos: dict[str, list[dict]] = defaultdict(list)
    texts: dict[str, list[str]] = defaultdict(list)
    desc: dict[tuple[str, str], str] = {}
    for _, row in revisions.iterrows():
        celex = str(row.get("CELEX", "") or "")
        try:
            onto = json.loads(row["ontology"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(onto, dict):
            continue
        ontos[celex].append(onto)
        if isinstance(row.get("text"), str):
            texts[celex].append(row["text"])
        for c in onto.get("classes") or []:
            if isinstance(c, dict) and c.get("name") and c.get("description"):
                k = (celex, c["name"])
                if len(c["description"]) > len(desc.get(k, "")):
                    desc[k] = c["description"]

    parent = {cid: cid for cid in cls["cluster_id"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    min_rank = CONFIDENCE_RANK.get(cfg["min_confidence"], 1)
    all_verdicts, usage_total = [], Counter()
    for celex, group in cls.groupby("CELEX"):
        celex = str(celex)
        clusters = []
        for cid, cg in group.groupby("cluster_id"):
            name = Counter(cg["name"]).most_common(1)[0][0]
            types = Counter(t for t in cg["declared_type"] if t)
            clusters.append({"cluster_id": cid, "name": name,
                             "type": types.most_common(1)[0][0] if types else "",
                             "description": max((desc.get((celex, n), "") for n in cg["name"].unique()), key=len)})
        if len(clusters) < 2:
            continue
        pairs = candidate_pairs(clusters, float(cfg["name_sim"]), float(cfg["desc_sim"]))
        pairs.sort(key=lambda p: -max(p[2]["name_sim"], p[2]["desc_sim"], 1.0 if p[2]["acronym"] else 0.0))
        pairs = pairs[: int(cfg["max_pairs_per_act"])]
        if not pairs:
            continue
        sentences = list(dict.fromkeys(s for t in dict.fromkeys(texts[celex]) for s in _sentences(t)))
        for c in clusters:
            c["facts"] = _facts(ontos[celex], c["name"])
        payloads = []
        for pid, (i, j, _sig) in enumerate(pairs):
            a, b = clusters[i], clusters[j]
            snippets = {k: v for k, v in (("A", _snippet(sentences, a["name"])), ("B", _snippet(sentences, b["name"])),
                                           ("both", _snippet(sentences, a["name"], b["name"]))) if v}
            payloads.append(_pair_payload(pid, a, b, snippets))
        verdicts, usage = judge(payloads, celex, cfg)
        usage_total.update(usage)
        for pid, (i, j, sig) in enumerate(pairs):
            v = verdicts.get(pid)
            a, b = clusters[i], clusters[j]
            ok = accepted(v, payloads[pid]["act_text"], min_rank)
            rec = {"celex": celex, "a": a["name"], "b": b["name"], **sig,
                   "label": v["label"] if v else None, "confidence": v["confidence"] if v else None,
                   "basis": (v or {}).get("basis"), "evidence": (v or {}).get("evidence"), "joined": ok}
            all_verdicts.append(rec)
            if ok:
                ra, rb = find(a["cluster_id"]), find(b["cluster_id"])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    mask = ents["entity_type"] == "class"
    ents.loc[mask, "cluster_id"] = [find(c) for c in ents.loc[mask, "cluster_id"]]
    labels = Counter(v["label"] for v in all_verdicts)
    joined = sum(1 for v in all_verdicts if v["joined"])
    report.update({"model": cfg["model"], "prompt_version": PROMPT_VERSION, "pairs": len(all_verdicts),
                   "labels": dict(labels), "joined_pairs": joined, "usage": dict(usage_total),
                   "verdicts": all_verdicts})
    logger.info("Pair judge: %d pairs, %d joined as SAME, labels %s, usage %s.",
                len(all_verdicts), joined, dict(labels), dict(usage_total))
    return ents, report
