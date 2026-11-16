"""Act-level aliases taken from the act's own text, applied before entity resolution.

An act defines its acronyms once ("Crypto Approval Authority (CAA)", "the Animal and Plant
Health Inspection Service (hereinafter 'APHIS')"); chunks after the definition use only the
acronym. The LLM then names the same body "CAA" in one chunk and "CryptoApprovalAuthority" in
another, and the name-based merge cannot see that they are one entity. This module reads the
definitions from the chunk texts of each act and renames every entity named exactly by such an
acronym to its long form, inside that act only (an acronym can mean different things in
different acts).

Acronym detection follows Schwartz & Hearst (2003): the acronym has 2-8 capitals and its letters
occur in order in the words just before the parenthesis, the first letter starting a word.
"""
import json
import logging
import re
from collections import defaultdict
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ACT_ALIASES = True     # False -> merge sees the names exactly as generated

# legal-act series and nomenclature codes, not entities
SKIP_ACRONYMS = {"EU", "EC", "EEC", "EAEC", "CN", "HS", "OJ", "TFEU", "TEU"}
_SMALL = {"and", "of", "for", "on", "the", "in", "to", "at", "de", "des", "du"}
_DEF = re.compile(
    r"([A-Za-z][\w\- ,]{2,120}?)\s*\((?:hereinafter[^)]{0,30}?)?[‘'\"“]?([A-Z]{2,8})s?[’'\"”]?\)")

# JSON fields that hold an entity name (string) or a list of names
NAME_FIELDS = {"name", "party", "asset", "source", "target", "is_a", "instance_of", "entity"}
NAME_LIST_FIELDS = {"entities_involved"}


def _long_form(words: list[str], acr: str) -> str | None:
    """Shortest tail of `words` whose letters contain `acr` in order, starting at a word start."""
    letters = acr.lower()
    n_max = min(len(letters) + 5, 2 * len(letters) + 2, len(words))
    for n in range(1, n_max + 1):
        cand = words[-n:]
        if cand[0].lower() in _SMALL or not cand[0][0].isalpha() or cand[0][0].lower() != letters[0]:
            continue
        s = " ".join(cand).lower()
        i = 0
        for ch in letters:
            i = s.find(ch, i)
            if i < 0:
                break
            i += 1
        else:
            content = [w for w in cand if w.lower() not in _SMALL]
            if len(content) <= len(letters) + 1:
                return " ".join(cand)
    return None


def acronym_definitions(text: str) -> dict[str, str]:
    """{acronym: long form} for every acronym the text defines; the first definition wins."""
    out: dict[str, str] = {}
    for m in _DEF.finditer(text or ""):
        acr = m.group(2)
        if acr in SKIP_ACRONYMS or acr in out:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z\-]*", m.group(1))
        lf = _long_form(words, acr)
        if lf:
            out[acr] = lf
    return out


def _camel(phrase: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[\s\-]+", phrase) if w)


def act_alias_maps(revisions: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{CELEX: {name key of the acronym: long-form name}} from the chunk texts of each act."""
    if "text" not in revisions.columns or "CELEX" not in revisions.columns:
        return {}
    from .entity_resolution import group_key   # local import: entity_resolution builds tables at import

    maps: dict[str, dict[str, str]] = {}
    for celex, texts in revisions.groupby("CELEX")["text"]:
        text = "\n".join(dict.fromkeys(str(t) for t in texts if isinstance(t, str)))
        defs = acronym_definitions(text)
        if defs:
            maps[str(celex)] = {group_key("class", acr): _camel(lf) for acr, lf in defs.items()}
    return maps


def _rename(obj: Any, mapping: dict[str, str], key_of, counter: dict) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in NAME_FIELDS and isinstance(v, str):
                new = mapping.get(key_of(v))
                if new and new != v:
                    counter[f"{v} -> {new}"] += 1
                    v = new
            elif k in NAME_LIST_FIELDS and isinstance(v, list):
                v = [mapping.get(key_of(x), x) if isinstance(x, str) else x for x in v]
            else:
                v = _rename(v, mapping, key_of, counter)
            out[k] = v
        return out
    if isinstance(obj, list):
        return [_rename(x, mapping, key_of, counter) for x in obj]
    return obj


def apply_act_aliases(revisions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Rename acronym-named entities to their long form, per act. Returns the rewritten revisions
    and a report {CELEX: {"old -> new": count}}."""
    report: dict[str, dict[str, int]] = {}
    if not ACT_ALIASES or revisions.empty:
        return revisions, report
    maps = act_alias_maps(revisions)
    if not maps:
        return revisions, report
    from .entity_resolution import group_key

    def key_of(name: str) -> str:
        return group_key("class", name)

    out = revisions.copy()
    new_col = []
    for celex, txt in zip(out["CELEX"].astype(str), out["ontology"]):
        mapping = maps.get(celex)
        if not mapping:
            new_col.append(txt)
            continue
        try:
            onto = json.loads(txt)
        except (TypeError, json.JSONDecodeError):
            new_col.append(txt)
            continue
        counter: dict = defaultdict(int)
        onto = _rename(onto, mapping, key_of, counter)
        if counter:
            rep = report.setdefault(celex, defaultdict(int))
            for k, n in counter.items():
                rep[k] += n
        new_col.append(json.dumps(onto, ensure_ascii=False))
    out["ontology"] = new_col
    n = sum(sum(r.values()) for r in report.values())
    if n:
        logger.info("Act aliases: %d names renamed to their long form in %d acts.", n, len(report))
    return out, {c: dict(r) for c, r in report.items()}
