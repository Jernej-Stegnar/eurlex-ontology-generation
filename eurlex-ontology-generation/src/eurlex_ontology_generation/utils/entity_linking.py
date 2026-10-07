"""Entity linking: gazetteer lookup, class-vs-individual detection, optional Wikidata.

Order of evidence, strongest first:
  1. the seed gazetteer (resources/gazetteer.yml): known EU-law individuals
  2. deterministic name heuristics (acronyms, names built on a known individual)
  3. Wikidata (optional, network): P31 "instance of" vs P279 "subclass of"

Nothing here calls an LLM.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .vocabulary import RESOURCES_DIR, Vocabulary, split_identifier

logger = logging.getLogger(__name__)

DEFAULT_GAZETTEER_PATH = RESOURCES_DIR / "gazetteer.yml"

# head nouns that make "<Individual>_<noun>" a KIND of thing (a class), e.g.
# EGF_Mobilization, PRIMA_Activity: there are many mobilisations of the EGF.
_CLASS_HEAD_NOUNS = {
    "mobilisation", "mobilization", "application", "applicant", "payment", "decision",
    "contribution", "action", "activity", "participation", "participant", "funding",
    "financing", "proposal", "contract", "grant", "loan", "measure", "project",
    "beneficiary", "recipient", "user", "member", "call", "derogation", "monitoring",
    "access", "tool", "assistance", "operation", "implementation", "structure", "support",
}

# head nouns that make "<Individual>_<noun>" itself a single named thing:
# Cyprus_Government, Moldova_Reform_Roadmap, NationalBankOfMoldova
_INDIVIDUAL_HEAD_NOUNS = {
    "government", "territory", "roadmap", "policy", "strategy", "plan", "bank", "ministry",
    "parliament", "presidency", "authorities", "administration", "agenda", "secretariat",
}

# aliases too generic to be recognised inside a longer name
# ('EnergyUnionStrategy' is not about the European Union as an actor)
_GENERIC_EMBED_KEYS = {"union", "eu", "commission", "council", "parliament", "ombudsman",
                       "erasmus", "uk", "un"}


def entity_key(name: Any) -> str:
    """Spelling-insensitive key: 'Solidarity Fund' == 'SolidarityFund' == 'solidarity_fund'."""
    return "".join(split_identifier(str(name or "")))


@dataclass
class GazetteerEntry:
    name: str
    is_a: str
    core_type: str
    iri: str | None = None
    member_state: bool | None = None
    wikidata: str | None = None
    source: str = "gazetteer"

    def as_dict(self) -> dict:
        out = {"name": self.name, "is_a": self.is_a, "type": self.core_type,
               "kind": "individual", "link_source": self.source}
        if self.iri:
            out["iri"] = self.iri
        if self.member_state is not None:
            out["member_state"] = self.member_state
        if self.wikidata:
            out["wikidata"] = self.wikidata
        return out


class Gazetteer:
    def __init__(self, entries: list[GazetteerEntry], aliases: dict[str, str]):
        self.entries = {e.name: e for e in entries}
        self._index: dict[str, str] = {}
        for e in entries:
            self._index[entity_key(e.name)] = e.name
        for alias, canonical in aliases.items():
            self._index.setdefault(entity_key(alias), canonical)

    def lookup(self, name: Any) -> GazetteerEntry | None:
        canonical = self._index.get(entity_key(name))
        return self.entries.get(canonical) if canonical else None

    def alias_groups(self) -> dict[str, list[str]]:
        """canonical name -> list of known variants (for force-merging)."""
        groups: dict[str, list[str]] = {n: [] for n in self.entries}
        for key, canonical in self._index.items():
            groups[canonical].append(key)
        return groups

    def find_embedded(self, name: str) -> GazetteerEntry | None:
        """A known individual whose name is a proper prefix/infix of `name`
        ('Cyprus_Territory' -> Cyprus, 'EGF_Mobilization' -> EGF)."""
        tokens = split_identifier(name)
        for size in range(len(tokens) - 1, 0, -1):
            for start in range(0, len(tokens) - size + 1):
                key = "".join(tokens[start:start + size])
                hit = None if key in _GENERIC_EMBED_KEYS else self._index.get(key)
                if hit:
                    return self.entries[hit]
        return None


@lru_cache(maxsize=4)
def load_gazetteer(path: str | None = None, vocabulary_path: str | None = None) -> Gazetteer:
    from .vocabulary import load_vocabulary

    vocab = load_vocabulary(vocabulary_path)
    p = Path(path) if path else DEFAULT_GAZETTEER_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    prefix = raw.get("country_iri_prefix", "")
    entries: list[GazetteerEntry] = []
    aliases: dict[str, str] = {}
    for c in raw.get("countries", []):
        entries.append(GazetteerEntry(
            name=c["name"], is_a="Country", core_type=vocab.core_type_of("Country"),
            iri=f"{prefix}{c['iso3']}" if c.get("iso3") else None,
            member_state=bool(c.get("member_state")), wikidata=c.get("wikidata")))
        for a in c.get("aliases", []) or []:
            aliases[a] = c["name"]
    for e in raw.get("entities", []):
        core = vocab.core_type_of(e["is_a"])
        if core is None:
            raise ValueError(f"Gazetteer entry {e['name']}: unknown is_a {e['is_a']}")
        entries.append(GazetteerEntry(name=e["name"], is_a=e["is_a"], core_type=core,
                                      iri=e.get("iri"), wikidata=e.get("wikidata")))
        for a in e.get("aliases", []) or []:
            aliases[a] = e["name"]
    return Gazetteer(entries, aliases)


# ----------------------------------------------------------- class vs instance
def detect_kind(name: str, description: str, gazetteer: Gazetteer, vocab: Vocabulary,
                declared_kind: str | None = None) -> dict:
    """Decide whether `name` denotes a class or an individual.

    Returns {kind, confidence, evidence, related_individual?}. Confidence is
    'high' for gazetteer hits, 'medium' for structural evidence, 'low' for
    weak heuristics; 'low' results belong in a human review queue.
    """
    hit = gazetteer.lookup(name)
    if hit:
        return {"kind": "individual", "confidence": "high",
                "evidence": f"gazetteer entry {hit.name}", "gazetteer": hit.name}

    tokens = split_identifier(name)
    raw_parts = re.split(r"[_\s]+", str(name).strip())

    embedded = gazetteer.find_embedded(name)
    if embedded:
        # 'NationalBankOfMoldova': the head noun is the one before 'of'
        head = tokens[tokens.index("of") - 1] if "of" in tokens[1:] else (tokens[-1] if tokens else "")
        if head in _INDIVIDUAL_HEAD_NOUNS:
            return {"kind": "individual", "confidence": "medium",
                    "evidence": f"the '{head}' of individual {embedded.name}",
                    "related_individual": embedded.name}
        confidence = "medium" if (head in _CLASS_HEAD_NOUNS or head.rstrip("s") in _CLASS_HEAD_NOUNS) else "low"
        return {"kind": "class", "confidence": confidence,
                "evidence": f"kind of '{head}' specific to {embedded.name}",
                "related_individual": embedded.name}

    # acronym such as PRIMAIS / EFSD: a single proper name
    if len(raw_parts) == 1 and re.fullmatch(r"[A-Z][A-Z0-9]{2,}", raw_parts[0]):
        return {"kind": "individual", "confidence": "low", "evidence": "all-caps acronym"}

    if declared_kind in ("class", "individual"):
        return {"kind": declared_kind, "confidence": "low", "evidence": "declared by generator"}

    return {"kind": "class", "confidence": "low", "evidence": "default"}


# ---------------------------------------------------------------- Wikidata
class WikidataLinker:
    """Optional linker. Looks a label up on Wikidata and reads P31/P279.

    An item that has P31 (instance of) and no P279 (subclass of) is an
    individual; P279 without P31 is a class. Only exact label/alias matches
    are accepted, and the caller only consults Wikidata for names the
    gazetteer does not know (so 'Georgia' the country never reaches it).

    Results are cached in a JSON file so a corpus run makes each lookup once.
    """

    API = "https://www.wikidata.org/w/api.php"

    def __init__(self, cache_path: str, user_agent: str, max_lookups: int = 200,
                 pause_seconds: float = 0.2, timeout: float = 10.0):
        self.cache_path = Path(cache_path)
        self.user_agent = user_agent
        self.max_lookups = max_lookups
        self.pause_seconds = pause_seconds
        self.timeout = timeout
        self.lookups = 0
        self.cache: dict[str, Any] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}

    def _get(self, params: dict) -> dict:
        url = f"{self.API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def lookup(self, label: str) -> dict | None:
        text = re.sub(r"[_]+", " ", label).strip()
        key = text.lower()
        if key in self.cache:
            return self.cache[key]
        if self.lookups >= self.max_lookups:
            return None
        self.lookups += 1
        try:
            found = self._get({"action": "wbsearchentities", "search": text, "language": "en",
                               "format": "json", "limit": 5, "type": "item"})
            match = next((r for r in found.get("search", [])
                          if r.get("label", "").lower() == key
                          or any(a.lower() == key for a in r.get("aliases", []) or [])), None)
            result = None
            if match:
                ent = self._get({"action": "wbgetentities", "ids": match["id"], "props": "claims",
                                 "format": "json"})
                claims = ent.get("entities", {}).get(match["id"], {}).get("claims", {})
                p31 = [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                       for c in claims.get("P31", [])]
                result = {"qid": match["id"], "label": match.get("label"),
                          "description": match.get("description", ""),
                          "p31": [q for q in p31 if q],
                          "has_p31": bool(claims.get("P31")),
                          "has_p279": bool(claims.get("P279"))}
            self.cache[key] = result
            time.sleep(self.pause_seconds)
            return result
        except Exception as exc:  # network off, rate limit, ...
            logger.warning("Wikidata lookup failed for %r: %s", text, exc)
            return None

    def kind_of(self, label: str) -> tuple[str | None, dict | None]:
        r = self.lookup(label)
        if not r:
            return None, None
        if r["has_p31"] and not r["has_p279"]:
            return "individual", r
        if r["has_p279"] and not r["has_p31"]:
            return "class", r
        return None, r

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
