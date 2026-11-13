"""Tools the extraction agent can call. All offline and deterministic: they read
the act's own chunks (the same text the pipeline gives the model), the act
memory, the skill library and the gazetteer. No web, Wikidata or CELLAR calls,
so runs are reproducible and send nothing outward.

`submit_ontology` is the only way to finish. It validates the ontology
(schema, normalization, the review step's deterministic checks, harness
checks) and either accepts it or returns the problems so the model can fix
them; this replaces the separate review -> revise loop.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ...pipelines.ontology_review.ontology_checks import run_deterministic_checks
from ...utils.entity_linking import Gazetteer
from ...utils.llm_client import inline_refs
from ...utils.ontology_normalize import normalize_chunk_ontology
from ...utils.vocabulary import Vocabulary
from .library import Library
from .memory import ActMemory

MAX_TOOL_CHARS = 6000
STOPWORDS = set("the a an and or of to in on for by with from as at be is are shall may not this that "
                "which such any its their other under referred article paragraph".split())
HEADING = re.compile(r"^\s*(Article\s+\d+[a-z]?|ANNEX\s*[IVXLC]*|CHAPTER\s+[IVXLC\d]+|SECTION\s+\d+|TITLE\s+[IVXLC]+)\b.*$",
                     re.IGNORECASE | re.MULTILINE)
DEFINITION = re.compile(r"[‘'\"“]([^’'\"”]{2,80})[’'\"”]\s+(?:means|shall mean|refers to)\b[^;\n]{0,400}",
                        re.IGNORECASE)


@dataclass
class Act:
    celex: str
    act_name: str
    act_type: str
    date: str
    chunks: list[dict]                    # rows with chunk_id, chunk_number, chunk_text; sorted

    def index_of(self, chunk_id: str) -> int:
        for i, c in enumerate(self.chunks):
            if c["chunk_id"] == chunk_id:
                return i
        raise KeyError(chunk_id)


@dataclass
class EpisodeState:
    """Per-chunk state the tools share with the loop."""
    act: Act
    chunk: dict
    memory: ActMemory
    library: Library
    vocab: Vocabulary
    gazetteer: Gazetteer
    schema_model: Any
    max_submits: int = 3
    submits: int = 0
    accepted: dict | None = None          # ontology as submitted, once accepted
    last_valid: dict | None = None        # last schema-valid submission
    last_findings: list[dict] = field(default_factory=list)
    final_status: str = ""                # accepted | accepted_with_warnings
    skills_loaded: list[str] = field(default_factory=list)
    warned_once: bool = False


def _clip(text: str, n: int = MAX_TOOL_CHARS) -> str:
    return text if len(text) <= n else text[:n] + f"\n... [cut, {len(text) - n} more characters]"


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS]


# ------------------------------------------------------------ read tools
def act_outline(st: EpisodeState) -> str:
    lines = [f"CELEX {st.act.celex} | type {st.act.act_type} | date {st.act.date}",
             f"title: {st.act.act_name}", f"{len(st.act.chunks)} chunks; current chunk "
             f"{st.chunk['chunk_id']} is number {st.act.index_of(st.chunk['chunk_id']) + 1}", "headings:"]
    seen = set()
    for c in st.act.chunks:
        for m in HEADING.finditer(c["chunk_text"]):
            h = m.group(0).strip()[:100]
            if h.lower() not in seen:
                seen.add(h.lower())
                lines.append(f"- {h} [{c['chunk_id']}]")
    return _clip("\n".join(lines))


def read_chunk(st: EpisodeState, chunk_id: str) -> str:
    for c in st.act.chunks:
        if c["chunk_id"] == chunk_id:
            return _clip(c["chunk_text"])
    return f"error: {chunk_id} is not a chunk of act {st.act.celex}"


def neighbour_chunks(st: EpisodeState, k: int = 1) -> str:
    k = max(1, min(int(k or 1), 2))
    i = st.act.index_of(st.chunk["chunk_id"])
    parts = []
    for j in range(max(0, i - k), min(len(st.act.chunks), i + k + 1)):
        if j != i:
            c = st.act.chunks[j]
            parts.append(f"=== {c['chunk_id']} ({'before' if j < i else 'after'} the current chunk) ===\n{c['chunk_text']}")
    return _clip("\n\n".join(parts) or "(no neighbouring chunks)", 2 * MAX_TOOL_CHARS)


def search_act(st: EpisodeState, query: str, limit: int = 5) -> str:
    terms = _words(query)
    if not terms:
        return "error: query has no searchable words"
    hits = []
    for c in st.act.chunks:
        low = c["chunk_text"].lower()
        score = sum(low.count(t) for t in terms)
        if score:
            pos = min((low.find(t) for t in terms if t in low), default=0)
            hits.append((score, c["chunk_id"], c["chunk_text"][max(0, pos - 200):pos + 400]))
    hits.sort(key=lambda h: -h[0])
    if not hits:
        return "no matches in this act"
    return _clip("\n\n".join(f"[{cid}] ...{snip}..." for _, cid, snip in hits[:max(1, min(int(limit or 5), 10))]))


def find_definitions(st: EpisodeState) -> str:
    found = []
    for c in st.act.chunks:
        for m in DEFINITION.finditer(c["chunk_text"]):
            found.append(f"- {m.group(0).strip()} [{c['chunk_id']}]")
    return _clip("\n".join(dict.fromkeys(found)) or "no definitions found in this act")


def gazetteer_lookup(st: EpisodeState, name: str) -> str:
    e = st.gazetteer.lookup(name)
    if e is None:
        return json.dumps({"found": False})
    return json.dumps({"found": True, "name": e.name, "is_a": e.is_a, "type": e.core_type,
                       "kind": "individual"}, ensure_ascii=False)


def memory_read(st: EpisodeState, kind: str | None = None) -> str:
    return st.memory.read_text(kind or None)


def memory_write(st: EpisodeState, kind: str, key: str, value: str) -> str:
    return st.memory.write(kind, key, value, st.chunk["chunk_id"])


def load_skill(st: EpisodeState, name: str) -> str:
    body = st.library.load(name)
    if body is None:
        return f"error: no skill {name!r}; available: {sorted(st.library.skills) or 'none'}"
    if name not in st.skills_loaded:
        st.skills_loaded.append(name)
    return body


# ---------------------------------------------------------- check / submit
def _harness_findings(onto: dict, chunk_text: str) -> list[dict]:
    """Checks the review step does not do: is each rule anchored in THIS chunk?"""
    findings = []
    text_words = set(_words(chunk_text))
    for kind, rules in (onto.get("rights_model") or {}).items():
        for i, r in enumerate(rules or []):
            label = " ".join(str(r.get(k, "")) for k in ("action_label", "asset"))
            words = set(_words(label.replace("_", " ")))
            if words and not words & text_words:
                findings.append({"code": "rule_not_in_chunk", "severity": "warning",
                                 "message": f"rights_model.{kind}[{i}] ({r.get('action_label')}): none of its words "
                                            "appear in the current chunk. Keep only rules stated in this chunk; "
                                            "rules from neighbouring chunks belong to those chunks."})
    return findings


def check(st: EpisodeState, ontology: Any) -> tuple[dict | None, list[dict], list[dict], dict | None]:
    """-> (schema-valid ontology or None, blocking errors, warnings, normalized ontology)."""
    if not isinstance(ontology, dict):
        return None, [{"code": "not_an_object", "severity": "error", "message": "ontology must be a JSON object"}], [], None
    try:
        valid = st.schema_model.model_validate(ontology).model_dump()
    except ValidationError as exc:
        errs = [{"code": "schema", "severity": "error",
                 "message": f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"} for e in exc.errors()[:20]]
        return None, errs, [], None
    normalized, _changes = normalize_chunk_ontology(valid, st.vocab, st.gazetteer)
    _, findings = run_deterministic_checks(json.dumps(valid, ensure_ascii=False), st.chunk["chunk_text"])
    findings += _harness_findings(valid, st.chunk["chunk_text"])
    errors = [f for f in findings if f.get("severity") == "error"]
    warnings = [f for f in findings if f.get("severity") == "warning"]
    return valid, errors, warnings, normalized


def _report(errors: list[dict], warnings: list[dict]) -> str:
    out = [f"ERROR {f['code']}: {f['message']}" + (f" Fix: {f['fix']}" if f.get("fix") else "") for f in errors]
    out += [f"WARNING {f['code']}: {f['message']}" + (f" Fix: {f['fix']}" if f.get("fix") else "") for f in warnings]
    return "\n".join(out)


def check_ontology(st: EpisodeState, ontology: Any) -> str:
    valid, errors, warnings, _ = check(st, ontology)
    if valid is not None and not errors and not warnings:
        return "no problems found"
    return _clip(_report(errors, warnings))


def submit_ontology(st: EpisodeState, ontology: Any) -> str:
    st.submits += 1
    valid, errors, warnings, _ = check(st, ontology)
    st.last_findings = errors + warnings
    if valid is not None:
        st.last_valid = valid
    left = st.max_submits - st.submits
    if valid is None or errors:
        return _clip(f"REJECTED ({left} submits left). Fix these and submit again:\n{_report(errors, warnings)}")
    if warnings and not st.warned_once and left > 0:
        # first submission with warnings goes back once, like a review round
        st.warned_once = True
        return _clip("NOT YET ACCEPTED. Review these warnings against the text. Fix what is wrong, then submit "
                     f"again (unchanged if every warning is a false alarm). {left} submits left.\n"
                     + _report([], warnings))
    st.accepted = valid
    st.final_status = "accepted_with_warnings" if warnings else "accepted"
    return "ACCEPTED. The task for this chunk is finished."


# ------------------------------------------------------------- registry
def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or [],
                       "additionalProperties": False}}}


def tool_specs(schema_model: Any) -> list[dict]:
    onto_schema = inline_refs(schema_model.model_json_schema())
    onto_schema.pop("title", None)
    return [
        _fn("act_outline", "Act metadata (type, title, date) and the article/annex headings of the whole act, "
            "with the chunk each heading is in."),
        _fn("read_chunk", "Full text of any chunk of this act.",
            {"chunk_id": {"type": "string"}}, ["chunk_id"]),
        _fn("neighbour_chunks", "Text of the chunks before and after the current one (k = 1 or 2 on each side). "
            "For context only: extract rules only from the current chunk.",
            {"k": {"type": "integer", "minimum": 1, "maximum": 2}}),
        _fn("search_act", "Keyword search over this act's chunks, e.g. to find the paragraph a derogation refers to.",
            {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
        _fn("find_definitions", "All \"'X' means ...\" definitions in this act."),
        _fn("gazetteer_lookup", "Controlled name, upper class and kind of a known country, institution or body.",
            {"name": {"type": "string"}}, ["name"]),
        _fn("memory_read", "What you noted about this act in earlier chunks, plus the rules already extracted "
            "from earlier chunks (do not repeat those).",
            {"kind": {"type": "string", "enum": ["definition", "entity", "cross_ref", "note", "rules"]}}),
        _fn("memory_write", "Note something later chunks of this act will need: a definition, the canonical name "
            "of an entity, an open cross-reference (e.g. 'derogation from Art. 3(1)'), or a note.",
            {"kind": {"type": "string", "enum": ["definition", "entity", "cross_ref", "note"]},
             "key": {"type": "string"}, "value": {"type": "string"}}, ["kind", "key", "value"]),
        _fn("load_skill", "Read the full procedure of a skill listed in the system prompt.",
            {"name": {"type": "string"}}, ["name"]),
        _fn("check_ontology", "Check a draft ontology without submitting it. Returns schema errors and warnings.",
            {"ontology": onto_schema}, ["ontology"]),
        _fn("submit_ontology", "Submit the final ontology for the current chunk. This is the only way to finish. "
            "Returns ACCEPTED, or the problems to fix.",
            {"ontology": onto_schema}, ["ontology"]),
    ]


HANDLERS = {
    "act_outline": act_outline, "read_chunk": read_chunk, "neighbour_chunks": neighbour_chunks,
    "search_act": search_act, "find_definitions": find_definitions, "gazetteer_lookup": gazetteer_lookup,
    "memory_read": memory_read, "memory_write": memory_write, "load_skill": load_skill,
    "check_ontology": check_ontology, "submit_ontology": submit_ontology,
}


def call_tool(st: EpisodeState, name: str, args: dict | None) -> str:
    fn = HANDLERS.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return fn(st, **(args or {}))
    except TypeError as exc:
        return f"error: bad arguments for {name}: {exc}"
    except KeyError as exc:
        return f"error: {exc}"
