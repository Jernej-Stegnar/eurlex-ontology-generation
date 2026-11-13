"""Act memory: what the agent has learned about one act while reading it.

One JSON file per CELEX in the run's memory directory, so it survives across
batches of the same act. Written by the agent (definitions, entities,
cross-references, notes) and by the harness (one summary line per accepted
rule). Never shared between acts. Entries carry the chunk they came from, so
re-running a chunk first drops what that chunk wrote before (idempotent resume).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

KINDS = ("definition", "entity", "cross_ref", "note")
MAX_VALUE_CHARS = 400
MAX_READ_CHARS = 8000


class ActMemory:
    def __init__(self, celex: str, memory_dir: str | Path | None):
        self.celex = celex
        self.path = Path(memory_dir) / f"{celex}.json" if memory_dir else None
        self.entries: list[dict] = []     # {kind, key, value, chunk_id}
        self.rules: list[dict] = []       # {chunk_id, kind, party, action, asset}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries, self.rules = data.get("entries", []), data.get("rules", [])

    def forget_chunk(self, chunk_id: str) -> None:
        self.entries = [e for e in self.entries if e.get("chunk_id") != chunk_id]
        self.rules = [r for r in self.rules if r.get("chunk_id") != chunk_id]

    def write(self, kind: str, key: str, value: str, chunk_id: str) -> str:
        if kind not in KINDS:
            return f"error: kind must be one of {list(KINDS)}"
        key, value = str(key).strip()[:120], str(value).strip()[:MAX_VALUE_CHARS]
        if not key:
            return "error: key is empty"
        # one entry per (kind, key): a later write replaces the earlier one
        self.entries = [e for e in self.entries if not (e["kind"] == kind and e["key"].lower() == key.lower())]
        self.entries.append({"kind": kind, "key": key, "value": value, "chunk_id": chunk_id})
        return "ok"

    def add_rules(self, chunk_id: str, ontology: dict) -> None:
        rm = ontology.get("rights_model") or {}
        for kind in ("permissions", "prohibitions", "duties"):
            for r in rm.get(kind) or []:
                if isinstance(r, dict):
                    self.rules.append({"chunk_id": chunk_id, "kind": kind[:-1] if kind != "duties" else "duty",
                                       "party": r.get("party", ""), "action": r.get("action_label") or r.get("action", ""),
                                       "asset": r.get("asset", "")})

    def read_text(self, kind: str | None = None) -> str:
        lines = []
        for k in KINDS:
            if kind and k != kind:
                continue
            items = [e for e in self.entries if e["kind"] == k]
            if items:
                lines.append(f"## {k}")
                lines += [f"- {e['key']}: {e['value']} (from {e['chunk_id']})" for e in items]
        if (kind in (None, "rules")) and self.rules:
            lines.append("## rules already extracted from earlier chunks")
            lines += [f"- {r['chunk_id']}: {r['kind']} | {r['party']} | {r['action']} | {r['asset']}" for r in self.rules]
        text = "\n".join(lines) or "(act memory is empty)"
        if len(text) > MAX_READ_CHARS:
            text = text[-MAX_READ_CHARS:]
            text = "(older entries cut)\n" + text[text.find("\n") + 1:]
        return text

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"celex": self.celex, "entries": self.entries, "rules": self.rules},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
