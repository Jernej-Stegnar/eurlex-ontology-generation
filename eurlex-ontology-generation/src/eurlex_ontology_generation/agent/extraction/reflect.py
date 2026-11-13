"""Learning phase (T9): turn the extraction agent's errors on TRAINING acts into skills and lessons.

1. build_dossier(): from a scored training run (judged model rules + per-gold-row status, as
   written by eval_gold/score_train.py) and the chunk texts, a compact error report: gold rules
   missed, wrong type/actor/object, correct but incomplete (strict fails), and model rules that
   match no gold rule.
2. reflect(): the same model, in its own tool loop, reads the dossier plus the current library and
   edits a COPY of the library (write/delete skills, add/remove lessons). It never sees evaluation
   acts; text naming an evaluation or held-out act is refused.
The caller (runs/scripts/agent_learn.py) gates the edited library on the dev acts and commits it
only if the dev score improves. Extraction runs read the library and never write it.
"""
from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ...utils.llm_client import ChatTurn, llm_chat_messages
from .library import _NAME, MAX_LESSONS, MAX_SKILL_LINES, MAX_SKILLS, Library

MAX_SKILL_EDITS = 3      # per iteration, so each accepted change stays attributable
MAX_LESSON_EDITS = 6
GOLD_FIELDS = ["article", "rule_type", "assignee", "action", "target", "constraints", "flags"]


# ---------------------------------------------------------------- dossier
def _window(text: str, quote: str, w: int = 350) -> str:
    flat, q = re.sub(r"\s+", " ", text or ""), re.sub(r"\s+", " ", quote or "").strip()
    i = flat.lower().find(q.lower()[:80]) if q else -1
    if i < 0:
        return flat[:2 * w]
    return "..." + flat[max(0, i - w):i + len(q) + w] + "..."


def _rule(r) -> str:
    return (f"{r.get('rule_type', '')} | party: {r.get('party', '')} | action: {r.get('action_label') or r.get('action', '')}"
            f" | asset: {r.get('asset', '')} | constraints: {r.get('constraints', '') or '-'}")


def _gold(r) -> str:
    return " | ".join(f"{k}: {r.get(k, '') or '-'}" for k in GOLD_FIELDS)


def build_dossier(judged: pd.DataFrame, gold_status: pd.DataFrame, chunk_text: dict[str, str],
                  per_section: int = 15) -> str:
    """judged: score_run columns (chunk_id, rule_type, party, action_label, asset, constraints,
    provision, j_cls, j_gold_id, j_object, j_constraints, j_actor, j_note, j_strict);
    gold_status: gold rows with chunks, in_input, found, correct, strict."""
    j = judged.fillna("")
    g = gold_status.fillna("")
    g = g[g["in_input"].astype(str).str.lower() == "true"]
    gi = g.set_index("gold_id")
    flag = lambda s: s.str.contains("exception|nested_condition|sanction|amendment|validity")  # noqa: E731
    out = []
    n_model, n_matched = len(j), int((j.j_gold_id != "").sum())
    found = g.found.astype(str).str.lower() == "true"
    correct = g.correct.astype(str).str.lower() == "true"
    strict = g.strict.astype(str).str.lower() == "true"
    out.append("# Error dossier (training acts only)\n"
               f"Gold rules in input: {len(g)}; found {int(found.sum())}, correct {int(correct.sum())}, "
               f"complete (strict) {int(strict.sum())}. Model rules: {n_model}, matching a gold rule: {n_matched}.")
    eo = g[flag(g["flags"])]
    out.append("Hard cases (gold flagged exception / nested_condition / sanction / amendment / validity): "
               f"{len(eo)}, found {int(eo.found.astype(str).str.lower().eq('true').sum())}.")
    by_flag = []
    for fl in ["exception", "nested_condition", "sanction", "amendment", "validity"]:
        s = g[g["flags"].str.contains(fl)]
        if len(s):
            by_flag.append(f"{fl} {int(s.found.astype(str).str.lower().eq('true').sum())}/{len(s)}")
    out.append("Found by flag: " + ", ".join(by_flag))

    missed = g[~found].copy()
    missed["_eo"] = flag(missed["flags"])
    missed = missed.sort_values("_eo", ascending=False).head(per_section)
    out.append(f"\n## A. Gold rules the agent MISSED ({int((~found).sum())}; showing {len(missed)}, hard cases first)")
    for _, r in missed.iterrows():
        cid = (r.chunks.split(";") or [""])[0]
        emitted = j[j.chunk_id == cid]
        out.append(f"\n### {r.gold_id} (chunk {cid})\nGOLD: {_gold(r)}\nTEXT: {_window(chunk_text.get(cid, ''), r.quote)}\n"
                   f"AGENT EMITTED FOR THIS CHUNK ({len(emitted)} rules): "
                   + ("; ".join(_rule(x) for _, x in emitted.head(6).iterrows()) or "none"))

    wrong = j[j.j_cls.str.startswith("wrong")].head(per_section)
    out.append(f"\n## B. Matched but WRONG type/actor/object ({int(j.j_cls.str.startswith('wrong').sum())})")
    for _, r in wrong.iterrows():
        gr = gi.loc[r.j_gold_id] if r.j_gold_id in gi.index else {}
        out.append(f"\n### {r.j_gold_id} ({r.j_cls})\nAGENT: {_rule(r)}\nGOLD: {_gold(gr)}\nJUDGE: {r.j_note}")

    partial = j[(j.j_cls == "correct") & (j.j_strict.astype(str).str.lower() != "true")].head(per_section)
    n_partial = int(((j.j_cls == "correct") & (j.j_strict.astype(str).str.lower() != "true")).sum())
    out.append(f"\n## C. Correct but INCOMPLETE ({n_partial}): object or conditions missing")
    for _, r in partial.iterrows():
        gr = gi.loc[r.j_gold_id] if r.j_gold_id in gi.index else {}
        out.append(f"\n### {r.j_gold_id}\nAGENT: {_rule(r)}\nGOLD: {_gold(gr)}\n"
                   f"JUDGE: object={r.j_object}, constraints={r.j_constraints}; {r.j_note}")

    extra = j[j.j_gold_id == ""].head(per_section)
    out.append(f"\n## D. Agent rules matching NO gold rule ({int((j.j_gold_id == '').sum())}). Usually not a rule "
               "(recital, definition, scope, format), sometimes a rule the gold left out: judge from the text.")
    for _, r in extra.iterrows():
        out.append(f"\n### chunk {r.chunk_id}\nAGENT: {_rule(r)} | provision: {r.get('provision', '')}\n"
                   f"TEXT: {_window(chunk_text.get(r.chunk_id, ''), str(r.get('asset', '')))[:500]}")
    return "\n".join(out)


# ------------------------------------------------------------- reflection
SYSTEM = """You improve the procedures of an extraction agent that turns chunks of EU legal acts into ODRL rules (permissions, prohibitions, duties with party, action, asset and constraints). The agent works chunk by chunk with tools and finishes by submitting an ontology. It reads LESSONS (short notes, all shown in its system prompt) and SKILLS (procedures it loads by name when the description fits).

You get an error dossier from a run on TRAINING acts scored against hand-checked gold rules, plus the current lessons and skills. Change the library so the agent makes fewer of these errors on NEW acts:
- A skill is ONE recurring legal pattern (e.g. derogations and exceptions, time limits and validity, alternative conditions, amending acts, approvals with conditions): how to recognise it in the wording, how to encode it (which list, party, action, asset, constraints; separate rule or constraint), one short worked example taken from the dossier text, and pitfalls.
- A lesson is one sentence of general advice.
- Write procedures that generalise. Never write facts about one act, never name a specific act number. Examples may quote the dossier text.
- Fix the most frequent and most costly errors first (missed hard cases, wrong types, incomplete constraints). Do not encourage extracting non-rules: precision matters as much as recall.
- Prefer editing an existing skill to adding a near-duplicate. Keep skills short (max 150 lines) and the description one line that says when to load it.
- At most 3 skill writes or deletes and 6 lesson changes in this round. Then call finish with a short summary of what you changed and why.
Work only through the tools."""


@dataclass
class Edits:
    skills_written: list[str] = field(default_factory=list)
    skills_deleted: list[str] = field(default_factory=list)
    lessons_added: list[str] = field(default_factory=list)
    lessons_removed: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    summary: str = ""
    finished: bool = False


class LibraryEditor:
    """Edits a library directory (a copy, never the frozen one) under the reflection limits."""

    def __init__(self, lib_dir: Path, forbidden: list[str]):
        self.dir = Path(lib_dir)
        self.forbidden = [f.lower() for f in forbidden if f]
        self.e = Edits()

    def _lessons_path(self) -> Path:
        return self.dir / "LESSONS.md"

    def _lessons(self) -> tuple[list[str], list[str]]:
        p = self._lessons_path()
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else ["# Lessons"]
        head = [ln for ln in lines if not ln.startswith("- ")]
        return head, [ln[2:].strip() for ln in lines if ln.startswith("- ")]

    def _write_lessons(self, head, items) -> None:
        self._lessons_path().write_text("\n".join(head + [f"- {x}" for x in items]) + "\n", encoding="utf-8")

    def _leaks(self, text: str) -> str | None:
        low = text.lower()
        for f in self.forbidden:
            if f in low:
                return f
        return None

    def _skill_budget(self) -> bool:
        return len(self.e.skills_written) + len(self.e.skills_deleted) < MAX_SKILL_EDITS

    def write_skill(self, name: str, description: str, when_to_use: str, body: str) -> str:
        if not self._skill_budget():
            return f"refused: at most {MAX_SKILL_EDITS} skill writes/deletes per round"
        if not _NAME.match(name or ""):
            return "refused: name must be lowercase words joined by hyphens (2-49 characters)"
        if name == "writing-skills":
            return "refused: writing-skills is the format guide, not an extraction skill"
        if leak := self._leaks(" ".join([name, description, when_to_use, body])):
            self.e.refused.append(f"{name}: names {leak}")
            return f"refused: text names a specific act ({leak}); write general procedures"
        if len(body.splitlines()) > MAX_SKILL_LINES:
            return f"refused: body longer than {MAX_SKILL_LINES} lines"
        exists = (self.dir / name / "SKILL.md").exists()
        if not exists and len(Library(self.dir).skills) >= MAX_SKILLS:
            return f"refused: library already has {MAX_SKILLS} skills; edit or delete one"
        one = lambda s: " ".join(str(s).split())  # noqa: E731
        (self.dir / name).mkdir(parents=True, exist_ok=True)
        (self.dir / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {one(description)}\nwhen_to_use: {one(when_to_use)}\n---\n{body.strip()}\n",
            encoding="utf-8")
        self.e.skills_written.append(name)
        return f"ok: skill {name} {'replaced' if exists else 'created'}"

    def delete_skill(self, name: str) -> str:
        if not self._skill_budget():
            return f"refused: at most {MAX_SKILL_EDITS} skill writes/deletes per round"
        d = self.dir / str(name)
        if name == "writing-skills" or not (d / "SKILL.md").exists():
            return f"refused: no deletable skill {name!r}"
        shutil.rmtree(d)
        self.e.skills_deleted.append(name)
        return f"ok: skill {name} deleted"

    def add_lesson(self, text: str) -> str:
        if len(self.e.lessons_added) + len(self.e.lessons_removed) >= MAX_LESSON_EDITS:
            return f"refused: at most {MAX_LESSON_EDITS} lesson changes per round"
        text = " ".join(str(text).split())
        if not text or len(text) > 300:
            return "refused: a lesson is one sentence (1-300 characters)"
        if leak := self._leaks(text):
            self.e.refused.append(f"lesson: names {leak}")
            return f"refused: text names a specific act ({leak})"
        head, items = self._lessons()
        if len(items) >= MAX_LESSONS:
            return f"refused: already {MAX_LESSONS} lessons; remove one first"
        self._write_lessons(head, items + [text])
        self.e.lessons_added.append(text)
        return f"ok: lesson {len(items) + 1} added"

    def remove_lesson(self, number: int) -> str:
        if len(self.e.lessons_added) + len(self.e.lessons_removed) >= MAX_LESSON_EDITS:
            return f"refused: at most {MAX_LESSON_EDITS} lesson changes per round"
        head, items = self._lessons()
        try:
            gone = items.pop(int(number) - 1)
        except (ValueError, IndexError):
            return f"refused: no lesson {number}"
        self._write_lessons(head, items)
        self.e.lessons_removed.append(gone)
        return f"ok: lesson removed: {gone}"

    def finish(self, summary: str) -> str:
        self.e.summary, self.e.finished = str(summary)[:2000], True
        return "ok"


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req, "additionalProperties": False}}}


TOOLS = [
    _fn("write_skill", "Create or replace a skill (SKILL.md).",
        {"name": {"type": "string"}, "description": {"type": "string"}, "when_to_use": {"type": "string"},
         "body": {"type": "string"}}, ["name", "description", "when_to_use", "body"]),
    _fn("delete_skill", "Delete a skill that hurts or duplicates another.", {"name": {"type": "string"}}, ["name"]),
    _fn("add_lesson", "Add one lesson (one sentence).", {"text": {"type": "string"}}, ["text"]),
    _fn("remove_lesson", "Remove lesson number N (1-based, as listed).", {"number": {"type": "integer"}}, ["number"]),
    _fn("finish", "End the round with a short summary of the changes and why.", {"summary": {"type": "string"}}, ["summary"]),
]


def library_text(lib_dir: Path) -> str:
    lib = Library(lib_dir)
    lessons = "\n".join(f"{i}. {x}" for i, x in enumerate(lib.lessons, 1)) or "(none)"
    skills = "\n\n".join(f"### skill {s.name}\ndescription: {s.description}\nwhen_to_use: {s.when_to_use}\n{s.body}"
                         for s in lib.skills.values()) or "(none)"
    return f"## Current lessons\n{lessons}\n\n## Current skills\n{skills}"


def reflect(lib_dir: Path, dossier: str, *, backend: dict, model: str, forbidden: list[str],
            temperature: float = 0.2, seed: int | None = 42, max_steps: int = 10,
            llm: Callable[..., ChatTurn] | None = None, trace_path: Path | None = None) -> Edits:
    """Edit the library in lib_dir (a copy) from the dossier. Returns what was changed."""
    ed = LibraryEditor(lib_dir, forbidden)
    call = llm or (lambda msgs, tools, tc: llm_chat_messages(backend, model, msgs, tools=tools, tool_choice=tc,
                                                            temperature=temperature, seed=seed, max_tokens=8192))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": library_text(lib_dir) + "\n\n" + dossier}]
    handlers = {"write_skill": ed.write_skill, "delete_skill": ed.delete_skill, "add_lesson": ed.add_lesson,
                "remove_lesson": ed.remove_lesson, "finish": ed.finish}
    trace = []
    for step in range(max_steps):
        tc = {"type": "function", "function": {"name": "finish"}} if step == max_steps - 1 else "auto"
        turn = call(msgs, TOOLS, tc)
        msgs.append(turn.message)
        trace.append({"step": step, "tokens_in": turn.prompt_tokens, "tokens_out": turn.output_tokens,
                      "cache_hit": turn.cache_hit_tokens, "calls": [c.name for c in turn.tool_calls]})
        if not turn.tool_calls:
            msgs.append({"role": "user", "content": "Use the tools; call finish when done."})
            continue
        for c in turn.tool_calls:
            if c.parse_error:
                out = f"error: {c.parse_error}"
            elif c.name not in handlers:
                out = f"error: unknown tool {c.name}"
            else:
                try:
                    out = handlers[c.name](**(c.args or {}))
                except TypeError as exc:
                    out = f"error: bad arguments: {exc}"
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            trace[-1].setdefault("results", []).append(out[:200])
        if ed.e.finished:
            break
    if trace_path:
        trace_path.write_text(json.dumps({"trace": trace, "edits": ed.e.__dict__}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    return ed.e
