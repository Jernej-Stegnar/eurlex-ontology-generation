"""Skill library and lessons.

A skill is a folder `<skills_dir>/<name>/SKILL.md` with a front-matter block

    ---
    name: exception-handling
    description: one line, shown to the agent in its system prompt
    when_to_use: one line
    ---
    body: procedure, worked examples, pitfalls

Only names and descriptions go into the system prompt; the body is read on
demand through the load_skill tool. Lessons are short cross-act notes, one
per line starting with "- ", in `<skills_dir>/LESSONS.md`; all of them go into
the system prompt.

During extraction runs the library is read-only. It is written only by the
learning phase (scripts/agent_learn.py), which commits every accepted change.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

MAX_SKILLS = 15
MAX_SKILL_LINES = 150
MAX_LESSONS = 40
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


@dataclass
class Skill:
    name: str
    description: str
    when_to_use: str
    body: str


def _parse(text: str, folder: str) -> Skill:
    meta: dict[str, str] = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return Skill(name=meta.get("name", folder), description=meta.get("description", ""),
                 when_to_use=meta.get("when_to_use", ""), body=body.strip())


class Library:
    def __init__(self, skills_dir: str | Path | None):
        self.dir = Path(skills_dir) if skills_dir else None
        self.skills: dict[str, Skill] = {}
        self.lessons: list[str] = []
        if self.dir and self.dir.exists():
            for f in sorted(self.dir.glob("*/SKILL.md")):
                s = _parse(f.read_text(encoding="utf-8"), f.parent.name)
                self.skills[s.name] = s
            lessons = self.dir / "LESSONS.md"
            if lessons.exists():
                self.lessons = [ln[2:].strip() for ln in lessons.read_text(encoding="utf-8").splitlines()
                                if ln.startswith("- ") and ln[2:].strip()]

    def index_text(self) -> str:
        if not self.skills:
            return "(no skills yet)"
        return "\n".join(f"- {s.name}: {s.description}" + (f" Use when: {s.when_to_use}" if s.when_to_use else "")
                         for s in self.skills.values())

    def lessons_text(self) -> str:
        return "\n".join(f"- {x}" for x in self.lessons) if self.lessons else "(no lessons yet)"

    def load(self, name: str) -> str | None:
        s = self.skills.get(name)
        return None if s is None else s.body

    def content_hash(self) -> str:
        """sha256 over every file in the library, so each run records exactly what it used."""
        h = hashlib.sha256()
        if self.dir and self.dir.exists():
            for f in sorted(p for p in self.dir.rglob("*") if p.is_file()):
                h.update(f.relative_to(self.dir).as_posix().encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()[:16]

    def check_caps(self) -> list[str]:
        """Problems with the library's size and names (used by the learning gate)."""
        problems = []
        if len(self.skills) > MAX_SKILLS:
            problems.append(f"{len(self.skills)} skills > {MAX_SKILLS}")
        for s in self.skills.values():
            if not _NAME.match(s.name):
                problems.append(f"bad skill name {s.name!r}")
            if not s.description:
                problems.append(f"skill {s.name} has no description")
            if len(s.body.splitlines()) > MAX_SKILL_LINES:
                problems.append(f"skill {s.name} has more than {MAX_SKILL_LINES} lines")
        if len(self.lessons) > MAX_LESSONS:
            problems.append(f"{len(self.lessons)} lessons > {MAX_LESSONS}")
        return problems
