"""Learning phase: error dossier, library editor guard rails, reflection round with a scripted model."""
import json
import shutil
from pathlib import Path

import pandas as pd

from eurlex_ontology_generation.agent.extraction.library import Library
from eurlex_ontology_generation.agent.extraction.reflect import (
    MAX_SKILL_EDITS, LibraryEditor, build_dossier, library_text, reflect)

from .test_agent_extraction import FakeLLM

ROOT = Path(__file__).resolve().parents[1]


def _lib(tmp_path) -> Path:
    d = tmp_path / "lib"
    shutil.copytree(ROOT / "agent_skills", d)
    return d


def _scored():
    judged = pd.DataFrame([
        {"chunk_id": "X_1", "rule_type": "duty", "party": "MemberState", "action": "notify", "action_label": "notify_measures",
         "asset": "Measure", "constraints": "", "provision": "article", "j_cls": "correct", "j_gold_id": "G-1",
         "j_object": "partial", "j_constraints": "no", "j_actor": "same", "j_note": "deadline missing", "j_strict": "False"},
        {"chunk_id": "X_1", "rule_type": "duty", "party": "MemberState", "action": "postpone", "action_label": "postpone_notification",
         "asset": "notification", "constraints": "", "provision": "article", "j_cls": "wrong_type", "j_gold_id": "G-2",
         "j_object": "yes", "j_constraints": "yes", "j_actor": "same", "j_note": "gold is a permission", "j_strict": "False"},
        {"chunk_id": "X_2", "rule_type": "duty", "party": "Commission", "action": "publish", "action_label": "publish_report",
         "asset": "report", "constraints": "", "provision": "recital", "j_cls": "unmatched", "j_gold_id": "",
         "j_object": "", "j_constraints": "", "j_actor": "", "j_note": "", "j_strict": "False"},
    ])
    gold = pd.DataFrame([
        {"gold_id": "G-1", "article": "Art. 2(1)", "rule_type": "duty", "assignee": "Member States", "action": "notify",
         "target": "measures", "constraints": "by 31 December 2019", "flags": "validity", "quote": "Member States shall notify",
         "chunks": "X_1", "in_input": "True", "found": "True", "correct": "True", "strict": "False"},
        {"gold_id": "G-2", "article": "Art. 2(2)", "rule_type": "permission", "assignee": "Member States", "action": "postpone",
         "target": "notification", "constraints": "", "flags": "exception", "quote": "may postpone",
         "chunks": "X_1", "in_input": "True", "found": "True", "correct": "False", "strict": "False"},
        {"gold_id": "G-3", "article": "Art. 3", "rule_type": "permission", "assignee": "Member States", "action": "be exempt",
         "target": "notification", "constraints": "where products in Annex I", "flags": "exception ; nested_condition",
         "quote": "By way of derogation", "chunks": "X_2", "in_input": "True", "found": "False", "correct": "False", "strict": "False"},
        {"gold_id": "G-4", "article": "Art. 9", "rule_type": "none", "assignee": "", "action": "enter into force", "target": "",
         "constraints": "", "flags": "validity", "quote": "not in input", "chunks": "", "in_input": "False",
         "found": "False", "correct": "False", "strict": "False"},
    ])
    texts = {"X_1": "Member States shall notify the Commission by 31 December 2019. Member States may postpone it.",
             "X_2": "By way of derogation from Article 2, Member States may ... Whereas the Commission publishes reports."}
    return judged, gold, texts


def test_dossier_sections_and_counts():
    judged, gold, texts = _scored()
    d = build_dossier(judged, gold, texts)
    assert "Gold rules in input: 3; found 2, correct 1, complete (strict) 0. Model rules: 3, matching a gold rule: 2." in d
    assert "exception 1/2" in d and "validity 1/1" in d          # G-4 (not in input) excluded
    assert "## A. Gold rules the agent MISSED (1" in d and "### G-3 (chunk X_2)" in d and "By way of derogation" in d
    assert "### G-2 (wrong_type)" in d and "gold is a permission" in d
    assert "## C. Correct but INCOMPLETE (1)" in d and "constraints=no" in d
    assert "## D. Agent rules matching NO gold rule (1)" in d and "provision: recital" in d


def test_editor_guard_rails(tmp_path):
    lib = _lib(tmp_path)
    ed = LibraryEditor(lib, forbidden=["32015r1474", "2015/1474"])
    assert ed.write_skill("Bad Name", "d", "w", "b").startswith("refused")
    assert ed.write_skill("writing-skills", "d", "w", "b").startswith("refused")
    assert "names a specific act" in ed.write_skill("exceptions", "d", "w", "As in Regulation (EU) 2015/1474 ...")
    assert ed.write_skill("exceptions", "derogations", "'by way of derogation'", "Encode as a permission.").startswith("ok")
    assert ed.write_skill("validity", "dates", "'until'", "x\n" * 200).startswith("refused")   # too long
    assert ed.write_skill("validity", "dates", "'until'", "Put dates in constraints.").startswith("ok")
    assert ed.delete_skill("exceptions").startswith("ok")
    assert ed.write_skill("more", "d", "w", "b").startswith("refused: at most")                # budget of 3 used
    assert len(ed.e.skills_written) + len(ed.e.skills_deleted) == MAX_SKILL_EDITS
    assert ed.add_lesson("Recitals are not rules.").startswith("ok")
    assert ed.remove_lesson(5).startswith("refused") and ed.remove_lesson(1).startswith("ok")
    lib2 = Library(lib)
    assert sorted(lib2.skills) == ["validity", "writing-skills"] and lib2.lessons == [] and lib2.check_caps() == []


def test_reflection_round_with_scripted_model(tmp_path):
    lib = _lib(tmp_path)
    before = Library(lib).content_hash()
    body = "1. Recognise: 'by way of derogation', 'shall not apply'.\n2. Encode: separate permission rule.\n"
    llm = FakeLLM([
        [("write_skill", {"name": "exceptions", "description": "derogations and lifted duties",
                          "when_to_use": "text says by way of derogation", "body": body}),
         ("add_lesson", {"text": "Keep a derogation as its own permission rule, not a constraint."})],
        [("finish", {"summary": "added exceptions skill and one lesson"})],
    ])
    trace = tmp_path / "trace.json"
    e = reflect(lib, "# Error dossier\n...", backend={}, model="fake", forbidden=["32015r1474"], llm=llm, trace_path=trace)
    assert e.finished and e.skills_written == ["exceptions"] and len(e.lessons_added) == 1
    lib2 = Library(lib)
    assert lib2.skills["exceptions"].body.startswith("1. Recognise") and lib2.content_hash() != before
    assert "WORKING" not in llm.calls[0]["messages"][0]["content"]          # reflection prompt, not extraction prompt
    assert "## Current skills" in llm.calls[0]["messages"][1]["content"] and "Error dossier" in llm.calls[0]["messages"][1]["content"]
    assert json.loads(trace.read_text(encoding="utf-8"))["edits"]["summary"].startswith("added")
    assert "exceptions" in library_text(lib)


def test_reflection_forces_finish_on_last_step(tmp_path):
    lib = _lib(tmp_path)
    llm = FakeLLM([], default=lambda m, tc: [("finish", {"summary": "nothing"})] if isinstance(tc, dict) else [])
    e = reflect(lib, "d", backend={}, model="fake", forbidden=[], llm=llm, max_steps=3)
    assert e.finished and [c["tool_choice"] for c in llm.calls][-1]["function"]["name"] == "finish"
