"""Tests for the extraction agent harness with a scripted fake model (no Ollama, no API)."""
import copy
import json

import pytest

from eurlex_ontology_generation.agent.extraction import AgentConfig, run_act, run_chunk
from eurlex_ontology_generation.agent.extraction.library import Library
from eurlex_ontology_generation.agent.extraction.memory import ActMemory
from eurlex_ontology_generation.agent.extraction.tools import HANDLERS, Act, tool_specs
from eurlex_ontology_generation.utils.llm_client import ChatTurn, ToolCall
from eurlex_ontology_generation.utils.ontology_schema import build_ontology_model

CHUNKS = [
    {"chunk_id": "32099L0001_1", "chunk_number": 1,
     "chunk_text": "Article 1\nFor the purposes of this Directive, 'packaging' means any product used for containment."},
    {"chunk_id": "32099L0001_2", "chunk_number": 2,
     "chunk_text": "Article 2\n1. Member States shall notify the Commission of the measures by 31 December 2019."},
    {"chunk_id": "32099L0001_3", "chunk_number": 3,
     "chunk_text": "Article 3\nBy way of derogation from Article 2(1), Member States may postpone the notification."},
]


def make_act():
    return Act(celex="32099L0001", act_name="Test Directive", act_type="Directive", date="2099-01-01",
               chunks=copy.deepcopy(CHUNKS))


def onto(kind="duties", action_label="notify_measures", asset="Measure", party="MemberState"):
    rule = {"action": "notify", "action_label": action_label, "party": party, "assigner": "",
            "asset": asset, "provision": "article", "constraints": []}
    return {
        "ontology_name": "TestOntology", "description": "test",
        "odrl_alignment": {"core_entities": ["Party", "Asset"], "extensions": []},
        "classes": [
            {"name": "MemberState", "type": "Party", "is_a": "Country", "kind": "class", "description": "Member State"},
            {"name": "Measure", "type": "Asset", "is_a": "LegalAct", "kind": "class", "description": "measure"},
        ],
        "relationships": [],
        "rights_model": {"permissions": [], "prohibitions": [], "duties": [], kind: [rule]},
        "legal_patterns": [], "example_instances": [],
    }


class FakeLLM:
    """Plays back a script: each entry is a list of (tool name, args) or a callable(messages) -> that list."""

    def __init__(self, script, default=None):
        self.script, self.default = list(script), default
        self.calls = []

    def __call__(self, messages, tools, tool_choice):
        self.calls.append({"messages": copy.deepcopy(messages), "tool_choice": tool_choice,
                           "tool_names": [t["function"]["name"] for t in tools]})
        step = self.script.pop(0) if self.script else self.default
        if callable(step):
            step = step(messages, tool_choice)
        if isinstance(step, Exception):
            raise step
        calls, raw = [], []
        for i, (name, args) in enumerate(step):
            arguments = args if isinstance(args, str) else json.dumps(args)
            try:
                parsed, err = json.loads(arguments), None
            except json.JSONDecodeError as exc:
                parsed, err = None, str(exc)
            cid = f"c{len(self.calls)}_{i}"
            calls.append(ToolCall(id=cid, name=name, arguments=arguments, args=parsed, parse_error=err))
            raw.append({"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}})
        msg = {"role": "assistant", "content": "", **({"tool_calls": raw} if raw else {})}
        return ChatTurn(message=msg, content="", tool_calls=calls,
                        done_reason="tool_calls" if calls else "stop",
                        prompt_tokens=100, output_tokens=10, cache_hit_tokens=60)


def cfg(tmp_path, llm, **kw):
    return AgentConfig(backend={}, model="fake", system_prompt="SYSTEM {core_types}",
                       user_prompt_template="TEXT:\n{text}", skills_dir=str(tmp_path / "skills"),
                       memory_dir=str(tmp_path / "memory"), traces_dir=str(tmp_path / "traces"), llm=llm, **kw)


def submit(o):
    return [("submit_ontology", {"ontology": o})]


def tool_results(messages):
    return [m["content"] for m in messages if m.get("role") == "tool"]


# ----------------------------------------------------------------- loop
def test_happy_path_with_context_tools_and_memory(tmp_path):
    llm = FakeLLM([
        [("act_outline", {}), ("find_definitions", {})],
        [("memory_write", {"kind": "cross_ref", "key": "Art. 2(1)", "value": "notification duty"})],
        submit(onto()),
    ], default=submit(onto()))        # resubmit unchanged if the first submission bounces on warnings
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))

    assert res.status in ("accepted", "accepted_with_warnings")
    assert res.ontology["rights_model"]["duties"][0]["action_label"] == "notify_measures"
    assert res.normalized is not None and res.submits in (1, 2)
    assert (res.prompt_tokens, res.cache_hit_tokens) == (100 * res.steps, 60 * res.steps)
    results = tool_results(llm.calls[1]["messages"])
    assert "Directive" in results[0] and "Article 2" in results[0]          # outline
    assert "'packaging' means" in results[1]                                 # definitions
    # system prompt: vocabulary rendered + harness section; user prompt: chunk text + chunk suffix
    sys_msg, user_msg = llm.calls[0]["messages"][0]["content"], llm.calls[0]["messages"][1]["content"]
    assert "Party" in sys_msg and "WORKING AS AN AGENT" in sys_msg and "(no skills yet)" in sys_msg
    assert "Member States shall notify" in user_msg and "chunk 2 of 3" in user_msg
    mem = json.loads((tmp_path / "memory" / "32099L0001.json").read_text(encoding="utf-8"))
    assert mem["entries"][0]["key"] == "Art. 2(1)" and mem["rules"][0]["kind"] == "duty"
    trace = [json.loads(x) for x in (tmp_path / "traces" / "32099L0001_2.jsonl").read_text(encoding="utf-8").splitlines()]
    assert trace[0]["event"] == "start" and trace[-1]["event"] == "end" and trace[-1]["status"] == res.status


def test_schema_error_is_returned_and_fixed(tmp_path):
    bad = onto()
    bad["classes"][0]["type"] = "NotAType"
    llm = FakeLLM([submit(bad)], default=submit(onto()))
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))
    first = tool_results(llm.calls[1]["messages"])[0]
    assert first.startswith("REJECTED") and "classes.0.type" in first
    assert res.status in ("accepted", "accepted_with_warnings") and res.submits >= 2


def test_invalid_json_arguments_count_as_a_submit(tmp_path):
    llm = FakeLLM([[("submit_ontology", '{"ontology": {broken')]], default=submit(onto()))
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))
    assert "valid JSON" in tool_results(llm.calls[1]["messages"])[0]
    assert res.status.startswith("accepted") and res.submits >= 2


def test_last_turns_force_a_submission(tmp_path):
    def answer(messages, tool_choice):
        return submit(onto()) if isinstance(tool_choice, dict) else [("act_outline", {})]
    llm = FakeLLM([], default=answer)
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm, max_steps=5))
    choices = [c["tool_choice"] for c in llm.calls]
    assert choices[:3] == ["auto"] * 3
    assert all(c == {"type": "function", "function": {"name": "submit_ontology"}} for c in choices[3:])
    assert res.status.startswith("accepted") and res.steps <= 5


def test_plain_text_reply_gets_a_nudge(tmp_path):
    llm = FakeLLM([[]], default=submit(onto()))
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))
    assert llm.calls[1]["messages"][-1]["role"] == "user" and "submit_ontology" in llm.calls[1]["messages"][-1]["content"]
    assert res.status.startswith("accepted")


def test_never_valid_fails_after_max_submits(tmp_path):
    llm = FakeLLM([], default=submit({"nonsense": 1}))
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm, max_submits=3))
    assert res.status == "failed_no_valid_submission" and res.submits == 3 and res.ontology is None
    assert res.findings and res.findings[0]["severity"] == "error"


def test_fallback_keeps_last_valid_submission(tmp_path):
    bad = onto()
    bad["classes"][0]["type"] = "NotAType"
    warned = onto(action_label="define_packaging", asset="containment_product")   # valid, but rule_not_in_chunk
    llm = FakeLLM([submit(warned)], default=submit(bad))
    # the valid submission bounces once on its warning, then only invalid ones follow
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm, max_submits=2))
    assert res.status == "fallback_last_valid" and res.submits == 2
    assert res.ontology["rights_model"]["duties"][0]["action_label"] == "define_packaging"
    assert res.normalized is not None


def test_llm_error_fails_the_chunk_not_the_run(tmp_path):
    llm = FakeLLM([RuntimeError("HTTP 500: token repeat limit")])
    res = run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))
    assert res.status == "failed_llm_error" and "repeat limit" in res.detail and res.ontology is None


def test_rule_from_another_chunk_is_flagged(tmp_path):
    foreign = onto(action_label="define_packaging", asset="containment_product", party="MemberState")
    llm = FakeLLM([submit(foreign)], default=submit(onto()))
    run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))
    assert "rule_not_in_chunk" in tool_results(llm.calls[1]["messages"])[0]


# ---------------------------------------------------------- act level
def test_run_act_keeps_order_and_carries_memory(tmp_path):
    seen = {}

    def answer(messages, tool_choice):
        chunk = "_3" if "chunk 3 of 3" in messages[1]["content"] else "_2"
        if not any(m.get("role") == "tool" for m in messages):
            return [("memory_read", {})]
        seen.setdefault(chunk, tool_results(messages)[0])
        return submit(onto(kind="permissions", action_label="postpone_notification", asset="notification")
                      if chunk == "_3" else onto())
    llm = FakeLLM([], default=answer)
    out = run_act(make_act(), ["32099L0001_3", "32099L0001_2"], cfg(tmp_path, llm))
    assert [r.chunk_id for r in out] == ["32099L0001_2", "32099L0001_3"]
    assert "empty" in seen["_2"]
    assert "32099L0001_2: duty | MemberState | notify_measures" in seen["_3"]


def test_rerunning_a_chunk_replaces_its_memory(tmp_path):
    c = cfg(tmp_path, FakeLLM([], default=submit(onto())))
    run_chunk(make_act(), "32099L0001_2", c)
    run_chunk(make_act(), "32099L0001_2", c)
    mem = ActMemory("32099L0001", tmp_path / "memory")
    assert len(mem.rules) == 1


# ------------------------------------------------------------ library
def test_library_index_load_hash_and_caps(tmp_path):
    d = tmp_path / "skills" / "exception-handling"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: exception-handling\ndescription: derogations\nwhen_to_use: 'by way of derogation'\n"
                                "---\nEncode a derogation as a permission.\n", encoding="utf-8")
    (tmp_path / "skills" / "LESSONS.md").write_text("# L\n- recitals are not rules\n", encoding="utf-8")
    lib = Library(tmp_path / "skills")
    assert "exception-handling: derogations" in lib.index_text() and lib.lessons == ["recitals are not rules"]
    h = lib.content_hash()
    assert h == Library(tmp_path / "skills").content_hash() and lib.check_caps() == []

    llm = FakeLLM([[("load_skill", {"name": "exception-handling"}), ("load_skill", {"name": "nope"})]],
                  default=submit(onto()))
    res = run_chunk(make_act(), "32099L0001_3", cfg(tmp_path, llm))
    results = tool_results(llm.calls[1]["messages"])
    assert results[0].startswith("Encode a derogation") and "no skill 'nope'" in results[1]
    assert res.skills_loaded == ["exception-handling"] and res.library_hash == h
    assert "recitals are not rules" in llm.calls[0]["messages"][0]["content"]


def test_repo_start_library_is_valid():
    from pathlib import Path
    lib = Library(Path(__file__).resolve().parents[1] / "agent_skills")
    assert list(lib.skills) == ["writing-skills"] and lib.lessons == [] and lib.check_caps() == []


def test_tool_specs_cover_handlers_and_inline_the_schema():
    specs = tool_specs(build_ontology_model())
    assert sorted(s["function"]["name"] for s in specs) == sorted(HANDLERS)
    text = json.dumps(specs)
    assert "$ref" not in text and "$defs" not in text


def test_unknown_tool_and_bad_args_are_reported(tmp_path):
    llm = FakeLLM([[("no_such_tool", {}), ("read_chunk", {"wrong": 1}), ("read_chunk", {"chunk_id": "X_9"})]],
                  default=submit(onto()))
    run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))
    r = tool_results(llm.calls[1]["messages"])
    assert "unknown tool" in r[0] and "bad arguments" in r[1] and "not a chunk" in r[2]


@pytest.mark.parametrize("k,expected", [(1, ["_1", "_3"]), (2, ["_1", "_3"])])
def test_neighbour_chunks(tmp_path, k, expected):
    llm = FakeLLM([[("neighbour_chunks", {"k": k})]], default=submit(onto()))
    run_chunk(make_act(), "32099L0001_2", cfg(tmp_path, llm))
    text = tool_results(llm.calls[1]["messages"])[0]
    assert all(f"32099L0001{s}" in text for s in expected) and "Article 2\n1." not in text
