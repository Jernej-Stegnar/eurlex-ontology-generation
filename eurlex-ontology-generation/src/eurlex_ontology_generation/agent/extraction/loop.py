"""The extraction agent's episode loop.

One episode per chunk: the model gets the generation pipeline's own system
prompt and chunk prompt (so it starts from exactly what the single-prompt
pipeline knows), plus a harness section with the act's lessons and the skill
index. It may call tools for context and must finish with submit_ontology.
Chunks of one act run in order so that act memory carries over.

The LLM call is injectable (AgentConfig.llm) so tests can script the model.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...utils.entity_linking import load_gazetteer
from ...utils.llm_client import ChatTurn, llm_chat_messages
from ...utils.ontology_normalize import normalize_chunk_ontology
from ...utils.ontology_schema import build_ontology_model
from ...utils.vocabulary import load_vocabulary, render_prompt
from .library import Library
from .memory import ActMemory
from .tools import Act, EpisodeState, call_tool, tool_specs

logger = logging.getLogger(__name__)

AGENT_SECTION = """

WORKING AS AN AGENT
You work on one chunk of an EU act at a time, with tools. The chunk text is in the user message.
- Tools give you context: act_outline (act type, title, headings), neighbour_chunks, search_act, find_definitions, gazetteer_lookup, and the act memory (memory_read / memory_write) that carries what you learned in earlier chunks of the same act.
- Use context to understand the chunk: what a derogation derogates from, what a defined term means, who "the competent authority" is, whether a paragraph is a recital.
- Extract rules ONLY from the current chunk. A rule whose sentence is in a neighbouring chunk belongs to that chunk, not this one. Do not repeat rules memory_read lists as already extracted.
- Before submitting, note for later chunks anything they will need (memory_write): definitions, canonical entity names, open cross-references.
- Skills hold procedures for recurring legal patterns; load_skill reads one when its description fits.
- Finish by calling submit_ontology with the complete ontology object. Do not answer in plain text. If the submission comes back with problems, fix them and submit again.

LESSONS FROM EARLIER WORK
{lessons}

SKILLS (load_skill <name> for the full procedure)
{skills}
"""

CHUNK_SUFFIX = """

---
Current chunk: {chunk_id} (chunk {position} of {total} of act {celex}, {act_type}).
Do not reply with the JSON as text: pass the object above as the `ontology` argument of submit_ontology."""


@dataclass
class AgentConfig:
    backend: dict
    model: str
    system_prompt: str
    user_prompt_template: str
    temperature: float = 0.1
    seed: int | None = 42
    max_steps: int = 12
    max_tokens: int = 8192
    max_submits: int = 3
    vocabulary_path: str | None = None
    skills_dir: str | None = None
    memory_dir: str | None = None
    traces_dir: str | None = None
    llm: Callable[..., ChatTurn] | None = None      # (messages, tools, tool_choice) -> ChatTurn


@dataclass
class ChunkResult:
    chunk_id: str
    CELEX: str
    status: str                 # accepted | accepted_with_warnings | fallback_last_valid | failed_*
    ontology: dict | None       # as submitted (schema-valid); None if nothing valid was submitted
    normalized: dict | None     # after normalize_chunk_ontology (what merge should read)
    normalization_changes: list = field(default_factory=list)
    findings: list = field(default_factory=list)      # problems of the last submission
    steps: int = 0
    submits: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    done_reason: str = ""       # finish reason of the last LLM turn
    skills_loaded: list = field(default_factory=list)
    library_hash: str = ""
    seconds: float = 0.0
    detail: str = ""


def _default_llm(cfg: AgentConfig) -> Callable[..., ChatTurn]:
    def call(messages, tools, tool_choice):
        return llm_chat_messages(cfg.backend, cfg.model, messages, tools=tools, tool_choice=tool_choice,
                                 temperature=cfg.temperature, seed=cfg.seed, max_tokens=cfg.max_tokens)
    return call


class _Trace:
    def __init__(self, traces_dir: str | None, chunk_id: str):
        self.path = Path(traces_dir) / f"{chunk_id}.jsonl" if traces_dir else None
        self.rows: list[dict] = []

    def add(self, **row: Any) -> None:
        self.rows.append({"t": round(time.time(), 3), **row})

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.rows) + "\n",
                                 encoding="utf-8")


def run_chunk(act: Act, chunk_id: str, cfg: AgentConfig, *, library: Library | None = None,
              memory: ActMemory | None = None) -> ChunkResult:
    started = time.time()
    library = library or Library(cfg.skills_dir)
    memory = memory or ActMemory(act.celex, cfg.memory_dir)
    memory.forget_chunk(chunk_id)
    vocab = load_vocabulary(cfg.vocabulary_path)
    gazetteer = load_gazetteer(vocabulary_path=cfg.vocabulary_path)
    schema_model = build_ontology_model(cfg.vocabulary_path)
    chunk = act.chunks[act.index_of(chunk_id)]
    st = EpisodeState(act=act, chunk=chunk, memory=memory, library=library, vocab=vocab,
                      gazetteer=gazetteer, schema_model=schema_model, max_submits=cfg.max_submits)
    res = ChunkResult(chunk_id=chunk_id, CELEX=act.celex, status="", ontology=None, normalized=None,
                      library_hash=library.content_hash())
    llm = cfg.llm or _default_llm(cfg)
    tools = tool_specs(schema_model)
    trace = _Trace(cfg.traces_dir, chunk_id)

    system = render_prompt(cfg.system_prompt, cfg.vocabulary_path) + AGENT_SECTION.format(
        lessons=library.lessons_text(), skills=library.index_text())
    user = render_prompt(cfg.user_prompt_template, cfg.vocabulary_path, text=chunk["chunk_text"]) + CHUNK_SUFFIX.format(
        chunk_id=chunk_id, position=act.index_of(chunk_id) + 1, total=len(act.chunks),
        celex=act.celex, act_type=act.act_type)
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    trace.add(event="start", chunk_id=chunk_id, celex=act.celex, model=cfg.model, library_hash=res.library_hash,
              system_chars=len(system), user_chars=len(user))

    try:
        for step in range(cfg.max_steps):
            # the last two turns may only submit, so every episode ends with a submission
            force = step >= cfg.max_steps - 2
            tool_choice: Any = ({"type": "function", "function": {"name": "submit_ontology"}} if force else "auto")
            turn = llm(messages, tools, tool_choice)
            res.steps += 1
            res.prompt_tokens += turn.prompt_tokens
            res.output_tokens += turn.output_tokens
            res.cache_hit_tokens += turn.cache_hit_tokens
            res.done_reason = turn.done_reason
            messages.append(turn.message)
            trace.add(event="turn", step=step, forced=force, done_reason=turn.done_reason,
                      prompt_tokens=turn.prompt_tokens, output_tokens=turn.output_tokens,
                      cache_hit_tokens=turn.cache_hit_tokens, content=turn.content[:2000],
                      tool_calls=[{"name": c.name, "arguments": c.arguments[:20000]} for c in turn.tool_calls])

            if not turn.tool_calls:
                messages.append({"role": "user", "content": "Use the tools; finish by calling submit_ontology "
                                                            "with the ontology object."})
                continue
            for call in turn.tool_calls:
                res.tool_calls += 1
                if call.parse_error:
                    if call.name == "submit_ontology":
                        st.submits += 1
                    out = f"error: {call.parse_error}. Send the arguments again as valid JSON."
                else:
                    out = call_tool(st, call.name, call.args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
                trace.add(event="tool", name=call.name, result=out[:4000], result_chars=len(out))
            if st.accepted is not None or st.submits >= cfg.max_submits:
                break
    except Exception as exc:  # noqa: BLE001 - an API failure fails this chunk, not the batch
        logger.exception("agent episode failed for %s", chunk_id)
        res.detail = f"{type(exc).__name__}: {exc}"[:500]
        trace.add(event="error", detail=res.detail)

    res.submits, res.skills_loaded, res.findings = st.submits, st.skills_loaded, st.last_findings
    final = st.accepted if st.accepted is not None else st.last_valid
    if st.accepted is not None:
        res.status = st.final_status
    elif st.last_valid is not None:
        res.status = "fallback_last_valid"
    else:
        res.status = "failed_llm_error" if res.detail else "failed_no_valid_submission"
    if final is not None:
        res.ontology = final
        res.normalized, res.normalization_changes = normalize_chunk_ontology(final, vocab, gazetteer)
        memory.add_rules(chunk_id, final)
    memory.save()
    res.seconds = round(time.time() - started, 1)
    trace.add(event="end", **{k: v for k, v in asdict(res).items() if k not in ("ontology", "normalized")})
    trace.save()
    return res


def run_act(act: Act, chunk_ids: list[str], cfg: AgentConfig) -> list[ChunkResult]:
    """Run the agent over the given chunks of one act, in the act's chunk order."""
    library = Library(cfg.skills_dir)
    memory = ActMemory(act.celex, cfg.memory_dir)
    order = sorted(chunk_ids, key=act.index_of)
    return [run_chunk(act, cid, cfg, library=library, memory=memory) for cid in order]
