"""MCP server exposing the curation workbench to an agent (Claude Desktop,
Claude Code, Cursor, or any MCP client).

    python -m eurlex_ontology_generation.agent.mcp_server --root /path/to/eurlex-ontology-generation

The agent can read the ontology, the review queue and the source texts, run the
extraction step on a text with the configured backend, and *propose* decisions.
Every proposal is validated (re-refine + export + SHACL, optionally HermiT)
before it is stored. Accepting a decision is deliberately not a tool: people
accept or reject proposals with scripts/curation.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .workbench import Decision, Workbench, load_parameters

INSTRUCTIONS = """\
You curate an ODRL-aligned ontology extracted from EUR-Lex acts.
Work from the review queue (review_queue). For each item: read the entity or rule
(get_entity / get_rule), read the source text (search_text, get_chunk, act_metadata),
check identifiers (gazetteer_lookup, wikidata_lookup), then call validate_decision and,
if it passes, propose_decision. Ground every decision in the text: quote it in
`evidence` and cite CELEX numbers. Do not guess. If the text does not settle the
question, skip the item. You cannot accept decisions; a person reviews your proposals."""


def _server_class():
    try:                                    # MCP Python SDK 2.x
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:                     # MCP Python SDK 1.x
        try:
            from mcp.server.fastmcp import FastMCP
            return FastMCP
        except ImportError as exc:
            raise SystemExit("The MCP SDK is not installed: uv sync --extra agent") from exc


def build_server(root: str | Path, env: str = "local", proposer: str = "mcp-agent"):
    wb = Workbench(root, load_parameters(Path(root).resolve(), env))
    server = _server_class()(name="eurlex-ontology-curation", instructions=INSTRUCTIONS)
    tool = server.tool

    @tool()
    def review_queue(kind: str | None = None, limit: int = 20, offset: int = 0) -> dict:
        """Open review items. kind: entity_kind | type_conflict | class_promotion | rule_slot."""
        return wb.review_queue(kind, min(limit, 100), offset)

    @tool()
    def get_entity(name: str) -> dict:
        """A class or individual with its rules, relationships, examples and gazetteer entry."""
        return wb.get_entity(name)

    @tool()
    def get_rule(rule_id: str) -> dict:
        """A permission, prohibition or duty with source snippets from its acts."""
        return wb.get_rule(rule_id)

    @tool()
    def get_chunk(chunk_id: str) -> dict:
        """Full text of one source chunk."""
        return wb.get_chunk(chunk_id)

    @tool()
    def search_text(query: str, celex: str | None = None, limit: int = 5) -> list[dict]:
        """Keyword search over the source chunks, optionally within one act (CELEX)."""
        return wb.search_text(query, celex, min(limit, 20))

    @tool()
    def act_metadata(celex: str, online: bool = False) -> dict:
        """Act type, year, title and date for a CELEX number; online=True also queries CELLAR."""
        return wb.act_metadata(celex, online)

    @tool()
    def gazetteer_lookup(name: str) -> dict:
        """Curated identifier (EU body, country, fund) for a name or alias."""
        return wb.gazetteer_lookup(name)

    @tool()
    def wikidata_lookup(label: str) -> dict:
        """Wikidata item for a label and whether it is an individual or a class."""
        return wb.wikidata_lookup(label)

    @tool()
    def extract_chunk(text: str) -> dict:
        """Run the extraction step on a text with the configured backend, then normalise and check it."""
        return wb.extract_chunk(text)

    @tool()
    def check_ontology(ontology: dict, text: str) -> dict:
        """Deterministic checks (grounding, vocabulary, leakage) of a chunk ontology against its text."""
        return wb.check_ontology(ontology, text)

    @tool()
    def validate_decision(decision: Decision, reasoner: bool = False) -> dict:
        """Dry run: apply the decision, re-export and report the SHACL (and optional HermiT) delta."""
        return wb.validate_decision(decision.model_dump(exclude_none=True), reasoner)

    @tool()
    def propose_decision(decision: Decision, model: str | None = None, reasoner: bool = False) -> dict:
        """Validate and store a proposal. It is stored as pending, or rejected_by_validation if it
        adds SHACL violations or makes the ontology inconsistent. A person accepts it later."""
        return wb.propose_decision(decision.model_dump(exclude_none=True), proposer, model, reasoner)

    @tool()
    def list_decisions(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Curation log: pending | accepted | rejected | rejected_by_validation."""
        rows = wb.decisions(status)
        return [{k: v for k, v in r.items() if k != "validation"} for r in rows[-min(limit, 200):]]

    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("EURLEX_ONTOLOGY_ROOT", "."),
                    help="Project root (the folder with conf/ and data/).")
    ap.add_argument("--env", default="local", help="Kedro config environment for overrides.")
    ap.add_argument("--proposer", default=os.environ.get("CURATION_PROPOSER", "mcp-agent"))
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    args = ap.parse_args(argv)
    build_server(args.root, args.env, args.proposer).run(transport=args.transport)


if __name__ == "__main__":
    main()
