"""Review curation proposals (people only).

    python scripts/curation.py list [--status pending]
    python scripts/curation.py show dec_ab12cd34ef56
    python scripts/curation.py accept dec_ab12cd34ef56 --reviewer "Jane Doe"
    python scripts/curation.py reject dec_ab12cd34ef56 --reviewer "Jane Doe"
    python scripts/curation.py accept-all --reviewer "Jane Doe" --type set_typing   # bulk, after review

Accepted decisions are applied by the next run of the ontology_refinement pipeline:
    kedro run --pipeline ontology_refinement,ontology_export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eurlex_ontology_generation.agent.workbench import Workbench  # noqa: E402


def _summary(d: dict) -> str:
    target = d.get("entity") or d.get("rule_id") or f"{d.get('entities')} -> {d.get('into')}"
    detail = ", ".join(f"{k}={d[k]}" for k in ("is_a", "type_value", "kind") if d.get(k))
    return f"{d['id']}  {d['status']:<22} {d['type']:<15} {target}  {detail}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Review curation proposals.")
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--status", default="pending")
    sh = sub.add_parser("show")
    sh.add_argument("id")
    for name in ("accept", "reject"):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.add_argument("--reviewer", required=True)
    bulk = sub.add_parser("accept-all")
    bulk.add_argument("--reviewer", required=True)
    bulk.add_argument("--type", default=None, help="Only this decision type.")
    bulk.add_argument("--proposer", default=None, help="Only proposals from this proposer.")
    args = ap.parse_args(argv)

    wb = Workbench(args.root)
    if args.cmd == "list":
        rows = wb.decisions(None if args.status == "all" else args.status)
        for d in rows:
            print(_summary(d))
        print(f"{len(rows)} decision(s)")
    elif args.cmd == "show":
        hit = [d for d in wb.decisions() if d["id"] == args.id]
        if not hit:
            print(f"unknown decision {args.id}", file=sys.stderr)
            return 1
        print(json.dumps(hit[0], indent=2, ensure_ascii=False))
    elif args.cmd in ("accept", "reject"):
        d = wb.set_status(args.id, "accepted" if args.cmd == "accept" else "rejected", args.reviewer)
        print(_summary(d))
    elif args.cmd == "accept-all":
        rows = [d for d in wb.decisions("pending")
                if (args.type is None or d["type"] == args.type)
                and (args.proposer is None or d.get("proposer") == args.proposer)]
        for d in rows:
            print(_summary(wb.set_status(d["id"], "accepted", args.reviewer)))
        print(f"accepted {len(rows)} decision(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
