"""Export the refined ontology to Turtle and validate it with SHACL."""
from __future__ import annotations

import json
from pathlib import Path

from ...utils.rdf_export import export_rdf
from ...utils.rdf_validate import (
    reasoner_check,
    round_trip_check,
    summarize_counts,
    validate_graph,
)


def export_and_validate(refined: dict, base_iri: str, shapes_path: str | None,
                        vocabulary_path: str | None, fail_on_violation: bool,
                        run_reasoner: bool = True):
    graph, stats = export_rdf(refined, base_iri, vocabulary_path)
    conforms, report, summary = validate_graph(graph, shapes_path, base_iri)
    reasoner = reasoner_check(graph) if run_reasoner else {"available": False, "reason": "disabled"}
    summary = {
        "fail_on_violation": bool(fail_on_violation),
        "export": {k: v for k, v in stats.items() if k != "skipped"},
        "export_skipped": stats["skipped"],
        "graph": summarize_counts(graph),
        "turtle_round_trip": round_trip_check(graph),
        "shacl": summary,
        "owl_reasoner": reasoner,
    }
    return graph, report, summary


def quality_gate(summary: dict, turtle_path: str) -> bool:
    """Runs after the files are written, so a failing run still leaves the report."""
    if not summary.get("fail_on_violation"):
        return True
    counts = summary["shacl"]["results_by_severity"]
    if counts.get("Violation", 0):
        raise ValueError("SHACL violations found (see data/08_reporting/ontology_export_summary.json): "
                         + json.dumps(counts))
    reasoner = summary.get("owl_reasoner", {})
    if reasoner.get("available") and (not reasoner["consistent"] or reasoner.get("unsatisfiable_classes")):
        raise ValueError("OWL reasoner found the ontology inconsistent: " + json.dumps(reasoner))
    return True


def persist_export(graph, report, summary: dict, turtle_path: str, report_path: str,
                   summary_path: str) -> str:
    for path in (turtle_path, report_path, summary_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=turtle_path, format="turtle")
    report.serialize(destination=report_path, format="turtle")
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return turtle_path
