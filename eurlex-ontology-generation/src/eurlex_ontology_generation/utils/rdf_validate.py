"""SHACL validation of the exported ontology, with a readable summary.

`validate_graph` runs pySHACL with the project shapes (resources/shapes.ttl)
and returns the report graph plus a JSON-friendly summary grouped by
severity and message, with example focus nodes, so the numbers can go into
the evaluation report and CI.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, Namespace
from rdflib.namespace import RDFS

from .rdf_export import DEFAULT_BASE_IRI, namespaces
from .vocabulary import RESOURCES_DIR

SH = Namespace("http://www.w3.org/ns/shacl#")
DEFAULT_SHAPES_PATH = RESOURCES_DIR / "shapes.ttl"
MAX_EXAMPLES = 8


def load_shapes(shapes_path: str | None = None, base_iri: str = DEFAULT_BASE_IRI) -> Graph:
    text = Path(shapes_path or DEFAULT_SHAPES_PATH).read_text(encoding="utf-8")
    default_eo = str(namespaces(DEFAULT_BASE_IRI)["eo"])
    wanted_eo = str(namespaces(base_iri)["eo"])
    if wanted_eo != default_eo:
        text = text.replace(default_eo, wanted_eo)
    return Graph().parse(data=text, format="turtle")


def _short(node, data: Graph) -> str:
    label = data.value(node, RDFS.label)
    if label is not None:
        return str(label)
    s = str(node)
    return s.rsplit("/", 1)[-1].rsplit("#", 1)[-1] if s.startswith("http") else f"_:{s[:8]}"


def validate_graph(data: Graph, shapes_path: str | None = None,
                   base_iri: str = DEFAULT_BASE_IRI) -> tuple[bool, Graph, dict]:
    from pyshacl import validate

    shapes = load_shapes(shapes_path, base_iri)
    conforms, report, _ = validate(data_graph=data, shacl_graph=shapes, inference="none",
                                   abort_on_first=False, allow_warnings=True, meta_shacl=False,
                                   advanced=True, debug=False)
    if not isinstance(report, Graph):          # pySHACL could not run the shapes
        raise RuntimeError(f"SHACL validation could not run: {report}")
    by_severity: Counter = Counter()
    by_message: dict[tuple, dict] = {}
    for result in report.subjects(SH.resultSeverity, None):
        severity = str(report.value(result, SH.resultSeverity)).rsplit("#", 1)[-1]
        message = str(report.value(result, SH.resultMessage) or report.value(result, SH.sourceConstraintComponent))
        focus = report.value(result, SH.focusNode)
        by_severity[severity] += 1
        entry = by_message.setdefault((severity, message), {"severity": severity, "message": message,
                                                             "count": 0, "examples": []})
        entry["count"] += 1
        if focus is not None and len(entry["examples"]) < MAX_EXAMPLES:
            entry["examples"].append(_short(focus, data))
    violations = by_severity.get("Violation", 0)
    summary = {
        "conforms_without_violations": violations == 0,
        "shacl_conforms_strict": bool(conforms),
        "results_by_severity": dict(by_severity),
        "results": sorted(by_message.values(), key=lambda e: (
            {"Violation": 0, "Warning": 1, "Info": 2}.get(e["severity"], 3), -e["count"])),
    }
    return violations == 0, report, summary


def round_trip_check(data: Graph) -> dict:
    """Serialize to Turtle and parse back: catches IRIs/literals that break the syntax."""
    text = data.serialize(format="turtle")
    back = Graph().parse(data=text, format="turtle")
    return {"triples": len(data), "triples_after_round_trip": len(back), "ok": len(back) == len(data)}


def summarize_counts(data: Graph) -> dict:
    """Basic OWL/ODRL statistics of the exported graph."""
    from rdflib.namespace import OWL, RDF

    odrl = Namespace("http://www.w3.org/ns/odrl/2/")
    count = defaultdict(int)
    for _, _, o in data.triples((None, RDF.type, None)):
        count[str(o)] += 1
    return {
        "owl_classes": count[str(OWL.Class)],
        "named_individuals": count[str(OWL.NamedIndividual)],
        "object_properties": count[str(OWL.ObjectProperty)],
        "permissions": count[str(odrl.Permission)],
        "prohibitions": count[str(odrl.Prohibition)],
        "duties": count[str(odrl.Duty)],
        "constraints": count[str(odrl.Constraint)],
        "actions": count[str(odrl.Action)],
    }


def reasoner_check(data: Graph, timeout_note: str = "") -> dict:
    """OWL 2 consistency check with HermiT (via owlready2; needs Java).

    Returns {"available": False, ...} when owlready2 or Java is missing, so the
    pipeline still runs without it. Catches what SHACL cannot: logical
    contradictions that follow from the axioms (e.g. an individual that is
    both a Fund and a Country, which are under disjoint core classes).
    """
    import shutil
    import tempfile

    try:
        from owlready2 import (OwlReadyInconsistentOntologyError, World, sync_reasoner)
    except ImportError:
        return {"available": False, "reason": "owlready2 not installed (pip install owlready2)"}
    if not shutil.which("java"):
        return {"available": False, "reason": "Java not found (HermiT needs a JRE)"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ontology.nt"
        data.serialize(destination=str(path), format="nt", encoding="utf-8")
        world = World()
        onto = world.get_ontology(path.as_uri()).load()
        try:
            with onto:
                sync_reasoner(world, infer_property_values=False, debug=0)
        except OwlReadyInconsistentOntologyError as exc:
            return {"available": True, "reasoner": "HermiT", "consistent": False,
                    "error": str(exc)[:500]}
        unsat = [str(c.iri) for c in world.inconsistent_classes()]
        return {"available": True, "reasoner": "HermiT", "consistent": True,
                "unsatisfiable_classes": unsat}
