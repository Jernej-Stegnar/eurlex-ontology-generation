"""Tests for the OWL/ODRL Turtle export and SHACL validation."""
import json

import pytest
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from eurlex_ontology_generation.pipelines.ontology_refinement.nodes import refine_ontology
from eurlex_ontology_generation.utils.entity_linking import load_gazetteer
from eurlex_ontology_generation.utils.rdf_export import ODRL, WD, export_rdf, local_name, namespaces
from eurlex_ontology_generation.utils.rdf_validate import round_trip_check, validate_graph

MERGED = {
    "ontology_name": "M", "description": "",
    "classes": [
        {"name": "SolidarityFund", "type": "Party", "description": "a fund", "support": {"chunks": 3, "documents": 2}},
        {"name": "European_Commission", "type": "Party", "description": "", "support": {"chunks": 5, "documents": 4}},
        {"name": "Latvia", "type": "Legal_extension", "description": "", "support": {"chunks": 1, "documents": 1}},
        {"name": "Application", "type": "Legal_extension", "description": "an application", "support": {"chunks": 2, "documents": 1}},
    ],
    "relationships": [{"source": "SolidarityFund", "relation": "managed_by", "target": "European_Commission", "support": 1}],
    "rights_model": {
        "permissions": [{"action": "mobilize_fund", "party": "European_Commission", "asset": "SolidarityFund",
                         "constraints": [{"name": "deadline", "value": "by 1 March 2019"},
                                         {"name": "amount", "value": "EUR 50 000 000"}],
                         "support": {"chunks": 2}}],
        "duties": [{"action": "submit_application", "party": "Latvia", "asset": "Application",
                    "constraints": [], "support": {"chunks": 1}}],
        "prohibitions": [],
    },
    "legal_patterns": [], "example_instances": [],
}
NS = namespaces()
EO, ENT = NS["eo"], NS["ent"]


@pytest.fixture(scope="module")
def graph():
    refined, _ = refine_ontology(json.loads(json.dumps(MERGED)))
    g, stats = export_rdf(refined)
    return g, stats


def test_export_structure(graph):
    g, stats = graph
    assert stats["skipped"] == []
    fund = ENT["European_Union_Solidarity_Fund"]
    assert (fund, RDF.type, OWL.NamedIndividual) in g
    assert (fund, RDF.type, EO.Fund) in g
    assert (fund, OWL.sameAs, WD["Q5413352"]) in g
    latvia = ENT["Latvia"]
    assert (latvia, RDF.type, EO.Country) in g
    assert any(str(o).endswith("/country/LVA") for o in g.objects(latvia, URIRef("http://www.w3.org/2004/02/skos/core#exactMatch")))
    app = ENT["Application"]
    assert (app, RDF.type, OWL.Class) in g and (app, RDFS.subClassOf, EO.Document) in g
    assert (ENT["European_Commission"], EO.manages, fund) in g          # inverse flipped


def test_odrl_rules_and_typed_constraints(graph):
    g, _ = graph
    perms = list(g.subjects(RDF.type, ODRL.Permission))
    assert len(perms) == 1
    p = perms[0]
    assert (p, ODRL.assignee, ENT["European_Commission"]) in g
    assert str(g.value(p, ODRL.action)).endswith("/action/mobilise")
    rights = {(str(g.value(c, ODRL.leftOperand)).rsplit("/", 1)[-1], g.value(c, ODRL.rightOperand))
              for c in g.objects(p, ODRL.constraint)}
    assert ("dateTime", Literal("2019-03-01", datatype=XSD.date)) in rights
    assert ("payAmount", Literal("50000000", datatype=XSD.decimal)) in rights
    assert any(True for _ in g.subjects(ODRL.permission, p))


def test_round_trip(graph):
    assert round_trip_check(graph[0])["ok"]


def test_clean_graph_has_no_violations(graph):
    ok, _, summary = validate_graph(graph[0])
    assert ok, summary["results"]


def _messages(summary, severity):
    return " | ".join(r["message"] for r in summary["results"] if r["severity"] == severity)


def test_shacl_catches_structural_errors(graph):
    g = graph[0].__class__()
    for t in graph[0]:
        g.add(t)
    rule = next(g.subjects(RDF.type, ODRL.Duty))
    g.remove((rule, ODRL.action, None))                                   # rule without action
    c = BNode()
    g.add((rule, ODRL.constraint, c))
    g.add((c, RDF.type, ODRL.Constraint))
    g.add((c, ODRL.leftOperand, ODRL.dateTime))
    g.add((c, ODRL.operator, ODRL.approximately))                         # not an ODRL operator
    g.add((c, ODRL.rightOperand, Literal("soon")))
    bad = ENT["AuditPermission"]                                          # meta-model leakage + two core types
    g.add((bad, RDFS.label, Literal("AuditPermission")))
    g.add((bad, EO.entityKind, Literal("individual")))
    g.add((bad, EO.coreType, Literal("Party")))
    g.add((bad, RDF.type, OWL.NamedIndividual))
    g.add((bad, RDF.type, EO.Fund))
    g.add((bad, RDF.type, EO.Country))
    ok, _, summary = validate_graph(g)
    assert not ok
    violations = _messages(summary, "Violation")
    assert "exactly one odrl:action" in violations
    assert "ODRL 2.2 operators" in violations
    assert "meta-model term" in violations
    assert "two disjoint core classes" in violations
    assert "core type of its upper-ontology parent" in violations
    assert "not an xsd:date" in _messages(summary, "Warning")


def test_shacl_warns_on_slot_typing(graph):
    g = graph[0].__class__()
    for t in graph[0]:
        g.add(t)
    perm = next(g.subjects(RDF.type, ODRL.Permission))
    g.remove((perm, ODRL.target, None))
    g.add((perm, ODRL.target, ENT["Latvia"]))                             # a Party as target
    ok, _, summary = validate_graph(g)
    assert ok                                                             # warnings do not fail
    assert "odrl:target is a Party" in _messages(summary, "Warning")


def test_local_name():
    assert local_name("EU Solidarity Fund (2018)") == "EU_Solidarity_Fund_2018"
    assert local_name("2019 budget").startswith("e_")


def test_gazetteer_identifiers_are_well_formed():
    import re
    gaz = load_gazetteer()
    for entry in gaz.entries.values():
        if entry.wikidata:
            assert re.fullmatch(r"Q[1-9]\d*", entry.wikidata), entry.name
        if entry.iri:
            assert entry.iri.startswith("http://publications.europa.eu/resource/authority/"), entry.name
    assert len({e.wikidata for e in gaz.entries.values() if e.wikidata}) == \
        sum(1 for e in gaz.entries.values() if e.wikidata), "duplicate QIDs in gazetteer"


def test_reasoner_detects_disjointness_violation(graph):
    from eurlex_ontology_generation.utils.rdf_validate import reasoner_check

    ok = reasoner_check(graph[0])
    if not ok["available"]:
        pytest.skip(ok["reason"])
    assert ok["consistent"] and not ok["unsatisfiable_classes"]
    g = graph[0].__class__()
    for t in graph[0]:
        g.add(t)
    g.add((ENT["European_Union_Solidarity_Fund"], RDF.type, EO.Country))   # Fund and Country: disjoint
    bad = reasoner_check(g)
    assert bad["consistent"] is False


def test_iris_use_the_w3id_namespace(graph):
    g = graph[0]
    onto = URIRef("https://w3id.org/eurlex-odrl/ontology")
    assert (onto, RDF.type, OWL.Ontology) in g
    assert not any("example.org" in str(t) for triple in g for t in triple)
