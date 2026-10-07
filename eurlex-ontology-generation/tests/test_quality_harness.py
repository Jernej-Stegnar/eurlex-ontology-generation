"""Tests for the ontology quality harness: vocabulary, gazetteer, normalisation,
review scoring, refinement, evaluation and persistence. No LLM or Ollama needed."""
import json

import pandas as pd
import pytest

from eurlex_ontology_generation.pipelines.ontology_refinement.nodes import (
    check_consistency,
    refine_ontology,
)
from eurlex_ontology_generation.pipelines.ontology_review.ontology_checks import (
    compute_scores,
    count_warnings,
    format_findings,
    run_deterministic_checks,
)
from eurlex_ontology_generation.utils.entity_linking import (
    WikidataLinker,
    detect_kind,
    load_gazetteer,
)
from eurlex_ontology_generation.utils.evaluation import compute_metrics, gold_metrics
from eurlex_ontology_generation.utils.ontology_normalize import (
    normalize_chunk_ontology,
    split_composite,
)
from eurlex_ontology_generation.utils.ontology_schema import build_ontology_model
from eurlex_ontology_generation.utils.persist import append_dedup_csv
from eurlex_ontology_generation.utils.vocabulary import (
    load_vocabulary,
    parse_date,
    render_prompt,
)


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def gaz():
    return load_gazetteer()


# ------------------------------------------------------------ vocabulary
class TestVocabulary:
    @pytest.mark.parametrize("raw,action,obj", [
        ("mobilize_egf", "mobilise", "egf"),
        ("mobilised_fund", "mobilise", "fund"),
        ("provide_financial_contribution", "provide", "financial_contribution"),
        ("carry_out_audit", "implement", "audit"),
        ("submit", "submit", ""),
    ])
    def test_normalize_action(self, vocab, raw, action, obj):
        out = vocab.normalize_action(raw)
        assert (out["action"], out["action_object"], out["controlled"]) == (action, obj, True)

    def test_canonical_action_keeps_label_object(self, vocab):
        out = vocab.normalize_action("provide", "provide_financial_contribution")
        assert out["action"] == "provide" and out["action_object"] == "financial_contribution"

    def test_unknown_verb_is_uncontrolled(self, vocab):
        out = vocab.normalize_action("frobnicate_widgets")
        assert out["controlled"] is False and out["action"] == "frobnicate"

    @pytest.mark.parametrize("name,leaks", [
        ("AuditDuty", True), ("AuditPermissionTarget", True), ("Assignee", True),
        ("RightsHolder", True), ("Duty", True),
        ("ImportDuty", False), ("CustomsDuty", False), ("RuleOfLaw", False),
        ("NetworkOperator", False), ("MigrationPolicy", False),
    ])
    def test_metamodel_leakage(self, vocab, name, leaks):
        assert bool(vocab.metamodel_leakage(name)) is leaks

    def test_structure_legacy_constraint_amount(self, vocab):
        c = vocab.structure_constraint({"name": "flexibility_ceiling", "value": "EUR 600 000 000 (2011 prices)"})
        assert c["left_operand"] == "payAmount" and c["right_operand"] == "600000000" and c["unit"] == "EUR"

    def test_structure_legacy_constraint_deadline(self, vocab):
        c = vocab.structure_constraint({"name": "deadline", "value": "no later than 1 January 2019"})
        assert (c["left_operand"], c["operator"], c["right_operand"]) == ("dateTime", "lteq", "2019-01-01")

    def test_unmatched_constraint_falls_back_to_condition(self, vocab):
        c = vocab.structure_constraint({"name": "zzz", "value": "something qualitative"})
        assert c["left_operand"] == "condition" and c["controlled"] is False

    def test_relation_inverse_is_flipped(self, vocab):
        assert vocab.normalize_relation("funded_by", "A", "B") == ("funds", "B", "A", True)
        assert vocab.normalize_relation("weird_link", "A", "B")[3] is False

    def test_parse_date(self):
        assert parse_date("on 11 December 2018") == "2018-12-11"
        assert parse_date("March 3, 2019") == "2019-03-03"
        assert parse_date("31 February 2019") is None

    def test_every_upper_class_has_valid_core_type(self, vocab):
        assert all(spec["core_type"] in vocab.core_types for spec in vocab.upper.values())

    def test_render_prompt_fills_vocabulary(self):
        out = render_prompt("{upper_ontology}|{actions}|{text}", text="T")
        assert "EUInstitution" in out and "mobilise" in out and out.endswith("|T")


# ------------------------------------------------------------ gazetteer
class TestEntityLinking:
    def test_aliases(self, gaz):
        assert gaz.lookup("Moldova").name == "Republic_of_Moldova"
        assert gaz.lookup("SolidarityFund").name == "European_Union_Solidarity_Fund"
        assert gaz.lookup("Latvia").iri.endswith("/country/LVA")
        assert gaz.lookup("Latvia").core_type == "Party"
        assert gaz.lookup("EGF").core_type == "Asset"

    @pytest.mark.parametrize("name,kind", [
        ("Latvia", "individual"), ("EGF", "individual"), ("Cyprus_Government", "individual"),
        ("NationalBankOfMoldova", "individual"), ("EGF_Mobilization", "class"),
        ("SolidarityFundApplication", "class"), ("FinancialContribution", "class"),
        ("EnergyUnionStrategy", "class"),   # 'Union' is too generic to count as embedded
    ])
    def test_detect_kind(self, gaz, vocab, name, kind):
        assert detect_kind(name, "", gaz, vocab)["kind"] == kind

    def test_wikidata_linker_uses_cache_without_network(self, tmp_path):
        cache = tmp_path / "wd.json"
        cache.write_text(json.dumps({"foo bar": {"qid": "Q1", "label": "foo bar", "description": "",
                                                 "p31": ["Q5"], "has_p31": True, "has_p279": False}}))
        linker = WikidataLinker(str(cache), "test", max_lookups=0)
        kind, info = linker.kind_of("Foo_Bar")
        assert kind == "individual" and info["qid"] == "Q1"
        assert linker.kind_of("unknown label") == (None, None)   # no lookups allowed


# ------------------------------------------------------------ schema
def test_schema_rejects_invalid_type_and_accepts_example():
    model = build_ontology_model()
    good = {
        "ontology_name": "X", "description": "d",
        "odrl_alignment": {"core_entities": [], "extensions": []},
        "classes": [{"name": "European_Commission", "type": "Party", "is_a": "EUInstitution",
                     "kind": "individual", "description": "d"}],
        "rights_model": {"permissions": [], "prohibitions": [], "duties": [
            {"action": "submit", "action_label": "submit_report", "party": "European_Commission",
             "constraints": [{"left_operand": "dateTime", "operator": "lteq", "right_operand": "2019-01-01"}]}]},
    }
    model.model_validate(good)
    bad = json.loads(json.dumps(good))
    bad["classes"][0]["type"] = "ODRL_core"
    with pytest.raises(Exception):
        model.model_validate(bad)
    assert "enum" in json.dumps(model.model_json_schema())


# ------------------------------------------------------------ normalisation
LEGACY = {
    "ontology_name": "Flex", "description": "",
    "odrl_alignment": {"core_entities": [], "extensions": ["Latvia", "AuditDuty"]},
    "classes": [
        {"name": "FlexibilityInstrument", "type": "ODRL_core", "description": "an instrument"},
        {"name": "Latvia", "type": "Legal_extension", "description": "the state"},
        {"name": "AuditDuty", "type": "Duty", "description": "duty to audit"},
        {"name": "EuropeanParliamentAndCouncil", "type": "Party", "description": ""},
    ],
    "relationships": [{"source": "FlexibilityInstrument", "relation": "funded_by", "target": "Latvia"}],
    "rights_model": {
        "permissions": [{"action": "mobilize_flexibility_instrument", "party": "EuropeanParliamentAndCouncil",
                         "asset": "FlexibilityInstrument",
                         "constraints": [{"name": "deadline", "value": "1 January 2019"}]}],
        "duties": [{"action": "audit", "party": "Latvia", "asset": "AuditDuty", "constraints": []}],
    },
    "legal_patterns": [], "example_instances": [],
}


class TestNormalize:
    def test_normalize_legacy_chunk(self):
        out, changes = normalize_chunk_ontology(json.loads(json.dumps(LEGACY)))
        classes = {c["name"]: c for c in out["classes"]}
        assert "AuditDuty" not in classes
        assert "EuropeanParliamentAndCouncil" not in classes
        assert classes["FlexibilityInstrument"]["type"] == "Asset"
        assert classes["Latvia"] == {**classes["Latvia"], "type": "Party", "is_a": "Country", "kind": "individual"}
        perms = out["rights_model"]["permissions"]
        assert {p["party"] for p in perms} == {"European_Parliament", "Council"}
        assert perms[0]["action"] == "mobilise" and perms[0]["constraints"][0]["right_operand"] == "2019-01-01"
        assert out["rights_model"]["duties"][0]["asset"] == ""
        assert out["rights_model"]["prohibitions"] == []
        assert out["relationships"][0]["relation"] == "funds"
        assert out["relationships"][0]["source"] == "Latvia"
        # every referenced name is defined
        names = {c["name"] for c in out["classes"]}
        assert {"European_Parliament", "Council"} <= names
        assert out["odrl_alignment"]["extensions"] == sorted(
            c["name"] for c in out["classes"] if c["type"] == "Legal_extension")
        codes = {c["code"] for c in changes}
        assert {"metamodel_leakage_removed", "composite_class_split", "class_retyped",
                "action_normalized", "constraint_structured", "relation_normalized"} <= codes

    def test_normalized_output_passes_schema(self):
        out, _ = normalize_chunk_ontology(json.loads(json.dumps(LEGACY)))
        build_ontology_model().model_validate(out)

    def test_split_composite_requires_known_parts(self, gaz):
        assert split_composite("EuropeanParliamentAndCouncil", gaz) == ["European_Parliament", "Council"]
        assert split_composite("TermsAndConditions", gaz) is None


# ------------------------------------------------------------ review checks
class TestReviewChecks:
    TEXT = ("The Flexibility Instrument shall be mobilised. Latvia shall be audited by "
            "1 January 2019. The European Parliament and the Council decide.")

    def test_new_findings(self):
        _, findings = run_deterministic_checks(json.dumps(LEGACY), self.TEXT)
        codes = {f["code"] for f in findings}
        assert "metamodel_leakage" in codes
        assert "misclassified_core_class" in codes
        assert "gazetteer_type_mismatch" in codes          # Latvia typed Legal_extension
        assert "unstructured_constraint" in codes

    def test_info_findings_do_not_count(self):
        findings = [{"code": "uncontrolled_action", "severity": "info", "message": "m"}] * 5
        assert compute_scores(findings, 0) == (5, 5)
        assert format_findings(findings) == "(none)"

    def test_count_by_code(self):
        findings = [{"code": "misclassified_core_class", "severity": "warning", "message": str(i)}
                    for i in range(4)]
        assert count_warnings(findings, "code") == 1 and count_warnings(findings, "finding") == 4
        assert compute_scores(findings, 0, "code")[1] == 4
        assert compute_scores(findings, 0, "finding")[1] == 1

    def test_normalisation_clears_structural_findings(self):
        out, _ = normalize_chunk_ontology(json.loads(json.dumps(LEGACY)))
        _, findings = run_deterministic_checks(json.dumps(out), self.TEXT)
        codes = {f["code"] for f in findings}
        assert not codes & {"metamodel_leakage", "misclassified_core_class", "invalid_class_type",
                            "gazetteer_type_mismatch", "undefined_entities", "unstructured_constraint"}


# ------------------------------------------------------------ refinement
MERGED = {
    "ontology_name": "M", "description": "",
    "classes": [
        {"name": "SolidarityFund", "type": "Party", "description": "a fund", "support": {"chunks": 3, "documents": 2}},
        {"name": "EU Solidarity Fund", "type": "Asset", "description": "the fund", "support": {"chunks": 1, "documents": 1}},
        {"name": "European_Commission", "type": "Party", "description": "", "support": {"chunks": 5, "documents": 4}},
        {"name": "AuditPermissionTarget", "type": "Legal_extension", "description": "", "support": {"chunks": 1, "documents": 1}},
        {"name": "Application", "type": "Legal_extension", "description": "", "support": {"chunks": 2, "documents": 1}},
    ],
    "relationships": [{"source": "SolidarityFund", "relation": "managed_by", "target": "European_Commission", "support": 1}],
    "rights_model": {
        "permissions": [{"action": "mobilize_fund", "party": "European_Commission", "asset": "SolidarityFund",
                         "constraints": [], "support": {"chunks": 1}},
                        {"action": "mobilise_fund", "party": "European_Commission", "asset": "EU Solidarity Fund",
                         "constraints": [], "support": {"chunks": 2}}],
        "duties": [{"action": "review_application", "party": "European_Commission", "asset": "AuditPermissionTarget",
                    "constraints": [{"name": "deadline", "value": "1 March 2019"}], "support": {"chunks": 1}}],
        "prohibitions": [],
    },
    "legal_patterns": [], "example_instances": [],
}


class TestRefinement:
    def test_refine(self):
        refined, report = refine_ontology(json.loads(json.dumps(MERGED)))
        inds = {i["name"]: i for i in refined["individuals"]}
        assert inds["European_Union_Solidarity_Fund"]["type"] == "Asset"
        assert inds["European_Union_Solidarity_Fund"]["support"]["chunks"] == 4
        assert sorted(inds["European_Union_Solidarity_Fund"]["merged_from"]) == ["EU Solidarity Fund", "SolidarityFund"]
        assert "AuditPermissionTarget" not in {c["name"] for c in refined["classes"]}
        perms = refined["rights_model"]["permissions"]
        assert len(perms) == 1 and perms[0]["support"]["chunks"] == 3 and perms[0]["action"] == "mobilise"
        assert refined["relationships"][0]["relation"] == "manages"
        assert refined["relationships"][0]["source"] == "European_Commission"
        assert refined["rights_model"]["duties"][0]["asset"] is None
        assert report["consistency_issues"]["undefined_reference"] == []

    def test_check_consistency_flags_slots(self):
        onto = {"classes": [{"name": "Fund", "type": "Asset"}],
                "individuals": [{"name": "Commission", "type": "Party"}],
                "rights_model": {"duties": [{"action": "pay", "party": "Fund", "asset": "Commission"}]},
                "relationships": []}
        issues = check_consistency(onto)
        assert issues["party_not_party"] and issues["asset_is_party"]


# ------------------------------------------------------------ evaluation
def test_evaluation_before_after(tmp_path):
    before = compute_metrics(MERGED)
    refined, _ = refine_ontology(json.loads(json.dumps(MERGED)))
    after = compute_metrics(refined)
    assert before["leakage"]["count"] == 1 and after["leakage"]["count"] == 0
    assert after["actions"]["pct_rules_controlled"] == 100.0
    # raw generator output: agreement with the gazetteer is a real measurement
    assert before["typing"]["pct_known_entities_correct_type"] is not None
    # refined output: every known entity was typed BY the gazetteer, so the
    # agreement would be circular and is not reported
    assert after["typing"]["known_entities_typed_by_gazetteer"] == after["typing"]["known_entities"]
    assert after["typing"]["pct_known_entities_correct_type"] is None


def test_gold_metrics_never_scores_against_itself(tmp_path):
    refined, _ = refine_ontology(json.loads(json.dumps(MERGED)))
    gold = tmp_path / "gold.yml"
    gold.write_text(
        "entities:\n"
        "  - {name: SolidarityFund, type: Asset, is_a: Fund, kind: individual, source: expert}\n"
        "  - {name: Application, type: Asset, is_a: Document, kind: class, source: expert}\n"
        "  - {name: European_Commission, type: Party, is_a: EUInstitution, kind: individual}\n")
    g = gold_metrics(refined, str(gold))
    assert g["found_in_ontology"] == 3 and g["independent_labels"] == 2
    assert g["gazetteer_lookups"]["n"] == 1          # SolidarityFund: typed by the gazetteer
    assert g["held_out"]["n"] == 1                   # Application: typed by the typer
    assert g["held_out"]["is_a_accuracy"] == 100.0
    assert g["seed_smoke_test"]["n"] == 1            # no `source` -> seed, never headline


def test_annotation_sheet_and_kappa(tmp_path):
    from eurlex_ontology_generation.utils.evaluation import annotator_agreement, cohen_kappa, load_gold

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame([{"name": "X", "type": "Party", "is_a": "Country", "kind": "individual"},
                  {"name": "Y", "type": "Asset", "is_a": "Fund", "kind": "class"},
                  {"name": "Z", "type": "", "is_a": "", "kind": ""}]).to_csv(a, index=False)
    pd.DataFrame([{"name": "X", "type": "Party", "is_a": "Country", "kind": "individual"},
                  {"name": "Y", "type": "Party", "is_a": "Fund", "kind": "class"}]).to_csv(b, index=False)
    rows = load_gold(str(a))
    assert [r["name"] for r in rows] == ["X", "Y"] and rows[0]["source"] == "annotator"
    ag = annotator_agreement(str(a), str(b))
    assert ag["common_entities"] == 2 and ag["is_a_kappa"] == 1.0 and ag["type_raw_agreement"] == 50.0
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "b", "a"]) == 0.0


def test_typer_ablation_is_independent_of_gazetteer():
    from eurlex_ontology_generation.utils.evaluation import typer_ablation

    r = typer_ablation()
    assert r["canonical_names"]["n"] == len(load_gazetteer().entries)
    # the ablation must not be able to see the gazetteer: a perfect score would
    # mean the lookup leaked into the "held-out" path
    assert r["canonical_names"]["type_accuracy"] < 100.0


# ------------------------------------------------------------ persistence
def test_append_dedup_csv_keeps_rounds(tmp_path):
    path = tmp_path / "part_000.csv"
    pd.DataFrame([{"chunk_id": "a", "approved": False}]).to_csv(path, index=False)   # legacy file
    append_dedup_csv(path, pd.DataFrame([{"chunk_id": "a", "round": 1, "approved": True}]),
                     key=["chunk_id", "round"])
    append_dedup_csv(path, pd.DataFrame([{"chunk_id": "a", "round": 1, "approved": False}]),
                     key=["chunk_id", "round"])
    df = pd.read_csv(path)
    assert sorted(df["round"].tolist()) == [0, 1]
    assert df[df["round"] == 1]["approved"].tolist() == [False]


# ------------------------------------------------------------ review loop
def test_rereview_reads_revisions_and_records_round(tmp_path, monkeypatch):
    from eurlex_ontology_generation.pipelines.ontology_review import nodes as review_nodes
    from eurlex_ontology_generation.pipelines.ontology_revision.nodes import load_ontology_reviews

    normalized, _ = normalize_chunk_ontology(json.loads(json.dumps(LEGACY)))
    text = TestReviewChecks.TEXT
    pd.DataFrame([{"batch_id": 0, "chunk_id": "c1", "CELEX": "X", "text": text,
                   "ontology": json.dumps(normalized)}]).to_csv(tmp_path / "part_000.csv", index=False)

    empty_review = review_nodes.Review(overall_score=4, odrl_compliance=4, strengths=[], weaknesses=[],
                                       missing_concepts=[], unsupported_concepts=[], hallucinations=[])
    monkeypatch.setattr(review_nodes, "_call_judge", lambda **kw: (empty_review, None))
    thresholds = {"min_overall_score": 4, "min_odrl_compliance": 4, "max_verified_weaknesses": 1,
                  "max_warnings": 2, "count_by": "code"}
    out = review_nodes.review_ontologies(0, 1000, "unused", pd.DataFrame(), "m", 0.0, 1, 100, 100, 0,
                                         "sys", "{text}{ontology}{automatic_checks}", thresholds,
                                         {"max_workers": 1}, review_round=1, source="revisions",
                                         revisions_directory=str(tmp_path))
    assert out.loc[0, "round"] == 1 and out.loc[0, "status"] == "ok"

    reviews_dir = tmp_path / "reviews"
    review_nodes.persist_ontology_reviews(out.assign(approved=False), 0, 1000, str(reviews_dir))
    review_nodes.persist_ontology_reviews(out.assign(round=2, approved=True), 0, 1000, str(reviews_dir))
    latest = load_ontology_reviews(0, 1000, str(reviews_dir))
    assert len(latest) == 1 and int(latest.iloc[0]["round"]) == 2
