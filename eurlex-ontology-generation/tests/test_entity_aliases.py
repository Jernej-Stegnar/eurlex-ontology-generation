"""T13: merge keys for word order, contrast vetoes, act-text acronym aliases."""
import json

import pandas as pd

from eurlex_ontology_generation.utils.ontology_merge_common import entity_resolution as er
from eurlex_ontology_generation.utils.ontology_merge_common.act_aliases import (
    acronym_definitions,
    apply_act_aliases,
)


def key(name):
    return er.group_key("class", name)


def test_word_order_variants_share_a_key():
    assert key("RulesOfTradingVenue") == key("TradingVenueRules")
    assert key("CommitteeForRiskAssessment") == key("RiskAssessmentCommittee")
    assert key("ReportOfImportation") == key("ImportationReport")
    assert key("NotificationOfNonCompliance") == key("NonComplianceNotification")


def test_word_order_keeps_head_noun_distinct():
    assert key("HeadOfCommissionDepartment") != key("CommissionDepartment")
    assert key("UseOfEncryptingTechnology") != key("EncryptingTechnology")
    assert er.key_tokens("ScientificCommitteeOnOccupationalExposureLimits")[-1] == "committee"


def test_alias_table_still_resolves():
    # "Council of the European Union" is a hand-curated alias of Council
    assert er.ALIAS_DISPLAY.get(key("Council_of_the_European_Union")) == "Council"
    assert er.ALIAS_DISPLAY.get(key("CouncilOfTheEuropeanUnion")) == "Council"


def test_actions_are_not_reordered():
    assert er.group_key("action", "notify_of") == "notify_of"


def tok(n):
    return er.key_tokens(n)


def test_contrast_modifiers_never_merge():
    for a, b in [("EquivalenceAttestation", "NonEquivalenceAttestation"),
                 ("LightweightPlasticCarrierBag", "VeryLightweightPlasticCarrierBag"),
                 ("ElectronicallyTradedTransaction", "NonElectronicallyTradedTransaction")]:
        assert not er._mergeable(tok(a), tok(b), 0.99)


def test_different_modifiers_same_head_do_not_merge():
    assert not er._mergeable(tok("SmallEnterprise"), tok("MediumSizedEnterprise"), 0.95)
    assert not er._mergeable(tok("PlasticCarrierBag"), tok("LightweightPlasticCarrierBag"), 0.95)


def test_connector_and_plural_variants_still_merge():
    assert er._mergeable(tok("CarcinogenOrMutagen"), tok("CarcinogensAndMutagens"), 0.9)


TEXT_A = ("The Crypto Approval Authority (CAA) shall approve products. "
          "The Animal and Plant Health Inspection Service (hereinafter 'APHIS') shall inspect.")


def test_acronym_definitions():
    defs = acronym_definitions(TEXT_A + " Official Journal of the European Union (EU) L 1.")
    assert defs["CAA"] == "Crypto Approval Authority"
    assert defs["APHIS"] == "Animal and Plant Health Inspection Service"
    assert "EU" not in defs


def _row(celex, text, onto):
    return {"chunk_id": f"{celex}_0", "CELEX": celex, "text": text, "ontology": json.dumps(onto)}


def test_apply_act_aliases_renames_inside_the_act_only():
    onto = {
        "classes": [{"name": "CAA", "type": "Party"}],
        "relationships": [{"source": "CAA", "relation": "approves", "target": "Product"}],
        "rights_model": {"duties": [{"party": "CAA", "action": "approve", "asset": "Product"}]},
        "legal_patterns": [{"entities_involved": ["CAA", "Product"]}],
    }
    revisions = pd.DataFrame([
        _row("32018D0559", TEXT_A, onto),
        _row("32099R0001", "No definitions here.", onto),     # another act: CAA may mean something else
    ])
    out, report = apply_act_aliases(revisions)
    a = json.loads(out.loc[0, "ontology"])
    assert a["classes"][0]["name"] == "CryptoApprovalAuthority"
    assert a["relationships"][0]["source"] == "CryptoApprovalAuthority"
    assert a["rights_model"]["duties"][0]["party"] == "CryptoApprovalAuthority"
    assert a["legal_patterns"][0]["entities_involved"] == ["CryptoApprovalAuthority", "Product"]
    assert json.loads(out.loc[1, "ontology"])["classes"][0]["name"] == "CAA"
    assert report == {"32018D0559": {"CAA -> CryptoApprovalAuthority": 3}}


def test_apply_act_aliases_without_text_is_a_no_op():
    revisions = pd.DataFrame([{"chunk_id": "x_0", "CELEX": "x", "ontology": "{}"}])
    out, report = apply_act_aliases(revisions)
    assert out.equals(revisions) and report == {}
