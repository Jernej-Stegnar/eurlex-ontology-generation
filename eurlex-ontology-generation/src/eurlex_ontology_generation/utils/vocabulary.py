"""Controlled vocabulary: upper ontology, actions, constraints, relations, reserved terms.

Everything is loaded from ``resources/vocabulary.yml`` so that the generation
prompt, the JSON schema, the review checks, the revision post-processor and the
refinement pipeline share one definition.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
DEFAULT_VOCABULARY_PATH = RESOURCES_DIR / "vocabulary.yml"

OTHER_ACTION = "other"

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}


# ---------------------------------------------------------------- tokenising
def split_identifier(name: str) -> list[str]:
    """'EGF_MobilisationDecision' -> ['egf', 'mobilisation', 'decision']."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def lemma(token: str) -> str:
    """Very small verb/noun lemmatiser, good enough for snake_case actions."""
    t = token.lower()
    if len(t) > 5 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 5 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("ed") and not t.endswith("eed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def _forms(token: str) -> set[str]:
    """Surface forms a token may stand for: 'mobilised' -> {mobilised, mobilise, mobilis}."""
    t = token.lower()
    out = {t, lemma(t)}
    if t.endswith(("ed", "es")) and len(t) > 4:
        out.add(t[:-1])
    if t.endswith("ing") and len(t) > 5:
        out.add(t[:-3] + "e")
    return out


def _key(text: str) -> str:
    return "".join(split_identifier(text))


@dataclass
class Vocabulary:
    core_types: list[str]
    upper: dict[str, dict]
    actions: dict[str, dict]
    left_operands: dict[str, dict]
    operators: list[str]
    operator_cues: dict[str, list[str]]
    relations: dict[str, dict]
    reserved_always: set[str]
    duty_allowlist: set[str]
    bare_core_names: set[str]
    # derived indexes
    action_synonyms: list[tuple[list[str], str]] = field(default_factory=list)
    relation_index: dict[str, tuple[str, bool]] = field(default_factory=dict)
    keyword_to_upper: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------ properties
    @property
    def upper_names(self) -> list[str]:
        return list(self.upper)

    @property
    def action_names(self) -> list[str]:
        return list(self.actions)

    @property
    def relation_names(self) -> list[str]:
        return list(self.relations)

    @property
    def left_operand_names(self) -> list[str]:
        return list(self.left_operands)

    def core_type_of(self, is_a: str | None) -> str | None:
        entry = self.upper.get(is_a or "")
        return entry["core_type"] if entry else None

    # --------------------------------------------------------------- typing
    def guess_is_a(self, name: str, description: str = "") -> tuple[str, str]:
        """Pick an upper class from the head noun (last token), then from any
        token, then from the description. Returns (is_a, evidence)."""
        tokens = split_identifier(name)
        for tok in reversed(tokens[-1:]):
            for cand in (tok, lemma(tok)):
                if cand in self.keyword_to_upper:
                    return self.keyword_to_upper[cand], f"head noun '{tok}'"
        for tok in reversed(tokens[:-1]):
            for cand in (tok, lemma(tok)):
                if cand in self.keyword_to_upper:
                    return self.keyword_to_upper[cand], f"name token '{tok}'"
        for tok in split_identifier(description)[:12]:
            if tok in self.keyword_to_upper:
                return self.keyword_to_upper[tok], f"description word '{tok}'"
        return "Concept", "no keyword matched"

    # -------------------------------------------------------------- leakage
    def metamodel_leakage(self, name: str) -> str | None:
        """Reason string if `name` is an ODRL meta-model term used as a class."""
        tokens = split_identifier(name)
        if not tokens:
            return None
        if "".join(tokens) in self.bare_core_names:
            return f"'{name}' is a bare ODRL core term"
        joined_pairs = {a + b for a, b in zip(tokens, tokens[1:])}
        hit = (set(tokens) | joined_pairs) & self.reserved_always
        if hit:
            return f"'{name}' contains ODRL meta-model term '{sorted(hit)[0]}'"
        if tokens[-1] == "duty" and len(tokens) > 1:
            prefix = tokens[:-1]
            if not (set(prefix) & self.duty_allowlist):
                if self.match_action(prefix)[0] is not None:
                    return f"'{name}' is a reified duty (action '{'_'.join(prefix)}' + 'Duty')"
        return None

    # --------------------------------------------------------------- actions
    def match_action(self, tokens: list[str]) -> tuple[str | None, int]:
        """Longest synonym matching the start of `tokens`. Returns (canonical, n_tokens)."""
        forms = [_forms(t) for t in tokens]
        for syn, canonical in self.action_synonyms:      # sorted longest first
            n = len(syn)
            if len(tokens) >= n and all(syn[i] in forms[i] for i in range(n)):
                return canonical, n
        return None, 0

    def normalize_action(self, action: Any, label: Any = None) -> dict:
        """Map a free-text action to {action, action_object, action_label, controlled, included_in}.

        `action` may already be canonical (structured output) or free text
        such as 'provide_financial_contribution'. `label` is the generator's
        original wording when it used structured output.
        """
        raw_action = str(action or "").strip()
        raw_label = str(label or "").strip()
        if raw_action in self.actions:
            obj = ""
            if raw_label:
                toks = split_identifier(raw_label)
                canonical, n = self.match_action(toks)
                obj = "_".join(toks[n:]) if canonical == raw_action else ""
            return {"action": raw_action, "action_object": obj,
                    "action_label": raw_label or raw_action, "controlled": True,
                    "included_in": self.actions[raw_action]["included_in"]}

        text = raw_label if raw_action in ("", OTHER_ACTION) else raw_action
        tokens = split_identifier(text)
        if not tokens:
            return {"action": OTHER_ACTION, "action_object": "", "action_label": text,
                    "controlled": False, "included_in": "use"}
        canonical, n = self.match_action(tokens)
        if canonical:
            return {"action": canonical, "action_object": "_".join(tokens[n:]),
                    "action_label": "_".join(tokens), "controlled": True,
                    "included_in": self.actions[canonical]["included_in"]}
        # unknown verb: keep its lemma as the action, flag as uncontrolled
        return {"action": lemma(tokens[0]), "action_object": "_".join(tokens[1:]),
                "action_label": "_".join(tokens), "controlled": False, "included_in": "use"}

    # ----------------------------------------------------------- constraints
    def infer_left_operand(self, name: str, value: str = "") -> tuple[str | None, bool]:
        """(left_operand, controlled) for a legacy constraint name."""
        if name in self.left_operands:
            return name, True
        name_tokens = split_identifier(name)
        joined = "_".join(name_tokens)
        for operand, spec in self.left_operands.items():
            for kw in spec["keywords"]:
                if kw == joined or kw in name_tokens or ("_" in kw and kw in joined):
                    return operand, True
        if parse_date(value):
            return "dateTime", True
        if _CURRENCY.search(str(value)):
            return "payAmount", True
        if re.search(r"\d\s*(%|per\s?cent)", str(value), re.I):
            return "percentage", True
        return None, False

    def infer_operator(self, value: str) -> str:
        v = f" {str(value).lower()} "
        for operator, cues in self.operator_cues.items():
            if any(f" {cue} " in v for cue in cues):
                return operator
        return "eq"

    def structure_constraint(self, constraint: dict) -> dict | None:
        """Return an ODRL-style constraint dict, converting legacy {name, value} forms."""
        if not isinstance(constraint, dict):
            return None
        if constraint.get("left_operand"):
            left = str(constraint["left_operand"])
            operator = str(constraint.get("operator") or "eq")
            right = constraint.get("right_operand")
            return {"left_operand": left,
                    "operator": operator if operator in self.operators else "eq",
                    "right_operand": _normalize_value(left, right),
                    "unit": str(constraint.get("unit") or ""),
                    "controlled": left in self.left_operands}
        name = str(constraint.get("name") or "").strip()
        value = constraint.get("value")
        if not name or value in (None, ""):
            return None
        left, controlled = self.infer_left_operand(name, str(value))
        # unmatched qualitative conditions fall back to the generic extension
        # operand; `controlled: False` keeps them visible in the metrics
        if left is None and "condition" in self.left_operands:
            left = "condition"
        right, unit = _normalize_value(left, value), ""
        if left == "payAmount" and split_amount(value):
            right, unit = split_amount(value)
        return {"left_operand": left or name,
                "operator": self.infer_operator(str(value)),
                "right_operand": right,
                "unit": unit,
                "source_name": name,
                "controlled": controlled}

    # ------------------------------------------------------------- relations
    def normalize_relation(self, relation: Any, source: Any, target: Any) -> tuple[str | None, Any, Any, bool]:
        """(canonical_relation, source, target, controlled). Inverse synonyms swap ends."""
        if not isinstance(relation, str) or not relation.strip():
            return None, source, target, False
        key = "_".join(split_identifier(relation))
        if key in self.relation_index:
            canonical, inverse = self.relation_index[key]
            return (canonical, target, source, True) if inverse else (canonical, source, target, True)
        return key or None, source, target, False

    # ---------------------------------------------------------- prompt text
    def render_upper_ontology(self) -> str:
        lines = []
        for core in self.core_types:
            lines.append(f"{core}:")
            for name, spec in self.upper.items():
                if spec["core_type"] == core:
                    lines.append(f"  - {name}: {spec['description']}")
        return "\n".join(lines)

    def render_actions(self) -> str:
        return ", ".join(self.action_names)

    def render_left_operands(self) -> str:
        return ", ".join(self.left_operand_names)

    def render_relations(self) -> str:
        return ", ".join(self.relation_names)

    def prompt_variables(self) -> dict[str, str]:
        return {
            "core_types": ", ".join(self.core_types),
            "upper_ontology": self.render_upper_ontology(),
            "actions": self.render_actions(),
            "left_operands": self.render_left_operands(),
            "operators": ", ".join(self.operators),
            "relations": self.render_relations(),
        }


def render_prompt(template: str, vocabulary_path: str | None = None, **values) -> str:
    """Fill a prompt template with the controlled-vocabulary placeholders
    ({core_types}, {upper_ontology}, {actions}, {left_operands}, {operators},
    {relations}) plus any extra values such as {text}."""
    variables = load_vocabulary(vocabulary_path).prompt_variables()
    variables.update(values)
    return template.format(**variables)


# ------------------------------------------------------------------ values
def parse_date(value: Any) -> str | None:
    """ISO date for '1 January 2019', 'January 1, 2019', '2019-01-01'; else None."""
    s = str(value or "").strip()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        return m.group(0)
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", s)
    if m and m.group(2).lower() in _MONTHS:
        return _iso(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", s)
    if m and m.group(1).lower() in _MONTHS:
        return _iso(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
    return None


def _iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


_CURRENCY = re.compile(r"\b(EUR|ECU|USD|GBP)\b|€")
_AMOUNT = re.compile(r"(?:\b(EUR|ECU|USD|GBP)\b|€)\s*([\d][\d\s.,]*\d|\d)")


def split_amount(value: Any) -> tuple[str, str] | None:
    """'EUR 600 000 000 (2011 prices)' -> ('600000000', 'EUR')."""
    m = _AMOUNT.search(str(value or ""))
    if not m:
        return None
    number = re.sub(r"[\s,]", "", m.group(2))
    if number.count(".") > 1:          # 1.000.000 style thousands separators
        number = number.replace(".", "")
    return number, (m.group(1) or "EUR")


def _normalize_value(left_operand: str | None, value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if left_operand == "dateTime":
        return parse_date(text) or text
    return text


# ------------------------------------------------------------------ loading
def _build(raw: dict) -> Vocabulary:
    cons = raw["constraints"]
    reserved = raw["reserved_terms"]
    vocab = Vocabulary(
        core_types=list(raw["core_types"]),
        upper=dict(raw["upper_ontology"]),
        actions=dict(raw["actions"]),
        left_operands=dict(cons["left_operands"]),
        operators=list(cons["operators"]),
        operator_cues={k: list(v) for k, v in cons["operator_cues"].items()},
        relations=dict(raw["relations"]),
        reserved_always=set(reserved["always"]),
        duty_allowlist=set(reserved["duty_allowlist"]),
        bare_core_names={_key(n) for n in reserved["bare_core_names"]},
    )
    syns = []
    for canonical, spec in vocab.actions.items():
        for syn in [canonical, *spec.get("synonyms", [])]:
            syns.append((syn.split("_"), canonical))
    vocab.action_synonyms = sorted(syns, key=lambda s: -len(s[0]))
    for canonical, spec in vocab.relations.items():
        vocab.relation_index[canonical] = (canonical, False)
        for syn in spec.get("synonyms", []):
            vocab.relation_index.setdefault("_".join(split_identifier(syn)), (canonical, False))
        for syn in spec.get("inverse_synonyms", []):
            vocab.relation_index.setdefault("_".join(split_identifier(syn)), (canonical, True))
    for name, spec in vocab.upper.items():
        for kw in spec.get("keywords", []):
            vocab.keyword_to_upper.setdefault(kw, name)
    for name, spec in vocab.upper.items():
        if spec["core_type"] not in vocab.core_types:
            raise ValueError(f"Upper class {name} has unknown core_type {spec['core_type']}")
    return vocab


@lru_cache(maxsize=4)
def load_vocabulary(path: str | None = None) -> Vocabulary:
    p = Path(path) if path else DEFAULT_VOCABULARY_PATH
    with open(p, encoding="utf-8") as fh:
        return _build(yaml.safe_load(fh))
