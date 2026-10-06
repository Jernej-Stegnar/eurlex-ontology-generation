# Ontology quality harness

This branch adds a harness around the LLM steps so that typing, vocabulary and
review quality are enforced and measured in code, not left to the prompts.

## What changed, by issue

### 1. Inconsistent typing
| Layer | Where | What it does |
|---|---|---|
| Controlled vocabulary | `src/.../resources/vocabulary.yml` | One file defines class types (`Party`, `Asset`, `Legal_extension`), a 20-class upper ontology (`is_a`), canonical actions, ODRL constraint operands/operators, canonical relations and reserved ODRL terms. |
| Constrained decoding | `utils/ontology_schema.py`, `ontology_generation` | The JSON schema built from the vocabulary is passed to Ollama's `format`, so `type`, `is_a`, `kind`, `action`, `left_operand`, `operator` and `relation` are enums the model cannot leave (`structured_output: true`). |
| Prompt fix | `parameters_ontology_generation.yml` | The example no longer says `"type": "ODRL_core"` (the generator copied it; the reviewer then penalised it on almost every chunk). Prohibitions are in the schema again (the revision schema had dropped them). Recitals are flagged as non-binding. |
| Gazetteer | `src/.../resources/gazetteer.yml`, `utils/entity_linking.py` | Countries (with EU Vocabularies country IRIs), EU institutions and bodies, international organisations and named funds/programmes. A match fixes type, `is_a` and `kind` deterministically and force-merges aliases. |
| Deterministic typer | `utils/ontology_normalize.py` | Gazetteer > declared `is_a` > head-noun keyword (`...Fund` → `Fund`) > declared type. |
| Checks | `ontology_review/ontology_checks.py` | New findings: `invalid_class_type`, `gazetteer_type_mismatch`, `rule_slot_type` (party not a Party, asset is a Party). |
| Refinement | `pipelines/ontology_refinement` | Re-types the merged ontology globally and reports slot-typing conflicts that remain. |

### 2. Classes that should be individuals
`utils/entity_linking.detect_kind` decides class vs individual, in order:
gazetteer hit → name built on a known individual (`Cyprus_Government` is an
individual, `EGF_Mobilization` is a class of mobilisations of the EGF) →
all-caps acronym (low confidence) → declared `kind`. Optionally, low-confidence
decisions are checked on Wikidata (`P31` instance-of vs `P279` subclass-of,
exact label match only, cached; `ontology_refinement.wikidata.enabled`). EU
sources come first because EU-specific constructs are rarely in Wikidata.
The refined ontology has separate `classes` and `individuals` (with
`instance_of`), and low-confidence decisions go to `review_queue` in the report.

### 3. Meta-model leakage and uncontrolled vocabulary
- Leakage: class names built on ODRL structure (`AuditDuty`,
  `AuditPermissionTarget`, `Assignee`) are flagged by the review, removed by
  the normaliser and rejected by the merge. `ImportDuty`, `RuleOfLaw`,
  `NetworkOperator` are allowed (see `reserved_terms` in the vocabulary).
- Actions: free-text actions are split into a canonical verb plus an object
  (`mobilize_egf` → `mobilise` + `egf`), each verb mapped to its ODRL parent
  action. Unknown verbs are kept but marked `controlled: false`.
- Constraints: converted to `{left_operand, operator, right_operand, unit}`;
  dates become ISO, amounts become digits plus currency. Unmatched qualitative
  conditions fall back to the extension operand `condition` with
  `controlled: false`.
- Relations: closed set; inverse forms are flipped (`funded_by` → `funds`).

### 4. Review verdicts
- Scoring counts findings per distinct code (`approval_thresholds.count_by:
  code`): ten mistyped classes are one root cause, not ten. `finding`
  restores the old behaviour.
- `info` findings (vocabulary drift fixed in code) never affect approval.
- **Re-review loop**: `scripts/run_ontology_batches.py --review-rounds N`
  runs generate → review(0) → revise(1) → review(1) → … and stops when every
  chunk is approved. Reviews are stored per `(chunk_id, round)`; the revision
  reads the latest round.
- The merge joins the latest verdict and records `support.approved_chunks`
  for every class and rule; `exclude_unapproved: true` merges only approved
  chunks.

## Evaluation harness
```bash
uv run python scripts/evaluate_ontology.py \
  data/06_models/eurlex_odrl_ontology_merged.json \
  data/07_model_output/eurlex_odrl_ontology_refined.json \
  --reviews-dir data/04_feature/ontology_reviews \
  --gold conf/base/evaluation/gold_entities.yml \
  --out data/08_reporting/ontology_evaluation.json
```
Reports typing, kind, leakage, action/constraint/relation control,
consistency, linking and review approval per round, plus gold-set accuracy.
`conf/base/evaluation/gold_entities.yml` is a **seed** written with the
harness; replace it with expert annotations before trusting its numbers.

## Running
```bash
uv run python scripts/run_ontology_batches.py --max-batches 10 --review-rounds 2
uv run python scripts/run_ontology_merge.py          # partitions, global, refinement
uv run kedro run --pipelines ontology_refinement     # refinement only (no LLM)
uv run pytest tests/test_quality_harness.py
```

## Not done here
- Corporate-body IRIs and Wikidata QIDs in the gazetteer are left empty rather
  than typed from memory; fill them from the EU Vocabularies authority tables.
- OWL/Turtle export and SHACL shapes.
- Chunking on article boundaries (recitals are only flagged, not separated).
- The new prompts and schema have not been run against Ollama in this branch;
  run a few batches and compare with `evaluate_ontology.py`.
