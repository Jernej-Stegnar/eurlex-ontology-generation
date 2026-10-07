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

### Avoiding circular numbers
The gazetteer, keyword typer and seed gold set were written together, so any
score that compares one of them with another proves nothing. The evaluation
therefore separates:

| Number | What it compares | Evidence? |
|---|---|---|
| `typing.pct_known_entities_correct_type` | entity types vs gazetteer facts, **excluding entities the gazetteer typed** | yes on raw generator output; not reported (`null`) on the refined ontology, where every known entity was typed by the lookup |
| Held-out typer test (`typer_ablation`) | the deterministic typer with the gazetteer **switched off**, on every gazetteer name and alias | yes: the gazetteer's labels are facts about the world, not typer output |
| Gold `held_out` | independent labels (blind annotation sheets or `source: expert`) on entities **not** typed by the gazetteer | yes, this is the headline accuracy |
| Gold `gazetteer_lookups` | independent labels on gazetteer-typed entities | checks the gazetteer's content only |
| Gold `seed_smoke_test` | `conf/base/evaluation/gold_entities.yml` (`source: seed`) | no: regression test only |

Held-out typer result on the current vocabulary, measured once and not tuned
afterwards: **19% of the 95 canonical gazetteer names get the right core type
from keywords alone** (6% of countries, 64% of funds). The refined ontology's
correct typing of known entities comes from the gazetteer, and unknown named
entities depend on the generator's declared type plus the keyword fallback,
which is weak. Only blind annotation can measure that path:

```bash
uv run python scripts/make_annotation_sheet.py --n 120 --annotator A   # same seed -> same sample
uv run python scripts/make_annotation_sheet.py --n 120 --annotator B
# two people fill type / is_a / kind from the text (the sheet hides our labels), then:
uv run python scripts/evaluate_ontology.py data/07_model_output/eurlex_odrl_ontology_refined.json \
  --gold data/08_reporting/annotation/annotation_sheet_A.csv \
  --agreement data/08_reporting/annotation/annotation_sheet_A.csv data/08_reporting/annotation/annotation_sheet_B.csv
```
Report held-out accuracy only together with inter-annotator kappa; adjudicate
disagreements into a third sheet with `source` = `adjudicated`.

## OWL/ODRL export and validation
`uv run kedro run --pipelines ontology_export` (also run by
`scripts/run_ontology_merge.py`) turns the refined JSON into
`data/07_model_output/eurlex_odrl_ontology.ttl` and checks it three ways:

| Check | Tool | Fails the run on |
|---|---|---|
| Turtle round trip | rdflib | any triple lost |
| SHACL (`resources/shapes.ttl`) | pySHACL | any `sh:Violation` (`fail_on_violation`) |
| OWL 2 consistency | HermiT via owlready2 (optional, needs Java) | inconsistency or unsatisfiable class |

Graph layout: `eo:Party ≡ odrl:Party`, `eo:Asset ≡ odrl:Asset` and
`eo:LegalExtension` are pairwise disjoint; the 20 upper classes sit under them;
domain classes are `owl:Class` + `rdfs:subClassOf` an upper class; individuals
are `owl:NamedIndividual` typed with an upper class, `owl:sameAs` their Wikidata
item and `skos:exactMatch` their EU Vocabularies IRI. Rules are
`odrl:Permission/Prohibition/Duty` in one `odrl:Set`, with typed constraints
(`xsd:date`, `xsd:decimal`, `odrl:unit` = EU currency IRI).

### Namespace
IRIs live under the persistent namespace `https://w3id.org/eurlex-odrl/`
(`ontology#` vocabulary, `entity/`, `action/`, `policy/`, `shapes#`). To make
them resolve:
1. merge this branch into `main` (the redirects point at files on `main`);
2. fork https://github.com/perma-id/w3id.org, copy `publish/w3id/eurlex-odrl/`
   (`.htaccess` + `README.md`) to the root of the fork as `eurlex-odrl/`, and
   open a pull request.
RDF clients are then redirected (303) to the Turtle files, browsers to this
document.

SHACL severities: **Violation** = structurally broken (rule without action or
assignee, non-ODRL operator, entity under two disjoint core classes, core type
contradicting its parent, meta-model term as a name, dangling reference);
**Warning** = modelling problem needing a human decision (assignee not a Party,
target is a Party, duty from a recital, free-text `condition` constraint,
undated `dateTime`, action outside the vocabulary, isolated entity);
**Info** = individual without an external identifier.
Results: `data/08_reporting/ontology_export_summary.json` and `shacl_report.ttl`.

## Identifiers
`resources/gazetteer.yml` now carries Wikidata QIDs for 91 entries (collected
from web-search results whose page title exactly matched the label; ambiguous
cases have a `wikidata_note`) and EU Vocabularies IRIs for all countries plus the
Commission's corporate-body IRI. On a machine with internet access:
```bash
uv run python scripts/link_identifiers.py --verify                       # label + P31 check of every QID
uv run python scripts/link_identifiers.py --corporate-body-rdf <file> --write   # fill corporate-body IRIs
```

## Running
```bash
uv run python scripts/run_ontology_batches.py --max-batches 10 --review-rounds 2
uv run python scripts/run_ontology_merge.py          # partitions, global, refinement
uv run kedro run --pipelines ontology_refinement     # refinement only (no LLM)
uv run kedro run --pipelines ontology_export          # Turtle + SHACL + HermiT
uv sync --extra reasoner                              # owlready2 for the HermiT check
uv run pytest tests/test_quality_harness.py tests/test_rdf_export.py
```

## Not done here
- Corporate-body IRIs other than the Commission's (run `link_identifiers.py`
  with the downloaded authority table).
- Chunking on article boundaries (recitals are only flagged, not separated).
- The new prompts and schema have not been run against Ollama in this branch;
  run a few batches and compare with `evaluate_ontology.py`.
