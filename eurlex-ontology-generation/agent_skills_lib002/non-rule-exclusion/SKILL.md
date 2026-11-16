---
name: non-rule-exclusion
description: Recognising text that is not a normative rule (recitals, definitions, scope, annex biographical listings, entry-into-force boilerplate) and not extracting it.
when_to_use: When a chunk contains recitals, definitions, scope statements, annex biographical data, or "binding in its entirety" boilerplate.
---
# Non-rule exclusion (precision)

## Recognise
Not every normative-looking sentence is a rule. Do NOT extract:
- Recitals ("Having regard to...", "Whereas...", "The Commission, Having regard to the Treaty...").
- Definitions ("For the purposes of this Regulation, 'X' means...").
- Scope statements ("This Regulation shall apply to...").
- Annex biographical listings of designated persons/entities (names, dates of birth, passport numbers, "reasonable grounds to believe...").
- "This Regulation shall be binding in its entirety and directly applicable in all Member States."
- "Done at Brussels, ..." and signature blocks.

## Encode
- Emit no rule for these.
- Entry into force ("This Regulation shall enter into force on the twentieth day following...") is a validity event, not a permission/duty/prohibition; encode it as rule_type "none" with flag "validity" only if the pipeline expects it, otherwise skip.
- A bare "The Annex ... is amended in accordance with Annex II" is an amendment marker, not a permission/duty; do not assign an actor to it.

## Example
TEXT: "This Regulation shall be binding in its entirety and directly applicable in all Member States."
RULE: (none)

TEXT: "The Annex to Implementing Regulation (EU) No 540/2011 is amended in accordance with Annex II to this Regulation."
RULE: {"rule_type": "none", "assignee": "unspecified", "action": "amend", "target": "the Annex to Implementing Regulation (EU) No 540/2011", "constraints": ["in accordance with Annex II to this Regulation"], "flags": ["amendment"]}

## Pitfalls
- Extracting annex biographical data as "designate person" duties (a large false-positive cluster).
- Extracting recitals or definitions as duties.
- Assigning an actor (e.g. the Commission) to a bare "is amended" sentence when the text names none.
- Treating "binding in its entirety" as a rule.
