---
name: constraint-qualifiers
description: Copying adverbial qualifiers, purpose clauses, and prior-approval conditions into constraints instead of dropping them.
when_to_use: When a rule sentence contains qualifiers like "directly or indirectly", "knowingly and intentionally", "for their personal use only", "intended solely for", or a prior-approval/purpose condition.
---
# Constraint qualifiers

## Recognise
Rules are frequently narrowed by qualifiers that change their scope. These must be captured as constraints, not dropped:
- "directly or indirectly" (manner)
- "knowingly and intentionally" (mental state)
- "for their personal use only", "intended solely for humanitarian... use", "for use in de-mining operations" (purpose)
- "deliveries approved in advance by the competent authority of the Member State in question" (prior approval)
- "taking full account of the criteria set out in..." (reference condition)
- "where available", "where appropriate" (conditional)

## Encode
- Put each qualifier into the constraints list, verbatim or near-verbatim.
- Do NOT drop a qualifier just because the action/object already looks complete.
- Do NOT turn a prior-approval or purpose clause into a separate rule; it is a constraint on the main rule.
- When the same qualifier recurs across several sub-rules (e.g. "deliveries approved in advance..."), repeat it on each rule.

## Example
TEXT: "Article 1 shall also not apply to protective clothing ... temporarily exported to Sudan by UN personnel ... for their personal use only."
RULE: {"rule_type": "permission", "assignee": "UN personnel, personnel of the EU or its Member States, representatives of the media, humanitarian and development workers and associated personnel", "action": "temporarily export to Sudan", "target": "protective clothing, including flak jackets and military helmets", "constraints": ["temporarily exported", "for their personal use only"], "flags": ["exception"]}

TEXT: "...participate, knowingly and intentionally, in activities the object or effect of which is to circumvent the measures referred to in paragraph 2(a) or (b)."
RULE: {"rule_type": "prohibition", "assignee": "unspecified", "action": "participate in activities circumventing the measures", "target": "measures referred to in Art. 1(2)(a) or (b)", "constraints": ["knowingly and intentionally", "object or effect is circumvention"]}

## Pitfalls
- Omitting "directly or indirectly" / "knowingly and intentionally" (the most common incompleteness).
- Omitting purpose qualifiers ("intended solely for protective use", "for use in de-mining operations").
- Dropping the prior-approval constraint ("deliveries approved in advance by the competent authority").
- Inventing a named actor (e.g. MemberState) when the text leaves the actor unspecified.
