# Lessons (cross-act notes for the extraction agent)
Written only by the learning phase (scripts/agent_learn.py); one lesson per line starting with "- ".
All lessons go into the agent's system prompt. At most 40.
- Copy adverbial qualifiers like "directly or indirectly", "knowingly and intentionally", and "for their personal use only" into the constraints list verbatim rather than dropping them.
- "Include in the Annex" / "establish the list" is a duty, not a permission, and its target is the identifying information plus the "where available" condition.
- Do not extract recitals, definitions, scope statements, annex biographical listings, or "binding in its entirety" boilerplate as rules.
- When a sentence is passive or impersonal ("shall be used/developed/maintained") with no named actor, still extract the duty with assignee "unspecified".
- Record prior-approval and purpose clauses ("deliveries approved in advance", "intended solely for protective use") as constraints on the main rule, not as separate rules.
- Extract each indented item under a shared heading as its own rule, keeping the heading's actor, rather than collapsing them into one rule.
