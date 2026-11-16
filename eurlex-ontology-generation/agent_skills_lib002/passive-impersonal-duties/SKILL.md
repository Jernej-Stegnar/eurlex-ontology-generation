---
name: passive-impersonal-duties
description: Duties phrased in the passive voice or with an impersonal subject, where the actor is unspecified but the obligation is still a rule.
when_to_use: When a sentence uses "shall be + past participle" or an impersonal subject ("the system shall", "the document shall") with no named actor.
---
# Passive and impersonal duties

## Recognise
Operative obligations are often written without naming who must act:
- "The system shall use industry standards and be able to interact with public and private systems..."
- "The central SafeSeaNet shall be used for the distribution of electronic messages..."
- "The IFCD shall describe in detail the performance requirements..."
- "SafeSeaNet networks shall be used to distribute amongst Member States..."
- "The IFCD shall include rules for: ..." (followed by indented items)

These are still duties. Do NOT skip a chunk because no actor is named.

## Encode
- rule_type: duty
- assignee: "unspecified" (do not invent an actor such as the Commission or the Agency unless the text names one)
- action: the verb phrase after "shall be" (e.g. "use industry standards and interact with public and private systems", "be used for the distribution of", "describe in detail the performance requirements", "include rules for")
- target: the object of the action (e.g. "public and private systems used to create, provide or receive information within SafeSeaNet", "electronic messages and data exchanged or shared...", "an interface and functionalities control document (IFCD)")
- constraints: any qualifier clause ("where internationally-adopted rules allow...", "with an appropriate level of security", "in close cooperation with the Member States")

When a heading introduces a list of indented items each beginning with a verb ("provide appropriate guidance", "approve the interface...", "adopt guidelines", "liaise with..."), extract EACH indented item as its own duty, with the shared actor from the heading.

## Example
TEXT: "The central SafeSeaNet shall be used for the distribution of electronic messages and data exchanged or shared in accordance with this Directive and relevant Union legislation."
RULE: {"rule_type": "duty", "assignee": "unspecified", "action": "use the central SafeSeaNet for the distribution", "target": "electronic messages and data exchanged or shared in accordance with this Directive and relevant Union legislation", "constraints": []}

TEXT: "The Commission shall develop and maintain, in close cooperation with the Member States, an interface and functionalities control document (IFCD)."
RULE: {"rule_type": "duty", "assignee": "the Commission", "action": "develop and maintain", "target": "an interface and functionalities control document (IFCD)", "constraints": ["in close cooperation with the Member States"]}

## Pitfalls
- Emitting 0 rules for a whole chunk of passive operative duties (the most costly miss).
- Inventing an actor (Commission/Agency) when the text uses a passive or impersonal subject; use "unspecified".
- Collapsing several indented items under one heading into a single rule; extract each item separately.
- Treating "shall be used/developed/maintained" as non-normative description and skipping it.
