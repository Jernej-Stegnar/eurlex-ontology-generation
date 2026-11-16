---
name: writing-skills
description: Format and rules for skills in this library. Read only if you are asked to write or edit a skill.
when_to_use: in the learning phase, when proposing or editing a skill
---
A skill is one recurring legal pattern and how to encode it in the ontology the pipeline expects.

Format: a folder `<name>/SKILL.md` (name: lowercase words joined by hyphens) starting with front matter:
name, description (one line: what the pattern is), when_to_use (one line: the textual cue that triggers it).

Body, at most 150 lines:
1. Recognise: the wording that signals the pattern, with 2-3 real phrasings.
2. Encode: which rights_model list, party, action, asset and constraints to use; what NOT to extract.
3. Example: one short worked example from a training act: the sentence, then the rule(s) as JSON.
4. Pitfalls: the errors this skill exists to prevent.

Rules:
- One pattern per skill. Prefer editing an existing skill to adding a near-duplicate.
- Examples come only from training acts, never from evaluation acts.
- State procedures, not facts about one act.
