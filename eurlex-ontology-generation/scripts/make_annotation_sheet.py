"""Create a BLIND annotation sheet for building an independent gold set.

The sheet lists a reproducible random sample of entities from the refined
ontology with their name, the text the model gave as a description and up to
two source-text snippets. It deliberately does NOT show the pipeline's type,
is_a or kind, so annotators label from the text, not from our output.

  uv run python scripts/make_annotation_sheet.py --n 120 --annotator A
  uv run python scripts/make_annotation_sheet.py --n 120 --annotator B   # same sample (same seed)

Annotators fill the columns type / is_a / kind (allowed values are in the
`_allowed_values` sheet header and in ANNOTATION_GUIDE below). Then:

  uv run python scripts/evaluate_ontology.py data/07_model_output/eurlex_odrl_ontology_refined.json \
      --gold data/08_reporting/annotation/annotation_sheet_A.csv \
      --agreement data/08_reporting/annotation/annotation_sheet_A.csv data/08_reporting/annotation/annotation_sheet_B.csv

Sampling is stratified: entities the gazetteer did NOT type are
over-represented (they are what the typer and the LLM decide), and a small
share of gazetteer-typed entities is kept as a control.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eurlex_ontology_generation.utils.vocabulary import load_vocabulary  # noqa: E402

ANNOTATION_GUIDE = """\
type   Party (an actor) | Asset (a thing acted upon: legal act, fund, money, programme, resource, document)
       | Legal_extension (procedure, measure, event, sector, other legal concept)
is_a   one upper class: {upper}
kind   individual (ONE named thing: a country, an institution, a named fund)
       | class (a KIND of thing: Member State, application, financial contribution)
Leave a cell empty if the text does not let you decide; use `notes` for doubts.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ontology", default="data/07_model_output/eurlex_odrl_ontology_refined.json")
    p.add_argument("--n", type=int, default=120, help="Number of entities to sample.")
    p.add_argument("--control-share", type=float, default=0.15,
                   help="Share of the sample drawn from gazetteer-typed entities.")
    p.add_argument("--seed", type=int, default=20261007, help="Same seed = same sample for every annotator.")
    p.add_argument("--annotator", default="A")
    p.add_argument("--out-dir", default="data/08_reporting/annotation")
    args = p.parse_args()

    onto = json.loads(Path(args.ontology).read_text(encoding="utf-8"))
    entities = onto.get("classes", []) + onto.get("individuals", [])
    contexts: dict[str, list[str]] = {}
    for ex in onto.get("example_instances", []):
        ctx = (ex.get("source_context") or "").strip()
        if ctx:
            contexts.setdefault(ex.get("entity"), []).append(ctx[:300])

    rng = random.Random(args.seed)
    by_gaz = sorted((e for e in entities if e.get("type_source") == "gazetteer"), key=lambda e: e["name"])
    other = sorted((e for e in entities if e.get("type_source") != "gazetteer"), key=lambda e: e["name"])
    n_control = min(len(by_gaz), round(args.n * args.control_share))
    sample = rng.sample(other, min(len(other), args.n - n_control)) + rng.sample(by_gaz, n_control)
    rng.shuffle(sample)

    rows = [{
        "item": i + 1,
        "name": e["name"],
        "description": (e.get("description") or "")[:400],
        "context_1": (contexts.get(e["name"]) or [""])[0],
        "context_2": (contexts.get(e["name"]) or ["", ""])[1] if len(contexts.get(e["name"]) or []) > 1 else "",
        "type": "", "is_a": "", "kind": "", "notes": "",
        "annotator": args.annotator, "source": "annotator",
    } for i, e in enumerate(sample)]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / f"annotation_sheet_{args.annotator}.csv"
    pd.DataFrame(rows).to_csv(sheet, index=False, encoding="utf-8")
    vocab = load_vocabulary()
    (out / "ANNOTATION_GUIDE.txt").write_text(
        ANNOTATION_GUIDE.format(upper=", ".join(vocab.upper_names)), encoding="utf-8")
    print(f"Wrote {sheet} ({len(rows)} entities: {len(rows) - n_control} decided by typer/LLM, "
          f"{n_control} gazetteer controls) and {out / 'ANNOTATION_GUIDE.txt'}")


if __name__ == "__main__":
    main()
