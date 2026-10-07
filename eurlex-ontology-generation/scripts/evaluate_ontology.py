"""Evaluate one or more ontology JSON files and print/save the quality metrics.

Examples
  # one ontology
  uv run python scripts/evaluate_ontology.py data/07_model_output/eurlex_odrl_ontology_refined.json

  # before / after, plus review approval rates and the gold set
  uv run python scripts/evaluate_ontology.py \
      data/06_models/eurlex_odrl_ontology_merged.json \
      data/07_model_output/eurlex_odrl_ontology_refined.json \
      --reviews-dir data/04_feature/ontology_reviews \
      --gold conf/base/evaluation/gold_entities.yml \
      --out data/08_reporting/ontology_evaluation.json

Run it on every change to prompts, vocabulary, gazetteer or merge settings,
on the same slice of the corpus, and compare the numbers.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eurlex_ontology_generation.utils.evaluation import (  # noqa: E402
    compute_metrics,
    gold_metrics,
    headline_table,
    review_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ontologies", nargs="+", help="Ontology JSON files to evaluate.")
    parser.add_argument("--gold", default=None, help="Gold entity annotations (YAML).")
    parser.add_argument("--reviews-dir", default=None, help="Directory with review part_*.csv files.")
    parser.add_argument("--out", default=None, help="Write all metrics to this JSON file.")
    parser.add_argument("--shacl", action="store_true",
                        help="Also export refined ontologies to RDF and count SHACL results.")
    args = parser.parse_args()

    results = {}
    for path in args.ontologies:
        onto = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = compute_metrics(onto)
        if args.gold:
            metrics["gold"] = gold_metrics(onto, args.gold)
        if args.shacl and "individuals" in onto:
            from eurlex_ontology_generation.utils.rdf_export import export_rdf
            from eurlex_ontology_generation.utils.rdf_validate import validate_graph

            _, _, shacl = validate_graph(export_rdf(onto)[0])
            counts = shacl["results_by_severity"]
            metrics["shacl"] = {"violations": counts.get("Violation", 0),
                                "warnings": counts.get("Warning", 0), "infos": counts.get("Info", 0)}
        results[Path(path).name] = metrics

    print(headline_table(results))
    if args.gold:
        print()
        for label, m in results.items():
            g = m["gold"]
            print(f"gold [{label}]: found {g['found_in_ontology']}/{g['gold_entities']}  "
                  f"type {g['type_accuracy']}%  is_a {g['is_a_accuracy']}%  kind {g['kind_accuracy']}%")

    output = {"ontologies": results}
    if args.reviews_dir:
        output["review"] = review_metrics(args.reviews_dir)
        print()
        for rnd, r in output["review"].get("per_round", {}).items():
            print(f"review round {rnd}: {r['approved']}/{r['chunks']} approved ({r['pct_approved']}%), "
                  f"mean overall score {r['mean_overall_score']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
