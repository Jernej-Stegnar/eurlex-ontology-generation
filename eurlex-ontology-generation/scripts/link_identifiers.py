"""Verify and complete the gazetteer's external identifiers.

Run on a machine with internet access (the project's sandboxes cannot reach
wikidata.org or publications.europa.eu).

  # 1. check every Wikidata QID: English label/alias matches the entity and,
  #    for countries, P31 is 'country' (Q6256) or 'sovereign state' (Q3624078);
  #    also look up entries without a QID by exact label
  uv run python scripts/link_identifiers.py --verify

  # 2. fill EU corporate-body IRIs from the authority table. Download the
  #    "Corporate body" table (SKOS, RDF/XML or Turtle) from EU Vocabularies:
  #    https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/corporate-body
  uv run python scripts/link_identifiers.py --corporate-body-rdf corporatebodies-skos.rdf --write

Without --write the script only reports; with --write it edits
src/eurlex_ontology_generation/resources/gazetteer.yml in place (iri only for
entries that have none; QIDs are never overwritten, only reported).
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GAZETTEER = ROOT / "src" / "eurlex_ontology_generation" / "resources" / "gazetteer.yml"
USER_AGENT = "EurLexOntologyGeneration/0.1 (identifier verification)"
COUNTRY_TYPES = {"Q6256", "Q3624078"}          # country, sovereign state


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def entries() -> list[dict]:
    raw = yaml.safe_load(GAZETTEER.read_text(encoding="utf-8"))
    out = [{**c, "is_a": "Country"} for c in raw.get("countries", [])]
    return out + list(raw.get("entities", []))


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def verify(pause: float) -> list[dict]:
    report = []
    for e in entries():
        names = {key(e["name"].replace("_", " "))} | {key(a) for a in e.get("aliases", []) or []}
        qid = e.get("wikidata")
        row = {"name": e["name"], "wikidata": qid}
        try:
            if qid:
                data = get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
                ent = data["entities"][qid]
                labels = {key(ent.get("labels", {}).get("en", {}).get("value", ""))}
                labels |= {key(a["value"]) for a in ent.get("aliases", {}).get("en", [])}
                p31 = {c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                       for c in ent.get("claims", {}).get("P31", [])}
                row["label"] = ent.get("labels", {}).get("en", {}).get("value")
                row["label_ok"] = bool(names & labels)
                row["p31"] = sorted(q for q in p31 if q)
                row["type_ok"] = (bool(p31 & COUNTRY_TYPES) if e["is_a"] == "Country" else None)
                row["status"] = "ok" if row["label_ok"] and row["type_ok"] is not False else "CHECK"
            else:
                q = urllib.parse.urlencode({"action": "wbsearchentities", "search": e["name"].replace("_", " "),
                                            "language": "en", "format": "json", "limit": 5})
                hits = get_json(f"https://www.wikidata.org/w/api.php?{q}").get("search", [])
                exact = [h for h in hits if key(h.get("label", "")) in names]
                row["status"] = "SUGGESTION" if exact else "NOT_FOUND"
                row["suggestions"] = [{"qid": h["id"], "label": h.get("label"),
                                       "description": h.get("description")} for h in exact]
        except Exception as exc:  # network problems are reported per entry
            row["status"] = f"ERROR: {exc}"
        report.append(row)
        time.sleep(pause)
    return report


def corporate_bodies(rdf_path: str) -> dict[str, str]:
    """English prefLabel/altLabel key -> concept IRI from the SKOS authority table."""
    from rdflib import Graph
    from rdflib.namespace import SKOS

    g = Graph().parse(rdf_path)
    index = {}
    for prop in (SKOS.prefLabel, SKOS.altLabel):
        for concept, _, label in g.triples((None, prop, None)):
            if getattr(label, "language", None) in ("en", None):
                index.setdefault(key(label), str(concept))
    return index


def add_iris(index: dict[str, str], write: bool) -> list[dict]:
    found = []
    lines = GAZETTEER.read_text(encoding="utf-8").splitlines()
    for e in yaml.safe_load(GAZETTEER.read_text(encoding="utf-8")).get("entities", []):
        if e.get("iri"):
            continue
        for n in [e["name"].replace("_", " "), *(e.get("aliases") or [])]:
            iri = index.get(key(n))
            if iri and "/corporate-body/" in iri:
                found.append({"name": e["name"], "matched_label": n, "iri": iri})
                break
    if write:
        for f in found:
            pattern = re.compile(rf"^(\s*- \{{name: {re.escape(f['name'])},.*)\}}\s*$")
            for i, line in enumerate(lines):
                m = pattern.match(line)
                if m and "iri:" not in line:
                    lines[i] = f'{m.group(1)}, iri: "{f["iri"]}"}}'
        GAZETTEER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true", help="Check Wikidata QIDs (needs internet).")
    p.add_argument("--corporate-body-rdf", help="EU Vocabularies corporate-body table (SKOS file).")
    p.add_argument("--write", action="store_true", help="Write found corporate-body IRIs to the gazetteer.")
    p.add_argument("--pause", type=float, default=0.3, help="Seconds between Wikidata requests.")
    p.add_argument("--out", default="data/08_reporting/identifier_verification.json")
    args = p.parse_args()
    if not (args.verify or args.corporate_body_rdf):
        p.error("nothing to do: pass --verify and/or --corporate-body-rdf")

    result = {}
    if args.verify:
        rows = verify(args.pause)
        result["wikidata"] = rows
        bad = [r for r in rows if r["status"] != "ok"]
        print(f"Wikidata: {len(rows) - len(bad)}/{len(rows)} ok")
        for r in bad:
            print(f"  {r['status']:<11} {r['name']:<40} {r.get('wikidata') or ''} "
                  f"{r.get('label') or ''} {r.get('suggestions') or ''}")
    if args.corporate_body_rdf:
        found = add_iris(corporate_bodies(args.corporate_body_rdf), args.write)
        result["corporate_body"] = found
        print(f"Corporate-body IRIs found: {len(found)}" + (" (written)" if args.write else ""))
        for f in found:
            print(f"  {f['name']:<40} {f['iri']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.verify and any(r["status"].startswith(("CHECK", "ERROR")) for r in result["wikidata"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
