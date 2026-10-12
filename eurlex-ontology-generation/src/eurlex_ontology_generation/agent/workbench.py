"""Curation workbench: the operations an agent (or a person) uses to resolve the
hard cases the deterministic pipeline flags.

This module has no MCP dependency, so it can be unit-tested and reused from
scripts. `agent/mcp_server.py` exposes it as MCP tools.

Ground rules (enforced here, not left to the agent):
  * the agent never edits the ontology; it appends *proposals* to the
    curation log (data/05_model_input/curation/decisions.jsonl);
  * every proposal is validated before it is stored: the ontology is
    re-refined with the proposal applied, exported, and checked with SHACL
    (and optionally HermiT); a proposal that adds violations or makes the
    ontology inconsistent is stored as `rejected_by_validation`;
  * only a person accepts proposals (scripts/curation.py); the refinement
    pipeline applies accepted decisions only.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from ..pipelines.ontology_refinement.nodes import load_curation, refine_ontology
from ..utils.entity_linking import load_gazetteer
from ..utils.vocabulary import load_vocabulary

DecisionType = Literal["set_typing", "merge_entities", "remove_rule", "promote_class", "demote_class"]


class Decision(BaseModel):
    """A proposed curation decision. Fields used depend on `type`."""
    type: DecisionType
    entity: str | None = Field(None, description="Entity name (set_typing, promote_class, demote_class).")
    is_a: str | None = Field(None, description="Upper-ontology class (set_typing).")
    type_value: Literal["Party", "Asset", "Legal_extension"] | None = Field(
        None, description="Core type (set_typing), only if no is_a is given.")
    kind: Literal["class", "individual"] | None = Field(None, description="(set_typing)")
    entities: list[str] | None = Field(None, description="Names to merge (merge_entities).")
    into: str | None = Field(None, description="Target name (merge_entities).")
    rule_id: str | None = Field(None, description="Rule id (remove_rule).")
    rationale: str = Field(..., min_length=10, description="Why: cite the text or source.")
    evidence: list[str] = Field(default_factory=list, description="Quotes, CELEX numbers, URLs.")
    item_id: str | None = Field(None, description="Review-queue item this resolves.")

    @model_validator(mode="after")
    def _fields_for_type(self):
        need = {"set_typing": ["entity"], "promote_class": ["entity"], "demote_class": ["entity"],
                "merge_entities": ["entities", "into"], "remove_rule": ["rule_id"]}[self.type]
        missing = [f for f in need if not getattr(self, f)]
        if missing:
            raise ValueError(f"{self.type} needs {missing}")
        if self.type == "set_typing" and not (self.is_a or self.type_value or self.kind):
            raise ValueError("set_typing needs at least one of is_a, type_value, kind")
        return self


def load_parameters(root: Path, env: str = "local") -> dict:
    """Parameters as Kedro would load them: conf/base/parameters*.yml, overridden
    by conf/<env>/parameters*.yml, with ${globals:...} resolved from globals.yml."""
    conf = root / "conf"
    globals_: dict = {}
    for d in (conf / "base", conf / env):
        g = d / "globals.yml"
        if g.exists():
            globals_.update(yaml.safe_load(g.read_text(encoding="utf-8")) or {})
    params: dict = {}
    for d in (conf / "base", conf / env):
        for f in sorted(d.glob("parameters*.yml")):
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            for k, v in data.items():
                params[k] = {**params.get(k, {}), **v} if isinstance(v, dict) and isinstance(params.get(k), dict) else v

    def resolve(node):
        if isinstance(node, dict):
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        if isinstance(node, str):
            m = re.fullmatch(r"\$\{globals:([\w.]+)\}", node.strip())
            if m:
                cur: Any = globals_
                for part in m.group(1).split("."):
                    cur = cur[part]
                return resolve(cur)
        return node

    return resolve(params)


class Workbench:
    def __init__(self, root: str | Path = ".", params: dict | None = None):
        self.root = Path(root).resolve()
        self.params = params if params is not None else load_parameters(self.root)
        p = self.params
        self.merged_path = self.root / p["ontology_refinement"]["input_path"]
        self.curation_path = self.root / p["ontology_refinement"]["curation_path"]
        self.min_docs = int(p["ontology_refinement"].get("min_documents_for_class", 3))
        self.refined_path = self.root / "data/07_model_output/eurlex_odrl_ontology_refined.json"
        self.report_path = self.root / "data/08_reporting/ontology_refinement_report.json"
        self.base_iri = p.get("ontology_export", {}).get("base_iri", "https://w3id.org/eurlex-odrl/")
        self.vocab = load_vocabulary()
        self.gazetteer = load_gazetteer()
        self._baseline: dict | None = None

    # ------------------------------------------------------------ state
    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _current(self) -> tuple[dict, dict]:
        """Refined ontology and report for the current accepted decisions."""
        if self.refined_path.exists() and self.report_path.exists():
            return self._read_json(self.refined_path), self._read_json(self.report_path)
        return refine_ontology(self._read_json(self.merged_path), curation=load_curation(str(self.curation_path)),
                               min_documents_for_class=self.min_docs)

    def decisions(self, status: str | None = None) -> list[dict]:
        if not self.curation_path.exists():
            return []
        rows = [json.loads(line) for line in self.curation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if status is None or r.get("status") == status]

    # ------------------------------------------------------------ texts
    @cached_property
    def _texts(self) -> pd.DataFrame:
        """Source chunks known locally: revision/review files carry the chunk text."""
        frames = []
        for d in ("data/04_feature/ontology_revisions", "data/04_feature/ontology_reviews"):
            for f in sorted((self.root / d).glob("part_*.csv")):
                frames.append(pd.read_csv(f, usecols=lambda c: c in ("chunk_id", "CELEX", "text"), encoding="utf-8"))
        parts = self.root / "data/03_primary/eurlex_chunks_parts"
        for f in sorted(parts.glob("part_*.csv")):
            df = pd.read_csv(f, encoding="utf-8")
            if "chunk_text" in df.columns:
                frames.append(df.rename(columns={"chunk_text": "text"})[["chunk_id", "CELEX", "text"]])
        if not frames:
            return pd.DataFrame(columns=["chunk_id", "CELEX", "text"])
        return pd.concat(frames, ignore_index=True).dropna(subset=["text"]).drop_duplicates("chunk_id")

    def get_chunk(self, chunk_id: str) -> dict:
        hit = self._texts[self._texts["chunk_id"] == chunk_id]
        if hit.empty:
            return {"error": f"chunk {chunk_id} not found locally"}
        r = hit.iloc[0]
        return {"chunk_id": r["chunk_id"], "CELEX": r["CELEX"], "text": r["text"]}

    def search_text(self, query: str, celex: str | None = None, limit: int = 5) -> list[dict]:
        df = self._texts
        if celex:
            df = df[df["CELEX"] == celex]
        words = [w for w in re.split(r"[\W_]+", query.lower()) if len(w) > 2]
        if not words:
            return []
        scored = []
        for _, r in df.iterrows():
            t = str(r["text"]).lower()
            score = sum(t.count(w) for w in words)
            if score:
                i = min((t.find(w) for w in words if w in t), default=0)
                scored.append((score, {"chunk_id": r["chunk_id"], "CELEX": r["CELEX"],
                                       "snippet": str(r["text"])[max(0, i - 200): i + 400]}))
        return [s for _, s in sorted(scored, key=lambda x: -x[0])[:limit]]

    # ------------------------------------------------------------ lookups
    def review_queue(self, kind: str | None = None, limit: int = 20, offset: int = 0,
                     include_decided: bool = False) -> dict:
        _, report = self._current()
        decided = {d.get("item_id") for d in self.decisions() if d.get("status") != "rejected_by_validation"}
        items = [i for i in report.get("review_queue", [])
                 if (kind is None or i["kind"] == kind) and (include_decided or i["id"] not in decided)]
        return {"total": len(items), "items": items[offset: offset + limit],
                "kinds": sorted({i["kind"] for i in report.get("review_queue", [])})}

    def get_entity(self, name: str) -> dict:
        refined, _ = self._current()
        ents = {e["name"]: e for e in refined.get("classes", []) + refined.get("individuals", [])}
        e = ents.get(name)
        if e is None:
            g = self.gazetteer.lookup(name)
            return {"error": f"entity {name} not in the ontology",
                    "gazetteer": g.as_dict() if g else None,
                    "similar": sorted(n for n in ents if name.lower()[:5] in n.lower())[:10]}
        rules = [{"id": r.get("id"), "kind": k, "action": r["action"], "action_label": r.get("action_label"),
                  "role": "party" if r.get("party") == name else "asset", "sources": r.get("sources", [])[:5]}
                 for k, rs in refined["rights_model"].items() for r in rs if name in (r.get("party"), r.get("asset"))]
        rels = [r for r in refined.get("relationships", []) if name in (r["source"], r["target"])]
        examples = [x for x in refined.get("example_instances", []) if x.get("entity") == name][:3]
        g = self.gazetteer.lookup(name)
        return {"entity": e, "kind": "individual" if "instance_of" in e else "class", "rules": rules[:30],
                "relationships": rels[:30], "examples": examples, "gazetteer": g.as_dict() if g else None}

    def get_rule(self, rule_id: str) -> dict:
        refined, _ = self._current()
        for kind, rs in refined["rights_model"].items():
            for r in rs:
                if r.get("id") == rule_id:
                    query = " ".join(filter(None, [r.get("action_label"), r.get("party"), r.get("asset")]))
                    snippets = []
                    for celex in r.get("sources", [])[:3]:
                        snippets += self.search_text(query, celex=celex, limit=1)
                    return {"kind": kind, "rule": r, "source_snippets": snippets}
        return {"error": f"rule {rule_id} not found"}

    def gazetteer_lookup(self, name: str) -> dict:
        g = self.gazetteer.lookup(name)
        return g.as_dict() if g else {"found": False}

    def act_metadata(self, celex: str, online: bool = False) -> dict:
        from ..utils.rdf_export import parse_celex

        out = {"celex": celex, **parse_celex(celex),
               "eurlex_url": f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"}
        corpus = self.root / self.params.get("ontology_export", {}).get(
            "act_metadata_path", "data/02_intermediate/eurlex_corpus.csv")
        if corpus.exists():
            df = pd.read_csv(corpus, usecols=lambda c: c in ("CELEX", "Act_name", "Act_type", "Subject_matter",
                                                            "Date_document"), encoding="utf-8")
            hit = df[df["CELEX"] == celex]
            if not hit.empty:
                out["corpus"] = {k: (None if pd.isna(v) else v) for k, v in hit.iloc[0].to_dict().items()}
        if online:
            out["cellar"] = cellar_metadata(celex)
        return out

    def wikidata_lookup(self, label: str) -> dict:
        from ..utils.entity_linking import WikidataLinker

        linker = WikidataLinker(str(self.root / "data/02_intermediate/wikidata_cache.json"),
                                "EurLexOntologyGeneration/0.1 (curation workbench)", max_lookups=20)
        kind, info = linker.kind_of(label)
        linker.save()
        return {"kind": kind, "item": info} if info else {"found": False}

    # ------------------------------------------------------------ extraction / checks
    def extract_chunk(self, text: str) -> dict:
        """Run the generation step on one text with the configured backend, then
        the deterministic normalisation and checks."""
        from ..pipelines.ontology_generation.nodes import extract_json
        from ..pipelines.ontology_review.ontology_checks import run_deterministic_checks
        from ..utils.llm_client import llm_chat
        from ..utils.ontology_normalize import normalize_chunk_ontology
        from ..utils.ontology_schema import build_ontology_model
        from ..utils.vocabulary import render_prompt

        p = self.params["ontology_generation"]
        response = llm_chat(p["llm_backend"], p["model"], render_prompt(p["system_prompt"]),
                            render_prompt(p["user_prompt_template"], text=text), p["temperature"],
                            schema_model=build_ontology_model() if p.get("structured_output") else None,
                            json_mode=p.get("json_mode", True))
        raw = extract_json(response.content)
        normalized, changes = normalize_chunk_ontology(raw)
        _, findings = run_deterministic_checks(json.dumps(normalized), text)
        return {"model": p["model"], "backend": p["llm_backend"].get("provider"),
                "ontology": normalized, "normalization_changes": changes, "findings": findings}

    def check_ontology(self, ontology: dict, text: str) -> dict:
        from ..pipelines.ontology_review.ontology_checks import run_deterministic_checks
        from ..utils.ontology_normalize import normalize_chunk_ontology

        _, raw_findings = run_deterministic_checks(json.dumps(ontology), text)
        normalized, changes = normalize_chunk_ontology(ontology)
        _, findings = run_deterministic_checks(json.dumps(normalized), text)
        return {"findings_as_given": raw_findings, "findings_after_normalization": findings,
                "normalization_changes": changes}

    # ------------------------------------------------------------ validation
    def _evaluate(self, curation: list[dict], reasoner: bool) -> dict:
        from ..utils.rdf_export import export_rdf
        from ..utils.rdf_validate import reasoner_check, validate_graph

        refined, report = refine_ontology(self._read_json(self.merged_path), curation=curation,
                                          min_documents_for_class=self.min_docs)
        graph, _ = export_rdf(refined, self.base_iri)
        _, _, shacl = validate_graph(graph, base_iri=self.base_iri)
        out = {"shacl": shacl["results_by_severity"],
               "messages": {f"{r['severity']}: {r['message']}": r["count"] for r in shacl["results"]},
               "counts": report["counts"]}
        if reasoner:
            out["owl"] = reasoner_check(graph)
        return out

    def validate_decision(self, decision: dict, reasoner: bool = False) -> dict:
        d = Decision.model_validate(decision).model_dump(exclude_none=True)
        problems = self._check_references(d)
        if problems:
            return {"ok": False, "problems": problems}
        accepted = self.decisions("accepted")
        if self._baseline is None:
            self._baseline = self._evaluate(accepted, reasoner=False)
        after = self._evaluate(accepted + [{**d, "status": "accepted"}], reasoner)
        before = self._baseline
        delta = {k: after["messages"].get(k, 0) - before["messages"].get(k, 0)
                 for k in set(after["messages"]) | set(before["messages"])}
        delta = {k: v for k, v in delta.items() if v}
        new_violations = after["shacl"].get("Violation", 0) - before["shacl"].get("Violation", 0)
        inconsistent = reasoner and after.get("owl", {}).get("available") and not after["owl"].get("consistent", True)
        return {"ok": new_violations <= 0 and not inconsistent, "new_violations": new_violations,
                "inconsistent": bool(inconsistent), "shacl_delta": delta,
                "counts_before": before["counts"], "counts_after": after["counts"]}

    def _check_references(self, d: dict) -> list[str]:
        refined, _ = self._current()
        names = {e["name"] for e in refined.get("classes", []) + refined.get("individuals", [])}
        rule_ids = {r.get("id") for rs in refined["rights_model"].values() for r in rs}
        problems = []
        for n in ([d.get("entity")] if d.get("entity") else []) + (d.get("entities") or []):
            if n not in names:
                problems.append(f"unknown entity {n}")
        if d.get("rule_id") and d["rule_id"] not in rule_ids:
            problems.append(f"unknown rule {d['rule_id']}")
        if d.get("is_a") and d["is_a"] not in self.vocab.upper:
            problems.append(f"is_a must be one of {self.vocab.upper_names}")
        return problems

    def propose_decision(self, decision: dict, proposer: str, model: str | None = None,
                         reasoner: bool = False) -> dict:
        validation = self.validate_decision(decision, reasoner)
        d = Decision.model_validate(decision).model_dump(exclude_none=True)
        record = {**d, "id": f"dec_{uuid.uuid4().hex[:12]}",
                  "status": "pending" if validation.get("ok") else "rejected_by_validation",
                  "proposer": proposer, "model": model,
                  "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "validation": validation}
        self.curation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.curation_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"id": record["id"], "status": record["status"], "validation": validation}

    def set_status(self, decision_id: str, status: str, reviewer: str) -> dict:
        """Accept or reject a pending decision (people only; not exposed over MCP)."""
        if status not in ("accepted", "rejected"):
            raise ValueError("status must be accepted or rejected")
        rows = self.decisions()
        hit = None
        for r in rows:
            if r["id"] == decision_id:
                if r["status"] == "rejected_by_validation" and status == "accepted":
                    raise ValueError("a decision that failed validation cannot be accepted; propose a new one")
                r.update(status=status, reviewer=reviewer,
                         reviewed=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                hit = r
        if hit is None:
            raise KeyError(decision_id)
        tmp = self.curation_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(self.curation_path)
        self._baseline = None
        return hit


# ------------------------------------------------------------------ CELLAR
CELLAR_SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"


def cellar_metadata(celex: str, timeout: float = 20.0) -> dict:
    """Best-effort metadata from the Publications Office SPARQL endpoint (CDM).
    Needs internet access; returns {"error": ...} when unreachable."""
    import urllib.parse
    import urllib.request

    if not re.fullmatch(r"[0-9CE]\d{4}[A-Z]{1,2}\d+(\(\d+\))?", celex):
        return {"error": "not a CELEX number"}
    query = f"""
    PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
    SELECT ?work ?date ?force ?end ?title WHERE {{
      ?work cdm:resource_legal_id_celex "{celex}"^^<http://www.w3.org/2001/XMLSchema#string> .
      OPTIONAL {{ ?work cdm:work_date_document ?date }}
      OPTIONAL {{ ?work cdm:resource_legal_date_entry-into-force ?force }}
      OPTIONAL {{ ?work cdm:resource_legal_date_end-of-validity ?end }}
      OPTIONAL {{ ?expr cdm:expression_belongs_to_work ?work ;
                        cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> ;
                        cdm:expression_title ?title }}
    }} LIMIT 5"""
    url = f"{CELLAR_SPARQL}?{urllib.parse.urlencode({'query': query, 'format': 'application/sparql-results+json'})}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json",
                                                   "User-Agent": "EurLexOntologyGeneration/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            rows = json.loads(resp.read().decode("utf-8"))["results"]["bindings"]
    except Exception as exc:
        return {"error": f"CELLAR unreachable: {exc}"}
    if not rows:
        return {"error": "no CELLAR work found for this CELEX"}
    r = rows[0]
    return {k: r[k]["value"] for k in ("work", "date", "force", "end", "title") if k in r}
