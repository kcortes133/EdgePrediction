"""Look up Ubergraph IC and Monarch KG annotation counts for ontology terms.

Used for the Discussion example (GO:0005654 "nucleoplasm") and to fill the
corpus-size placeholder N in the Methods.

For each term it reports:
  - Ubergraph normalizedInformationContent (the score editICNodes.py filtered on)
  - Ubergraph normalizedSubClassInformationContent (is-a-only variant, for comparison)
  - Ubergraph referenceCount = |D(t)|, the number of classes pointing at t
  - N, the Ubergraph corpus size, derived from nIC = 100 * (1 - ln|D(t)| / ln N)
  - from the local KG edges file: edges with the term as object, by predicate,
    and the number of distinct annotated entities, by ID prefix

Usage (from the repo root):
    python src/analysis/termICLookup.py GO:0005654
    python src/analysis/termICLookup.py GO:0005654 HP:0000118 --skip-kg
    python src/analysis/termICLookup.py GO:0005654 --count-n   # exact N (slow query)

The Ubergraph part needs internet access to https://ubergraph.apps.renci.org/sparql.
The KG part streams the 4 GB edges TSV once (a few minutes).
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

from SPARQLWrapper import SPARQLWrapper, JSON

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KG_DIR = os.path.join(PROJECT_ROOT, "monarch-kg-Sept2025", "monarch-kg-Sept2025")
EDGES_FILE = os.path.join(KG_DIR, "monarch-kg_edges.tsv")
OUTDIR = os.path.join(PROJECT_ROOT, "analysis", "results", "term_ic")

SPARQL_ENDPOINT = "https://ubergraph.apps.renci.org/sparql"
OBO = "http://purl.obolibrary.org/obo/"
VOCAB = "http://reasoner.renci.org/vocab/"

# Gene-to-term annotation predicates (GO cellular component / function / process)
ANNOTATION_PREDICATES = {
    "biolink:located_in", "biolink:is_active_in", "biolink:colocalizes_with",
    "biolink:part_of", "biolink:enables", "biolink:contributes_to",
    "biolink:actively_involved_in", "biolink:acts_upstream_of_or_within",
    "biolink:has_phenotype", "biolink:gene_associated_with_condition", "biolink:causes",
}


def curie_to_iri(curie):
    prefix, local = curie.split(":", 1)
    return f"{OBO}{prefix}_{local}"


def sparql(query):
    s = SPARQLWrapper(SPARQL_ENDPOINT)
    s.setMethod("POST")
    s.setReturnFormat(JSON)
    s.addCustomHttpHeader("User-Agent", "TRIM-termICLookup/1.0")
    s.setQuery(query)
    return s.query().convert()["results"]["bindings"]


def ubergraph_scores(curies):
    values = " ".join(f"<{curie_to_iri(c)}>" for c in curies)
    q = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?t ?label ?nic ?scnic ?rc WHERE {{
      VALUES ?t {{ {values} }}
      OPTIONAL {{ ?t rdfs:label ?label }}
      OPTIONAL {{ ?t <{VOCAB}normalizedInformationContent> ?nic }}
      OPTIONAL {{ ?t <{VOCAB}normalizedSubClassInformationContent> ?scnic }}
      OPTIONAL {{ ?t <{VOCAB}referenceCount> ?rc }}
    }}"""
    out = {}
    for row in sparql(q):
        curie = row["t"]["value"].replace(OBO, "").replace("_", ":", 1)
        get = lambda k, f=float: f(row[k]["value"]) if k in row else None
        rec = {
            "label": row.get("label", {}).get("value"),
            "normalizedInformationContent": get("nic"),
            "normalizedSubClassInformationContent": get("scnic"),
            "referenceCount_D(t)": get("rc", lambda v: int(float(v))),
        }
        nic, rc = rec["normalizedInformationContent"], rec["referenceCount_D(t)"]
        # nIC = 100 * (1 - ln D / ln N)  =>  ln N = ln D / (1 - nIC/100)
        if nic is not None and rc and rc > 1 and nic < 100:
            rec["N_derived"] = round(math.exp(math.log(rc) / (1 - nic / 100)))
        out[curie] = rec
    return out


def ubergraph_count_n():
    q = f"SELECT (COUNT(?t) AS ?n) WHERE {{ ?t <{VOCAB}normalizedInformationContent> ?x }}"
    return int(sparql(q)[0]["n"]["value"])


def kg_annotations(curies):
    """Stream the edges file once; count edges with each term as object."""
    targets = set(curies)
    by_pred = defaultdict(Counter)
    subjects = defaultdict(set)
    annot_subjects = defaultdict(set)
    as_subject = Counter()
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    with open(EDGES_FILE, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_s, i_p, i_o = header.index("subject"), header.index("predicate"), header.index("object")
        for line in f:
            if not any(t in line for t in targets):  # cheap pre-filter
                continue
            cols = line.rstrip("\n").split("\t")
            s, p, o = cols[i_s], cols[i_p], cols[i_o]
            if o in targets:
                by_pred[o][p] += 1
                subjects[o].add(s)
                if p in ANNOTATION_PREDICATES:
                    annot_subjects[o].add(s)
            if s in targets:
                as_subject[s] += 1
    out = {}
    for t in curies:
        out[t] = {
            "edges_as_object": sum(by_pred[t].values()),
            "edges_as_subject": as_subject[t],
            "edges_as_object_by_predicate": dict(by_pred[t].most_common()),
            "distinct_annotated_entities": len(annot_subjects[t]),
            "distinct_annotated_entities_by_prefix": dict(
                Counter(x.split(":", 1)[0] for x in annot_subjects[t]).most_common()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("terms", nargs="+", help="CURIEs, e.g. GO:0005654")
    ap.add_argument("--skip-ubergraph", action="store_true")
    ap.add_argument("--skip-kg", action="store_true")
    ap.add_argument("--count-n", action="store_true", help="also run the (slow) exact count of N")
    args = ap.parse_args()

    result = {t: {} for t in args.terms}
    if not args.skip_ubergraph:
        for t, rec in ubergraph_scores(args.terms).items():
            result[t]["ubergraph"] = rec
        if args.count_n:
            result["_ubergraph_N"] = ubergraph_count_n()
    if not args.skip_kg:
        for t, rec in kg_annotations(args.terms).items():
            result[t]["monarch_kg"] = rec

    os.makedirs(OUTDIR, exist_ok=True)
    out_file = os.path.join(OUTDIR, "term_ic_lookup.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
