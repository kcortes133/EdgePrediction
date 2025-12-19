"""
Compute gene–disease phenotypic similarity using Ubergraph precomputed IC values.

Optimizations:
 - Uses ubergraph:subClassOfClosure for fast ancestor retrieval
 - Batches SPARQL queries
 - Caches results locally to avoid repeated queries
"""

from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import json
import os

UBERGRAPH = "https://ubergraph.apps.renci.org/sparql"
CACHE_FILE = "phenotype_cache.json"


# -------------------------------
# SPARQL Utilities
# -------------------------------
def run_sparql(query):
    """Run a SPARQL query and return JSON results."""
    sparql = SPARQLWrapper(UBERGRAPH)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    return sparql.query().convert()["results"]["bindings"]


# -------------------------------
# Cache Utilities
# -------------------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {"ancestors": {}, "ic": {}}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# -------------------------------
# Ancestor Retrieval (batched)
# -------------------------------
def get_ancestors_batch(phenotype_ids, cache):
    """Retrieve ancestors for a batch of phenotypes using closure table."""
    missing = [pid for pid in phenotype_ids if pid not in cache["ancestors"]]
    if not missing:
        return cache["ancestors"]

    values = " ".join(f"<http://purl.obolibrary.org/obo/{pid.replace(':', '_')}>" for pid in missing)
    query = f"""
    SELECT DISTINCT ?phenotype ?ancestor WHERE {{
      VALUES ?phenotype {{ {values} }}
      ?phenotype <http://purl.obolibrary.org/obo/ubergraph#subClassOfClosure> ?ancestor .
      FILTER(STRSTARTS(STR(?ancestor), "http://purl.obolibrary.org/obo/HP_") ||
             STRSTARTS(STR(?ancestor), "http://purl.obolibrary.org/obo/MP_"))
    }}
    """
    results = run_sparql(query)
    for r in results:
        ph = r["phenotype"]["value"].split("/")[-1].replace("_", ":")
        anc = r["ancestor"]["value"].split("/")[-1].replace("_", ":")
        cache["ancestors"].setdefault(ph, []).append(anc)

    # ensure each term includes itself
    for pid in missing:
        cache["ancestors"].setdefault(pid, []).append(pid)

    save_cache(cache)
    return cache["ancestors"]


# -------------------------------
# IC Retrieval (batched)
# -------------------------------
def get_precomputed_ic_batch(phenotype_ids, cache):
    """Retrieve normalized IC scores for all phenotypes."""
    missing = [pid for pid in phenotype_ids if pid not in cache["ic"]]
    if not missing:
        return cache["ic"]

    values = " ".join(f"<http://purl.obolibrary.org/obo/{pid.replace(':', '_')}>" for pid in missing)
    query = f"""
    SELECT ?phenotype ?ic WHERE {{
      VALUES ?phenotype {{ {values} }}
      ?phenotype <http://reasoner.renci.org/vocab/normalizedInformationContent> ?ic .
    }}
    """
    results = run_sparql(query)
    for r in results:
        pid = r["phenotype"]["value"].split("/")[-1].replace("_", ":")
        cache["ic"][pid] = float(r["ic"]["value"])

    # default 0 if not found
    for pid in missing:
        cache["ic"].setdefault(pid, 0.0)

    save_cache(cache)
    return cache["ic"]


# -------------------------------
# Resnik Similarity
# -------------------------------
def resnik_similarity(gene_phens, disease_phens, ic, ancestors):
    """Resnik similarity = max IC among shared ancestors."""
    best_ic = 0.0
    for g in gene_phens:
        for d in disease_phens:
            common = set(ancestors.get(g, [])) & set(ancestors.get(d, []))
            if not common:
                continue
            ic_common = max(ic.get(t, 0.0) for t in common)
            if ic_common > best_ic:
                best_ic = ic_common
    return best_ic


# -------------------------------
# Main Similarity Pipeline
# -------------------------------
def compute_gene_disease_similarity(genes_dict, diseases_dict, threshold=50.0):
    """
    Compute Resnik phenotypic similarity using Ubergraph’s precomputed IC.
    Returns gene–disease pairs with similarity >= threshold.
    """
    cache = load_cache()

    # Collect all phenotype IDs
    all_phens = set(p for phens in genes_dict.values() for p in phens) | \
                set(p for phens in diseases_dict.values() for p in phens)

    print(f"Fetching ancestors for {len(all_phens)} phenotypes (cached where possible)...")
    ancestors = get_ancestors_batch(all_phens, cache)

    print("Retrieving precomputed IC scores (cached where possible)...")
    ic = get_precomputed_ic_batch(all_phens, cache)

    results = []
    print("Computing Resnik similarities...")
    for gene, g_phens in genes_dict.items():
        for disease, d_phens in diseases_dict.items():
            sim = resnik_similarity(g_phens, d_phens, ic, ancestors)
            if sim >= threshold:
                results.append((gene, disease, sim))

    return sorted(results, key=lambda x: x[2], reverse=True)


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    genes = {
        "HGNC:14313": ["HP:0001250", "HP:0001263"],   # seizures, developmental delay
        "HGNC:1100": ["HP:0001249"]                   # intellectual disability
    }

    diseases = {
        "MONDO:0005148": ["HP:0001250", "HP:0001249"],  # epileptic encephalopathy
        "MONDO:0005012": ["HP:0000707"]                 # autism spectrum disorder
    }

    results = compute_gene_disease_similarity(genes, diseases, threshold=45.0)

    print("\nGene–Disease pairs with similarity >= 45:")
    for g, d, s in results:
        print(f"{g} ↔ {d} : {s:.2f}")
