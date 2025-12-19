from oaklib import get_adapter
import rareDiseaseSubsets
from queries import *
from neo4jConnection import Neo4jConnection
from neo4jConfig import configDict
import json, os
from operator import itemgetter

# -------------------------------
# Ancestor-based Jaccard similarity
# -------------------------------
def get_ancestors(terms):
    # -------------------------------
    # Load UPheno ontology remotely
    # -------------------------------
    # This pulls from OBO PURL; no local file needed
    adapter = get_adapter("sqlite:obo:upheno")
    """Return set of ancestors + self for each phenotype term."""
    ancestors = set()
    for t in terms:
        for a in adapter.ancestors(t, reflexive=True):
            ancestors.add(a)
    return ancestors

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)
import math
from collections import Counter

# -----------------------------------------------------------
# Existing code assumed available:
# get_ancestors(), jaccard_similarity()
# -----------------------------------------------------------


# ===========================================================
# Compute Information Content (IC)
# ===========================================================
_ic_cache = {}  # global cache keyed by adapter id


def compute_information_content(adapter):
    """
    Compute IC for all ontology terms using descendant frequency.
    IC(t) = -log(freq(t) / total)
    Works with SqlImplementation (no corpus required).
    """

    # Check if we’ve already computed IC for this ontology
    if id(adapter) in _ic_cache:
        return _ic_cache[id(adapter)]

    print("🔧 Computing Information Content (IC) for ontology...")

    counts = Counter()
    total = 0
    all_terms = list(adapter.entities(filter_obsoletes=True))

    for t in all_terms:
        descs = list(adapter.descendants(t, reflexive=True))
        for d in descs:
            counts[d] += 1
            total += 1

    ic = {t: -math.log(counts[t] / total) for t in counts if counts[t] > 0}
    _ic_cache[id(adapter)] = ic
    print(f"✅ Computed IC for {len(ic)} terms.")
    return ic


# ===========================================================
# Manual Resnik similarity using IC
# ===========================================================
def resnik_similarity_manual(adapter, t1, t2, ic):
    """
    Compute Resnik similarity manually using IC of MICA (most informative common ancestor).
    Works even if adapter is a SqlImplementation.
    """
    common_ancs = set(adapter.ancestors(t1, reflexive=True)) & set(adapter.ancestors(t2, reflexive=True))
    if not common_ancs:
        return 0.0
    return max(ic.get(a, 0.0) for a in common_ancs)


# ===========================================================
# Set-based phenotype similarity (Jaccard or Resnik)
# ===========================================================
def computeSimilarity(ortho_phens, disease_phens, method="jaccard", adapter=None):
    """
    Compute semantic similarity between two phenotype sets.

    Parameters
    ----------
    ortho_phens : list[str]
        Phenotype IDs for the gene.
    disease_phens : list[str]
        Phenotype IDs for the disease.
    method : str
        'jaccard' (default) or 'resnik'.
    adapter : Ontology adapter (required for 'resnik').
    """

    if method == "jaccard":
        ortho_anc = get_ancestors(ortho_phens)
        disease_anc = get_ancestors(disease_phens)
        return jaccard_similarity(ortho_anc, disease_anc)

    elif method == "resnik":
        if adapter is None:
            raise ValueError("Adapter must be provided for Resnik similarity.")

        ic = compute_information_content(adapter)
        # Compute best-match-average (BMA) across phenotype sets
        sims = []
        for p1 in ortho_phens:
            best = max(resnik_similarity_manual(adapter, p1, p2, ic) for p2 in disease_phens)
            sims.append(best)
        for p2 in disease_phens:
            best = max(resnik_similarity_manual(adapter, p2, p1, ic) for p1 in ortho_phens)
            sims.append(best)
        return sum(sims) / len(sims) if sims else 0.0

    else:
        raise ValueError(f"Unsupported similarity method: {method}")
'''

def computeSimilarity(ortho_phens, disease_phens):
    """
    Compute semantic similarity between two phenotype sets
    using ancestor-based Jaccard index.
    """
    ortho_anc = get_ancestors(ortho_phens)
    disease_anc = get_ancestors(disease_phens)
    return jaccard_similarity(ortho_anc, disease_anc)
'''

# Step 2: Get Phenotypes for Ortholog Genes
# -------------------------------
def getPhenotypes(gene_id):
    query = nameGenePhen_query(gene_id)
    response = conn.query(query, db=DB_NAME)
    return [item for sublist in json.loads(json.dumps(response)) for item in sublist]

# -------------------------------
# Step 3: Get Disease Phenotypes
# -------------------------------
def getDiseasePhenotypes():
    query = "MATCH (d:`biolink:Disease`)-[:`biolink:has_phenotype`]-(p:`biolink:PhenotypicFeature`) RETURN d.id, collect(p.id)"
    response = conn.query(query, db=DB_NAME)
    return [(row[0], row[1]) for row in response]

# -------------------------------
# Save Gene → Phenotype mappings
# -------------------------------

def save_gene_phenotypes_to_file(gene_ids, filename="gene_phenotypes.json"):
    """
    Query Neo4j for all phenotypes of specified genes and save to a JSON file.

    Parameters
    ----------
    gene_ids : list[str]
        List of gene identifiers (e.g., ["HGNC:11138", "HGNC:5173"])
    filename : str
        Output filename (default: "gene_phenotypes.json")
    """

    all_gene_phens = {}

    for gid in gene_ids:
        query = f"""
        MATCH (g:`biolink:Gene` {{id: '{gid}'}})-[:`biolink:has_phenotype`]-(p:`biolink:PhenotypicFeature`)
        RETURN collect(DISTINCT p.id)
        """
        response = conn.query(query, db=DB_NAME)
        phens = response[0][0] if response and response[0] else []
        all_gene_phens[gid] = phens

    # Save to file
    with open(filename, "w") as f:
        json.dump(all_gene_phens, f, indent=2)

    print(f"✅ Saved {len(all_gene_phens)} genes and their phenotypes to {filename}")


# -------------------------------
# Load Gene → Phenotype mappings
# -------------------------------

def load_gene_phenotypes_from_file(filename="gene_phenotypes.json"):
    """
    Load gene-phenotype mappings from JSON file.
    Returns a dict: {gene_id: [phenotype_ids]}.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "r") as f:
        return json.load(f)

# -------------------------------
# Save Gene → Phenotype mappings
# -------------------------------

def save_dis_phenotypes_to_file(dis_ids, filename="gene_phenotypes.json"):
    """
    Query Neo4j for all phenotypes of specified genes and save to a JSON file.

    Parameters
    ----------
    dis_ids : list[str]
        List of disease identifiers (e.g., ["MONDO:11138", "MONDO:5173"])
    filename : str
        Output filename (default: "gene_phenotypes.json")
    """

    all_disease_phens = {}

    for did in dis_ids:
        query = f"""
        MATCH (g:`biolink:Disease` {{id: '{did}'}})-[:`biolink:has_phenotype`]-(p:`biolink:PhenotypicFeature`)
        RETURN collect(DISTINCT p.id)
        """
        response = conn.query(query, db=DB_NAME)
        phens = response[0][0] if response and response[0] else []
        all_disease_phens[did] = phens

    # Save to file
    with open(filename, "w") as f:
        json.dump(all_disease_phens, f, indent=2)

    print(f"✅ Saved {len(all_disease_phens)} genes and their phenotypes to {filename}")
    return all_disease_phens


# -------------------------------
# Use precomputed file for similarity
# -------------------------------

def get_top_similar_genes_from_file(disease_id, filename="gene_phenotypes.json", n=100):
    """
    Compute top n genes similar to a given disease using a local file
    of precomputed gene→phenotype mappings.
    """
    adapter = get_adapter("sqlite:obo:upheno")

    # Step 1 — Get disease phenotypes from Neo4j
    disease_query = f"""
    MATCH (d:`biolink:Disease` {{id: '{disease_id}'}})-[:`biolink:has_phenotype`]-(p:`biolink:PhenotypicFeature`)
    RETURN collect(DISTINCT p.id)
    """
    disease_response = conn.query(disease_query, db=DB_NAME)
    if not disease_response or not disease_response[0]:
        print(f"No phenotypes found for {disease_id}")
        return []
    disease_phens = disease_response[0][0]

    # Step 2 — Load gene phenotypes from file
    gene_phens_dict = load_gene_phenotypes_from_file(filename)

    # Step 3 — Compute similarity
    similarities = []
    for gene_id, phens in gene_phens_dict.items():
        sim = computeSimilarity(phens, disease_phens, method='jaccard', adapter=adapter)
        #similarities.append((gene_id, sim))
        if sim > 0.4:
            similarities.append((gene_id, sim))

    # Step 4 — Rank and return top n
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


# -------------------------------
# Connect to Neo4j
# -------------------------------
conn = Neo4jConnection(
    uri=configDict['uri'],
    user=configDict['user'],
    pwd=configDict['pwd']
)

DB_NAME = "monarch-20250217"

#genes = rareDiseaseSubsets.getGenesWODisConns('removeHMI_edges/removedEdges.tsv', 'Genes Without Disease Conn.txt')
#save_gene_phenotypes_to_file(genes)

