#!/usr/bin/env python3
"""
Compute scores for shortest paths in a GRAPE graph:
- Paths are obtained using GRAPE's native k-shortest path search.
- Score = 100 for each gene node + IC(ontology node)
- For each gene:
      compute score_sum and score_avg = score_sum / path_length
- Only paths with length > 2 are considered.
- Return top K genes by their best scoring path.
"""

from typing import Dict, List, Any, Iterable, Tuple
import requests
from ensmallen import Graph
import math

#############################
# SPARQL IC Fetch Utilities
#############################
# TODO: download all ic values into dictionary for quick recall
# pilot study; FA, castleman,
# how to leverage information content
# pick one disease
# scope waaay back

SPARQL_TEMPLATE = """
SELECT ?phenotype ?ic WHERE {
  VALUES ?phenotype { %VALUES% }
  ?phenotype <http://reasoner.renci.org/vocab/normalizedInformationContent> ?ic .
}
"""

def fetch_ic_values(uris: Iterable[str], endpoint: str, batch_size: int = 200) -> Dict[str, float]:
    """Fetch normalizedInformationContent values for a set of phenotype URIs."""
    headers = {"Accept": "application/sparql-results+json"}
    uris = list(uris)
    ic_map = {}

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    for batch in chunks(uris, batch_size):
        formatted = " ".join(f"<{u}>" for u in batch)
        query = SPARQL_TEMPLATE.replace("%VALUES%", formatted)

        r = requests.post(endpoint, data={"query": query}, headers=headers)
        r.raise_for_status()
        rows = r.json().get("results", {}).get("bindings", [])

        for row in rows:
            uri = row["phenotype"]["value"]
            ic_val = float(row["ic"]["value"])
            ic_map[uri] = ic_val

    return ic_map


#############################
# Node Type Identification
#############################

def is_gene(node_name: str, node_type: str) -> bool:
    """Determine if a node is a gene based on GRAPE node type annotation."""
    if node_type is None:
        return False

    nt = node_type.lower()
    return "gene" in nt or "biolink:gene" in nt.lower()


#############################
# Path Scoring
#############################

def score_path(
    path: List[str],
    graph: Graph,
    ic_map: Dict[str, float]
) -> Tuple[float, float, Dict[str, float]]:
    """
    Score a path:
      - Gene node → +100
      - Phenotype node → +IC value
    Returns:
      score_sum, score_avg, breakdown
    """
    breakdown = {}
    score_sum = 0.0

    for node in path:
        node_type = graph.get_node_type_from_node_name(node)

        if is_gene(node, node_type):
            breakdown[node] = 100.0
            score_sum += 100.0
        else:
            # Ontology/phenotype node: check IC map
            ic_val = ic_map.get(node, 0.0)
            breakdown[node] = ic_val
            score_sum += ic_val

    score_avg = score_sum / len(path)
    return score_sum, score_avg, breakdown


#############################
# Main Function
#############################

def top_k_gene_disease_pairs_grape(
    graph: Graph,
    disease_id: str,
    gene_list: Iterable[str],
    sparql_endpoint: str,
    n_paths_per_gene: int = 3,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    For each gene:
       - Get k-shortest paths using GRAPE (length > 2 only)
       - Compute IC-based scores
       - Keep best path (by score_avg)
    Return top K by score_avg.
    """

    # 1) Gather phenotype node names for IC lookup
    all_non_gene_nodes = set()

    # Temporary storage of paths before scoring
    candidate_paths = {}

    for gene in gene_list:
        # GRAPE returns list of list-of-nodes
        paths = graph.get_k_shortest_path_node_names_from_node_names(disease_id, gene, n_paths_per_gene)
        # Filter paths of length <= 2
        valid = [p for p in paths if len(p) > 2]
        if not valid:
            continue

        candidate_paths[gene] = valid

        for path in valid:
            for node in path:
                geneType = ['HGNC', 'MGI', 'ZFIN', 'RGD', 'Xenbase']
                node_type = node.split(":")[0]
                if node_type not in geneType:
                    all_non_gene_nodes.add(node)

    # 2) Retrieve IC values
    ic_map = {}
    if all_non_gene_nodes:
        ic_map = fetch_ic_values(all_non_gene_nodes, sparql_endpoint)

    # 3) Score paths & pick best per gene
    results = []

    for gene, paths in candidate_paths.items():
        best_avg = -math.inf
        best_sum = None
        best_path = None
        best_breakdown = None

        for path in paths:
            score_sum, score_avg, breakdown = score_path(path, graph, ic_map)

            if score_avg > best_avg:
                best_avg = score_avg
                best_sum = score_sum
                best_path = path
                best_breakdown = breakdown

        results.append({
            "gene": gene,
            "score_avg": best_avg,
            "score_sum": best_sum,
            "best_path": best_path,
            "path_breakdown": best_breakdown
        })
        print(gene, score_avg, score_sum, best_path, best_breakdown)


    # 4) Sort by best score_avg
    results = sorted(results, key=lambda x: x["score_avg"], reverse=True)

    return results[:top_k]

def load_disease_gene_map(filepath: str, deduplicate: bool = False):
    """
    Load a CSV file with columns:
        Rare Disease,Gene
        MONDO:0007691,HGNC:9118
        ...

    Returns:
        dict mapping disease_id -> list of gene_ids

    Parameters:
        filepath: path to CSV file
        deduplicate: if True, remove duplicate genes for a disease
    """
    disease_to_genes = {}

    with open(filepath, "r") as f:
        # Skip header
        header = next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue

            disease, gene = line.split(",")

            if disease not in disease_to_genes:
                disease_to_genes[disease] = gene

    return disease_to_genes


###########################################
# Example Usage (fill in your file paths)
###########################################
if __name__ == "__main__":

    sparql_endpoint = "https://ubergraph.renci.org/sparql"
    kgFolder = 'monarch-kg-Sept2025/'
    edgesF = kgFolder + 'monarch-kg_edges.tsv'
    nodesF = kgFolder + 'monarch-kg_nodes.tsv'

    # Example graph load (your version from question):
    gTest = Graph.from_csv(
        directed=False,
        node_path=nodesF,
        edge_path= edgesF,
        node_list_separator='\t',
        edge_list_separator='\t',
        verbose=True,
        nodes_column='id',
        node_list_node_types_column='category',
        default_node_type='biolink:NamedThing',
        sources_column='subject',
        destinations_column='object',
        edge_list_edge_types_column='predicate',
        name='Monarch KG Test'
    )

    d2G = load_disease_gene_map('Rare Disease Gene Associations.csv', deduplicate=True)
    for disease_id in d2G.keys():
        #
        results = top_k_gene_disease_pairs_grape(
            graph=gTest,
            disease_id=disease_id,
            gene_list=d2G.values(),
            sparql_endpoint=sparql_endpoint,
            n_paths_per_gene=3,
            top_k=10
        )
        #
        print('-----' + disease_id + '-----'  + '\n')
        print(d2G[disease_id])
        print(results)
