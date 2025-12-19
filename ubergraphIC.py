import csv
import requests
from collections import defaultdict

UBERGRAPH = "https://ubergraph.apps.renci.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "KG-IC-Filter-Script/1.0"
}

#########################################################
# 1. SPARQL IC Fetching (batch, cached, POST)
#########################################################

def expand_curie(curie):
    """Expand GO:0000001 → http://purl.obolibrary.org/obo/GO_0000001"""
    if ":" not in curie:
        return None
    prefix, local = curie.split(":")
    return f"http://purl.obolibrary.org/obo/{prefix}_{local}"

def fetch_ic_batch(curie_list):
    """Fetch information content for a batch (50–200 recommended)."""
    iris = [f"<{expand_curie(c)}>" for c in curie_list if expand_curie(c)]
    if not iris:
        return {}

    values_block = " ".join(iris)

    query = f"""
    SELECT ?term ?ic
    WHERE {{
      VALUES ?term {{ {values_block} }}
      ?term <http://purl.obolibrary.org/obo/OMO_0005000> ?ic .
    }}
    """

    response = requests.post(
        UBERGRAPH,
        data={"query": query},
        headers=HEADERS
    )

    if response.status_code != 200:
        raise RuntimeError(f"SPARQL error {response.status_code}: {response.text}")

    results = response.json()["results"]["bindings"]

    ic_map = {}
    for r in results:
        iri = r["term"]["value"]
        curie = iri.replace("http://purl.obolibrary.org/obo/", "").replace("_", ":")
        ic_map[curie] = float(r["ic"]["value"])

    return ic_map


#########################################################
# 2. Master IC Cache
#########################################################

def compute_ic_for_terms(all_terms, batch_size=100):
    """Compute IC for all ontology terms while caching results."""
    ic_cache = {}
    terms = list(all_terms)

    for i in range(0, len(terms), batch_size):
        batch = terms[i:i+batch_size]
        print(f"Querying batch {i//batch_size+1}: {len(batch)} terms")

        try:
            batch_ic = fetch_ic_batch(batch)
        except Exception as e:
            print(f"SPARQL batch error: {e}")
            continue

        ic_cache.update(batch_ic)

    return ic_cache


#########################################################
# 3. Load KG nodes and edges
#########################################################

def load_nodes(nodes_file):
    nodes = {}
    with open(nodes_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            nodes[row["id"]] = row
    return nodes

def load_edges(edges_file):
    edges = []
    with open(edges_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            edges.append(row)
    return edges


#########################################################
# 4. Filter nodes based on IC threshold
#########################################################

def filter_nodes_by_ic(nodes, ic_map, threshold):
    removed = set()
    for node_id in nodes:
        if node_id in ic_map and ic_map[node_id] < threshold:
            removed.add(node_id)
    return removed


#########################################################
# 5. Remove nodes + edges and write filtered output
#########################################################

def filter_kg(nodes, edges, removed_nodes, out_nodes, out_edges):
    # Filter nodes
    with open(out_nodes, "w", encoding="utf-8", newline="") as fn:
        writer = None
        for nid, row in nodes.items():
            if nid in removed_nodes:
                continue
            if writer is None:
                writer = csv.DictWriter(fn, fieldnames=row.keys(), delimiter="\t")
                writer.writeheader()
            writer.writerow(row)

    # Filter edges
    with open(out_edges, "w", encoding="utf-8", newline="") as fe:
        writer = None
        for row in edges:
            src = row["subject"]
            tgt = row["object"]
            if src in removed_nodes or tgt in removed_nodes:
                continue
            if writer is None:
                writer = csv.DictWriter(fe, fieldnames=row.keys(), delimiter="\t")
                writer.writeheader()
            writer.writerow(row)


#########################################################
# 6. Main pipeline
#########################################################

def run_filter_kg(
    kg_folder,
    threshold=1.0,
    batch_size=100
):
    nodesF = f"{kg_folder}/monarch-kg_nodes.tsv"
    edgesF = f"{kg_folder}/monarch-kg_edges.tsv"

    print("Loading nodes/edges…")
    nodes = load_nodes(nodesF)
    edges = load_edges(edgesF)

    print(f"Loaded {len(nodes):,} nodes and {len(edges):,} edges")

    # Collect ontology nodes (GO, MONDO, HP, etc.)
    ontology_nodes = [
        n for n in nodes
        if n.startswith(("GO:", "HP:", "MONDO:", "UBERON:", "CHEBI:"))
    ]

    print(f"Ontology terms to query IC for: {len(ontology_nodes):,}")

    # Compute IC with caching
    ic_map = compute_ic_for_terms(ontology_nodes, batch_size=batch_size)

    print(f"IC retrieved for {len(ic_map):,} ontology nodes")

    # Filter nodes by IC
    removed = filter_nodes_by_ic(nodes, ic_map, threshold)

    print(f"Nodes removed: {len(removed):,}")

    # Output files
    out_nodes = f"{kg_folder}/monarch-kg_nodes.filtered.tsv"
    out_edges = f"{kg_folder}/monarch-kg_edges.filtered.tsv"

    print("Filtering KG…")
    filter_kg(nodes, edges, removed, out_nodes, out_edges)

    # Summary
    print("\n===== SUMMARY =====")
    print(f"IC threshold: {threshold}")
    print(f"Nodes removed: {len(removed):,}")
    print(f"Nodes kept: {len(nodes) - len(removed):,}")
    print(f"Edges kept: see filtered file")
    print("===================")

    return removed


#########################################################
# Example execution
#########################################################

if __name__ == "__main__":
    run_filter_kg(
        kg_folder="monarch-kg-Sept2025",
        threshold=1.0,
        batch_size=100
    )
