import csv
import json
import time
import logging
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
import matplotlib.pyplot as plt

# ===============================================================
# CONFIG
# ===============================================================
IC_THRESHOLD = 80        # Remove terms with IC < threshold
BATCH_SIZE = 200          # Batch size for SPARQL VALUES
DELAY = 0.2               # Delay between SPARQL calls

KG_DIR = Path("monarch-kg-Sept2025")
NODES_FILE = KG_DIR / "monarch-kg_nodes.tsv"
EDGES_FILE = KG_DIR / "monarch-kg_edges.tsv"

OUTPUT_NODES = KG_DIR / "monarch-kg_nodes.filtered_80.tsv"
OUTPUT_EDGES = KG_DIR / "monarch-kg_edges.filtered_80.tsv"

LOG_FILE = "kg_ic_filter.log"
SUMMARY_FILE = "kg_ic_filter_summary.txt"

ONTOLOGIES = {
    "GO": "http://purl.obolibrary.org/obo/GO_",
    "HP": "http://purl.obolibrary.org/obo/HP_",
    "MONDO": "http://purl.obolibrary.org/obo/MONDO_",
}

SPARQL_ENDPOINT = "https://ubergraph.apps.renci.org/sparql"

SPARQL_TEMPLATE = """
SELECT ?phenotype ?ic WHERE {
  VALUES ?phenotype { %VALUES% }
  ?phenotype <http://reasoner.renci.org/vocab/normalizedInformationContent> ?ic .
}
"""

# ===============================================================
# LOGGING
# ===============================================================

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===============================================================
# SPARQL IC QUERYING
# ===============================================================

def query_ic_batch(iris):
    """
    Query IC values for a batch of IRIs using Ubergraph.
    """
    values_clause = " ".join(f"<{iri}>" for iri in iris)
    query = SPARQL_TEMPLATE.replace("%VALUES%", values_clause)

    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setMethod("POST")
    sparql.setReturnFormat(JSON)
    sparql.setRequestMethod("POST")
    sparql.addCustomHttpHeader("User-Agent", "ICBatchQuery/1.0")
    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
    except Exception as e:
        logging.error(f"SPARQL ERROR for batch: {e}")
        return {}
    ic_map = {}
    for row in results["results"]["bindings"]:
        iri = row["phenotype"]["value"].replace("http://purl.obolibrary.org/obo/", "").replace('_', ":")
        ic_val = float(row["ic"]["value"])
        ic_map[iri] = ic_val

    return ic_map


def query_ic_for_ontology(term_iris):
    """
    Query IC for a list of IRIs, batching the VALUES clause.
    """
    ic_results = {}

    for i in range(0, len(term_iris), BATCH_SIZE):
        batch = term_iris[i:i+BATCH_SIZE]
        ic_batch = query_ic_batch(batch)
        ic_results.update(ic_batch)
        time.sleep(DELAY)

    return ic_results


# ===============================================================
# LOAD NODES + EXTRACT IRIs BY ONTOLOGY
# ===============================================================

def load_nodes():
    """
    Load the entire nodes TSV into memory.
    Returns: dict[id] = row_dict
    """
    nodes = {}
    with open(NODES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            nodes[row["id"]] = row
    return nodes


def group_nodes_by_ontology(nodes):
    """
    Group node IRIs by ontology prefix.
    Returns: dict[ontology] = [iri1, iri2, ...]
    """
    ont_groups = {ont: [] for ont in ONTOLOGIES}
    curieStart = "http://purl.obolibrary.org/obo/"
    for node_id in nodes:
        iri = node_id
        for ont, prefix in ONTOLOGIES.items():
            if iri.startswith(ont):
                ont_groups[ont].append(curieStart+iri.replace(":", "_"))

    return ont_groups


# ===============================================================
# FILTER NODES + EDGES
# ===============================================================

def filter_and_save(nodes, low_ic_set):
    """
    Remove nodes with IC < k and edges attached to them.
    """

    # -----------------------------
    # Filter nodes
    # -----------------------------
    kept_nodes = {nid: row for nid, row in nodes.items() if nid not in low_ic_set}

    # -----------------------------
    # Write filtered nodes
    # -----------------------------
    with open(NODES_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_NODES, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            if row["id"] not in low_ic_set:
                writer.writerow(row)

    # -----------------------------
    # Filter edges
    # -----------------------------
    removed_edges = 0
    with open(EDGES_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_EDGES, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            subj = row["subject"]
            obj = row["object"]
            if subj not in low_ic_set and obj not in low_ic_set:
                writer.writerow(row)
            else:
                removed_edges += 1

    return kept_nodes, removed_edges


def plot_ic_histogram(ic_map, ontology_name="Ontology", bins=50):
    ic_values = list(ic_map.values())

    plt.figure(figsize=(8, 5))
    plt.hist(ic_values, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f"Information Content Distribution for {ontology_name}")
    plt.xlabel("Information Content (IC)")
    plt.ylabel("Number of Terms")
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def plot_multi_ontology_ic_histogram(ic_maps, bins=50):
    """
    Plot multiple ontologies' IC distributions in a single figure.

    Parameters:
        ic_maps: dict
            Keys are ontology names, values are dicts {term: IC}.
        bins: int
            Number of bins for histogram.
    """
    plt.figure(figsize=(10, 6))

    colors = ["skyblue", "salmon", "lightgreen", "orange", "purple"]
    i=0
    for ont, ic_map in ic_maps.items():
        ic_values = list(ic_map.values())
        plt.hist(
            ic_values,
            bins=bins,
            alpha=0.5,
            label=ont,
            color=colors[i % len(colors)],
            edgecolor='black'
        )
        i+=1

    plt.title("Information Content Distribution by Ontology")
    plt.xlabel("Information Content (IC)")
    plt.ylabel("Number of Terms")
    plt.legend()
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.show()




# ===============================================================
# MAIN WORKFLOW
# ===============================================================

def main():

    print("Loading nodes..")
    logging.info("Loading nodes...")
    nodes = load_nodes()

    print("Grouping nodes..")
    logging.info("Grouping nodes by ontology...")
    ont_groups = group_nodes_by_ontology(nodes)

    low_ic_set = set()
    full_ic_map = {}
    ontICMaps = {}

    # -----------------------------------------------------------
    # Process each ontology separately
    # -----------------------------------------------------------
    for ont, iris in ont_groups.items():
        if not iris:
            continue
        print("Processing", ont)
        logging.info(f"Querying IC for ontology {ont} with {len(iris)} terms...")

        ic_map = query_ic_for_ontology(iris)
        full_ic_map.update(ic_map)

        plot_ic_histogram(ic_map, ontology_name=ont, bins=50)
        ontICMaps[ont] = ic_map

        # Save only those below threshold
        ont_low_ic = {iri: ic for iri, ic in ic_map.items() if ic < IC_THRESHOLD}
        low_ic_set.update(ont_low_ic.keys())

        out_file = f"low_ic_{ont}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ont_low_ic, f, indent=2)
        print(f"Wrote {len(low_ic_set)} nodes to {out_file}")
        logging.info(f"{len(ont_low_ic)} {ont} terms below IC {IC_THRESHOLD}. Saved to {out_file}")
    plot_multi_ontology_ic_histogram(ontICMaps, bins=50)
    # -----------------------------------------------------------
    # Filtering KG
    # -----------------------------------------------------------
    logging.info("Filtering nodes/edges...")
    print("Filtering nodes/edges...")

    kept_nodes, removed_edges = filter_and_save(nodes, low_ic_set)

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------
    with open(SUMMARY_FILE, "w") as f:
        f.write("=== KG FILTER SUMMARY ===\n")
        f.write(f"Total nodes: {len(nodes)}\n")
        f.write(f"Removed nodes: {len(low_ic_set)}\n")
        f.write(f"Remaining nodes: {len(kept_nodes)}\n")
        f.write(f"Removed edges: {removed_edges}\n")

    logging.info("Filtering complete.")
    logging.info(f"Removed {len(low_ic_set)} nodes and {removed_edges} edges.")


if __name__ == "__main__":
    main()
