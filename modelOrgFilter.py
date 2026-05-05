#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

kg_folder = 'monarch-kg-Sept2025/'

NODES_FILE = kg_folder + "monarch-kg_nodes.tsv"
EDGES_FILE = kg_folder + "monarch-kg_edges.tsv"

OUT_NODES_FILE = "monarch-kg-Sept2025/monarch-kg_nodes_human_plus_agnostic1.tsv"
OUT_EDGES_FILE = "monarch-kg-Sept2025/monarch-kg_edges_human_plus_agnostic1.tsv"

HUMAN_TAXON = "NCBITaxon:9606"


def keep_node(row):
    """
    Return True if node should be kept.

    Keep if:
      - in_taxon == HUMAN
      - OR in_taxon is empty / missing (species-agnostic)
    """
    in_taxon = row.get("in_taxon", "").strip()

    if not in_taxon:
        # Species-agnostic node (e.g. GO, pathways, ontology terms)
        return True

    return in_taxon == HUMAN_TAXON


def filter_nodes(nodes_in, nodes_out):
    """
    Filter nodes and return set of retained node IDs.
    """
    kept_node_ids = set()
    removed_nonhuman = 0

    with open(nodes_in, newline="", encoding="utf-8") as fin, \
            open(nodes_out, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            if keep_node(row):
                writer.writerow(row)
                kept_node_ids.add(row["id"])
            else:
                removed_nonhuman += 1

    print(f"Removed {removed_nonhuman:,} explicitly non-human nodes")
    return kept_node_ids

def filter_edges(edges_in, edges_out, valid_node_ids, removed_edge_ids):
    """
    Remove edges where:
      - subject or object is not in valid_node_ids
      - OR edge id is listed in removed_edge_ids
    """
    kept = 0
    removed = 0
    removed_by_id = 0
    removed_by_node = 0

    with open(edges_in, newline="", encoding="utf-8") as fin, \
         open(edges_out, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            edge_id = row.get("id")
            subj = row.get("subject")
            obj = row.get("object")

            if subj in removed_edge_ids or obj in removed_edge_ids:
                removed += 1
                removed_by_id += 1
                continue

            if subj not in valid_node_ids or obj not in valid_node_ids:
                removed += 1
                removed_by_node += 1
                continue

            writer.writerow(row)
            kept += 1

    print(f"Removed by explicit edge list: {removed_by_id:,}")
    print(f"Removed by node filtering: {removed_by_node:,}")

    return kept, removed


def load_removed_edge_ids(removed_edges_file):
    """
    Load edge IDs to exclude.
    """
    removed_ids = set()

    with open(removed_edges_file, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            edge_O= row.get("object")
            edge_S = row.get("subject")
            removed_ids.add(edge_S)
            removed_ids.add(edge_O)

    return removed_ids


def main():
    if not Path(NODES_FILE).exists() or not Path(EDGES_FILE).exists():
        sys.exit("ERROR: Input files not found")

    print("Filtering nodes (human + species-agnostic)...")
    kept_nodes = filter_nodes(NODES_FILE, OUT_NODES_FILE)
    print(f"Kept {len(kept_nodes):,} total nodes")
    testEdges = load_removed_edge_ids('TP_hgnc_mondo_edges.tsv')
    print("Filtering edges...")
    kept_edges, removed_edges = filter_edges(
        EDGES_FILE, OUT_EDGES_FILE, kept_nodes, testEdges
    )

    print(f"Kept {kept_edges:,} edges")
    print(f"Removed {removed_edges:,} edges")


if __name__ == "__main__":
    main()
