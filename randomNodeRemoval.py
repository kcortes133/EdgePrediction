import argparse
import pandas as pd
import random


def load_protected_nodes(tp_file):
    """Load nodes that cannot be removed (HGNC genes in TP file)."""
    tp_df = pd.read_csv(tp_file, sep="\t", dtype=str)
    protected = set(tp_df["subject"].dropna().unique()) | set(tp_df['object'].dropna().unique())
    return protected


def select_nodes_to_remove(nodes_file, n, protected_nodes, seed=None):
    """Randomly select n removable nodes."""
    nodes_df = pd.read_csv(nodes_file, sep="\t", dtype=str)

    all_nodes = set(nodes_df["id"])
    removable_nodes = list(all_nodes - protected_nodes)

    if n > len(removable_nodes):
        raise ValueError(
            f"Requested removal of {n} nodes but only {len(removable_nodes)} are removable."
        )

    if seed is not None:
        random.seed(seed)

    removed_nodes = set(random.sample(removable_nodes, n))
    return nodes_df, removed_nodes


def filter_nodes(nodes_df, removed_nodes, output_file):
    """Write nodes excluding removed ones."""
    filtered = nodes_df[~nodes_df["id"].isin(removed_nodes)]
    filtered.to_csv(output_file, sep="\t", index=False)

def filter_edges(edges_file, removed_nodes, output_file, stats_output_file=None):
    """Remove edges connected to removed nodes and report stats."""
    edges_df = pd.read_csv(edges_file, sep="\t", dtype=str)

    # Identify edges to remove
    to_remove_mask = (
        edges_df["subject"].isin(removed_nodes) |
        edges_df["object"].isin(removed_nodes)
    )

    removed_edges_df = edges_df[to_remove_mask]
    kept_edges_df = edges_df[~to_remove_mask]

    # Total edges removed
    total_removed = len(removed_edges_df)
    print(f"Total edges removed: {total_removed}")

    # Count edges per node (subject + object)
    subject_counts = removed_edges_df["subject"].value_counts()
    object_counts = removed_edges_df["object"].value_counts()

    # Combine counts
    edge_counts_per_node = subject_counts.add(object_counts, fill_value=0).astype(int)

    # Keep only removed nodes (optional but cleaner)
    edge_counts_per_node = edge_counts_per_node[
        edge_counts_per_node.index.isin(removed_nodes)
    ]

    # Convert to DataFrame
    stats_df = edge_counts_per_node.reset_index()
    stats_df.columns = ["node_id", "edges_removed"]

    # Sort for readability
    stats_df = stats_df.sort_values(by="edges_removed", ascending=False)

    # Save stats if requested
    if stats_output_file:
        stats_df.to_csv(stats_output_file, sep="\t", index=False)

    # Save filtered edges
    kept_edges_df.to_csv(output_file, sep="\t", index=False)
    bad_edges = kept_edges_df[
        kept_edges_df["subject"].isin(removed_nodes) |
        kept_edges_df["object"].isin(removed_nodes)
        ]

    print(f"Edges that SHOULD have been removed but weren't: {len(bad_edges)}")

    if len(bad_edges) > 0:
        print(bad_edges.head())
    return total_removed, stats_df

def write_removed_nodes(removed_nodes, output_file):
    """Write removed node IDs to a file."""
    df = pd.DataFrame({"id": list(sorted(removed_nodes))})
    df.to_csv(output_file, sep="\t", index=False)


def main():
    removed_nodes_file = 'monarch-kg-Sept2025/monarch-kg-Sept2025/removed_nodes_Rand7_80.tsv'
    edge_stats_file = 'monarch-kg-Sept2025/monarch-kg-Sept2025/removed_edge_stats7.tsv'
    tp_edges = 'TP_hgnc_mondo_edges.tsv'
    n = 6448
    nodes = 'monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_nodes.tsv'
    out_nodes = 'monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_nodes7_Rand_80.tsv'

    edges = 'monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_edges.tsv'
    out_edges = 'monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_edges7_Rand_80.tsv'
    #seed = 42
    seed = 7
    print("Loading protected nodes...")
    protected_nodes = load_protected_nodes(tp_edges)

    print("Selecting nodes to remove...")
    nodes_df, removed_nodes = select_nodes_to_remove(
        nodes,
        n,
        protected_nodes,
        seed
    )

    print(f"Removing {len(removed_nodes)} nodes")
    print("Writing removed nodes...")
    write_removed_nodes(removed_nodes, removed_nodes_file)

    print("Filtering nodes file...")
    filter_nodes(nodes_df, removed_nodes, out_nodes)

    print("Filtering edges file...")
    print("Filtering edges file...")
    total_removed, stats_df = filter_edges(
        edges,
        removed_nodes,
        out_edges,
        edge_stats_file
    )
    print(total_removed)
    print(stats_df)
    print("Done.")


if __name__ == "__main__":
    main()