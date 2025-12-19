from ensmallen import Graph

def get_n_shortest_paths(graph: Graph, gene_id: str, disease_id: str, n: int = 5):
    """
    Get the n shortest paths between a gene and disease node using GRAPE.

    Parameters
    ----------
    graph : Graph
        The GRAPE graph object (already loaded).
    gene_id : str
        The ID of the gene node.
    disease_id : str
        The ID of the disease node.
    n : int
        Number of shortest paths to retrieve.

    Returns
    -------
    list[list[str]]
        A list of paths, where each path is a list of node IDs.
    """
    # GRAPE provides k-shortest path search
    # This method internally uses Dijkstra/Yen algorithms for weighted/unweighted graphs
    paths = graph.get_k_shortest_path_node_names_from_node_names(
        gene_id, disease_id, n
    )

    # Convert each path to a list of node names
    return paths
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
                disease_to_genes[disease] = []

    return disease_to_genes


# Example usage
if __name__ == "__main__":
    # Example: Load the graph from CSV (Monarch-style KG export)
    graph = Graph.from_csv(
        edge_path="edges.csv",  # columns: source, destination, [weight]
        directed=False
    )

    d2G = load_disease_gene_map('Rare Disease Gene Associations.csv')
    diseases = d2G.keys()
    genes = d2G.values()


    n = 5
    for disease in diseases:
        for gene in genes:
            paths = get_n_shortest_paths(graph, gene, disease, n=n)

        for i, path in enumerate(paths, 1):
            print(f"Path {i}: {' → '.join(path)}")
