from ensmallen import Graph
from embiggen.similarities.dag_resnik import DAGResnik
from oaklib import get_adapter
import pandas as pd

from neo4jConnection import Neo4jConnection
from neo4jConfig import configDict
import queries


# establishing connection with neo4j
conn = Neo4jConnection(uri=configDict['uri'],
                       user=configDict['user'],
                       pwd=configDict['pwd'])
DB_NAME = configDict['db']

import matplotlib.pyplot as plt
import pandas as pd

from neo4jConnection import Neo4jConnection
from neo4jConfig import configDict

# -----------------------------
# Neo4j Query Function
# -----------------------------
def get_disease_gene_counts(conn, db):
    """
    Returns list of tuples: (disease_id, gene_count)
    """
    query = """
    MATCH (d:`biolink:Disease`)
    OPTIONAL MATCH (d)--(:`biolink:Gene`)
    RETURN d.id AS disease_id, count(*) AS gene_count
    """

    result = conn.query(query, db=db)
    return [(record["disease_id"], record["gene_count"]) for record in result]


# -----------------------------
# Main Script
# -----------------------------
def main():
    # Connect
    conn = Neo4jConnection(
        uri=configDict['uri'],
        user=configDict['user'],
        pwd=configDict['pwd']
    )
    DB_NAME = configDict['db']

    # Run query
    print("Querying Neo4j for disease→gene counts...")
    data = get_disease_gene_counts(conn, DB_NAME)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["disease_id", "gene_count"])

    selected_rows = df.loc[df["gene_count"] > 100]
    print(selected_rows)

    # -----------------------------
    # Plot distribution of gene counts
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(df["gene_count"], bins=30)
    plt.xlabel("Number of Genes Connected to a Disease")
    plt.ylabel("Number of Diseases")
    plt.title("Distribution of Gene–Disease Connection Counts")
    plt.tight_layout()
    plt.show()

    # Print top of table
    print(df.head())

    # Optionally save results
    df.to_csv("disease_gene_counts.csv", index=False)
    print("Saved: disease_gene_counts.csv")


if __name__ == "__main__":
    main()
