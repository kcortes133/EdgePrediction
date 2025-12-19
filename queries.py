"""
queries.py

This module contains Cypher queries for exploring the Monarch Initiative
Knowledge Graph (Neo4j). Each query is documented with its purpose and output.
"""

# -------------------------------
# Basic Queries
# -------------------------------

# Get all Human protien codeing Geens
nameHGNCProtienCode_query = """
MATCH (m:`biolink:Gene`)
WHERE m.id STARTS WITH "HGNC" and any (t in m.type where t= 'SO:0001217')
RETURN m.id
"""

nameDisease_query = """
MATCH (d:`biolink:Disease`)--(g:`biolink:Gene`)
RETURN d.id, g.id
"""

# Get all human genes (HGNC identifiers)
nameHGNC_query = """
MATCH (m:`biolink:Gene`)
WHERE m.id STARTS WITH "HGNC"
RETURN m.id
"""
#  Returns all HGNC gene IDs in the graph.


# Get orthologs for all human genes
nameHGNCOrthos_query = """
MATCH (m:`biolink:Gene`)-[:`biolink:orthologous_to`]-(n:`biolink:Gene`)
WHERE m.id STARTS WITH "HGNC"
RETURN n.id, m.id
"""
# For each human gene (HGNC), return its orthologous partner(s).


# Get phenotypes associated with human genes
nameHGNCPhens_query = """
MATCH (m:`biolink:Gene`)-[:`biolink:has_phenotype`]-(n:`biolink:PhenotypicFeature`)
WHERE m.id STARTS WITH "HGNC"
RETURN n.id, m.id
"""
#  For each human gene (HGNC), return linked phenotypic features.


# Get all phenotypes, their id and namespace i.e. human phenotype ontology (HPO)
namePhens_query = """
MATCH (phenotype:`biolink:PhenotypicFeature`)
          RETURN phenotype.id, phenotype.namespace"""


# -------------------------------
# Gene-Specific Queries
# -------------------------------
def numgeneOrtho_query(gene: str) -> str:
    """
    Find orthologs for a specific gene.

    Args:
        gene (str): Gene ID (e.g., "HGNC:12345").

    Returns:
        str: Cypher query string.
    """
    return f"""
    MATCH (m:`biolink:Gene` {{id: "{gene}"}})-[:`biolink:orthologous_to`]-(n:`biolink:Gene`)
    RETURN count(*)
    """

def namesgeneOrthos_query(gene: str) -> str:
    """
    Find orthologs for a specific gene.

    Args:
        gene (str): Gene ID (e.g., "HGNC:12345").

    Returns:
        str: Cypher query string.
    """
    return f"""
    MATCH (m:`biolink:Gene` {{id: "{gene}"}})-[:`biolink:orthologous_to`]-(n:`biolink:Gene`)
    RETURN n.id, m.id
    """


def numgeneDis_query(gene: str) -> str:
    """
    Count how many diseases a given gene is associated with.
    """
    return f"""
    MATCH (m:`biolink:Gene` {{id: "{gene}"}})--(n:`biolink:Disease`)
    RETURN count(*)
    """


def numDisGene_query(gene: str) -> str:
    """
    Count how many genes a given disease is associated with.
    """
    return f"""
    MATCH (m:`biolink:Disease` {{id: "{gene}"}})--(n:`biolink:Gene`)
    RETURN count(*)
    """


def numGenePhens_query(gene: str) -> str:
    """
    Count how many phenotypes are associated with a given gene.
    """
    return f"""
    MATCH (m:`biolink:Gene` {{id: "{gene}"}})--(n:`biolink:PhenotypicFeature`)
    RETURN count(*)
    """


def numGenePhen_query(gene: str) -> str:
    """
    Count phenotype associations where the gene explicitly has a 'has_phenotype' edge.
    """
    return f"""
    MATCH (m:`biolink:Gene` {{id: "{gene}"}})-[:`biolink:has_phenotype`]->(n:`biolink:PhenotypicFeature`)
    RETURN count(*)
    """

def nameGenePhen_query(gene: str) -> str:
    """
    Get phenotype associations where the gene explicitly has a 'has_phenotype' edge.
    """
    return f"""
    MATCH (m:`biolink:Gene` {{id: "{gene}"}})-[:`biolink:has_phenotype`]->(n:`biolink:PhenotypicFeature`)
    RETURN n.id
    """

def nameDisPhen_query(dis: str) -> str:
    """
    Get phenotype associations where the disease explicitly has a 'has_phenotype' edge.
    """
    return f"""
    MATCH (m:`biolink:Disease` {{id: "{dis}"}})-[:`biolink:has_phenotype`]->(n:`biolink:PhenotypicFeature`)
    RETURN n.id
    """
# -------------------------------
# Organism-Level Queries
# -------------------------------

def numGeneOrthoTaxon_query(org1: str, org2: str) -> str:
    """
    Count the number of orthologous gene pairs between two organisms.

    Args:
        org1 (str): Taxon ID of first organism.
        org2 (str): Taxon ID of second organism.
    """
    return f"""
    MATCH (n:`biolink:Gene` {{in_taxon: "{org1}"}})-[:`biolink:orthologous_to`]-
          (m:`biolink:Gene` {{in_taxon: "{org2}"}})
    RETURN count(*)
    """


def numOrgPhens_query(org_prefix: str) -> str:
    """
    Count the number of phenotype associations for all genes in a given organism.

    Args:
        org_prefix (str): Prefix of gene IDs for the organism (e.g., "HGNC", "ZFIN").
    """
    return f"""
    MATCH (n:`biolink:Gene`)
    WHERE n.id STARTS WITH "{org_prefix}"
    MATCH (n)-[:`biolink:has_phenotype`]->()
    RETURN count(*)
    """


def numUPHENOrg_query(org1: str, org2: str) -> str:
    """
    Count cross-species phenotype connections via uPheno.

    Args:
        org1 (str): Prefix of phenotype IDs for organism 1.
        org2 (str): Prefix of phenotype IDs for organism 2.
    """
    return f"""
    MATCH (n:`biolink:PhenotypicFeature`)
    WHERE n.id STARTS WITH "{org1}"
    MATCH (n)--(u:`biolink:PhenotypicFeature` WHERE u.id STARTS WITH "UPHENO")--
          (m:`biolink:PhenotypicFeature` WHERE m.id STARTS WITH "{org2}")
    RETURN count(*)
    """


def get_gene_edge_counts_by_taxon():
    """
    Query Neo4j for the number of edges connected to each gene,
    then aggregate results by taxon.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per taxon containing:
        - taxon: the NCBITaxon identifier
        - total_edges: total number of edges across all genes in that taxon
        - avg_edges: average number of edges per gene in that taxon
        - gene_count: number of genes in that taxon
    """
    query = '''
    MATCH (g:`biolink:Gene`)-[e]-()
    RETURN g.in_taxon AS taxon, count(e) AS edge_count
    '''
    response = conn.query(query, db='monarch-20250217')

    # Convert Neo4j response to DataFrame
    df = pd.DataFrame(response)

    # Aggregate by taxon
    agg_df = (
        df.groupby("taxon")
          .agg(
              total_edges=("edge_count", "sum"),
              avg_edges=("edge_count", "mean"),
              gene_count=("edge_count", "count")
          )
          .reset_index()
    )

    return agg_df

# -------------------------------
# Phenotypic feature counts by type
# -------------------------------
def get_phenotypic_feature_counts_by_type():
    """
    Query Neo4j for the number of phenotypic features,
    grouped by their type.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per phenotypic feature type containing:
        - type: the category/type of the phenotypic feature
        - feature_count: number of phenotypic features of this type
    """
    query = '''
    MATCH (g:`biolink:PhenotypicFeature`)
    RETURN g.type AS type, count(g) AS feature_count
    '''
    response = conn.query(query, db=configDict['db'])

    df = pd.DataFrame(response)
    return df



def get_gene_edge_counts_raw():
    """
    Query Neo4j for the number of edges connected to each gene.

    Returns
    -------
    pd.DataFrame
        A DataFrame with:
        - edge_count: number of edges connected to the gene
        - taxon: the NCBITaxon identifier for the gene
    """
    query = '''
    MATCH (g:`biolink:Gene`)-[e]-()
    RETURN count(e) AS edge_count, g.in_taxon AS taxon
    '''
    response = conn.query(query, db=configDict['db'])
    return pd.DataFrame(response)


# -------------------------------
# Disease Queries
# -------------------------------
def numDisGene_query(dis: str) -> str:
    """
    Count how many genes a given disease is associated with.
    """
    return f"""
    MATCH (m:`biolink:Disease` {{id: "{dis}"}})--(n:`biolink:Gene`)
    RETURN count(*)
    """


def numDisPhen_query(dis: str) -> str:
    """
    Count how many phenotypess a given disease is associated with.
    """
    return f"""
    MATCH (m:`biolink:Disease` {{id: "{dis}"}})--(n:`biolink:PhenotypicFeature`)
    RETURN count(*)
    """

def numDisGeneotypes_query(dis: str) -> str:
    """
    Count how many genotypes a given disease is associated with.
    """
    return f"""
    MATCH (m:`biolink:Disease` {{id: "{dis}"}})--(n:`biolink:Genotype`)
    RETURN count(*)
    """

def numDisGO_query(dis: str) -> str:
    """
    Count how many go terms a given disease is associated with.
    """
    return f"""
    MATCH (m:`biolink:Disease` {{id: "{dis}"}})--(n:`biolink:BiologicalProcess` where n.id starts with "GO:")
    RETURN count(*)
    """

def nameDisPhen_query(dis: str) -> str:
    """
    Get phenotype associations where the disease explicitly has a 'has_phenotype' edge.
    """
    return f"""
    MATCH (m:`biolink:Disease` {{id: "{dis}"}})-[:`biolink:has_phenotype`]->(n:`biolink:PhenotypicFeature`)
    RETURN n.id
    """


def numDisGeneOrtho_query(dis: str) -> str:
    """
    Find orthologs for a specific gene.

    Args:
        gene (str): Gene ID (e.g., "HGNC:12345").

    Returns:
        str: Cypher query string.
    """
    return f"""
    MATCH (d:`biolink:Disease` {{id: "{dis}"}})--(m:`biolink:Gene`)-[:`biolink:orthologous_to`]-(n:`biolink:Gene`)
    RETURN count(*)
    """

def namesgeneDisease_query(dis: str) -> str:
    """
    Find genes for a specific disease.

    Args:
        dis (str): Disease ID (e.g., "MONDO:12345").

    Returns:
        str: Cypher query string.
    """
    return f"""
    MATCH (d:`biolink:Disease` {{id: "{dis}"}})--(g:`biolink:Gene`)
    RETURN g.id
    """


