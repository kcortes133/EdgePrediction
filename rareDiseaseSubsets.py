import csv
import holoviews as hv
import matplotlib.pyplot as plt
import queries
import json
from neo4jConnection import Neo4jConnection
from neo4jConfig import configDict
from queries import *
import pandas as pd
from upsetplot import from_indicators, UpSet
from SPARQLWrapper import SPARQLWrapper, JSON
import math
from itertools import product
from collections import defaultdict


# establishing connection with neo4j
conn = Neo4jConnection(uri=configDict['uri'],
                       user=configDict['user'],
                       pwd=configDict['pwd'])
DB_NAME = configDict['db']

def getRareDiseases():
    rareDiseases = []
    with open('rareDiseases.tsv', 'r') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        header = next(reader)
        for row in reader:
            diseaseID = row[0]
            rareDiseases.append(diseaseID)
    return rareDiseases

def getAllDiseaseswGenes():
    """Return all human genes with HGNC IDs."""
    response = conn.query(nameDisease_query, db=DB_NAME)
    print(nameDisease_query)
    return [item for sublist in json.loads(json.dumps(response)) for item in sublist]

def getAllHGenes():
    """Return all human genes with HGNC IDs."""
    response = conn.query(nameHGNC_query, db=DB_NAME)
    return [item for sublist in json.loads(json.dumps(response)) for item in sublist]


def getAllOrthos():
    """Return all human genes that have orthologs in other organisms."""
    response = conn.query(nameHGNCOrthos_query, db=DB_NAME)
    orthologs = [item for sublist in json.loads(json.dumps(response)) for item in sublist]
    return set(filter(lambda x: x.startswith('HGNC'), orthologs))


def getOrthoTypeCount(humanGwithOrtho):
    """Count ortholog types (by taxon prefix) for each human gene."""
    humanOrthoTypes = {}
    for gene in humanGwithOrtho:
        query = namesgeneOrthos_query( gene)  # substitute
        response = conn.query(query, db=DB_NAME)
        orthologs = [item for sublist in json.loads(json.dumps(response)) for item in sublist]

        humanOrthoTypes[gene] = {}
        for o in orthologs:
            moType = o.split(':')[0]
            humanOrthoTypes[gene][moType] = humanOrthoTypes[gene].get(moType, 0) + 1
    return humanOrthoTypes


def hasPhenotype(genes):
    """Split genes into those with and without phenotypic annotations."""
    genesWithPhenotype, genesWithoutPhenotype = [], []
    for g in genes:
        query = numGenePhens_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            genesWithPhenotype.append(g)
        else:
            genesWithoutPhenotype.append(g)
    return set(genesWithPhenotype), set(genesWithoutPhenotype)


def hasDiseaseAnnotation(genes):
    """Split genes into those with and without disease annotations."""
    genesWithDisease, genesWithoutDisease = [], []
    for g in genes:
        query = numgeneDis_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            genesWithDisease.append(g)
        else:
            genesWithoutDisease.append(g)
    return set(genesWithDisease), set(genesWithoutDisease)

def hasOrthoAnnotation(dis):
    """Split genes into those with and without disease annotations."""
    disWithOrtho, disWithoutOrtho = [], []
    for g in dis:
        query = numDisGeneOrtho_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            disWithOrtho.append(g)
        else:
            disWithoutOrtho.append(g)
    return set(disWithOrtho), set(disWithoutOrtho)


def disHasPhenAnnotation(genes):
    """Split genes into those with and without disease annotations."""
    disWPhen, disWOPhen = [], []
    for g in genes:
        query = numDisPhen_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            disWPhen.append(g)
        else:
            disWOPhen.append(g)
    return set(disWPhen), set(disWOPhen)



def disHasGeneAnnotation(genes):
    """Split genes into those with and without disease annotations."""
    disWGene, disWOGene = [], []
    for g in genes:
        query = numDisGene_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            disWGene.append(g)
        else:
            disWOGene.append(g)
    return set(disWGene), set(disWOGene)



def disHasGenotype(genes):
    """Split genes into those with and without disease annotations."""
    disWGeneT, disWOGeneT = [], []
    for g in genes:
        query = numDisGeneotypes_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            disWGeneT.append(g)
        else:
            disWOGeneT.append(g)
    return set(disWGeneT), set(disWOGeneT)


def disHasGOAnnotation(genes):
    """Split genes into those with and without disease annotations."""
    disWGO, disWOGO = [], []
    for g in genes:
        query = numDisGO_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            disWGO.append(g)
        else:
            disWOGO.append(g)
    return set(disWGO), set(disWOGO)
# -------------------------------
# Sankey Visualization
# -------------------------------


def orthoSankey():
    allOrthologs = getAllOrthos()
    humanOrthoTypes = getOrthoTypeCount(allOrthologs)

    onlyOneOrtho = []
    multiOrthoOneOrg = []
    for hgene in humanOrthoTypes:
        if len(humanOrthoTypes[hgene]) == 1:
            modelOrg = list(humanOrthoTypes[hgene].keys())[0]
            if humanOrthoTypes[hgene][modelOrg] == 1:
                onlyOneOrtho.append(hgene)
            else: multiOrthoOneOrg.append(hgene)

'''
    multiOrthologs = set(allOrthologs) - set(onlyOneOrtho) - set(multiOrthoOneOrg)
    gwithPhen, gWOPhen = hasPhenotype(list(multiOrthologs))
    gwithPhenOO, gWOPhenOO = hasPhenotype(onlyOneOrtho)
    gwithPhenMO, gWOPhenMO = hasPhenotype(multiOrthoOneOrg)


    gwithDis, gWODis = hasDiseaseAnnotation(list(multiOrthologs))
    gwithDisOO, gWODisOO = hasDiseaseAnnotation(onlyOneOrtho)
    gwithDisMO, gWODisMO = hasDiseaseAnnotation(multiOrthoOneOrg)

    sanKeyData = []
    sanKeyData.append(['Human Genes with Orthologs', 'Genes with Many Orthologs', len(multiOrthologs)])
    sanKeyData.append(['Human Genes with Orthologs', 'Genes with One Ortholog', len(onlyOneOrtho)])
    sanKeyData.append(['Human Genes with Orthologs', 'Genes with Orthologs from One Organism', len(multiOrthoOneOrg)])

    sanKeyData.append(['Genes with Many Orthologs', 'Has Disease Annotation', len(gwithDis)])
    sanKeyData.append(['Genes with One Ortholog', 'Has Disease Annotation', len(gwithDisOO)])
    sanKeyData.append(['Genes with Orthologs from One Organism', 'Has Disease Annotation', len(gwithDisMO)])

    sanKeyData.append(['Genes with Many Orthologs', 'No Disease Annotation', len(gWODis)])
    sanKeyData.append(['Genes with One Ortholog', 'No Disease Annotation', len(gWODisOO)])
    sanKeyData.append(['Genes with Orthologs from One Organism', 'No Disease Annotation', len(gWODisMO)])

    sanKeyData.append(['Has Disease Annotation', 'Has Phenotypic Feature Many Orthologs', len(gwithDis.intersection(gwithPhen)) ])
    sanKeyData.append(['Has Disease Annotation', 'Has Phenotypic Feature One Ortholog', len(gwithDisOO.intersection(gwithPhenOO))])
    sanKeyData.append(['Has Disease Annotation', 'Has Phenotypic Feature Orthologs from One Organism', len(gwithDisMO.intersection(gwithPhenMO))])

    sanKeyData.append(['No Disease Annotation', 'No Phenotypic Feature Many Orthologs', len(gWODis.intersection(gWOPhen))])
    sanKeyData.append(['No Disease Annotation', 'No Phenotypic Feature One Ortholog', len(gWODisOO.intersection(gWOPhenOO))])
    sanKeyData.append(['No Disease Annotation', 'No Phenotypic Feature Orthologs from One Organism', len(gWODisMO.intersection(gWOPhenMO))])
'''
# get rare disease subset from each



#rareDisSubset = getRareDiseases()
#disSubset = getAllHGenes()
def diseaseGenePairs(disSubset):
    dgpairs = []
    for r in disSubset:
        query = namesgeneDisease_query(r)
        response = conn.query(query, db=DB_NAME)
        genes = [item for sublist in json.loads(json.dumps(response)) for item in sublist]
        if genes:
            for g in genes:
                dgpairs.append((r, g))
    headers = ["Disease", "Gene"]

    # Specify the output CSV file name
    output_filename = "Disease Gene Associations.csv"

    # Open the CSV file in write mode
    with open(output_filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header row (if defined)
        if headers:
            csv_writer.writerow(headers)

        # Write each pair as a row in the CSV
        for pair in dgpairs:
            csv_writer.writerow(pair)


def diseaseAnnotationFile(rareDisSubset):
    # how many diseases with gene assoc where gene has an ortholog
    diseaseWithGene, diseaseWOGene = disHasGeneAnnotation(rareDisSubset)
    disGeneHasOrtho, disGeneNoOrtho = hasOrthoAnnotation(diseaseWithGene)
    diseaseWithPhen, diseaseWOPhen = disHasPhenAnnotation(rareDisSubset)
    diseaseWithGenotype, diseaseWOGenotype = disHasGenotype(rareDisSubset)
    diseaseWithGO, diseaseWOGO = disHasGOAnnotation(rareDisSubset)

    data = {'Disease': list(rareDisSubset)}
    df = pd.DataFrame(data)

    df['Has Gene'] = df['Disease'].isin(diseaseWithGene).astype(int)
    df['Has Gene with Ortholog'] = df['Disease'].isin(disGeneHasOrtho).astype(int)
    df['Has Phenotype'] = df['Disease'].isin(diseaseWithPhen).astype(int)
    df['Has Genotype'] = df['Disease'].isin(diseaseWithGenotype).astype(int)
    df['Has GO'] = df['Disease'].isin(diseaseWithGO).astype(int)

    df.to_csv('Disease Annotation.csv', index=False)

# can use these as test cohorts to begin
# RDs with Gene and Phenotypes:  5676
# RDs with Gene and no Phenotypes:  445
# RD Gene has Ortholog:  6097
# RD Gene does not have Ortholog:  24



def summary():
    diseaseWithPhen, diseaseWOPhen = disHasPhenAnnotation(rareDisSubset)
    diseaseWithGene, diseaseWOGene = disHasGeneAnnotation(rareDisSubset)
    diseaseWithGenotype, diseaseWOGenotype = disHasGenotype(rareDisSubset)
    diseaseWithGO, diseaseWOGO = disHasGOAnnotation(rareDisSubset)
    print(len(rareDisSubset))
    print('Rare Diseases with annotation, Diseases Without')
    print('Phenotype: ',len(diseaseWithPhen), len(diseaseWOPhen))
    print('Gene: ',len(diseaseWithGene), len(diseaseWOGene))
    print('Genotype: ', len(diseaseWithGenotype), len(diseaseWOGenotype))
    print('GO Term: ', len(diseaseWithGO), len(diseaseWOGO))

def createSankey():
    sankeyData = []
    #sankeyData.append(['','', len()])
    sankeyData.append(['Rare Diseases','With Associated Gene', len(diseaseWithGene)])
    sankeyData.append(['Rare Diseases','Without Associated Gene', len(diseaseWOGene)])
    sankeyData.append(['Without Associated Gene','Has Phenotype Association', len(diseaseWOGene.intersection(diseaseWithPhen))])
    sankeyData.append(['Without Associated Gene','Has GO Association', len(diseaseWOGene.intersection(diseaseWithGO))])
    sankeyData.append(['Without Associated Gene','Has Genotype Association', len(diseaseWOGene.intersection(diseaseWithGenotype))])

    df = pd.DataFrame(sankeyData, columns=['source', 'target', 'value'])
    df.to_csv('Rare Disease Association Overview.csv')
    hv.extension('matplotib')
    hv.output(fig='svg')
    sankey = hv.Sankey(df, label='Rare Diseases')
    sankey.opts(label_position='left', edge_color='target', cmap='tab20')
    hv.save(sankey, 'RD-sankey.svg')
    hv.render(sankey, backend='matplotlib')
    plt.show()


def annotationUpset():
    df = pd.read_csv("Rare Disease Annotation.csv")

    # Select the binary annotation columns
    category_cols = ["Has Gene", "Has Gene with Ortholog", "Has Phenotype", "Has Genotype", "Has GO"]

    # Convert to boolean (0/1 → False/True)
    df[category_cols] = df[category_cols].astype(bool)

    # Build UpSet data (each MONDO disease is an observation, membership given by category cols)
    upset_data = from_indicators(category_cols, df[category_cols])

    # --- Create the UpSet plot ---
    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",  # largest intersections first
        orientation="horizontal"
    )

    # Draw
    upset.plot()
    plt.suptitle("UpSet Plot of Rare Disease Annotations", fontsize=14)
    plt.show()

#annotationUpset()
from neo4j import GraphDatabase

def get_gene_disease_paths(gene_id, disease_id, max_length=3):
    """
    Retrieve paths between a given gene and disease in Neo4j.

    :param gene_id: Gene identifier (e.g. "HGNC:1234")
    :param disease_id: Disease identifier (e.g. "MONDO:0001234")
    :param max_length: Maximum path length (hops) to explore
    :return: List of paths (nodes + relationships)
    """
    query = f"""
    MATCH p = (g:`biolink:Gene` {{id:"{gene_id}"}})
              -[*1..{max_length}]-
              (d:`biolink:Disease` {{id:"{disease_id}"}})
    RETURN p
    """
    return query

# Example usage
'''
if __name__ == "__main__":
    gene = "HGNC:14313"
    disease = "MONDO:0033485"
    pathsQuery = get_gene_disease_paths(gene, disease, max_length=4)
    response = conn.query(pathsQuery)
    print(pathsQuery)
    print(response)
    paths = [record["p"] for record in response]
    metapaths = {}
    for idx, path in enumerate(paths, 1):
        print(f"\nPath {idx}:")
        metaP = []
        for node in path.nodes:
            print(f"  Node: {node['id']} ({list(node.labels)})")
        for rel in path.relationships:
            metaP.append(rel.start_node['id'].split(":")[0])
            metaP.append(rel.type)
            metaP.append(rel.end_node['id'].split(":")[0])

            print(f"  Relationship: {rel.type} from {rel.start_node['id']} → {rel.end_node['id']}")
        mP = ' '.join(metaP)
        if mP not in metapaths:
            metapaths[mP] = 1
        else:
            metapaths[mP] += 1
        print(mP)

    #print(metapaths)
    #print(len(metapaths))
    print(sorted(metapaths.items(), key=lambda x: x[1], reverse=False))
'''


def analyze_metapaths(pairs_file: str, max_length: int = 4):
    """
    For each RareDisease-Gene pair in the file, query paths in Neo4j
    and increment meta-path counts across all pairs.

    Parameters
    ----------
    pairs_file : str
        CSV file with columns: Rare Disease, Gene
    max_length : int, optional
        Maximum path length to search in the graph (default = 4)

    Returns
    -------
    metapaths : dict
        Dictionary of meta-paths and their frequencies across all pairs
    """

    # Load pairs
    pairs = pd.read_csv(pairs_file)

    # Store metapaths across all pairs
    metapaths = {}
    c=0

    for _, row in pairs.iterrows():
        if c > 5:
            break
        #disease = row["Rare Disease"]
        disease = row["object"]
        #gene = row["Gene"]
        gene = row["subject"]

        print(f"\n=== Processing pair: {gene} - {disease} ===")

        # Build query
        pathsQuery = get_gene_disease_paths(gene, disease, max_length=max_length)
        response = conn.query(pathsQuery)

        paths = [record["p"] for record in response]

        for idx, path in enumerate(paths, 1):
            metaP = []

            # Show nodes
            for node in path.nodes:
                print(f"  Node: {node['id']} ({list(node.labels)})")

            # Extract metapath
            for rel in path.relationships:
                metaP.append(rel.start_node['id'].split(":")[0])
                metaP.append(rel.type)
                metaP.append(rel.end_node['id'].split(":")[0])
                print(f"  Relationship: {rel.type} from {rel.start_node['id']} → {rel.end_node['id']}")

            mP = " ".join(metaP)

            # Increment counts
            if mP not in metapaths:
                metapaths[mP] = 1
            else:
                metapaths[mP] += 1

        c+=1

    # Print sorted summary
    print("\n=== Final Meta-path Frequencies ===")
    print(sorted(metapaths.items(), key=lambda x: x[1], reverse=True))
    output_csv = ('Negative Set Rare Disease Metapath Frequencies '+  str(c) +'.csv')
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["MetaPath", "Count"])
        for mp, count in sorted(metapaths.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([mp, count])

    print(f"\nMeta-path counts saved to {output_csv}")

    return metapaths


# TODO: get genes without a disease relationship
# TODO: for each disease filter a list of genes that are more likely
# TODO: get metapaths surrounding rare disease - gene pairs and common disease - gene pairs Use to subset gene space

def getAllHGenes():
    """Return all human genes with HGNC IDs."""
    response = conn.query(nameHGNC_query, db=DB_NAME)
    return [item for sublist in json.loads(json.dumps(response)) for item in sublist]

def hasDiseaseAnnotation(genes):
    """Split genes into those with and without disease annotations."""
    genesWithDisease, genesWithoutDisease = [], []
    for g in genes:
        query = numgeneDis_query(g)
        response = conn.query(query, db=DB_NAME)
        count = [item for sublist in json.loads(json.dumps(response)) for item in sublist][0]
        if int(count) > 0:
            genesWithDisease.append(g)
        else:
            genesWithoutDisease.append(g)
    return set(genesWithDisease), set(genesWithoutDisease)


def getGenesWODisConns(removedEdgesFile, genesWODisConnsFile):
    genes = []
    with open(genesWODisConnsFile, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            genes.append(row[0].strip())
    with open(removedEdgesFile, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[-2].startswith("HGNC"):
                genes.append(row[-2])

    response = conn.query(nameHGNCProtienCode_query, db=DB_NAME)
    names = [item for sublist in json.loads(json.dumps(response)) for item in sublist]
    protienCodNoDisGenes = set(genes).intersection(set(names))

    return protienCodNoDisGenes


import pandas as pd
import math
'''
if __name__ == "__main__":
    diseases = getAllDiseaseswGenes()
    diseaseAnnotationFile(diseases)
    # Load CSV
    wonkyGenes = ['MONDO:0014343', 'MONDO:0100038', 'MONDO:0008377', 'MONDO:0008542', "MONDO:0007542"]
    df1 = pd.read_csv("Disease Annotation.csv")
    df = df1[~df1['Disease'].isin(wonkyGenes)]
    print(f"Original rows: {len(df1)}, Filtered rows: {len(df)}")

    # Feature columns (categories)
    categories = [
        "Has Gene",
        "Has Gene with Ortholog",
        "Has Phenotype",
        "Has Genotype",
        "Has GO"
    ]

    sampled_frames = []

    for cat in categories:
        # Get positive rows
        positives = df[df[cat] == 1]

        # Number to sample (~15%)
        n = math.ceil(len(positives) * 0.15)

        if n > 0:
            sampled = positives.sample(n=n, random_state=42)
            sampled_frames.append(sampled)

    # Combine and drop duplicates
    final_subset = pd.concat(sampled_frames).drop_duplicates()

    # Save to CSV
    final_subset.to_csv("subset_15_percent_each_categoryAll.csv", index=False)

    print("Selected rows:", len(final_subset))
    print(final_subset)
import pandas as pd'''
import pandas as pd


def merge_and_count(pair_file, rare_file, output_file):
    """
    pair_file: TSV with columns [disease, gene]
    rare_file: CSV/TSV with rare disease annotations, with column 'Rare Disease'

    Output: merged subset + printed annotation counts
    """

    # --- Load data files ---
    #df_pairs = pd.read_csv(pair_file, sep="\t", header=None,
    #                       names=["Rare Disease", "Gene"], dtype=str)
    # Load the TSV file
    df = pd.read_csv(pair_file, sep="\t")

    # Extract subject–object pairs
    df_pairs = df[["object", "subject"]]


    df_rare = pd.read_csv(rare_file, sep=None, engine="python", dtype=str)

    # --- Subset to diseases present in the pair file ---
    subset = df_rare[df_rare["Rare Disease"].isin(df_pairs["object"])]
    print(subset)

    # --- Merge gene information ---
    #merged = subset.merge(df_pairs, on="Rare Disease", how="left")

    # --- Save output ---
    #merged.to_csv(output_file, index=False)
    subset.to_csv(output_file, index=False)
    print(f"Saved merged subset to: {output_file}\n")

    # --- Print annotation counts ---
    print("=== Annotation Counts ===")
    annotation_cols = [
        "Has Gene",
        "Has Gene with Ortholog",
        "Has Phenotype",
        "Has Genotype",
        "Has GO",
    ]

    for col in annotation_cols:
        if col in subset.columns:
            count = (subset[col] == "1").sum()
            print(f"{col}: {count}")

    print("\nTotal rare diseases in final subset:", len(subset))

    return subset


if __name__ == "__main__":

    merge_and_count('removedEdges.tsv', 'Rare Disease Annotation.csv', 'Testing Subset Rare.csv')
