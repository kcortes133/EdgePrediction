import csv
import random

import pandas as pd
import pandas as pd

def removeSpecifiedGeneDiseaseEdges(disease_file_1, disease_file_2, edge_file,
                                    outfolder):
    """
    Removes human-gene → disease edges from an edge table, saves results to TSVs.

    Args:
        disease_file_1 (str): first disease CSV
        disease_file_2 (str): second disease CSV
        edge_file (str): edge TSV file

    Returns:
        removed_edges (list of lists)
        kept_edges (list of lists)
    """
    kept_file = outfolder + 'keptEdges.tsv'
    removed_file = outfolder + 'removedEdges.tsv'
    # ------------------------------------------
    # Load the two disease files
    # ------------------------------------------
    df1 = pd.read_csv(disease_file_1)
    df2 = pd.read_csv(disease_file_2)

    # Detect disease/MONDO column automatically
    possible_cols = ["Rare Disease", "Disease", "MONDO", "mondo", "disease"]

    def extract_mondo_ids(df, name):
        for col in possible_cols:
            if col in df.columns:
                print(f"[INFO] Using column '{col}' from {name}")
                return set(df[col].dropna().astype(str).unique())
        raise ValueError(f"No MONDO disease column found in {name}. Columns = {df.columns}")

    mondo1 = extract_mondo_ids(df1, disease_file_1)
    mondo2 = extract_mondo_ids(df2, disease_file_2)
    mondo_ids = mondo1.union(mondo2)
    print(f"[INFO] Total MONDO IDs loaded: {len(mondo_ids)}")

    # ------------------------------------------
    # Load edges
    # ------------------------------------------
    edges = pd.read_csv(edge_file, sep="\t", dtype=str)
    if "subject" not in edges.columns or "object" not in edges.columns:
        raise ValueError("Edge file must contain 'subject' and 'object' columns.")

    # ------------------------------------------
    # Identify HGNC → MONDO edges for removal
    # ------------------------------------------
    mask_remove = (
        edges["subject"].str.startswith("HGNC:", na=False) &
        edges["object"].isin(mondo_ids)
    )

    removed_df = edges[mask_remove]
    kept_df = edges[~mask_remove]

    print(f"[INFO] Removed edges: {len(removed_df)}")
    print(f"[INFO] Kept edges: {len(kept_df)}")

    # ------------------------------------------
    # Convert to LISTS
    # ------------------------------------------
    removed_edges = [list(removed_df.columns)] + removed_df.values.tolist()
    kept_edges = [list(kept_df.columns)] + kept_df.values.tolist()

    # ------------------------------------------
    # Save to TSV
    # ------------------------------------------
    removed_df.to_csv(removed_file, sep="\t", index=False)
    kept_df.to_csv(kept_file, sep="\t", index=False)
    print(f"[INFO] Saved removed edges to {removed_file}")
    print(f"[INFO] Saved kept edges to {kept_file}")

    return removed_edges, kept_edges

def removeSpecifiedGeneDiseaseEdges1(disease_file_1, disease_file_2, edge_file):
    """
    Removes human-gene → disease edges from an edge table.

    Inputs:
        disease_file_1 (str): path to first disease CSV (must contain column 'Disease')
        disease_file_2 (str): path to second disease CSV (must contain column 'Disease')
        edge_file (str): path to edges TSV file

    Returns:
        kept_edges (list of dict): edges that were kept
        removed_edges (list of dict): edges that were removed
    """

    # -------------------------
    # Load disease files
    # -------------------------
    df1 = pd.read_csv(disease_file_1)
    df2 = pd.read_csv(disease_file_2)

    # Combine MONDO IDs
    mondo_ids = set(df1["Rare Disease"].unique()).union(
        set(df2["Disease"].unique())
    )

    # -------------------------
    # Load edges
    # -------------------------
    edges = pd.read_csv(edge_file, sep="\t", dtype=str)

    # -------------------------
    # Determine which edges to remove
    # Remove if:
    #   subject starts with "HGNC:"
    #   AND object is in disease list
    # -------------------------
    mask_remove = (
        edges["subject"].str.startswith("HGNC:", na=False) &
        edges["object"].isin(mondo_ids)
    )
    removed = edges[mask_remove]
    kept = edges[~mask_remove]
    print(removed)

    # -------------------------
    # Convert to list of dicts
    # -------------------------
    #removed_edges = removed.to_list(orient="records")
    #kept_edges = kept.to_list(orient="records")

    return removed, kept




###
# Remove random HGNC ↔ MONDO edges
# remove all has mode of inheritance edges
###
def removeRandomHumanGeneDiseaseEdges(edgesF):
    removedEdges = []
    wonkyGenes = ['MONDO:0014343', 'MONDO:0100038', 'MONDO:0008377', 'MONDO:0008542', "MONDO:0007542"]
    keptEdges = []
    c = 0
    hmiEdges = 0
    disGenepairs = []
    with open(edgesF, 'r', encoding='utf8') as f:
        rd = csv.reader(f, delimiter='\t')
        for row in rd:
            if c == 0:
                header = row
            elif row[2] == 'biolink:has_mode_of_inheritance':
                hmiEdges += 1
            else:
                src, dst = row[-1], row[-2]
                # Keep only human gene ↔ disease edges
                if src not in wonkyGenes and dst not in wonkyGenes:
                    if (src.split(':')[0] == 'HGNC' or dst.split(':')[0] == 'HGNC'):
                        if(src.split(':')[0] == 'MONDO' or dst.split(':')[0] == 'MONDO'):
                            if random.randint(0, 20) < 5:  # ~10%
                                removedEdges.append(row)
                                disGenepairs.append((src, dst))
                            else:
                                keptEdges.append(row)
                        else:
                            keptEdges.append(row)
            c += 1

    removedEdges[:0] = [header]
    keptEdges[:0] = [header]

    with open('disGenePairs.tsv', 'w', encoding='utf8') as f:
        for dg in disGenepairs:
            src, dst = dg[0], dg[1]
            f.write(src + '\t' + dst + '\n')

    print('removed has mode of inheritance edges : ', hmiEdges)
    return removedEdges, keptEdges


# randomly remove edges from the Rare Disease Gene Associations.csv File
# can use annotations file after prediction to see if diff subsets get predicted better
def removeRandomRareGeneDiseaseEdges(edgesF, pairsF):
    # Load disease–gene pairs to remove
    pairs_to_remove = set()
    hmiEdges = 0
    with open(pairsF, "r", encoding="utf8") as f:
        rd = csv.reader(f, delimiter=",")
        header = next(rd, None)  # skip header if present
        for row in rd:
            if not row:
                continue
            gene, disease = row[0].strip(), row[1].strip()
            pairs_to_remove.add((gene, disease))
            pairs_to_remove.add((disease, gene))  # allow either direction
    removed_edges, kept_edges = [], []
    with open(edgesF, "r", encoding="utf8") as f:
        rd = csv.reader(f, delimiter="\t")
        header = next(rd)
        for row in rd:
            src, dst = row[-1], row[-2]
            if (src, dst) in pairs_to_remove or (dst, src) in pairs_to_remove:
                if random.randint(0, 9) < 1:  # ~10%
                    removed_edges.append(row)
                else: kept_edges.append(row)
            elif row[2] == 'biolink:has_mode_of_inheritance':
                hmiEdges += 1
            else:
                kept_edges.append(row)

    # Prepend header back
    removed_edges.insert(0, header)
    kept_edges.insert(0, header)

    return removed_edges, kept_edges


import pandas as pd


def remove_edges(source_file, remove_file, output_file,
                 key_cols=["subject", "predicate", "object"]):
    """
    Remove edges from source_file that appear in remove_file based on key columns.

    Parameters
    ----------
    source_file : str
        Path to the main edges file.
    remove_file : str
        Path to the edges that should be removed.
    output_file : str
        Path where the filtered edges will be saved.
    key_cols : list
        Columns used to identify matching edges. Default = subject–predicate–object.
    """

    print("📥 Loading files...")
    df = pd.read_csv(source_file, sep="\t", dtype=str)
    df_remove = pd.read_csv(remove_file, sep="\t", dtype=str)

    print(f"🔑 Matching on columns: {key_cols}")

    # Create merge keys
    df["_key"] = df[key_cols].agg("||".join, axis=1)
    df_remove["_key"] = df_remove[key_cols].agg("||".join, axis=1)

    # Filter out edges that exist in remove set
    remove_keys = set(df_remove["_key"])
    filtered_df = df[~df["_key"].isin(remove_keys)].drop(columns=["_key"])

    print(f"🧹 Removed {len(df) - len(filtered_df)} edges.")
    print(f"💾 Saving to: {output_file}")

    filtered_df.to_csv(output_file, sep="\t", index=False)

    return output_file


#
###
# Write edges to files
###
def writeRemovedEdgeFiles(folder, removedE, keptE):
    with open(folder+'removedEdges.tsv', 'w', encoding='utf8', newline='') as of:
        csv.writer(of, delimiter='\t').writerows(removedE)
    with open(folder+'keptEdges.tsv', 'w', encoding='utf8', newline='') as of:
        csv.writer(of, delimiter='\t').writerows(keptE)
