import os
import pandas as pd

def summarize_gene_ranks(base_dir, output_file="model_summary.tsv"):
    """
    Summarize gene ranking performance across model folders.

    Parameters
    ----------
    base_dir : str
        Directory containing model folders.
    output_file : str
        Output TSV file.
    """

    results = []

    # Loop through each model folder
    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name)
        file_path = os.path.join(model_path, "gene_ranks.tsv")

        if not os.path.isdir(model_path) or not os.path.exists(file_path):
            continue

        print(f"Processing {model_name}...")

        # Load file
        df = pd.read_csv(file_path, sep="\t")

        # Ensure classification column exists
        if "classification" not in df.columns:
            print(f"Skipping {model_name}: no classification column")
            continue

        # Counts
        missing = (df["classification"] == "MISSING_TRUE_GENE").sum()

        top10 = df["classification"].str.contains("Top10-TP", na=False).sum()

        # Top50 includes Top10 as well
        top50 = df["classification"].str.contains("Top50-TP", na=False).sum() + top10

        outside_top50 = df["classification"].str.contains(
            "TP-outside-Top50", na=False
        ).sum()

        total = len(df)

        results.append({
            "model": model_name,
            "total": total,
            "missing_true_gene": missing,
            "top10_tp": top10,
            "top50_tp": top50,
            "tp_outside_top50": outside_top50,
            "pct_top10": top10 / total if total else 0,
            "pct_top50": top50 / total if total else 0,
            "pct_missing": missing / total if total else 0,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by performance (Top10 or Top50)
    results_df = results_df.sort_values(by="pct_top10", ascending=False)

    # Save
    results_df.to_csv(output_file, sep="\t", index=False)

    print("\n=== Summary ===")
    print(results_df)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    outdir = 'comparison_figures_rand.tsv'

    summarize_gene_ranks(base_dir, outdir)

import os
import pandas as pd


def normalize_pair(a, b):
    """Return an order-independent representation of a pair."""
    return frozenset([a, b])


def load_tp_pairs(tp_file):
    """
    Load TP gene-disease pairs into a set (order-independent).
    """
    tp_df = pd.read_csv(tp_file, sep="\t")

    tp_pairs = set(
        normalize_pair(row["subject"], row["object"])
        for _, row in tp_df.iterrows()
    )

    return tp_pairs
import pandas as pd


def check_and_remove_duplicates(df, gene_col="gene", disease_col="disease", strict=True):
    """
    Check for duplicate gene-disease pairs (order-independent) after filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe.
    gene_col : str
        Column name for gene.
    disease_col : str
        Column name for disease.
    strict : bool
        If True, raise error when duplicates exist.
        If False, drop duplicates and continue.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed (if strict=False).
    """

    # Normalize to order-independent pairs
    df = df.copy()
    df["_pair"] = [
        tuple(sorted((g, d)))
        for g, d in zip(df[gene_col], df[disease_col])
    ]

    # Find duplicates
    dup_mask = df.duplicated(subset="_pair", keep=False)
    duplicates = df[dup_mask]

    if not duplicates.empty:
        print(f"Found {duplicates.shape[0]} duplicate rows "
              f"across {duplicates['_pair'].nunique()} unique pairs.")

        if strict:
            raise ValueError("Duplicate gene-disease pairs found after filtering.")
        else:
            # Drop duplicates, keep first occurrence
            df = df.drop_duplicates(subset="_pair", keep="first")

    return df.drop(columns=["_pair"])

def summarize_gene_ranks(base_dir, tp_file, output_file="model_summary.tsv"):
    """
    Summarize gene ranking performance across model folders,
    restricted to TP gene-disease pairs (order-independent).
    """

    results = []

    # Load TP pairs once
    tp_pairs = load_tp_pairs(tp_file)

    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name)
        file_path = os.path.join(model_path, "gene_ranks.tsv")

        if not os.path.isdir(model_path) or not os.path.exists(file_path):
            continue
        dups = find_duplicate_pairs(file_path)
        print("DUPES: ", len(dups))

        # or nicely formatted

        print(f"Processing {model_name}...")

        df = pd.read_csv(file_path, sep="\t")

        required_cols = {"gene", "disease", "classification"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {model_name}: missing required columns")
            continue

        # --- ORDER-INDEPENDENT FILTER ---
        df["pair"] = [
            normalize_pair(g, d) for g, d in zip(df["gene"], df["disease"])
        ]
        df = df[df["pair"].isin(tp_pairs)]

        # --- CHECK DUPLICATES ---
        df = check_and_remove_duplicates(df, strict=False)
        if df.empty:
            print(f"Skipping {model_name}: no matching TP pairs")
            continue

        # Counts
        missing = (df["classification"] == "MISSING_TRUE_GENE").sum()

        top10 = df["classification"].str.contains("Top10-TP", na=False).sum()

        # Top50 includes Top10
        top50 = (
            df["classification"].str.contains("Top50-TP", na=False).sum()
            + top10
        )

        outside_top50 = df["classification"].str.contains(
            "TP-outside-Top50", na=False
        ).sum()

        total = len(df)

        results.append({
            "model": model_name,
            "total": total,
            "missing_true_gene": missing,
            "top10_tp": top10,
            "top50_tp": top50,
            "tp_outside_top50": outside_top50,
            "pct_top10": top10 / total if total else 0,
            "pct_top50": top50 / total if total else 0,
            "pct_missing": missing / total if total else 0,
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(by="pct_top10", ascending=False)

    results_df.to_csv(output_file, sep="\t", index=False)

    print("\n=== Summary ===")
    print(results_df)
    print(f"\nSaved to: {output_file}")


def find_duplicate_pairs(file_path):
    """
    Identify duplicate gene-disease pairs in a TSV file,
    ignoring order (A,B == B,A).

    Parameters
    ----------
    file_path : str
        Path to TSV file with columns: gene, disease

    Returns
    -------
    duplicates : dict
        Mapping of pair -> count for pairs that appear more than once
    """

    df = pd.read_csv(file_path, sep="\t")

    if not {"gene", "disease"}.issubset(df.columns):
        raise ValueError("File must contain 'gene' and 'disease' columns")

    # Normalize to order-independent pairs
    pairs = [
        tuple(sorted((g, d)))
        for g, d in zip(df["gene"], df["disease"])
    ]

    counts = Counter(pairs)

    # Keep only duplicates
    duplicates = {pair: count for pair, count in counts.items() if count > 1}

    return duplicates


def report_duplicates(file_path):
    """
    Print duplicate pairs in a readable format.
    """
    duplicates = find_duplicate_pairs(file_path)

    if not duplicates:
        print("No duplicate pairs found.")
        return

    print("Duplicate gene-disease pairs:\n")
    for (g, d), count in duplicates.items():
        print(f"{g} - {d}: {count} times")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) + os.sep
    tp_file = "TP_hgnc_mondo_edges.tsv"
    output_file = "comparison_figures_randTP_filtered.tsv"

    summarize_gene_ranks(base_dir, tp_file, output_file)
