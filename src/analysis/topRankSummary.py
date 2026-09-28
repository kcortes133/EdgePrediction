import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RESULTS_GLOB = os.path.join(PROJECT_ROOT, "results_*_ranks")
GENE_RANKS_FILE = "gene_ranks.tsv"

TOP10 = 10
TOP50 = 50
TOP_N_IMPROVED = 10

ORDERED_SUBSETS = ["none", "40", "60", "80", "100"]

OUTDIR = os.path.join(PROJECT_ROOT, "comparison_figures", "TopRankSummary")
os.makedirs(OUTDIR, exist_ok=True)


# ======================
# HELPERS
# ======================
def get_subset(folder):
    name = os.path.basename(folder)
    prefix, suffix = RESULTS_GLOB.replace(PROJECT_ROOT, "").lstrip(os.sep).split("*")
    return name[len(prefix):len(name) - len(suffix)]


def normalize_pairs(df):
    """gene_ranks.tsv stores gene/disease in whichever of the two columns happened to be
    the source/destination, so the id prefix (HGNC/MONDO) is the only reliable signal."""
    disease_col_is_gene = df["disease"].str.startswith("HGNC")

    out = df.copy()
    out["gene"] = df["disease"].where(disease_col_is_gene, df["gene"])
    out["disease"] = df["gene"].where(disease_col_is_gene, df["disease"])
    return out


# ======================
# LOAD RESULTS
# ======================
def load_subset_ranks(folder):
    """Load one subset's gene_ranks.tsv, collapsing the gene->disease and disease->gene
    directional rows for each pair down to a single best (lowest) rank."""
    path = os.path.join(folder, GENE_RANKS_FILE)
    if not os.path.exists(path):
        print(f"Skipping {folder}: no {GENE_RANKS_FILE} found")
        return None

    df = normalize_pairs(pd.read_csv(path, sep="\t"))
    return df.groupby(["gene", "disease"], as_index=False)["rank"].min()


def load_all_subsets():
    subset_ranks = {}
    for folder in sorted(glob.glob(RESULTS_GLOB)):
        ranks = load_subset_ranks(folder)
        if ranks is not None:
            subset_ranks[get_subset(folder)] = ranks
    return subset_ranks


def build_rank_matrix(subset_ranks):
    """One row per gene-disease pair, one column per subset holding that subset's best rank."""
    merged = None
    for subset, df in subset_ranks.items():
        renamed = df.rename(columns={"rank": subset})
        merged = renamed if merged is None else merged.merge(renamed, on=["gene", "disease"], how="outer")
    return merged


# ======================
# TOP10 / TOP50 SUMMARY
# ======================
def summarize_top_fractions(subset_ranks):
    rows = []
    for subset, df in subset_ranks.items():
        total = len(df)
        top10 = int((df["rank"] <= TOP10).sum())
        top50 = int((df["rank"] <= TOP50).sum())

        rows.append({
            "subset": subset,
            "total_pairs": total,
            "top10_count": top10,
            "top10_frac": top10 / total if total else 0.0,
            "top50_count": top50,
            "top50_frac": top50 / total if total else 0.0,
        })

    summary = pd.DataFrame(rows)
    summary["subset"] = pd.Categorical(summary["subset"], ORDERED_SUBSETS, ordered=True)
    return summary.sort_values("subset")


def plot_top_fractions(summary_df):
    plt.figure(figsize=(8, 6))
    plt.plot(summary_df["subset"], summary_df["top10_frac"], marker="o", label="Top-10 fraction")
    plt.plot(summary_df["subset"], summary_df["top50_frac"], marker="o", label="Top-50 fraction")
    plt.xlabel("KG Subset")
    plt.ylabel("Fraction of gene-disease pairs")
    plt.title("Top-10 / Top-50 Fraction by Subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top_fractions_by_subset.png"))
    plt.close()


# ======================
# IMPROVEMENT TRACKING
# ======================
def compute_improvement(rank_matrix, from_subset, to_subset):
    """improvement = rank(from) - rank(to); positive means the pair's rank got better (lower)."""
    df = rank_matrix.dropna(subset=[from_subset, to_subset]).copy()
    df["improvement"] = df[from_subset] - df[to_subset]
    return df


def top_improved(rank_matrix, from_subset, to_subset, n=TOP_N_IMPROVED):
    df = compute_improvement(rank_matrix, from_subset, to_subset)
    return df.sort_values("improvement", ascending=False).head(n)


def least_improved_or_worse(rank_matrix, from_subset, to_subset, n=TOP_N_IMPROVED):
    df = compute_improvement(rank_matrix, from_subset, to_subset)
    return df.sort_values("improvement", ascending=True).head(n)


def flag_continuously_worsening(rank_matrix):
    """Pairs whose rank gets strictly worse (higher) at every step none->40->60->80->100."""
    df = rank_matrix.dropna(subset=ORDERED_SUBSETS).copy()

    worsens_every_step = pd.Series(True, index=df.index)
    for prev, curr in zip(ORDERED_SUBSETS[:-1], ORDERED_SUBSETS[1:]):
        worsens_every_step &= df[curr] > df[prev]

    return df[worsens_every_step]


def plot_pair_rank_lines(df, title, filename):
    """Line plot of each pair's rank across subsets (none->40->60->80->100)."""
    if df.empty:
        print(f"Nothing to plot for {filename}, skipping.")
        return

    plt.figure(figsize=(9, 6))
    for _, row in df.iterrows():
        label = f"{row['gene']} - {row['disease']}"
        plt.plot(ORDERED_SUBSETS, [row[s] for s in ORDERED_SUBSETS], marker="o", label=label)

    plt.xlabel("KG Subset")
    plt.ylabel("Rank (lower = better)")
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename))
    plt.close()


# ======================
# MAIN
# ======================
def main():
    print("Loading gene ranks for all subsets...")
    subset_ranks = load_all_subsets()

    print("Summarizing top10/top50 fractions...")
    summary = summarize_top_fractions(subset_ranks)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTDIR, "top_rank_summary.tsv"), sep="\t", index=False)
    plot_top_fractions(summary)

    print("Building rank matrix across subsets...")
    rank_matrix = build_rank_matrix(subset_ranks)
    rank_matrix.to_csv(os.path.join(OUTDIR, "rank_matrix_all_subsets.tsv"), sep="\t", index=False)

    print("Finding top improved pairs (none -> 80)...")
    improved_80 = top_improved(rank_matrix, "none", "80")
    improved_80.to_csv(os.path.join(OUTDIR, "top10_improved_none_to_80.tsv"), sep="\t", index=False)
    plot_pair_rank_lines(
        improved_80,
        "Top 10 Most Improved (none -> 80): Rank by Subset",
        "top10_improved_none_to_80.png"
    )

    print("Finding top improved pairs (none -> 100)...")
    improved_100 = top_improved(rank_matrix, "none", "100")
    improved_100.to_csv(os.path.join(OUTDIR, "top10_improved_none_to_100.tsv"), sep="\t", index=False)
    plot_pair_rank_lines(
        improved_100,
        "Top 10 Most Improved (none -> 100): Rank by Subset",
        "top10_improved_none_to_100.png"
    )

    print("Finding least improved / worsening pairs (none -> 100)...")
    worst = least_improved_or_worse(rank_matrix, "none", "100")
    worst.to_csv(os.path.join(OUTDIR, "least_improved_or_worse_none_to_100.tsv"), sep="\t", index=False)
    plot_pair_rank_lines(
        worst,
        "Least Improved / Worst (none -> 100): Rank by Subset",
        "least_improved_or_worse_none_to_100.png"
    )

    print("Flagging pairs that worsen at every subset step (none->40->60->80->100)...")
    continuously_worse = flag_continuously_worsening(rank_matrix)
    continuously_worse.to_csv(os.path.join(OUTDIR, "continuously_worsening_pairs.tsv"), sep="\t", index=False)
    print(f"Found {len(continuously_worse)} pairs that worsen at every step.")

    print("Done.")
    print(f"Outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
