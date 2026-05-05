import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# CONFIG
# ======================
RESULTS_GLOB = "results_ranks_*"
GENE_RANKS_FILE = "edge_predictions.tsv"
CONFUSION_FILE = "confusion_summary.tsv"
RARE_DISEASE_FILE = "Rare Disease Annotation.csv"

OUTDIR = "comparison_figures/Perceptron"
os.makedirs(OUTDIR, exist_ok=True)

ORDERED_SUBSETS = ["none", "40", "60", "80", "100"]
TOP10 = 10
TOP50 = 50

ANNOT_COLS = [
    "Has Gene",
    "Has Gene with Ortholog",
    "Has Phenotype",
    "Has Genotype",
    "Has GO"
]

sns.set(style="whitegrid", font_scale=1.1)

# ======================
# HELPERS
# ======================
def get_subset(folder):
    return folder.replace(RESULTS_GLOB, "")


def enforce_subset_order(df):
    df["subset"] = pd.Categorical(df["subset"], ORDERED_SUBSETS, ordered=True)
    return df.sort_values("subset")


# ======================
# LOAD RESULTS
# ======================
def load_results():
    results = {}

    for folder in sorted(glob.glob(RESULTS_GLOB)):
        subset = get_subset(folder)
        print(folder)
        print(subset)

        gene_path = os.path.join(folder, GENE_RANKS_FILE)
        if not os.path.exists(gene_path):
            continue

        results[subset] = pd.read_csv(gene_path, sep="\t")
        print(results[subset].head())

    return results


# ======================
# BUILD COMBINED DF
# ======================
def build_combined_df(results):
    rows = []

    for subset, df in results.items():
        tmp = df.copy()
        print(subset)
        tmp["subset"] = subset
        rows.append(tmp)

    combined = pd.concat(rows, ignore_index=True)
    return enforce_subset_order(combined)


# ======================
# LOAD RARE DISEASE DATA
# ======================
def load_rare_annotations():
    rare = pd.read_csv(RARE_DISEASE_FILE)

    # convert to binary (safety)
    for col in ANNOT_COLS:
        rare[col] = rare[col].fillna(0).astype(int)

    return rare.set_index("Rare Disease")


# ======================
# MERGE + FILTER RARE
# ======================
def get_rare_df(combined, rare_map):
    print(combined)
    print(rare_map)
    df = combined.merge(
        rare_map,
        left_on="destinations",
        right_index=True,
        how="inner"   # only rare diseases
    )
    return df


# ======================
# ANALYSIS: PER-ANNOTATION PERFORMANCE
# ======================
def compute_annotation_performance(df):
    rows = []

    for subset in ORDERED_SUBSETS:
        print(subset, " in ordered subset")
        sub = df[df["subset"] == subset]

        for col in ANNOT_COLS:
            group = sub[sub[col] == 1]

            if len(group) == 0:
                continue

            rows.append({
                "subset": subset,
                "annotation": col,
                "n": len(group),
                "top10_frac": (group["rank"] <= TOP10).mean(),
                "top50_frac": (group["rank"] <= TOP50).mean(),
                "median_rank": group["rank"].median()
            })

    return pd.DataFrame(rows)


# ======================
# PLOTTING
# ======================
def plot_annotation_top10(df):
    plt.figure(figsize=(10,6))

    for ann in ANNOT_COLS:
        sub = df[df["annotation"] == ann]
        if sub.empty:
            continue

        plt.plot(
            sub["subset"],
            sub["top10_frac"],
            marker="o",
            label=ann
        )

    plt.xlabel("KG Subset")
    plt.ylabel("Top-10 Fraction")
    plt.title("Top-10 Performance by Annotation Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "annotation_top10.png"))
    plt.close()


def plot_annotation_top50(df):
    plt.figure(figsize=(10,6))

    for ann in ANNOT_COLS:
        sub = df[df["annotation"] == ann]
        if sub.empty:
            continue

        plt.plot(
            sub["subset"],
            sub["top50_frac"],
            marker="o",
            label=ann
        )

    plt.xlabel("KG Subset")
    plt.ylabel("Top-50 Fraction")
    plt.title("Top-50 Performance by Annotation Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "annotation_top50.png"))
    plt.close()


def plot_annotation_rank_distribution(df):
    plt.figure(figsize=(12,7))

    for ann in ANNOT_COLS:
        sub = df[df[ann] == 1]
        if sub.empty:
            continue

        plt.hist(
            sub["rank"],
            bins=100,
            alpha=0.4,
            label=ann,
            log=True
        )

    plt.xlabel("Rank")
    plt.ylabel("Count (log scale)")
    plt.title("Rank Distribution by Annotation Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "annotation_rank_distribution.png"))
    plt.close()


def plot_annotation_boxplot(df):
    plt.figure(figsize=(12,6))

    melted = df.melt(
        id_vars=["subset", "rank"],
        value_vars=ANNOT_COLS,
        var_name="annotation",
        value_name="present"
    )

    melted = melted[melted["present"] == 1]

    sns.boxplot(
        data=melted,
        x="annotation",
        y="rank",
        hue="subset",
        showfliers=False
    )

    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.title("Rank Distribution by Annotation Type and Subset")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "annotation_boxplot.png"))
    plt.close()


# ======================
# MAIN
# ======================
def main():
    print("Loading results...")
    results = load_results()

    print("Combining...")
    combined = build_combined_df(results)

    print("Loading rare disease annotations...")
    rare_map = load_rare_annotations()

    print("Filtering rare diseases...")
    rare_df = get_rare_df(combined, rare_map)

    print("Computing annotation performance...")
    ann_perf = compute_annotation_performance(rare_df)
    print(ann_perf)

    ann_perf.to_csv(
        os.path.join(OUTDIR, "annotation_performance.tsv"),
        sep="\t",
        index=False
    )

    print("Plotting...")
    plot_annotation_top10(ann_perf)
    plot_annotation_top50(ann_perf)
    plot_annotation_rank_distribution(rare_df)
    plot_annotation_boxplot(rare_df)

    print("Done.")
    print(f"Outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()