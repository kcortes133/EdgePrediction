import os
import glob

import matplotlib.colors
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_GLOB = "results_*_ranks"
ANNOTATION_FILE = "analysis/data/Rare Disease Annotation.csv"
OUTPUT_FILE = "analysis/data/rare_disease_analysis.tsv"

# -----------------------------
# LOAD ANNOTATIONS
# -----------------------------
ann = pd.read_csv(ANNOTATION_FILE)

# standardize column names
ann = ann.rename(columns={
    "Rare Disease": "disease"
})

# ensure string format
ann["disease"] = ann["disease"].astype(str)

# annotation columns
annotation_cols = [
    "Has Gene",
    "Has Gene with Ortholog",
    "Has Phenotype",
    "Has Genotype",
    "Has GO"
]

"""
# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def classify_rank(rank):
    if rank <= 10:
        return "Top10"
    elif rank <= 50:
        return "Top50"
    else:
        return "OutsideTop50"
        """

def classify_prediction(score):
    if float(score) >= 0.5:
        return "True Positive"
    else:
        return "False Negative"

def load_subset(folder):
    """Load gene_ranks and edge_predictions and merge."""
    gene_ranks_file = os.path.join(folder, "gene_ranks.tsv")
    edge_preds_file = os.path.join(folder, "edge_predictions.tsv")

    if not os.path.exists(gene_ranks_file):
        return None

    #gr = pd.read_csv(gene_ranks_file, sep="\t")
    ep = pd.read_csv(edge_preds_file, sep="\t")

    # standardize column names
    #gr = gr.rename(columns={"disease": "disease", "gene": "gene"})
    df = ep.rename(columns={"sources": "gene", "destinations": "disease"})

    # merge predictions onto ranks
    #df = pd.merge(gr, ep[["gene", "disease", "prediction"]],
    #              on=["gene", "disease"], how="left")
    print(df)

    return df


def summarize(df, subset_name):
    """Compute summary stats grouped by annotations."""
    disease_set = set(ann["disease"])

    def normalize_row(row):
        if row["disease"] in disease_set:
            return row["disease"], row["gene"]
        elif row["gene"] in disease_set:
            return row["gene"], row["disease"]
        else:
            return None, None

    normalized = df.apply(
        lambda r: normalize_row(r), axis=1, result_type="expand"
    )
    normalized.columns = ["disease_norm", "gene_norm"]

    df["disease"] = normalized["disease_norm"]
    df["gene"] = normalized["gene_norm"]
    # drop rows without a valid disease
    df = df.dropna(subset=["disease"])
    df = df.drop_duplicates()

    # merge with annotations
    df = pd.merge(df, ann, on="disease", how="inner")

    if df.empty:
        return pd.DataFrame()

    # classify ranks
    #df["rank_group"] = df["rank"].apply(classify_rank)
    df["Pred"] = df["prediction"].apply(classify_prediction)
    print(df["Pred"])


    results = []

    # loop over annotation columns
    for col in annotation_cols:
        grouped = df.groupby(col)

        for val, g in grouped:
            result = {
                "subset": subset_name,
                "annotation": col,
                "value": val,
                "n": len(g),
                #"mean_rank": g["rank"].mean(),
                ##"median_rank": g["rank"].median(),
                "median_prediction": g["prediction"].median(),
                #"Top10": (g["rank_group"] == "Top10").sum(),
                #"Top50": (g["rank_group"] == "Top50").sum(),
                #"OutsideTop50": (g["rank_group"] == "OutsideTop50").sum(),
                "TP": (g["Pred"] == "True Positive").sum(),
                "FN": (g["Pred"] == "False Negative").sum(),
            }
            results.append(result)

    return pd.DataFrame(results)


# -----------------------------
# MAIN LOOP
# -----------------------------
all_results = []

for folder in sorted(glob.glob(RESULTS_GLOB)):
    print(f"Processing {folder}...")

    df = load_subset(folder)
    if df is None:
        continue

    subset_name = os.path.basename(folder)

    summary_df = summarize(df, subset_name)
    if not summary_df.empty:
        all_results.append(summary_df)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Saved results to {OUTPUT_FILE}")
else:
    print("No results generated.")



import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = "analysis/data/rare_disease_analysis.tsv"
OUTPUT_DIR = "analysis/results/annotation_subset_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_FILE, sep="\t")

# Ensure correct types
df["subset"] = df["subset"].astype(str)
df["annotation"] = df["annotation"].astype(str)
df["value"] = df["value"].astype(int)

# Order subsets (important for plotting trends)
subset_order = sorted(df["subset"].unique())

annotation_types = df["annotation"].unique()

# -----------------------------
# HELPER: PLOT FUNCTION
# -----------------------------
def plot_metric(metric, ylabel, filename, log_scale=False):
    plt.figure()

    for annotation in annotation_types:
        for val in [0, 1]:
            subset_vals = []

            for subset in subset_order:
                sub = df[
                    (df["annotation"] == annotation) &
                    (df["value"] == val) &
                    (df["subset"] == subset)
                ]

                if len(sub) == 0:
                    subset_vals.append(None)
                else:
                    subset_vals.append(sub[metric].values[0])

            plt.plot(subset_order, subset_vals, marker='o',
                     label=f"{annotation}={val}")

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Subset")
    plt.ylabel(ylabel)
    plt.title(ylabel + " Across Subsets")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
"""
# -----------------------------
# 1. MEAN RANK TREND
# -----------------------------
plot_metric(
    metric="mean_rank",
    ylabel="Mean Rank",
    filename="mean_rank_trend.png",
    log_scale=True
)

# -----------------------------
# 2. MEDIAN RANK TREND
# -----------------------------
plot_metric(
    metric="median_rank",
    ylabel="Median Rank",
    filename="median_rank_trend.png",
    log_scale=True
)
"""
# -----------------------------
# 3. MEAN PREDICTION SCORE
# -----------------------------
plot_metric(
    metric="TP",
    ylabel="Mean Prediction Score",
    filename="mean_prediction_trend.png",
    log_scale=False
)
"""
# -----------------------------
# 4. TOP10 PROPORTION
# -----------------------------
def plot_topk(k_col, ylabel, filename):
    plt.figure()

    for annotation in annotation_types:
        for val in [0, 1]:
            vals = []

            for subset in subset_order:
                sub = df[
                    (df["annotation"] == annotation) &
                    (df["value"] == val) &
                    (df["subset"] == subset)
                ]

                if len(sub) == 0:
                    vals.append(None)
                else:
                    total = sub["n"].values[0]
                    if total == 0:
                        vals.append(None)
                    else:
                        vals.append(sub[k_col].values[0] / total)

            plt.plot(subset_order, vals, marker='o',
                     label=f"{annotation}={val}")

    plt.xlabel("Subset")
    plt.ylabel(ylabel)
    plt.title(ylabel + " Across Subsets")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# -----------------------------
# 5. TOP10 & TOP50
# -----------------------------
plot_topk("Top10", "Top10 Proportion", "top10_trend.png")
plot_topk("Top50", "Top50 Proportion", "top50_trend.png")

print(f"Plots saved to: {OUTPUT_DIR}")
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = "analysis/data/rare_disease_analysis.tsv"
OUTPUT_DIR = "analysis/results/heatmaps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_FILE, sep="\t")

df["subset"] = df["subset"].astype(str)
df["annotation"] = df["annotation"].astype(str)
df["value"] = df["value"].astype(int)

# Create combined label (row axis of heatmap)
df["group"] = df["annotation"] + " = " + df["value"].astype(str)

subset_order = ['results_none_ranks', 'results_40_ranks', 'results_60_ranks', 'results_80_ranks', 'results_100_ranks']


# -----------------------------
# HELPER FUNCTION
# -----------------------------
def make_heatmap(metric, title, filename, log_scale=False, normalize=False):
    pivot = df.pivot(index="group", columns="subset", values=metric)
    pivot = pivot.reindex(columns=subset_order)
    if normalize:
        pivot = pivot.apply(lambda row: row / row.max() if row.max() != 0 else row, axis=1)

    plt.figure(figsize=(10, 6))

    if log_scale:
        log_norm = matplotlib.colors.LogNorm(vmin=pivot.min().min(), vmax=pivot.max().max())
        sns.heatmap(pivot, norm=log_norm, cmap="viridis_r", annot=True, fmt=".2f")
        #plt.yscale("log")
    else:
        sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".3f")

    plt.title(title)
    plt.xlabel("Subset")
    plt.ylabel("Annotation Type")
    plt.show()
    #plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
"""
# -----------------------------
# 1. MEAN RANK HEATMAP
# -----------------------------
make_heatmap(
    metric="mean_rank",
    title="Mean Rank Across Subsets (Lower = Better)",
    filename="heatmap_mean_rank.png",
)
"""
# -----------------------------
# 2. MEDIAN RANK HEATMAP
# -----------------------------
make_heatmap(
    metric="median_prediction",
    title="Median Pred Across Subsets",
    filename="heatmap_median_rank.png",
    log_scale=False,
)

# -----------------------------
# 3. MEAN PREDICTION HEATMAP
# -----------------------------
df["TP_ratio"] = df["TP"]/ (df["TP"]+df["FN"])
make_heatmap(
    metric="TP_ratio",
    title="True Positive Ratio",
    filename="heatmap_prediction.png"
)

# -----------------------------
# 4. TOP10 PROPORTION HEATMAP
# -----------------------------
df["top10_ratio"] = df["Top10"] / df["n"]
df["top50_ratio"] = (df["Top50"] +df["Top10"])/ df["n"]

make_heatmap(
    metric="top10_ratio",
    title="Top10 Proportion Across Subsets",
    filename="heatmap_top10.png"
)

# -----------------------------
# 5. TOP50 PROPORTION HEATMAP
# -----------------------------
make_heatmap(
    metric="top50_ratio",
    title="Top50 Proportion Across Subsets",
    filename="heatmap_top50.png"
)

print(f"Heatmaps saved to: {OUTPUT_DIR}")