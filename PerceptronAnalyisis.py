import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------
RESULTS_GLOB = "results_*"
GENE_RANKS_FILE = "gene_ranks.tsv"
CONFUSION_FILE = "confusion_summary.tsv"
EDGE_PREDS_FILE = "edge_predictions.tsv"

RARE_DISEASE_FILE = "Rare Disease Annotation.csv"

OUTDIR = "comparison_figures/perceptron"
os.makedirs(OUTDIR, exist_ok=True)

TOP10_THRESHOLD = 10
TOP50_THRESHOLD = 50
# ==============================
# Load TP edges for uniqueness
# ==============================
TP_FILE = "TP_hgnc_mondo_edges.tsv"
tp_edges = pd.read_csv(TP_FILE, sep="\t")

# Frequency counts from TP set ONLY
tp_disease_counts = (
    tp_edges["object"]
    .value_counts()
    .rename("tp_disease_count")
)

tp_gene_counts = (
    tp_edges["subject"]
    .value_counts()
    .rename("tp_gene_count")
)


# -----------------------
# Helpers
# -----------------------
def load_results():
    """Load all result folders into a structured dict."""
    results = {}

    for folder in sorted(glob.glob(RESULTS_GLOB)):
        subset = folder.replace("results_", "")

        gene_ranks_path = os.path.join(folder, GENE_RANKS_FILE)
        conf_path = os.path.join(folder, CONFUSION_FILE)
        edge_path = os.path.join(folder, EDGE_PREDS_FILE)

        if not os.path.exists(gene_ranks_path):
            continue

        results[subset] = {
            "gene_ranks": pd.read_csv(gene_ranks_path, sep="\t"),
            "confusion": pd.read_csv(conf_path, sep="\t") if os.path.exists(conf_path) else None,
            "edges": pd.read_csv(edge_path, sep="\t") if os.path.exists(edge_path) else None,
        }

    return results


def classify_rank(rank):
    if rank <= TOP10_THRESHOLD:
        return "Top10"
    elif rank <= TOP50_THRESHOLD:
        return "Top50"
    else:
        return "OutsideTop50"


# -----------------------
# Load data
# -----------------------
results = load_results()
rare_df = pd.read_csv(RARE_DISEASE_FILE)

rare_diseases = set(rare_df["Rare Disease"].dropna())

# -----------------------
# Annotation stratification
# -----------------------
ANNOT_COLS = [
    "Has Gene",
    "Has Gene with Ortholog",
    "Has Phenotype",
    "Has Genotype",
    "Has GO"
]

rare_df["annotation_score"] = rare_df[ANNOT_COLS].sum(axis=1)

def annotation_bin(score):
    if score == 0:
        return "None"
    elif score <= 2:
        return "Low"
    elif score <= 4:
        return "Medium"
    else:
        return "High"

rare_df["annotation_bin"] = rare_df["annotation_score"].apply(annotation_bin)

rare_annotation_map = (
    rare_df
    .set_index("Rare Disease")[["annotation_score", "annotation_bin"]]
)

# -----------------------
# Aggregate metrics
# -----------------------
summary_rows = []

for subset, data in results.items():
    df = data["gene_ranks"].copy()
    df["rank_class"] = df["rank"].apply(classify_rank)

    summary_rows.append({
        "subset": subset,
        "Top10_frac": (df["rank"] <= 10).mean(),
        "Top50_frac": (df["rank"] <= 50).mean(),
        "median_rank": df["rank"].median(),
        "mean_rank": df["rank"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTDIR, "ranking_summary.tsv"), sep="\t", index=False)


# -----------------------
# FIGURE 1: Top-10 / Top-50 performance
# -----------------------
plt.figure()
plt.bar(summary_df["subset"], summary_df["Top10_frac"])
plt.ylabel("Fraction of TPs in Top-10")
plt.xlabel("KG Subset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "top10_fraction.png"))
plt.close()

plt.figure()
plt.bar(summary_df["subset"], summary_df["Top50_frac"])
plt.ylabel("Fraction of TPs in Top-50")
plt.xlabel("KG Subset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "top50_fraction.png"))
plt.close()


# -----------------------
# FIGURE 2: Rank distributions
# -----------------------
plt.figure()
for subset, data in results.items():
    plt.hist(
        data["gene_ranks"]["rank"],
        bins=100,
        alpha=0.5,
        label=subset,
        log=True
    )

plt.xlabel("Gene Rank")
plt.ylabel("Count (log scale)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rank_distribution.png"))
plt.close()


# -----------------------
# FIGURE 3: AUROC / AUPRC
# -----------------------
auroc_rows = []
for subset, data in results.items():
    if data["confusion"] is not None:
        row = data["confusion"].iloc[0]
        auroc_rows.append({
            "subset": subset,
            "auroc": row["auroc"],
            "auprc": row["auprc"]
        })

auroc_df = pd.DataFrame(auroc_rows)

plt.figure()
plt.plot(auroc_df["subset"], auroc_df["auroc"], marker="o", label="AUROC")
plt.plot(auroc_df["subset"], auroc_df["auprc"], marker="o", label="AUPRC")
plt.ylabel("Score")
plt.xlabel("KG Subset")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "auroc_auprc.png"))
plt.close()


# -----------------------
# RARE DISEASE ANALYSIS
# -----------------------
rare_summary = []

for subset, data in results.items():
    df = data["gene_ranks"]
    rare_df_subset = df[df["disease"].isin(rare_diseases)]

    rare_summary.append({
        "subset": subset,
        "n_rare": len(rare_df_subset),
        "rare_top10_frac": (rare_df_subset["rank"] <= 10).mean(),
        "rare_top50_frac": (rare_df_subset["rank"] <= 50).mean(),
        "rare_median_rank": rare_df_subset["rank"].median()
    })

rare_summary_df = pd.DataFrame(rare_summary)
rare_summary_df.to_csv(
    os.path.join(OUTDIR, "rare_disease_ranking_summary.tsv"),
    sep="\t",
    index=False
)

df = data["gene_ranks"].copy()

# Join annotation info
df = df.merge(
    rare_annotation_map,
    left_on="disease",
    right_index=True,
    how="left"
)

# Keep only rare diseases for stratified analysis
df_rare = df[df["annotation_bin"].notna()]

# -----------------------
# FIGURE 4: Rare disease ranking performance
# -----------------------
plt.figure()
plt.bar(
    rare_summary_df["subset"],
    rare_summary_df["rare_top10_frac"]
)
plt.ylabel("Rare diseases in Top-10")
plt.xlabel("KG Subset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rare_top10_fraction.png"))
plt.close()

plt.figure()
plt.bar(
    rare_summary_df["subset"],
    rare_summary_df["rare_top50_frac"]
)
plt.ylabel("Rare diseases in Top-50")
plt.xlabel("KG Subset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rare_top50_fraction.png"))
plt.close()

# -----------------------
# Stratified rare disease performance
# -----------------------
strat_rows = []

for subset, data in results.items():
    df = data["gene_ranks"].copy()
    df = df.merge(
        rare_annotation_map,
        left_on="disease",
        right_index=True,
        how="left"
    )

    df = df[df["annotation_bin"].notna()]

    for ann_bin, g in df.groupby("annotation_bin"):
        strat_rows.append({
            "subset": subset,
            "annotation_bin": ann_bin,
            "n": len(g),
            "top10_frac": (g["rank"] <= 10).mean(),
            "top50_frac": (g["rank"] <= 50).mean(),
            "median_rank": g["rank"].median()
        })

strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv(
    os.path.join(OUTDIR, "rare_disease_stratified_performance.tsv"),
    sep="\t",
    index=False
)
plt.figure()

for ann_bin in ["None", "Low", "Medium", "High"]:
    sub = strat_df[strat_df["annotation_bin"] == ann_bin]
    if not sub.empty:
        plt.plot(
            sub["subset"],
            sub["top10_frac"],
            marker="o",
            label=ann_bin
        )

plt.xlabel("KG Subset")
plt.ylabel("Top-10 Fraction")
plt.title("Rare Disease Gene Ranking vs Annotation Richness")
plt.xticks(rotation=45)
plt.legend(title="Annotation Level")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rare_annotation_stratified_top10.png"))
plt.close()

print("Analysis complete.")
print(f"Figures and tables written to: {OUTDIR}")
# -----------------------
# Per-disease rank delta
# -----------------------
rank_tables = []

for subset, data in results.items():
    df = data["gene_ranks"][["disease", "rank"]].copy()
    df["subset"] = subset
    rank_tables.append(df)

rank_long = pd.concat(rank_tables, ignore_index=True)

rank_wide = rank_long.pivot_table(
    index="disease",
    columns="subset",
    values="rank"
)
BASELINE = "none"   # change if needed

for col in rank_wide.columns:
    if col != BASELINE:
        rank_wide[f"delta_vs_{BASELINE}_{col}"] = (
            rank_wide[col] - rank_wide[BASELINE]
        )
rank_wide = rank_wide.merge(
    rare_annotation_map,
    left_index=True,
    right_index=True,
    how="left"
)

rank_wide.to_csv(
    os.path.join(OUTDIR, "per_disease_rank_deltas.tsv"),
    sep="\t"
)
DELTA_COL = "delta_vs_none_40"

delta_df = rank_wide[
    rank_wide["annotation_bin"].notna() &
    rank_wide[DELTA_COL].notna()
]

plt.figure()
for ann_bin in ["None", "Low", "Medium", "High"]:
    vals = delta_df[delta_df["annotation_bin"] == ann_bin][DELTA_COL]
    plt.boxplot(
        vals,
        positions=[["None","Low","Medium","High"].index(ann_bin)],
        widths=0.6
    )

plt.axhline(0, linestyle="--")
plt.xticks(
    range(4),
    ["None", "Low", "Medium", "High"]
)
plt.ylabel("Rank Δ (lower is better)")
plt.title("Effect of KG Subsetting on Rare Disease Gene Ranking")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rank_delta_by_annotation.png"))
plt.close()
# -----------------------
# Long-form delta table
# -----------------------
BASELINE = "none"

delta_rows = []

for subset in rank_wide.columns:
    if subset in ["annotation_score", "annotation_bin"]:
        continue
    if subset == BASELINE:
        continue
    if not subset.startswith("delta_vs"):
        continue

    # Extract subset name from column
    target_subset = subset.replace(f"delta_vs_{BASELINE}_", "")

    tmp = rank_wide[[
        subset,
        "annotation_bin"
    ]].copy()

    tmp = tmp.rename(columns={subset: "rank_delta"})
    tmp["subset"] = target_subset

    delta_rows.append(tmp)

delta_long = pd.concat(delta_rows, ignore_index=True)

delta_long.to_csv(
    os.path.join(OUTDIR, "rank_delta_long.tsv"),
    sep="\t",
    index=False
)
plt.figure()
def signed_log(x):
    return np.sign(x) * np.log10(np.abs(x) + 1)

order = sorted(delta_long["subset"].unique())

for ann_bin in ["None", "Low", "Medium", "High"]:
    sub = delta_long[delta_long["annotation_bin"] == ann_bin]

    if sub.empty:
        continue

    medians = (
        sub.groupby("subset")["rank_delta"]
        .median()
        .reindex(order)
    )

    medians = medians.apply(signed_log)

    plt.plot(
        order,
        medians,
        marker="o",
        label=ann_bin
    )

plt.axhline(0, linestyle="--")
plt.xlabel("KG Subset")
plt.ylabel("Median Rank Δ vs Full KG")
plt.title("Effect of KG Subsetting on Rare Disease Gene Ranking")
plt.xticks(rotation=45)
plt.legend(title="Annotation Richness")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rank_delta_trends_by_subset.png"))
plt.close()
plt.figure()

for ann_bin in ["None", "Low", "Medium", "High"]:
    sub = delta_long[delta_long["annotation_bin"] == ann_bin]
    if sub.empty:
        continue

    stats = (
        sub.groupby("subset")["rank_delta"]
        .agg(["mean", "sem"])
        .sort_index()
    )

    plt.errorbar(
        stats.index,
        stats["mean"],
        yerr=stats["sem"],
        marker="o",
        capsize=3,
        label=ann_bin
    )

plt.axhline(0, linestyle="--")
plt.xlabel("KG Subset")
plt.ylabel("Mean Rank Δ ± SEM")
plt.title("Rank Delta Trends Across KG Subsets")
plt.xticks(rotation=45)
plt.legend(title="Annotation Richness")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rank_delta_trends_mean_sem.png"))
plt.close()

# -----------------------
# Improved / worsened counts (rare diseases)
# -----------------------
BASELINE = "none"

count_rows = []

for col in rank_wide.columns:
    if not col.startswith(f"delta_vs_{BASELINE}_"):
        continue

    subset = col.replace(f"delta_vs_{BASELINE}_", "")

    sub = rank_wide[
        rank_wide["annotation_bin"].notna() &
        rank_wide[col].notna()
    ]

    count_rows.append({
        "subset": subset,
        "improved": (sub[col] < 0).sum(),
        "worsened": (sub[col] > 0).sum(),
        "unchanged": (sub[col] == 0).sum(),
        "total": len(sub)
    })

count_df = pd.DataFrame(count_rows).sort_values("subset")
count_df.to_csv(
    os.path.join(OUTDIR, "rare_disease_improve_worsen_counts.tsv"),
    sep="\t",
    index=False
)

plt.figure()
plt.bar(
    count_df["subset"],
    count_df["improved"]
)

plt.xlabel("KG Subset")
plt.ylabel("Number of Rare Diseases")
plt.title("Rare Diseases with Improved Gene Rank After KG Subsetting")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rare_diseases_improved_counts.png"))
plt.close()
plt.figure()
plt.bar(
    count_df["subset"],
    count_df["worsened"]
)

plt.xlabel("KG Subset")
plt.ylabel("Number of Rare Diseases")
plt.title("Rare Diseases with Worse Gene Rank After KG Subsetting")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rare_diseases_worsened_counts.png"))
plt.close()
# -----------------------
# FIGURE: Net rank improvement
# -----------------------
count_df["net_improvement"] = count_df["improved"] - count_df["worsened"]

plt.figure()
plt.bar(
    count_df["subset"],
    count_df["net_improvement"]
)

plt.axhline(0, linestyle="--")
plt.xlabel("KG Subset")
plt.ylabel("Net # Rare Diseases (Improved − Worsened)")
plt.title("Net Effect of KG Subsetting on Rare Disease Gene Ranking")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rare_disease_net_improvement.png"))
plt.close()


# -----------------------
# FIGURE: Diverging improved vs worsened
# -----------------------
plt.figure()

plt.bar(
    count_df["subset"],
    count_df["improved"],
    label="Improved"
)

plt.bar(
    count_df["subset"],
    -count_df["worsened"],
    label="Worsened"
)

plt.axhline(0, linestyle="--")
plt.xlabel("KG Subset")
plt.ylabel("Number of Rare Diseases")
plt.title("Rare Disease Rank Changes Under KG Subsetting")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rare_disease_diverging_improve_worsen.png"))
plt.close()


import seaborn as sns

# ----------------------
# Combine results_* into one dataframe
# ----------------------
ORDERED_SUBSETS = ["none", "40", "60", "80", "100"]

combined_rows = []

for subset in ORDERED_SUBSETS:
    if subset not in results:
        print(f"WARNING: results_{subset} not found, skipping")
        continue

    df = results[subset]["gene_ranks"].copy()
    df["subset"] = subset
    df["Top10_flag"] = df["rank"] <= 10
    df["TopN"] = np.where(
        df["rank"] <= 10, "Top10",
        np.where(df["rank"] <= 50, "Top50", "OutsideTop50")
    )

    combined_rows.append(df)

combined_df = pd.concat(combined_rows, ignore_index=True)
# ==============================
# Attach TP-based uniqueness labels
# ==============================

combined_df = combined_df.merge(
    tp_disease_counts,
    left_on="disease",
    right_index=True,
    how="left"
)

combined_df = combined_df.merge(
    tp_gene_counts,
    left_on="gene",
    right_index=True,
    how="left"
)

# Fill NaNs (should not happen for TP rows, but safe)
combined_df[["tp_disease_count", "tp_gene_count"]] = (
    combined_df[["tp_disease_count", "tp_gene_count"]]
    .fillna(0)
)

combined_df["disease_uniqueness"] = np.where(
    combined_df["tp_disease_count"] > 1,
    "Non-unique disease (TP)",
    "Unique disease (TP)"
)

combined_df["gene_uniqueness"] = np.where(
    combined_df["tp_gene_count"] > 1,
    "Non-unique gene (TP)",
    "Unique gene (TP)"
)

# enforce categorical order (CRITICAL for plotting order)
combined_df["subset"] = pd.Categorical(
    combined_df["subset"],
    categories=ORDERED_SUBSETS,
    ordered=True
)

# ----------------------
# Plotting
# ----------------------
sns.set(style="whitegrid", font_scale=1.1)

palette = dict(zip(
    ORDERED_SUBSETS,
    sns.color_palette("tab10", n_colors=len(ORDERED_SUBSETS))
))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# ---- (1) Histogram of ranks ----
sns.histplot(
    data=combined_df,
    x="rank",
    hue="subset",
    bins=30,
    element="step",
    common_norm=False,
    palette=palette,
    ax=axes[0]
)
axes[0].set_title("Histogram of Gene Prediction Ranks")
axes[0].set_xlabel("Rank of True Gene")
axes[0].set_ylabel("Count")

# ---- (2) CDF of ranks ----
sns.ecdfplot(
    data=combined_df,
    x="rank",
    hue="subset",
    palette=palette,
    ax=axes[1]
)
axes[1].set_title("CDF of Gene Prediction Ranks")
axes[1].set_xlabel("Rank of True Gene")
axes[1].set_ylabel("Cumulative Probability")

# ---- (3) Score vs Rank (Top10 highlighted) ----
for subset in ORDERED_SUBSETS:
    sub = combined_df[combined_df["subset"] == subset]

    # Non-Top10
    axes[2].scatter(
        sub[~sub["Top10_flag"]]["rank"],
        sub[~sub["Top10_flag"]]["score"],
        alpha=0.5,
        color=palette[subset],
        label=subset
    )

    # Top10 highlighted
    axes[2].scatter(
        sub[sub["Top10_flag"]]["rank"],
        sub[sub["Top10_flag"]]["score"],
        s=70,
        edgecolor="black",
        linewidth=1.3,
        facecolor=palette[subset],
        label=f"{subset} Top10"
    )

axes[2].set_title("Prediction Score vs True Gene Rank (Top10 Highlighted)")
axes[2].set_xlabel("Rank of True Gene")
axes[2].set_ylabel("Prediction Score")
axes[2].legend(fontsize=9)

# ---- (4) Top-N counts ----
sns.countplot(
    data=combined_df,
    x="TopN",
    hue="subset",
    order=["Top10", "Top50", "OutsideTop50"],
    hue_order=ORDERED_SUBSETS,
    palette=palette,
    ax=axes[3]
)
axes[3].set_title("Number of Predictions by Top-N Category")
axes[3].set_xlabel("Top-N Category")
axes[3].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rank_comparison.png"), dpi=300)
plt.show()

# ==============================
# NEW FIGURE: FP + Rank–Score
# ==============================
# ----------------------
# Per-disease FP analysis
# ----------------------
disease_fp = (
    combined_df
    .assign(FalsePositive = combined_df["rank"] > TOP50_THRESHOLD)
    .groupby(["subset", "disease"])["FalsePositive"]
    .mean()
    .reset_index()
)

disease_fp["subset"] = pd.Categorical(
    disease_fp["subset"],
    categories=ORDERED_SUBSETS,
    ordered=True
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ---- (A) Per-disease FP boxplots ----
sns.boxplot(
    data=disease_fp,
    x="subset",
    y="FalsePositive",
    order=ORDERED_SUBSETS,
    palette=palette,
    showfliers=False,
    ax=axes[0]
)

axes[0].set_title("Per-Disease False Positive Rate")
axes[0].set_xlabel("Training Data Removal (%)")
axes[0].set_ylabel("Fraction of Runs Where\nTrue Gene Outside Top50")
axes[0].set_ylim(0, 1)

# ---- (B) Rank vs score ----
for subset in ORDERED_SUBSETS:
    sub = combined_df[combined_df["subset"] == subset]

    axes[1].scatter(
        sub["rank"],
        sub["score"],
        alpha=0.4,
        s=25,
        color=palette[subset],
        label=subset
    )

axes[1].axvline(10, linestyle="--", color="black", linewidth=1)
axes[1].axvline(50, linestyle=":", color="black", linewidth=1)

axes[1].set_xscale("log")
axes[1].set_title("Prediction Score vs True Gene Rank")
axes[1].set_xlabel("Rank of True Gene (log scale)")
axes[1].set_ylabel("Prediction Score")
axes[1].legend(title="Subset", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "false_positives_and_rank_score.png"), dpi=300)
plt.show()
# ----------------------
# Configuration
# ----------------------
ORDERED_SUBSETS = ["none", "40", "60", "80", "100"]
TOP50_THRESHOLD = 50

sns.set(style="whitegrid", font_scale=1.1)

palette = dict(zip(
    ORDERED_SUBSETS,
    sns.color_palette("tab10", len(ORDERED_SUBSETS))
))

# enforce subset order
combined_df["subset"] = pd.Categorical(
    combined_df["subset"],
    categories=ORDERED_SUBSETS,
    ordered=True
)

# ----------------------
# Per-disease false positives
# ----------------------
disease_fp = (
    combined_df
    .assign(FalsePositive=combined_df["rank"] > TOP50_THRESHOLD)
    .groupby(["subset", "disease"])["FalsePositive"]
    .mean()
    .reset_index()
)

# ----------------------
# Standalone multi-panel figure
# ----------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ---- (A) Rank distribution (boxplots) ----
sns.boxplot(
    data=combined_df,
    x="subset",
    y="rank",
    order=ORDERED_SUBSETS,
    palette=palette,
    showfliers=False,
    ax=axes[0]
)

axes[0].set_yscale("log")
axes[0].set_title("Distribution of True Gene Rank")
axes[0].set_xlabel("Training Data Removal (%)")
axes[0].set_ylabel("Rank (log scale)")

# ---- (B) Per-disease false positive rate ----
sns.boxplot(
    data=disease_fp,
    x="subset",
    y="FalsePositive",
    order=ORDERED_SUBSETS,
    palette=palette,
    showfliers=False,
    ax=axes[1]
)

axes[1].set_title("Per-Disease False Positive Rate")
axes[1].set_xlabel("Training Data Removal (%)")
axes[1].set_ylabel("Fraction of Diseases\n(True Gene Outside Top50)")
axes[1].set_ylim(0, 1)

# ---- (C) Rank vs score ----
for subset in ORDERED_SUBSETS:
    sub = combined_df[combined_df["subset"] == subset]
    axes[2].scatter(
        sub["rank"],
        sub["score"],
        alpha=0.4,
        s=25,
        color=palette[subset],
        label=subset
    )

axes[2].axvline(10, linestyle="--", color="black", linewidth=1)
axes[2].axvline(50, linestyle=":", color="black", linewidth=1)

axes[2].set_xscale("log")
axes[2].set_title("Prediction Score vs True Gene Rank")
axes[2].set_xlabel("Rank of True Gene (log scale)")
axes[2].set_ylabel("Prediction Score")
axes[2].legend(title="Subset", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "standalone_fp_rank_score_boxplots.png"), dpi=300)
plt.show()
import os
import glob
import pandas as pd
import numpy as np

# ----------------------
# Configuration
# ----------------------
RESULTS_GLOB = "results_*"
CONFUSION_FILE = "confusion_summary.tsv"
OUTDIR = "comparison_figures"
os.makedirs(OUTDIR, exist_ok=True)

ORDERED_SUBSETS = ["none", "40", "60", "80", "100"]

# ----------------------
# Load + aggregate
# ----------------------
rows = []

for folder in sorted(glob.glob(RESULTS_GLOB)):
    subset = folder.replace("results_", "")
    if subset not in ORDERED_SUBSETS:
        continue

    path = os.path.join(folder, CONFUSION_FILE)
    if not os.path.exists(path):
        print(f"WARNING: missing {path}")
        continue

    df = pd.read_csv(path, sep="\t")

    # Expect exactly one row per file
    row = df.iloc[0].to_dict()
    row["subset"] = subset
    rows.append(row)

summary_df = pd.DataFrame(rows)

# enforce subset order
summary_df["subset"] = pd.Categorical(
    summary_df["subset"],
    categories=ORDERED_SUBSETS,
    ordered=True
)
summary_df = summary_df.sort_values("subset")

# ----------------------
# Derived metrics
# ----------------------
summary_df["Precision"] = summary_df["TP"] / (summary_df["TP"] + summary_df["FP"])
summary_df["Recall"] = summary_df["TP"] / (summary_df["TP"] + summary_df["FN"])
summary_df["Specificity"] = summary_df["TN"] / (summary_df["TN"] + summary_df["FP"])
summary_df["Accuracy"] = (
    summary_df["TP"] + summary_df["TN"]
) / (
    summary_df[["TP", "FP", "FN", "TN"]].sum(axis=1)
)
summary_df["F1"] = 2 * (
    summary_df["Precision"] * summary_df["Recall"]
) / (
    summary_df["Precision"] + summary_df["Recall"]
)

# ----------------------
# Column order (paper-friendly)
# ----------------------
final_cols = [
    "subset",
    "TP", "FP", "TN", "FN",
    "Precision", "Recall", "Specificity", "Accuracy", "F1",
    "auroc", "auprc", "threshold"
]

summary_df = summary_df[final_cols]

# ----------------------
# Save
# ----------------------
out_path = os.path.join(OUTDIR, "confusion_summary_aggregated.tsv")
summary_df.to_csv(out_path, sep="\t", index=False)

print(f"Saved aggregated confusion summary to: {out_path}")
print(summary_df.round(4))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------
# Configuration
# ----------------------
SUMMARY_FILE = "comparison_figures/confusion_summary_aggregated.tsv"
OUTDIR = "comparison_figures"
ORDERED_SUBSETS = ["none", "40", "60", "80", "100"]

sns.set(style="whitegrid", font_scale=1.1)

# ----------------------
# Load data
# ----------------------
df = pd.read_csv(SUMMARY_FILE, sep="\t")

df["subset"] = pd.Categorical(
    df["subset"],
    categories=ORDERED_SUBSETS,
    ordered=True
)
df = df.sort_values("subset")

# ----------------------
# Long format for plotting
# ----------------------
conf_long = df.melt(
    id_vars="subset",
    value_vars=["TP", "FP", "FN", "TN"],
    var_name="Outcome",
    value_name="Count"
)

# consistent outcome order
conf_long["Outcome"] = pd.Categorical(
    conf_long["Outcome"],
    categories=["TP", "FP", "FN", "TN"],
    ordered=True
)

# ----------------------
# Plot
# ----------------------
plt.figure(figsize=(11, 6))

sns.barplot(
    data=conf_long,
    x="subset",
    y="Count",
    hue="Outcome"
)

plt.title("Confusion Matrix Components by Training Subset")
plt.xlabel("Training Data Removal (%)")
plt.ylabel("Count")
plt.legend(title="Outcome")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "confusion_matrix_by_subset.png"), dpi=300)
plt.show()
tp_disease_perf = (
    combined_df
    .groupby(["subset", "disease_uniqueness"])
    .agg(
        n=("rank", "size"),
        top10_frac=("rank", lambda x: (x <= 10).mean()),
        top50_frac=("rank", lambda x: (x <= 50).mean()),
        median_rank=("rank", "median")
    )
    .reset_index()
)

tp_disease_perf.to_csv(
    os.path.join(OUTDIR, "tp_unique_vs_nonunique_disease_performance.tsv"),
    sep="\t",
    index=False
)
tp_gene_perf = (
    combined_df
    .groupby(["subset", "gene_uniqueness"])
    .agg(
        n=("rank", "size"),
        top10_frac=("rank", lambda x: (x <= 10).mean()),
        top50_frac=("rank", lambda x: (x <= 50).mean()),
        median_rank=("rank", "median")
    )
    .reset_index()
)

tp_gene_perf.to_csv(
    os.path.join(OUTDIR, "tp_unique_vs_nonunique_gene_performance.tsv"),
    sep="\t",
    index=False
)
