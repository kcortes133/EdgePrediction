import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

############################################
# USER PARAMETERS
############################################
ROOT_DIR = "."
HMI_FILENAME = "HMI_top.csv"
TOP10_FILENAME = "top10results.csv"
RARE_FILE = "Testing Subset Rare.csv"

############################################
# HELPERS
############################################
def extract_rank(classification):
    if pd.isna(classification):
        return np.nan
    m = re.search(r"rank=(\d+)", classification)
    return int(m.group(1)) if m else np.nan

############################################
# LOAD ALL APPROACHES
############################################
hmi_dfs = []
conf_dfs = []

for folder in sorted(os.listdir(ROOT_DIR)):
    path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(path):
        continue

    hmi_path = os.path.join(path, HMI_FILENAME)
    conf_path = os.path.join(path, TOP10_FILENAME)

    if os.path.exists(hmi_path):
        df = pd.read_csv(hmi_path)
        df["Approach"] = folder
        df["rank"] = df["classification"].apply(extract_rank)
        df["Top10"] = df["classification"].str.contains("Top10", na=False)
        hmi_dfs.append(df)

    if os.path.exists(conf_path):
        cdf = pd.read_csv(conf_path)
        cdf["Approach"] = folder
        conf_dfs.append(cdf)

combined_df = pd.concat(hmi_dfs, ignore_index=True)
confusion_df = pd.concat(conf_dfs, ignore_index=True)

palette = sns.color_palette("tab10", combined_df["Approach"].nunique())
color_map = dict(zip(combined_df["Approach"].unique(), palette))

############################################
# ========== FIGURE 1: OVERALL ==========
############################################
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (A) Rank histogram
sns.histplot(
    data=combined_df,
    x="rank",
    hue="Approach",
    bins=50,
    element="step",
    common_norm=False,
    ax=axes[0,0]
)
axes[0,0].set_title("A. True Gene Rank Distribution")

# (B) Top-10 recovery rate
top10_rate = (
    combined_df.groupby("Approach")["Top10"]
    .mean()
    .reset_index()
)

sns.barplot(
    data=top10_rate,
    x="Approach",
    y="Top10",
    palette=color_map,
    ax=axes[0,1]
)
axes[0,1].set_title("B. Top-10 True Gene Recovery Rate")
axes[0,1].set_ylim(0,1)

# (C) Score vs Rank
for approach, sub in combined_df.groupby("Approach"):
    axes[1,0].scatter(
        sub.loc[~sub["Top10"], "rank"],
        sub.loc[~sub["Top10"], "score"],
        alpha=0.4,
        color=color_map[approach],
        label=approach
    )
    axes[1,0].scatter(
        sub.loc[sub["Top10"], "rank"],
        sub.loc[sub["Top10"], "score"],
        s=120,
        edgecolor="black",
        linewidth=1.2,
        color=color_map[approach]
    )

axes[1,0].set_title("C. Prediction Score vs True Rank")
axes[1,0].set_xlabel("Rank")
axes[1,0].set_ylabel("Score")

# (D) Rank CDF
for approach, sub in combined_df.groupby("Approach"):
    ranks = np.sort(sub["rank"].dropna())
    axes[1,1].plot(
        ranks,
        np.arange(len(ranks)) / len(ranks),
        label=approach,
        color=color_map[approach]
    )

axes[1,1].set_title("D. CDF of True Gene Rank")
axes[1,1].set_xlabel("Rank")
axes[1,1].set_ylabel("Fraction ≤ Rank")
axes[1,1].legend()

plt.tight_layout()
plt.show()

############################################
# LOAD RARE DISEASE ANNOTATIONS
############################################
rare_df = pd.read_csv(RARE_FILE).rename(columns={"Rare Disease": "disease"})
annotation_cols = [
    "Has Gene",
    "Has Gene with Ortholog",
    "Has Phenotype",
    "Has Genotype",
    "Has GO"
]
rare_df[annotation_cols] = rare_df[annotation_cols].astype(int)

rare_preds = combined_df.merge(rare_df, on="disease", how="inner")

############################################
# RARE DISEASE METRICS
############################################
rare_summary = []
annotation_results = []

for approach, sub in rare_preds.groupby("Approach"):
    tp = sub["Top10"].sum()
    fn = rare_df.shape[0] - tp
    rare_summary.append({
        "Approach": approach,
        "Recall": tp / (tp + fn)
    })

for ann in annotation_cols:
    total = rare_df[rare_df[ann] == 1].shape[0]
    for approach, sub in rare_preds[rare_preds[ann] == 1].groupby("Approach"):
        tp = sub["Top10"].sum()
        fn = total - tp
        annotation_results.append({
            "Annotation": ann,
            "Approach": approach,
            "Recall": tp / (tp + fn)
        })

rare_summary_df = pd.DataFrame(rare_summary)
annotation_perf_df = pd.DataFrame(annotation_results)

############################################
# ========== FIGURE 2: RARE DISEASE ==========
############################################
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (A) Rare disease recall
sns.barplot(
    data=rare_summary_df,
    x="Approach",
    y="Recall",
    palette=color_map,
    ax=axes[0,0]
)
axes[0,0].set_title("A. Rare Disease Recall")
#axes[0,0].set_ylim(0,1)

# (B) Annotation recall
sns.barplot(
    data=annotation_perf_df,
    x="Annotation",
    y="Recall",
    hue="Approach",
    ax=axes[0,1]
)
axes[0,1].set_title("B. Recall by Annotation Type")
#axes[0,1].set_ylim(0,1)
axes[0,1].tick_params(axis="x", rotation=30)

# (C) Precision–Recall (annotation)
annotation_perf_df["Precision"] = annotation_perf_df["Recall"]  # TP/(TP+FN)
sns.scatterplot(
    data=annotation_perf_df,
    x="Recall",
    y="Precision",
    hue="Annotation",
    style="Approach",
    s=120,
    ax=axes[1,0]
)
axes[1,0].set_title("C. Annotation-Level Precision–Recall")
#axes[1,0].set_xlim(0,1)
#axes[1,0].set_ylim(0,1)

# (D) Annotation coverage
coverage = rare_df[annotation_cols].mean().reset_index()
coverage.columns = ["Annotation", "Coverage"]

sns.barplot(
    data=coverage,
    x="Annotation",
    y="Coverage",
    ax=axes[1,1]
)
axes[1,1].set_title("D. Annotation Coverage in Rare Diseases")
axes[1,1].set_ylim(0,1)
axes[1,1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_confusion_matrices_from_folders(
    root_dir,
    filename="top10results.csv",
    normalize=False,
    plot=True
):
    """
    Compare confusion matrices across all subfolders containing a
    top10results CSV file.

    Parameters
    ----------
    root_dir : str
        Parent directory containing approach subfolders.
    filename : str
        Name of top10results file (default matches your pipeline).
    normalize : bool
        If True, normalize counts to proportions per approach.
    plot : bool
        If True, generate comparison bar plot.

    Returns
    -------
    confusion_df : pd.DataFrame
        Tidy dataframe with TP / FP / FN / TN per approach.
    """

    records = []

    for folder in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path):
            continue

        file_path = os.path.join(path, filename)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        counts = df["classification"].value_counts()

        for label in ["TP", "FP", "FN", "TN"]:
            records.append({
                "Approach": folder,
                "Class": label,
                "Count": counts.get(label, 0)
            })

    if not records:
        raise RuntimeError("No top10results files found.")

    confusion_df = pd.DataFrame(records)

    if normalize:
        totals = confusion_df.groupby("Approach")["Count"].transform("sum")
        confusion_df["Value"] = confusion_df["Count"] / totals
    else:
        confusion_df["Value"] = confusion_df["Count"]

    if plot:
        plt.figure(figsize=(10,6))
        sns.barplot(
            data=confusion_df,
            x="Class",
            y="Value",
            hue="Approach"
        )
        ylabel = "Proportion" if normalize else "Count"
        plt.ylabel(ylabel)
        plt.title("Confusion Matrix Comparison Across Prediction Approaches")
        plt.tight_layout()
        plt.show()

    return confusion_df


conf_df = compare_confusion_matrices_from_folders(
    root_dir=".",
    normalize=True
)
