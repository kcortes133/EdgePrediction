#!/usr/bin/env python3
"""
Refreshed F1 / classification-metric comparison.

Produces THREE separate comparison figure sets:
  1. standard_refresh/      - analysis/results/results_{40,60,80,100,none} + ModelOrgs
  2. rank_based_refresh/    - results_{40,60,80,100,none}_ranks (rank-based negative sampling)
  3. random_baseline_refresh/ - the 3 random-embedding perceptron runs (IC80 only),
                                compared to each other

Groups 1 and 2 each also get the random baseline drawn in as a reference
line (mean +/- std across the 3 random seeds) so real models can be judged
against a null model.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]

STANDARD_DIRS = {
    "None": BASE_DIR / "analysis/results/results_none",
    "40":   BASE_DIR / "analysis/results/results_40",
    "60":   BASE_DIR / "analysis/results/results_60",
    "80":   BASE_DIR / "analysis/results/results_80",
    "100":  BASE_DIR / "analysis/results/results_100",
}

RANK_DIRS = {
    "IC40_ranks":    BASE_DIR / "results_40_ranks",
    "IC60_ranks":    BASE_DIR / "results_60_ranks",
    "IC80_ranks":    BASE_DIR / "results_80_ranks",
    "IC100_ranks":   BASE_DIR / "results_100_ranks",
    "IC_none_ranks": BASE_DIR / "results_none_ranks",
}

RANDOM_DIRS = {
    "Random_seed1":  BASE_DIR / "monarch_percep_rand_80",
    "Random_seed7":  BASE_DIR / "monarch_percep_rand7_80",
    "Random_seed13": BASE_DIR / "monarch_percep_rand13_80",
}

OUT_ROOT = BASE_DIR / "analysis/results/comparison_figures"

REF_METRICS = ["F1", "Accuracy", "MCC", "Precision", "Recall"]


# ======================
# METRICS
# ======================
def safe_div(num, denom):
    return np.where(denom == 0, np.nan, num / denom)


def compute_metrics(df):
    df = df.copy()
    TP, FP, TN, FN = df["TP"], df["FP"], df["TN"], df["FN"]

    df["Precision"]   = safe_div(TP, TP + FP)
    df["Recall"]      = safe_div(TP, TP + FN)
    df["Specificity"] = safe_div(TN, TN + FP)
    df["Accuracy"]    = safe_div(TP + TN, TP + TN + FP + FN)

    df["F1"] = safe_div(
        2 * df["Precision"] * df["Recall"],
        df["Precision"] + df["Recall"]
    )

    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    df["MCC"] = safe_div(numerator, denominator)

    return df


def add_uncertainty(df):
    df = df.copy()

    def binomial_se(p, n):
        return np.sqrt((p * (1 - p)) / n)

    df["Precision_se"] = binomial_se(df["Precision"], df["TP"] + df["FP"])
    df["Recall_se"]    = binomial_se(df["Recall"], df["TP"] + df["FN"])
    df["Accuracy_se"]  = binomial_se(df["Accuracy"], df["TP"] + df["TN"] + df["FP"] + df["FN"])

    return df


# ======================
# LOAD
# ======================
def load_group(dirs):
    rows = []
    for subset_label, folder in dirs.items():
        f = folder / "confusion_summary.tsv"
        if not f.exists():
            print(f"  [WARN] missing {f}, skipping")
            continue
        df = pd.read_csv(f, sep="\t")
        row = df.iloc[0].to_dict()
        row["subset"] = subset_label
        rows.append(row)

    if not rows:
        return None

    gdf = pd.DataFrame(rows)
    gdf["subset"] = pd.Categorical(gdf["subset"], categories=list(dirs.keys()), ordered=True)
    gdf = gdf.sort_values("subset")
    gdf = compute_metrics(gdf)
    gdf = add_uncertainty(gdf)
    return gdf


# ======================
# PLOTTING
# ======================
def plot_metric(df, metric, ylabel, outpath, random_ref=None):
    plt.figure(figsize=(8, 5))

    x = df["subset"].astype(str)
    y = df[metric]

    se_col = metric + "_se"
    if se_col in df.columns:
        plt.errorbar(x, y, yerr=df[se_col], marker="o", capsize=4, label="Model")
    else:
        plt.plot(x, y, marker="o", label="Model")

    if random_ref is not None and metric in random_ref:
        mean, std = random_ref[metric]
        plt.axhline(mean, color="red", linestyle="--", linewidth=2, label="Random baseline (mean, IC80, n=3 seeds)")
        plt.axhspan(mean - std, mean + std, color="red", alpha=0.15)

    plt.xlabel("IC Threshold")
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs IC Threshold")
    plt.xticks(rotation=0, ha="center")
    if random_ref is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_precision_recall(df, outpath, random_ref=None):
    plt.figure(figsize=(8, 5))

    x = df["subset"].astype(str)

    plt.errorbar(x, df["Precision"], yerr=df["Precision_se"], marker="o", label="Precision", capsize=4)
    plt.errorbar(x, df["Recall"], yerr=df["Recall_se"], marker="o", label="Recall", capsize=4)

    if random_ref is not None:
        if "Precision" in random_ref:
            mean, std = random_ref["Precision"]
            plt.axhline(mean, color="tab:blue", linestyle="--", alpha=0.6, label="Random Precision")
        if "Recall" in random_ref:
            mean, std = random_ref["Recall"]
            plt.axhline(mean, color="tab:orange", linestyle="--", alpha=0.6, label="Random Recall")

    plt.xlabel("Subset")
    plt.ylabel("Score")
    plt.title("Precision vs Recall")
    plt.legend()
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ======================
# GROUP RUNNER
# ======================
def run_group(dirs, out_subdir, random_ref=None):
    print(f"\n=== {out_subdir} ===")
    gdf = load_group(dirs)
    if gdf is None:
        print("  No data found, skipping group.")
        return None

    outdir = OUT_ROOT / out_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    gdf.to_csv(outdir / "metrics.tsv", sep="\t", index=False)

    plot_metric(gdf, "F1", "F1 Score", outdir / "f1.png", random_ref=random_ref)
    plot_metric(gdf, "Accuracy", "Accuracy", outdir / "accuracy.png", random_ref=random_ref)
    plot_metric(gdf, "MCC", "MCC", outdir / "mcc.png", random_ref=random_ref)
    plot_precision_recall(gdf, outdir / "precision_recall.png", random_ref=random_ref)

    print(f"  {len(gdf)} subsets -> {outdir}")
    print(gdf[["subset", "Precision", "Recall", "F1", "MCC", "Accuracy"]].to_string(index=False))
    return gdf


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Random baseline runs, compared to each other
    random_df = run_group(RANDOM_DIRS, "random_baseline_refresh", random_ref=None)

    random_ref = None
    if random_df is not None and len(random_df) > 0:
        random_ref = {}
        for m in REF_METRICS:
            vals = random_df[m].astype(float)
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            random_ref[m] = (mean, std)
        print("\nRandom baseline reference (mean +/- std across seeds):")
        for m, (mean, std) in random_ref.items():
            print(f"  {m}: {mean:.4f} +/- {std:.4f}")

    # 2) Standard (non-ranked negative sampling) results, with random baseline overlaid
    run_group(STANDARD_DIRS, "standard_refresh", random_ref=random_ref)

    # 3) Rank-based negative sampling results, with random baseline overlaid
    run_group(RANK_DIRS, "rank_based_refresh", random_ref=random_ref)

    print(f"\nDone. Outputs under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
