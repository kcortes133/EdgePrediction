#!/usr/bin/env python3

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    df["MCC"] = safe_div(numerator, denominator)

    return df


# ======================
# UNCERTAINTY (BINOMIAL SE)
# ======================
def add_uncertainty(df):
    df = df.copy()

    def binomial_se(p, n):
        return np.sqrt((p * (1 - p)) / n)

    df["Precision_se"] = binomial_se(df["Precision"], df["TP"] + df["FP"])
    df["Recall_se"]    = binomial_se(df["Recall"], df["TP"] + df["FN"])
    df["Accuracy_se"]  = binomial_se(df["Accuracy"], df["TP"] + df["TN"] + df["FP"] + df["FN"])

    return df


# ======================
# LOAD FILES
# ======================
def load_confusion_files(input_path):
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "**/confusion_summary.tsv"), recursive=True)
    else:
        files = [input_path]

    rows = []

    for f in files:
        df = pd.read_csv(f, sep="\t")

        if not {"TP","FP","TN","FN"}.issubset(df.columns):
            continue

        row = df.iloc[0].to_dict()
        subset = os.path.basename(os.path.dirname(f))
        row["subset"] = subset
        rows.append(row)

    return pd.DataFrame(rows)


# ======================
# PLOTTING
# ======================
def plot_metric(df, metric, ylabel, outpath):
    plt.figure()

    x = df["subset"]
    y = df[metric]

    if metric + "_se" in df.columns:
        plt.errorbar(x, y, yerr=df[metric + "_se"], marker="o", capsize=4)
    else:
        plt.plot(x, y, marker="o")

    plt.xlabel("Subset")
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs Subset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_precision_recall(df, outpath):
    plt.figure()

    plt.errorbar(
        df["subset"], df["Precision"],
        yerr=df["Precision_se"],
        marker="o",
        label="Precision",
        capsize=4
    )

    plt.errorbar(
        df["subset"], df["Recall"],
        yerr=df["Recall_se"],
        marker="o",
        label="Recall",
        capsize=4
    )

    plt.xlabel("Subset")
    plt.ylabel("Score")
    plt.title("Precision vs Recall")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ======================
# MAIN
# ======================
def main():
    input = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    outdir = 'comparison_figures'
    os.makedirs(outdir, exist_ok=True)
    print("Loading data...")
    df = load_confusion_files(input)

    if df.empty:
        print("No valid files found.")
        return

    print("Computing metrics...")
    df = compute_metrics(df)

    print("Adding uncertainty...")
    df = add_uncertainty(df)
    # sort subsets naturally if possible
    #df = df.sort_values("subset")


    # save table
    df.to_csv(os.path.join(outdir, "metrics.tsv"), sep="\t", index=True)

    print("Plotting...")
    plot_metric(df, "F1", "F1 Score", os.path.join(outdir, "f1.png"))
    plot_metric(df, "Accuracy", "Accuracy", os.path.join(outdir, "accuracy.png"))
    plot_metric(df, "MCC", "MCC", os.path.join(outdir, "mcc.png"))
    plot_precision_recall(df, os.path.join(outdir, "precision_recall.png"))

    print("Done.")
    print(f"Outputs in: {outdir}")


if __name__ == "__main__":
    main()