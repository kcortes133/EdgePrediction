#!/usr/bin/env python3
"""
Regenerate the TRIM manuscript's result tables and figures from perceptron
run folders.

Inputs  (per subset, written by src/ml/perceptronBatch.py):
    <runs>/results_{none,40,60,80,100}/confusion_summary.tsv
    <runs>/results_{none,40,60,80,100}/gene_ranks.tsv
    data/TP_hgnc_mondo_edges.tsv
    analysis/data/Rare Disease Annotation.csv
    monarch_percep_rand{,7,13}_80/confusion_summary.tsv   (random control)

Outputs (in --out):
    table2_classification.tsv   Table 2 (F1, AUROC, TP/FN/FP/TN)
    ranking_summary.tsv         median rank, top-10 / top-50 fractions (all + rare)
    fig4c_confusion_counts.png  Figure 4C
    fig5_topk.png               Figure 5A (all diseases) and 5B (rare diseases)

Before summarising, every gene_ranks.tsv is checked against the TP test set
and against the other subsets, so results from a stale run are reported
instead of silently mixed in.

Usage:
    python src/analysis/paper_figures.py [--runs analysis/results] [--out analysis/results/paper]
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ["none", "40", "60", "80", "100"]
LABELS = {"none": "None", "40": "40", "60": "60", "80": "80", "100": "100"}
RANDOM_DIRS = ["monarch_percep_rand_80", "monarch_percep_rand7_80", "monarch_percep_rand13_80"]


def load_confusion(folder):
    c = pd.read_csv(folder / "confusion_summary.tsv", sep="\t").iloc[0]
    tp, fn, fp, tn = (int(c[k]) for k in ("TP", "FN", "FP", "TN"))
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "F1": 2 * tp / (2 * tp + fp + fn), "AUROC": float(c["auroc"])}


def load_ranks(folder):
    df = pd.read_csv(folder / "gene_ranks.tsv", sep="\t")
    df = df[pd.to_numeric(df["rank"], errors="coerce").notna()].copy()
    df["rank"] = df["rank"].astype(int)
    df["pair"] = list(zip(df["disease"], df["gene"]))
    return df


def check_consistency(ranks, tp_pairs):
    ok = True
    ref = set(ranks[SUBSETS[0]]["pair"])
    for s, df in ranks.items():
        pairs = set(df["pair"])
        outside = len(pairs - tp_pairs)
        if outside:
            ok = False
            print(f"WARNING [{s}]: {outside} of {len(pairs)} ranked pairs are not in the TP test set "
                  f"(stale run?)", file=sys.stderr)
        if pairs != ref:
            ok = False
            print(f"WARNING [{s}]: ranked pairs differ from subset '{SUBSETS[0]}' "
                  f"({len(pairs ^ ref)} pairs differ)", file=sys.stderr)
    return ok


def topk_summary(df):
    r = df["rank"]
    return {"n_pairs": len(r), "median_rank": float(r.median()),
            "top10": float((r <= 10).mean()), "top50": float((r <= 50).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "analysis/results"),
                    help="Folder containing results_<subset>/ run folders")
    ap.add_argument("--out", default=str(ROOT / "analysis/results/paper"))
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Table 2 ------------------------------------------------------
    t2 = pd.DataFrame({LABELS[s]: load_confusion(runs / f"results_{s}") for s in SUBSETS}).T
    t2 = t2[["F1", "AUROC", "TP", "FN", "FP", "TN"]]
    t2.index.name = "Subset"
    t2.to_csv(out / "table2_classification.tsv", sep="\t", float_format="%.3f")

    rand_f1 = [load_confusion(ROOT / d)["F1"] for d in RANDOM_DIRS if (ROOT / d).exists()]
    if rand_f1:
        print(f"Random-removal control F1: {np.mean(rand_f1):.3f} ± {np.std(rand_f1, ddof=1):.3f} "
              f"(n={len(rand_f1)})")
    print("\nTable 2\n", t2.round(3).to_string())

    # ---- Ranking ------------------------------------------------------
    tp = pd.read_csv(ROOT / "data/TP_hgnc_mondo_edges.tsv", sep="\t")
    tp_pairs = set(zip(tp["object"], tp["subject"])) | set(zip(tp["subject"], tp["object"]))
    ranks = {s: load_ranks(runs / f"results_{s}") for s in SUBSETS}
    consistent = check_consistency(ranks, tp_pairs)

    rare = pd.read_csv(ROOT / "analysis/data/Rare Disease Annotation.csv")
    rare_ids = set(rare["Rare Disease"])

    rows = []
    for s in SUBSETS:
        df = ranks[s]
        rows.append({"subset": LABELS[s], "diseases": "all", **topk_summary(df)})
        rows.append({"subset": LABELS[s], "diseases": "rare",
                     **topk_summary(df[df["disease"].isin(rare_ids)])})
    rk = pd.DataFrame(rows)
    rk.to_csv(out / "ranking_summary.tsv", sep="\t", index=False, float_format="%.3f")
    print("\nRanking summary\n", rk.round(3).to_string(index=False))

    # ---- Figure 4C ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SUBSETS)); w = 0.2
    colors = {"TP": "#2e7d32", "FN": "#a5d6a7", "FP": "#1565c0", "TN": "#90caf9"}
    for i, k in enumerate(["TP", "FN", "FP", "TN"]):
        ax.bar(x + (i - 1.5) * w, t2[k].values, w, label=k, color=colors[k])
    ax.set_xticks(x, [LABELS[s] for s in SUBSETS]); ax.set_xlabel("IC threshold")
    ax.set_ylabel("Count"); ax.legend(ncol=4, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig4c_confusion_counts.png", dpi=300); plt.close(fig)

    # ---- Figure 5 -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    shades = ["#c6dbef", "#9ecae1", "#6baed6", "#8e3a80", "#3182bd"]
    for ax, grp, title in zip(axes, ["all", "rare"], ["A  All diseases", "B  Rare diseases"]):
        sub = rk[rk["diseases"] == grp].set_index("subset")
        for i, s in enumerate(SUBSETS):
            vals = sub.loc[LABELS[s], ["top10", "top50"]].values * 100
            ax.bar(np.arange(2) + (i - 2) * 0.16, vals, 0.16, label=LABELS[s], color=shades[i])
        ax.set_xticks([0, 1], ["Top 10", "Top 50"]); ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axes[0].set_ylabel("True gene ranked within top k"); axes[0].legend(title="IC threshold", frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig5_topk.png", dpi=300); plt.close(fig)

    print(f"\nOutputs written to {out}")
    if not consistent:
        print("\nWARNING: ranked pairs are inconsistent across subsets; see messages above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
