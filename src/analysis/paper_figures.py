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
    fig3_scores_confusion.*     Figure 3 (A/B score histograms, C confusion counts)
    fig4_topk.*                 Figure 4 (A all diseases, B rare diseases)
    fig5_f1_control.*           Figure 5 (F1 vs random-node-removal control)

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
    # Each TP edge is ranked in both orientations; only disease (MONDO) -> candidate genes
    # is the task described in the paper. The swapped rows rank a MONDO id among genes.
    df = df[df["disease"].astype(str).str.startswith("MONDO")]
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
            "top10": float((r <= 10).mean()), "top20": float((r <= 20).mean()),
            "top50": float((r <= 50).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "analysis/results"),
                    help="Folder containing results_<subset>/ run folders")
    ap.add_argument("--suffix", default="",
                    help='Run-folder suffix, e.g. "_ranks" for results_<subset>_ranks/')
    ap.add_argument("--out", default=str(ROOT / "analysis/results/paper"))
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Table 2 ------------------------------------------------------
    t2 = pd.DataFrame({LABELS[s]: load_confusion(runs / f"results_{s}{args.suffix}") for s in SUBSETS}).T
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
    ranks = {s: load_ranks(runs / f"results_{s}{args.suffix}") for s in SUBSETS}
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

    # ---- Figures (numbered as in the manuscript) -----------------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    NAMES = {s: ("None" if s == "none" else f"nIC{s}") for s in SUBSETS}
    POS, NEG = "#2a6fb0", "#e8892b"

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Figure 3: (A, B) perceptron score histograms, (C) confusion-matrix counts
    fig = plt.figure(figsize=(7.1, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.45, wspace=0.25)
    bins = np.linspace(0, 1, 21)
    for k, (s, letter) in enumerate([("none", "A"), ("80", "B")]):
        ax = fig.add_subplot(gs[0, k])
        ep = pd.read_csv(runs / f"results_{s}{args.suffix}" / "edge_predictions.tsv", sep="\t")
        ax.hist(ep.loc[ep.label == 1, "prediction"], bins=bins, color=POS, alpha=0.6, label="Positive pairs")
        ax.hist(ep.loc[ep.label == 0, "prediction"], bins=bins, color=NEG, alpha=0.6, label="Negative pairs")
        ax.axvline(0.5, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel("Perceptron score"); ax.set_ylabel("Gene–disease pairs" if k == 0 else "")
        ax.set_title(f"{letter}  {NAMES[s]}", loc="left", fontweight="bold")
        if k == 0: ax.legend(frameon=False, loc="upper left")
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(SUBSETS)); w = 0.19
    colors = {"TP": "#2e7d32", "FN": "#a5d6a7", "FP": "#1565c0", "TN": "#90caf9"}
    for i, k in enumerate(["TP", "FN", "FP", "TN"]):
        ax.bar(x + (i - 1.5) * w, t2[k].values, w * 0.92, label=k, color=colors[k])
    ax.set_xticks(x, [NAMES[s] for s in SUBSETS]); ax.set_xlabel("Subset")
    ax.set_ylabel("Gene–disease pairs"); ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("C", loc="left", fontweight="bold")
    save(fig, "fig3_scores_confusion")

    # Figure 4: top-10 / top-20 / top-50 (A all diseases, B rare diseases)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    shades = ["#c6dbef", "#9ecae1", "#6baed6", "#8e3a80", "#3182bd"]
    ks = ["top10", "top20", "top50"]
    for ax, grp, title in zip(axes, ["all", "rare"], ["A  All diseases", "B  Rare diseases"]):
        sub = rk[rk["diseases"] == grp].set_index("subset")
        for i, s in enumerate(SUBSETS):
            vals = sub.loc[LABELS[s], ks].values * 100
            ax.bar(np.arange(len(ks)) + (i - 2) * 0.16, vals, 0.15, label=NAMES[s], color=shades[i])
        n = int(sub["n_pairs"].iloc[0])
        ax.set_xticks(range(len(ks)), ["Top 10", "Top 20", "Top 50"])
        ax.set_title(f"{title} (n = {n:,})", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    axes[0].set_ylabel("Pairs with true gene in top k"); axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig4_topk")

    # Figure 5: F1 per subset vs random-node-removal control
    if rand_f1:
        m, sd = np.mean(rand_f1), np.std(rand_f1, ddof=1)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.axhspan(m - sd, m + sd, color="#d62728", alpha=0.12, lw=0)
        ax.axhline(m, color="#d62728", ls="--", lw=1.2, label=f"Random removal (n = {len(rand_f1)}): {m:.3f} ± {sd:.3f}")
        ax.plot(range(len(SUBSETS)), t2["F1"].values, "o-", color="#2a6fb0", lw=2, ms=6, label="nIC-pruned subsets")
        for i, v in enumerate(t2["F1"].values):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.set_xticks(range(len(SUBSETS)), [NAMES[s] for s in SUBSETS]); ax.set_xlabel("Subset")
        ax.set_ylabel("F1"); ax.legend(frameon=False, fontsize=8, loc="lower left")
        lo = min(t2["F1"].min(), m - sd); hi = max(t2["F1"].max(), m + sd)
        ax.set_ylim(lo - 0.02, hi + 0.02)
        save(fig, "fig5_f1_control")

    print(f"\nOutputs written to {out}")
    if not consistent:
        print("\nWARNING: ranked pairs are inconsistent across subsets; see messages above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
