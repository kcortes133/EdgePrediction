import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
BENCH_FILE = "PRIME_edgePred_benchmark_DPRMG.csv"
OUT_DIR = "PRIMEbenchmark_summary_plots"

CORE_METRICS = [
    "auroc",
    "auprc",
    "f1_score",
    "balanced_accuracy"
]

EDGE_EMB_PARAM_COL = "('model_parameters', 'edge_embeddings')"
RANDOM_STATE_COL = "('model_parameters', 'random_state')"

# =========================
# Helpers
# =========================
def parse_list_like(val):
    """Parse stringified lists like ['Hadamard'] into readable labels."""
    if pd.isna(val):
        return "None"
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list) and len(parsed) > 0:
            return "+".join(map(str, parsed))
        return "None"
    except Exception:
        return str(val)


def find_eval_column(columns):
    for c in columns:
        if c.lower().strip() in {"evaluation_mode", "mode", "split"}:
            return c
    return None


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(BENCH_FILE)
    df.columns = [c.strip() for c in df.columns]

    # -------------------------
    # Filter to test split
    # -------------------------
    eval_col = find_eval_column(df.columns)
    if eval_col:
        df = df[df[eval_col].astype(str).str.lower() == "test"].copy()

    # -------------------------
    # Parse edge embeddings
    # -------------------------
    if EDGE_EMB_PARAM_COL in df.columns:
        df["edge_embedding"] = df[EDGE_EMB_PARAM_COL].apply(parse_list_like)
    else:
        df["edge_embedding"] = "Unknown"

    # -------------------------
    # Parse random state
    # -------------------------
    if RANDOM_STATE_COL in df.columns:
        df["random_state"] = df[RANDOM_STATE_COL].fillna("NA").astype(str)
    else:
        df["random_state"] = "NA"

    # -------------------------
    # Build labels
    # -------------------------
    df["model_label"] = (
        df["model_name"].astype(str)
        + " (" + df["library_name"].astype(str) + ")"
    )

    df["run_label"] = (
        df["model_label"]
        + " | emb=" + df["edge_embedding"]
        + " | rs=" + df["random_state"]
        + " | h=" + df["holdout_number"].astype(str)
    )

    metrics = [m for m in CORE_METRICS if m in df.columns]

    # =========================================================
    # Figure 1: Per-run performance (model + params)
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        ax.bar(df["run_label"], df[metric])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=90)

    fig.suptitle(
        "Edge Prediction Performance by Model + Parameters (Test)",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "HLL_by_model_parameters.png"))
    plt.close(fig)

    # =========================================================
    # Figure 2: Mean performance by edge embedding
    # =========================================================
    grouped = (
        df.groupby("edge_embedding")[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
    axes2 = axes2.flatten()

    for ax, metric in zip(axes2, metrics):
        ax.bar(grouped["edge_embedding"], grouped[metric])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Mean score")
        ax.tick_params(axis="x", rotation=30)

    fig2.suptitle(
        "Mean Performance by Edge Embedding (Test)",
        fontsize=16
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(os.path.join(OUT_DIR, "HLL_by_edge_embedding_mean.png"))
    plt.close(fig2)


if __name__ == "__main__":
    main()
