import os
import pandas as pd

# ----------------------
# Configuration
# ----------------------
SUBSETS = ["none", "40", "60", "80"]
RESULTS_DIR_TEMPLATE = "results_{}"
GENE_RANKS_FILE = "gene_ranks.tsv"

OUTDIR = "comparison_figures"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------
# Load gene_ranks for each subset
# ----------------------
dfs = {}

for subset in SUBSETS:
    path = os.path.join(
        RESULTS_DIR_TEMPLATE.format(subset),
        GENE_RANKS_FILE
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path, sep="\t")

    # keep only what we need
    df = df[["disease", "gene", "rank", "score"]].copy()
    df = df.rename(columns={
        "rank": f"rank_{subset}",
        "score": f"score_{subset}"
    })

    dfs[subset] = df

# ----------------------
# Merge across subsets
# ----------------------
merged = dfs["none"]
for subset in SUBSETS[1:]:
    merged = merged.merge(
        dfs[subset],
        on=["disease", "gene"],
        how="inner"
    )

print(f"Merged pairs across all subsets: {len(merged)}")

# ----------------------
# Improvement criteria
# ----------------------
rank_improves = (
    (merged["rank_none"] > merged["rank_40"]) &
    (merged["rank_40"] > merged["rank_60"]) &
    (merged["rank_60"] > merged["rank_80"])
)

score_improves = (
    (merged["score_none"] < merged["score_40"]) &
    (merged["score_40"] < merged["score_60"]) &
    (merged["score_60"] < merged["score_80"])
)

improving = merged[rank_improves & score_improves].copy()

print(f"Strictly improving gene–disease pairs: {len(improving)}")

# ----------------------
# Optional: improvement magnitudes
# ----------------------
improving["rank_gain_none_to_80"] = (
    improving["rank_none"] - improving["rank_80"]
)

improving["score_gain_none_to_80"] = (
    improving["score_80"] - improving["score_none"]
)

# sort by strongest improvement
improving = improving.sort_values(
    by=["rank_gain_none_to_80", "score_gain_none_to_80"],
    ascending=False
)

# ----------------------
# Save
# ----------------------
out_path = os.path.join(
    OUTDIR,
    "improving_pairs_none_40_60_80.tsv"
)

improving.to_csv(out_path, sep="\t", index=False)

print(f"Saved improving pairs to: {out_path}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------
# Configuration
# ----------------------
INFILE = "comparison_figures/improving_pairs_none_40_60_80.tsv"
OUTDIR = "comparison_figures"
os.makedirs(OUTDIR, exist_ok=True)

SUBSETS = ["none", "40", "60", "80"]
X_LABELS = ["none", "40", "60", "80"]

sns.set(style="whitegrid", font_scale=1.1)

# ----------------------
# Load data
# ----------------------
df = pd.read_csv(INFILE, sep="\t")

if df.empty:
    raise ValueError("No improving pairs found — cannot create figure.")

# Select top 10 by rank improvement (primary) then score gain
top10 = (
    df
    .sort_values(
        by=["rank_gain_none_to_80", "score_gain_none_to_80"],
        ascending=False
    )
    .head(10)
    .copy()
)

# ----------------------
# Reshape for plotting
# ----------------------
plot_rows = []

for _, row in top10.iterrows():
    pair_label = f"{row['gene']} – {row['disease']}"

    for subset in SUBSETS:
        plot_rows.append({
            "pair": pair_label,
            "subset": subset,
            "rank": row[f"rank_{subset}"]
        })

plot_df = pd.DataFrame(plot_rows)

plot_df["subset"] = pd.Categorical(
    plot_df["subset"],
    categories=SUBSETS,
    ordered=True
)

# ----------------------
# Plot
# ----------------------
plt.figure(figsize=(11, 7))

sns.lineplot(
    data=plot_df,
    x="subset",
    y="rank",
    hue="pair",
    marker="o",
    linewidth=2
)

plt.gca().invert_yaxis()  # lower rank = better
plt.yscale("log")

plt.title("Top 10 Most Improved Gene–Disease Pairs")
plt.xlabel("Training Data Removal (%)")
plt.ylabel("True Gene Rank (log scale)")
plt.legend(
    title="Gene – Disease",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "top10_improved_gene_disease_pairs.png"),
    dpi=300
)
plt.show()
plt.figure(figsize=(8, 6))

sns.barplot(
    data=top10,
    x="rank_gain_none_to_80",
    y=top10.apply(lambda r: f"{r['gene']} – {r['disease']}", axis=1),
    orient="h"
)

plt.title("Magnitude of Rank Improvement (none → 80)")
plt.xlabel("Rank Improvement (Δ rank)")
plt.ylabel("Gene – Disease Pair")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "top10_rank_improvement_magnitude.png"),
    dpi=300
)
plt.show()


from ensmallen import Graph

SUBSETS = ["none", "40", "60", "80", "100"]
kg_dict = {}

kgFile = 'monarch-kg-Sept2025/monarch-kg_'
out_folder_template = "results_{}/"

pairs_file = "comparison_figures/improving_pairs_none_40_60_80.tsv"
top_pairs = pd.read_csv(pairs_file, sep="\t").head(10)

# List of tuples: (gene, disease)
pairs_list = list(zip(top_pairs["gene"], top_pairs["disease"]))

summary_rows = []
path_summary = []
for subset in SUBSETS:
    print(f"Loading KG for subset: {subset}")
    out_folder = out_folder_template.format(subset)

    nodesF = kgFile + "nodes.filtered_"+ subset + ".tsv"
    edgesF = kgFile + "edges.filtered_" + subset + ".TestSet.tsv"

    if subset == "none":
        nodesF = kgFile + "nodes.tsv"
        edgesF = kgFile + "edges.TestSet.tsv"
    kg = Graph.from_csv(
        directed=False,
        node_path=nodesF,
        edge_path=edgesF,
        node_list_separator="\t",
        edge_list_separator="\t",
        verbose=True,
        nodes_column="id",
        node_list_node_types_column="category",
        default_node_type="biolink:NamedThing",
        sources_column="subject",
        destinations_column="object",
        edge_list_edge_types_column="predicate",
        name=f"Monarch KG {subset}"
    )

    for gene, disease in pairs_list:
        # Extract neighborhoods
        gene_neighbors = set(kg.get_neighbour_node_names_from_node_name(gene))
        disease_neighbors = set(kg.get_neighbour_node_names_from_node_name(disease))

        paths = kg.get_k_shortest_path_node_names_from_node_names(
            src_node_name=gene,
            dst_node_name=disease,
            k=10
        )
        # Intersection (shared nodes)
        shared_neighbors = gene_neighbors & disease_neighbors
        rank = 1
        for path in paths:
            path_summary.append({
                "subset": subset,
                "gene": gene,
                "disease": disease,
                "path_rank": rank,
                "path_nodes": path,
                "path_length": len(path),
            }

            )
            rank+=1

        # Record metrics
        summary_rows.append({
            "subset": subset,
            "gene": gene,
            "disease": disease,
            "gene_degree": len(gene_neighbors),
            "disease_degree": len(disease_neighbors),
            "shared_neighbors": len(shared_neighbors),
            "total_neighbors": len(gene_neighbors | disease_neighbors)
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("comparison_figures/kg_neighborhood_summary.tsv", sep="\t", index=False)

path_df = pd.DataFrame(path_summary)
path_df.to_csv("comparison_figures/kg_top10_shortest_paths.tsv", sep="\t", index=False)

