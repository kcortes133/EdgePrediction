import pandas as pd

# -----------------------------
# File paths
# -----------------------------
tp_edges_file = "TP_hgnc_mondo_edges.tsv"
rd_gene_assoc_file = "Rare Disease Gene Associations.csv"
rd_annotations_file = "Rare Disease Annotation.csv"
output_csv = "TP_test_set_annotation_summary.csv"

# -----------------------------
# Load data
# -----------------------------
tp_df = pd.read_csv(tp_edges_file, sep="\t")
rd_gene_df = pd.read_csv(rd_gene_assoc_file)
rd_annot_df = pd.read_csv(rd_annotations_file)

# -----------------------------
# Global rare disease count
# -----------------------------
total_rare_diseases = rd_annot_df["Rare Disease"].nunique()

# -----------------------------
# True positive test set stats
# -----------------------------
tp_diseases = set(tp_df["object"])
tp_genes = set(tp_df["subject"])
tp_pairs = tp_df[["subject", "object"]].drop_duplicates()

num_tp_diseases = len(tp_diseases)          # ← rare diseases in test set
num_tp_genes = len(tp_genes)
num_tp_pairs = len(tp_pairs)

# -----------------------------
# Annotation coverage (TP diseases)
# -----------------------------
tp_annotations = rd_annot_df[
    rd_annot_df["Rare Disease"].isin(tp_diseases)
]

annotation_summary = {
    "Has Gene": tp_annotations["Has Gene"].sum(),
    "Has Gene with Ortholog": tp_annotations["Has Gene with Ortholog"].sum(),
    "Has Phenotype": tp_annotations["Has Phenotype"].sum(),
    "Has Genotype": tp_annotations["Has Genotype"].sum(),
    "Has GO": tp_annotations["Has GO"].sum(),
}

# -----------------------------
# Gene association overlap
# -----------------------------
tp_gene_assoc = rd_gene_df[
    rd_gene_df["Rare Disease"].isin(tp_diseases)
]

num_assoc_pairs = len(
    tp_gene_assoc[["Rare Disease", "Gene"]].drop_duplicates()
)
num_assoc_diseases = tp_gene_assoc["Rare Disease"].nunique()
num_assoc_genes = tp_gene_assoc["Gene"].nunique()

# -----------------------------
# Build summary table
# -----------------------------
summary_rows = [
    {
        "Metric": "Total rare diseases (full dataset)",
        "Value": total_rare_diseases
    },
    {
        "Metric": "Rare diseases in TP test set",
        "Value": num_tp_diseases
    },
    {
        "Metric": "Total genes in TP test set",
        "Value": num_tp_genes
    },
    {
        "Metric": "Total disease–gene pairs in TP test set",
        "Value": num_tp_pairs
    },
    {
        "Metric": "Diseases with known gene associations (TP set)",
        "Value": num_assoc_diseases
    },
    {
        "Metric": "Genes in known associations (TP set)",
        "Value": num_assoc_genes
    },
    {
        "Metric": "Known disease–gene association pairs (TP set)",
        "Value": num_assoc_pairs
    },
]

# Annotation coverage rows
for annot, count in annotation_summary.items():
    summary_rows.append({
        "Metric": f"Diseases with {annot} (TP set)",
        "Value": count
    })

# -----------------------------
# Non-unique disease / gene counts
# -----------------------------
total_rows = len(tp_df)

# Diseases (MONDO)
total_disease_occurrences = tp_df["object"].shape[0]
unique_diseases = tp_df["object"].nunique()
non_unique_diseases = total_disease_occurrences - unique_diseases

# Genes (HGNC)
total_gene_occurrences = tp_df["subject"].shape[0]
unique_genes = tp_df["subject"].nunique()
non_unique_genes = total_gene_occurrences - unique_genes

summary_rows.extend([
    {
        "Metric": "Total disease occurrences (TP set)",
        "Value": total_disease_occurrences
    },
    {
        "Metric": "Unique diseases (TP set)",
        "Value": unique_diseases
    },
    {
        "Metric": "Non-unique disease occurrences (TP set)",
        "Value": non_unique_diseases
    },
    {
        "Metric": "Total gene occurrences (TP set)",
        "Value": total_gene_occurrences
    },
    {
        "Metric": "Unique genes (TP set)",
        "Value": unique_genes
    },
    {
        "Metric": "Non-unique gene occurrences (TP set)",
        "Value": non_unique_genes
    },
])

summary_df = pd.DataFrame(summary_rows)

# -----------------------------
# Write to CSV
# -----------------------------
summary_df.to_csv(output_csv, index=False)

print(f"Summary written to: {output_csv}")
