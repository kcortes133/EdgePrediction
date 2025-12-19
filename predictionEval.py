import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators, from_memberships


def make_upset_plots(results_file: str, annotations_file: str, out_prefix: str):
    # --- Load results ---
    results = pd.read_csv(results_file)

    # Extract disease IDs per classification
    fn_diseases = results.loc[results["classification"] == "FN", "object"].unique()
    tp_diseases = results.loc[results["classification"] == "TP", "object"].unique()

    # --- Load annotations ---
    annots = pd.read_csv(annotations_file)

    # Ensure MONDO IDs align
    annots = annots.rename(columns={"Rare Disease": "disease"})

    # --- Helper to build dataframe for upset ---
    def prepare_subset(diseases, label):
        subset = annots[annots["disease"].isin(diseases)].copy()
        if subset.empty:
            print(f"⚠️ No matches for {label} in annotation file")
            return None
        # convert to boolean indicators
        subset = subset.set_index("disease")
        subset = subset.astype(bool)
        return subset

    fn_subset = prepare_subset(fn_diseases, "FN")
    tp_subset = prepare_subset(tp_diseases, "TP")

    # --- Make plots ---
    if fn_subset is not None:
        upset_fn = from_indicators(fn_subset.columns, fn_subset)
        UpSet(upset_fn, show_counts=True).plot()
        plt.title("False Negatives (FN) disease annotations")
        plt.savefig(out_prefix + "_FN_upset.png", dpi=300, bbox_inches="tight")
        plt.close()

    if tp_subset is not None:
        upset_tp = from_indicators(tp_subset.columns, tp_subset)
        UpSet(upset_tp, show_counts=True).plot()
        plt.title("True Positives (TP) disease annotations")
        plt.savefig(out_prefix + "_TP_upset.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("✅ UpSet plots saved.")


def make_classification_upset1(results_file: str, annotations_file: str, out_file: str):
    # --- Load results ---
    results = pd.read_csv(results_file)

    # Keep only FN and TP rows
    results = results[results["classification"].isin(["FN", "TP"])]

    # --- Load annotations ---
    annots = pd.read_csv(annotations_file)
    annots = annots.rename(columns={"Rare Disease": "disease"})

    # Merge results with annotations
    merged = results.merge(
        annots, left_on="object", right_on="disease", how="inner"
    )

    if merged.empty:
        print("⚠️ No overlap between results and annotations")
        return

    # Drop duplicates (multiple gene-disease edges → one disease annotation per classification)
    merged = merged.drop_duplicates(subset=["disease", "classification"])

    # Convert annotation columns to boolean
    category_cols = [
        "Has Gene", "Has Gene with Ortholog", "Has Phenotype", "Has Genotype", "Has GO"
    ]
    merged[category_cols] = merged[category_cols].astype(bool)

    # Prepare dataframe for upsetplot
    upset_data = from_indicators(category_cols, merged, data=merged["classification"])

    # Plot
    plt.figure(figsize=(10, 6))
    UpSet(
        upset_data,
        show_counts=True,
        show_percentages=False,
        element_size=None,
        sort_by="cardinality",
    ).plot()
    plt.suptitle("UpSet Plot of Disease Annotations by Classification (FN vs TP)")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Stratified UpSet plot saved to {out_file}")


# Example usage


def make_classification_upset(results_file: str, annotations_file: str, out_file: str):
    # --- Load results ---
    results = pd.read_csv(results_file)
    results = results[results["classification"].isin(["FN", "TP"])]

    # --- Load annotations ---
    annots = pd.read_csv(annotations_file)
    annots = annots.rename(columns={"Rare Disease": "disease"})

    # Merge results with annotations
    merged = results.merge(
        annots, left_on="object", right_on="disease", how="inner"
    )
    if merged.empty:
        print("⚠️ No overlap between results and annotations")
        return

    # Convert annotation columns to boolean
    category_cols = [
        "Has Gene", "Has Gene with Ortholog", "Has Phenotype", "Has Genotype", "Has GO"
    ]
    merged[category_cols] = merged[category_cols].astype(bool)

    # Build memberships for upsetplot
    memberships = []
    for _, row in merged.iterrows():
        cats = tuple(cat for cat in category_cols if row[cat])
        if cats:
            memberships.append((cats, row["classification"]))

    # Prepare dataframe for stacked bars
    data = pd.DataFrame(memberships, columns=["membership", "classification"])
    upset_data = (
        data.groupby(["membership", "classification"])
        .size()
        .unstack(fill_value=0)
    )

    # Convert to proper format for UpSet
    upset_ready = from_memberships(upset_data.index, data=upset_data)

    # Plot with stacked bars
    plt.figure(figsize=(10, 6))
    upset = UpSet(
        upset_ready,
        show_counts=True,
        sort_by="cardinality",
    )
    upset.plot()

    # Overlay stacked bars manually
    #ax = upset["intersections"].axes["intersections"]
    upset_data.plot(
        kind="bar",
        stacked=True,
        #ax=ax,
        color={"FN": "red", "TP": "green"},
        legend=True,
    )

    plt.suptitle("UpSet Plot of Disease Annotations (Stacked FN vs TP)")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Color-coded stacked UpSet plot saved to {out_file}")


# Example usage

make_classification_upset(
    results_file="results_with_rare_flag.csv",
    annotations_file="Rare Disease Annotation.csv",
    out_file="FN_TP_combined_upset20_1.png"
)

def mark_rare_diseases(results_file: str, rare_file: str, output_file: str, summary_file):
    """
    Mark whether the object (disease) in results is in the rare disease list.

    Parameters
    ----------
    results_file : str
        CSV file with subject, predicate, object, score, classification.
    rare_file : str
        CSV file containing rare diseases in the first column (e.g. 'Rare Disease').
    output_file : str
        File path to save results with a new 'is_rare' column.
    """
    # Load results
    results = pd.read_csv(results_file)

    # Load rare disease list (assumes first column holds MONDO IDs)
    rare_df = pd.read_csv(rare_file)
    rare_diseases = set(rare_df.iloc[:, 0])  # first column

    # Check membership
    results["is_rare"] = results["object"].isin(rare_diseases)

    true_count = results['is_rare'].sum()
    print(true_count)
    # Save updated results
    results.to_csv(output_file, index=False)
    print(f"✅ Annotated results written to {output_file}")


    # --- Overall counts ---
    overall_counts = results["classification"].value_counts().to_dict()
    print("Overall counts:", overall_counts)

    # --- Rare disease counts ---
    rare_results = results[results["is_rare"]]
    rare_counts = rare_results["classification"].value_counts().to_dict()
    print("Rare disease counts:", rare_counts)

    # --- Per rare disease breakdown ---
    per_disease = (
        rare_results.groupby("object")["classification"]
        .value_counts()
        .unstack(fill_value=0)
    )
    per_disease.to_csv(summary_file)

    print(f"✅ Annotated results saved to {output_file}")
    print(f"✅ Rare disease summary saved to {summary_file}")

    return overall_counts, rare_counts, per_disease


# Example usage
mark_rare_diseases(
    results_file='testing20/top10results.csv',
    rare_file="Rare Disease Gene Associations.csv",
    output_file="results_with_rare_flag.csv",
    summary_file="summary_with_rare_flag.csv"
)

