#!/usr/bin/env python3
"""
perceptronBatch.py
==================
End-to-end edge-prediction pipeline for gene-disease ranking.

IMPORTANT: This pipeline always retrains embeddings and models fresh for each
run. This is necessary because different KGs (e.g., human-only vs human+mouse)
require independent embeddings. Caching is NOT used.

Pipeline overview
-----------------
1. Load training KG, positive test edges, and negative test edges.
2. Validate that TP/TN edges are structured as gene -> disease (or disease -> gene,
   but consistent).
3. Train fresh node embeddings (FirstOrderLINE) and edge classifier (Perceptron).
4. Score all positive and negative test edges; compute AUROC, AUPRC.
5. Verify that true pairs from pos_pred match the TP file exactly.
6. Rank candidate genes for each (disease, gene) pair from the TP set.

Usage
-----
python perceptronBatch.py \\
    --nodes      data/monarch-kg-Sept2025/monarch-kg_nodes.tsv \\
    --train      data/monarch-kg-Sept2025/monarch-kg_edges.tsv \\
    --pos        data/TP_hgnc_mondo_edges.tsv \\
    --neg        data/TN_hgnc_mondo_edges1.tsv \\
    --candidates geneCandidates.txt \\
    --out        analysis/results/human_only \\
    --threshold  0.5
"""

import argparse
import csv
import logging
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
from ensmallen import Graph
from sklearn.metrics import average_precision_score, roc_auc_score

from embiggen.edge_prediction import PerceptronEdgePrediction
from embiggen.embedders.ensmallen_embedders import FirstOrderLINEEnsmallen

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 50_000   # Max genes scored per temp-graph call (keeps RAM bounded)

# ---------------------------------------------------------------------------
# Logging -- will be configured per-run to write to output directory
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def setup_logging(output_dir):
    """
    Configure logging to write to a file in the output directory.

    Parameters
    ----------
    output_dir : str
        Directory where logs will be written.
    """
    log_file = os.path.join(output_dir, "perceptron.log")
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Logging initialized to %s", log_file)


# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------

def load_graph(nodes_file, edges_file, name):
    """
    Load a KG from TSV node/edge files using the Ensmallen/GRAPE convention.

    Parameters
    ----------
    nodes_file : str
        Path to the nodes TSV (must have columns: id, category).
    edges_file : str
        Path to the edges TSV (must have columns: subject, predicate, object).
    name : str
        Logical name for the graph (used internally by Ensmallen).

    Returns
    -------
    Graph
        An undirected Ensmallen Graph object.
    """
    return Graph.from_csv(
        directed=False,
        node_path=nodes_file,
        edge_path=edges_file,
        node_list_separator="\t",
        edge_list_separator="\t",
        verbose=False,
        nodes_column="id",
        node_list_node_types_column="category",
        default_node_type="biolink:NamedThing",
        sources_column="subject",
        destinations_column="object",
        edge_list_edge_types_column="predicate",
        name=name,
    )


# ---------------------------------------------------------------------------
# Edge validation - ensure edges are structured as gene -> disease
# ---------------------------------------------------------------------------

def validate_edge_structure(edges_file, graph, node_types_map):
    """
    Validate that all edges in the file are structured as gene -> disease.

    Parameters
    ----------
    edges_file : str
        Path to the edges TSV file.
    graph : Graph
        The loaded graph object (for type validation).
    node_types_map : dict
        Mapping of node ID to node type/category.

    Returns
    -------
    tuple (bool, dict)
        (is_valid, summary)
        - is_valid: True if all edges are gene->disease or disease->gene consistent
        - summary: dict with counts of genes and diseases in the edges
    """
    gene_count = 0
    disease_count = 0
    violations = []

    with open(edges_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, 1):
            subject = row.get("subject")
            obj = row.get("object")

            if not subject or not obj:
                violations.append(f"Row {idx}: missing subject or object")
                continue

            subj_type = node_types_map.get(subject, "UNKNOWN")
            obj_type = node_types_map.get(obj, "UNKNOWN")

            # Check if one is a gene and the other is a disease
            is_subj_gene = "Gene" in subj_type or "HGNC" in subject
            is_subj_disease = "Disease" in subj_type or "MONDO" in subject
            is_obj_gene = "Gene" in obj_type or "HGNC" in obj
            is_obj_disease = "Disease" in obj_type or "MONDO" in obj

            if is_subj_gene and is_obj_disease:
                gene_count += 1
            elif is_subj_disease and is_obj_gene:
                disease_count += 1
            else:
                violations.append(
                    f"Row {idx}: subject={subject} ({subj_type}), "
                    f"object={obj} ({obj_type}) — not gene-disease pair"
                )

    is_valid = len(violations) == 0
    summary = {
        "total_edges": gene_count + disease_count,
        "gene_disease_edges": gene_count,
        "disease_gene_edges": disease_count,
        "violations": violations[:10],  # Log first 10 violations
        "is_valid": is_valid,
    }

    return is_valid, summary


def build_node_types_map(nodes_file):
    """
    Build a mapping of node ID -> node type/category.

    Parameters
    ----------
    nodes_file : str
        Path to the nodes TSV.

    Returns
    -------
    dict
        Mapping of node ID to category.
    """
    node_types = {}
    with open(nodes_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            node_id = row.get("id")
            category = row.get("category", "")
            if node_id:
                node_types[node_id] = category
    return node_types


# ---------------------------------------------------------------------------
# Verify true pairs from predictions match the TP file
# ---------------------------------------------------------------------------

def load_tp_pairs(tp_file):
    """
    Load all (gene, disease) pairs from the TP file.

    Assumes TP file has 'subject' and 'object' columns. Pairs are stored
    as frozensets to handle bidirectional edges.

    Parameters
    ----------
    tp_file : str
        Path to the TP edges TSV.

    Returns
    -------
    set of frozenset
        All unique gene-disease pairs from the TP file.
    """
    pairs = set()
    with open(tp_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            subject = row.get("subject")
            obj = row.get("object")
            if subject and obj:
                # Use frozenset to handle bidirectional edges
                pairs.add(frozenset([subject, obj]))
    return pairs


def verify_true_pairs(pos_pred, tp_pairs):
    """
    Verify that all (gene, disease) pairs from pos_pred are in the TP file.

    Parameters
    ----------
    pos_pred : pd.DataFrame
        Positive predictions with 'sources' (genes) and 'destinations' (diseases).
    tp_pairs : set of frozenset
        True pairs from the TP file.

    Returns
    -------
    tuple (bool, dict)
        (all_verified, report)
    """
    pred_pairs = set()
    mismatches = []

    for _, row in pos_pred.iterrows():
        gene = row["sources"]
        disease = row["destinations"]
        pair = frozenset([gene, disease])
        pred_pairs.add(pair)

        # Check if this pair is in the TP file
        if pair not in tp_pairs:
            mismatches.append((gene, disease))

    all_verified = len(mismatches) == 0
    report = {
        "total_pred_pairs": len(pred_pairs),
        "total_tp_pairs": len(tp_pairs),
        "all_verified": all_verified,
        "mismatches": mismatches[:10],  # Log first 10
        "mismatch_count": len(mismatches),
    }

    if mismatches:
        logger.warning(
            "WARNING: %d predicted pairs are not in the TP file (showing first 10):\n%s",
            len(mismatches),
            "\n".join([f"  {g} -> {d}" for g, d in mismatches[:10]])
        )

    return all_verified, report


# ---------------------------------------------------------------------------
# Gene/node validation
# ---------------------------------------------------------------------------

def filter_genes_in_nodes(nodes_file, genes):
    """
    Return only those gene IDs that are present in the nodes file.

    Uses a single streaming pass over the nodes TSV to avoid loading the
    full file into memory.

    Parameters
    ----------
    nodes_file : str
        Path to the nodes TSV.
    genes : list of str
        Candidate gene IDs to validate.

    Returns
    -------
    list of str
        Subset of genes that exist as node IDs in the graph.
    """
    gene_set = set(genes)
    remaining = set(gene_set)

    with open(nodes_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "id" not in reader.fieldnames:
            raise ValueError("Nodes file is missing an 'id' column.")
        for row in reader:
            remaining.discard(row["id"])
            if not remaining:
                break

    missing = remaining
    kept = gene_set - missing

    logger.info("Gene validation: before=%d  removed=%d  remaining=%d",
                len(gene_set), len(missing), len(kept))
    if missing:
        logger.info("Example missing genes: %s", list(missing)[:10])

    return list(kept)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Perceptron-based edge prediction and gene ranking pipeline. "
        "NOTE: Embeddings and models are retrained fresh each run."
    )
    parser.add_argument("--nodes", required=True,
                        help="Filtered nodes TSV for the training KG.")
    parser.add_argument("--train", required=True,
                        help="Filtered edges TSV for the training KG.")
    parser.add_argument("--pos", default="data/TP_hgnc_mondo_edges.tsv",
                        help="Positive (true-positive) test edges TSV.")
    parser.add_argument("--neg", default="data/TN_hgnc_mondo_edges1.tsv",
                        help="Negative (true-negative) test edges TSV.")
    parser.add_argument("--candidates", default="geneCandidates.txt",
                        help="Text file of candidate gene IDs (one per line).")
    parser.add_argument("--out", required=True,
                        help="Output directory (created if absent).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for confusion-matrix binarisation (default 0.5).")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    setup_logging(args.out)

    logger.info("=" * 70)
    logger.info("Pipeline start: %s", args.out)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load graphs
    # ------------------------------------------------------------------
    logger.info("Loading training and test graphs")
    gTrain = load_graph(args.nodes, args.train, "Train KG")
    gPos   = load_graph(args.nodes, args.pos,   "Positive Test KG")
    gNeg   = load_graph(args.nodes, args.neg,   "Negative Test KG")

    # ------------------------------------------------------------------
    # Validate edge structure
    # ------------------------------------------------------------------
    logger.info("Validating edge structure in TP/TN files")
    node_types = build_node_types_map(args.nodes)

    is_pos_valid, pos_summary = validate_edge_structure(args.pos, gPos, node_types)
    logger.info("Positive edges validation: %s (gene->disease: %d, disease->gene: %d)",
                "PASS" if is_pos_valid else "FAIL",
                pos_summary["gene_disease_edges"],
                pos_summary["disease_gene_edges"])
    if pos_summary["violations"]:
        logger.warning("TP violations: %s", pos_summary["violations"])

    is_neg_valid, neg_summary = validate_edge_structure(args.neg, gNeg, node_types)
    logger.info("Negative edges validation: %s (gene->disease: %d, disease->gene: %d)",
                "FAIL" if not is_neg_valid else "PASS",
                neg_summary["gene_disease_edges"],
                neg_summary["disease_gene_edges"])
    if neg_summary["violations"]:
        logger.warning("TN violations: %s", neg_summary["violations"])

    # ------------------------------------------------------------------
    # Load candidate genes
    # ------------------------------------------------------------------
    with open(args.candidates) as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d candidate genes", len(genes))

    # ------------------------------------------------------------------
    # ALWAYS train fresh embeddings and models
    # (No caching option - each run trains independently)
    # ------------------------------------------------------------------
    logger.info("Training FRESH FirstOrderLINE embeddings (128-dim, 10 epochs)")
    embedder = FirstOrderLINEEnsmallen(embedding_size=128, epochs=10)

    logger.info("Fitting FRESH PerceptronEdgePrediction (Hadamard edge embedding)")
    model = PerceptronEdgePrediction(edge_embeddings="Hadamard")
    model.fit(graph=gTrain, node_features=embedder)
    logger.info("Training complete")

    # ------------------------------------------------------------------
    # Score positive and negative test edges
    # ------------------------------------------------------------------
    logger.info("Scoring positive test edges")
    pos_pred = model.predict_proba(
        graph=gPos,
        node_features=embedder,
        support=gTrain,
        return_predictions_dataframe=True,
        return_node_names=True,
    )
    pos_pred["label"] = 1

    logger.info("Scoring negative test edges")
    neg_pred = model.predict_proba(
        graph=gNeg,
        node_features=embedder,
        support=gTrain,
        return_predictions_dataframe=True,
        return_node_names=True,
    )
    neg_pred["label"] = 0

    preds = pd.concat([pos_pred, neg_pred], ignore_index=True)
    preds.to_csv(os.path.join(args.out, "edge_predictions.tsv"), sep="\t", index=False)
    logger.info("Predictions saved to edge_predictions.tsv")

    # ------------------------------------------------------------------
    # Verify true pairs match the TP file
    # ------------------------------------------------------------------
    logger.info("Loading TP pairs from file and verifying predictions")
    tp_pairs = load_tp_pairs(args.pos)
    pairs_verified, verify_report = verify_true_pairs(pos_pred, tp_pairs)

    logger.info("True pairs verification: %s",
                "PASS" if pairs_verified else "FAIL")
    logger.info("  Predicted pairs: %d, TP file pairs: %d, Mismatches: %d",
                verify_report["total_pred_pairs"],
                verify_report["total_tp_pairs"],
                verify_report["mismatch_count"])

    # ------------------------------------------------------------------
    # Confusion matrix + ranking metrics
    # ------------------------------------------------------------------
    y_true  = preds["label"].astype(int).values
    y_score = preds["predictions"].astype(float).values
    y_hat   = (y_score >= args.threshold).astype(int)

    TP = int(((y_hat == 1) & (y_true == 1)).sum())
    FN = int(((y_hat == 0) & (y_true == 1)).sum())
    TN = int(((y_hat == 0) & (y_true == 0)).sum())
    FP = int(((y_hat == 1) & (y_true == 0)).sum())

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    logger.info("Evaluation metrics: AUROC=%.4f  AUPRC=%.4f  TP=%d FN=%d TN=%d FP=%d",
                auroc, auprc, TP, FN, TN, FP)

    with open(os.path.join(args.out, "confusion_summary.tsv"), "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["TP", "FN", "TN", "FP", "threshold", "auroc", "auprc"])
        writer.writerow([TP, FN, TN, FP, args.threshold, auroc, auprc])

    # ------------------------------------------------------------------
    # Score distribution plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(pos_pred["predictions"], bins=20, alpha=0.6, label="Positives")
    plt.hist(neg_pred["predictions"], bins=20, alpha=0.6, label="Negatives")
    plt.xlabel("Perceptron score")
    plt.ylabel("Frequency")
    plt.title("Score distributions: positives vs negatives")
    plt.legend()
    plt.savefig(os.path.join(args.out, "score_distributions.png"))
    plt.close()

    # ------------------------------------------------------------------
    # Build true (disease, gene) pairs from the positive test set
    # (using the TP file as the source of truth)
    # ------------------------------------------------------------------
    logger.info("Building true disease->gene pairs from TP file")
    true_pairs = list(set(zip(
        pos_pred["destinations"].values,   # disease
        pos_pred["sources"].values,        # true gene
    )))
    logger.info("Unique (disease, gene) pairs to rank: %d", len(true_pairs))

    # Merge true genes into the candidate pool
    genes = list(set(genes) | set(pos_pred["sources"].values))
    genes = filter_genes_in_nodes(args.nodes, genes)
    logger.info("Candidate pool after node validation: %d genes", len(genes))

    # ------------------------------------------------------------------
    # Gene ranking: two-pass approach for efficiency
    #
    # Pass 1: Find the true_gene_score
    # Pass 2: Count all genes with higher scores
    #
    # This is memory-efficient for supercomputer runs:
    # - Does not accumulate all scores in memory
    # - Only stores the true_gene_score
    # - Counts on-the-fly while processing chunks
    # ------------------------------------------------------------------
    results = []
    logger.info("Starting disease->gene ranking for %d pairs (two-pass approach)",
                len(true_pairs))

    with tempfile.TemporaryDirectory() as tmp_dir:
        for d_idx, (disease, true_gene) in enumerate(true_pairs, 1):
            logger.info("[%d/%d] Ranking genes for disease %s - PASS 1: Finding true_gene_score",
                        d_idx, len(true_pairs), disease)

            true_gene_score = None

            # ---- PASS 1: Find the true_gene_score ----
            for chunk_start in range(0, len(genes), CHUNK_SIZE):
                gene_chunk = genes[chunk_start: chunk_start + CHUNK_SIZE]

                # Skip if true_gene already found
                if true_gene_score is not None:
                    break

                # Temporary edge file: disease <-> each gene in this chunk
                edge_path = os.path.join(
                    tmp_dir, "chunk_%d_%d.tsv" % (d_idx, chunk_start)
                )
                with open(edge_path, "w", newline="") as fh:
                    writer = csv.writer(fh, delimiter="\t")
                    writer.writerow(["subject", "predicate", "object"])
                    writer.writerows(
                        (disease, "biolink:related_to", g) for g in gene_chunk
                    )

                gChunk = Graph.from_csv(
                    directed=False,
                    node_path=args.nodes,
                    edge_path=edge_path,
                    node_list_separator="\t",
                    edge_list_separator="\t",
                    verbose=False,
                    nodes_column="id",
                    node_list_node_types_column="category",
                    default_node_type="biolink:NamedThing",
                    sources_column="subject",
                    destinations_column="object",
                    edge_list_edge_types_column="predicate",
                    name="CandidateChunk",
                )

                chunk_preds = model.predict_proba(
                    graph=gChunk,
                    node_features=embedder,
                    support=gTrain,
                    return_predictions_dataframe=True,
                    return_node_names=True,
                )

                # Look for true_gene in this chunk
                for gene, score in zip(chunk_preds["sources"],
                                       chunk_preds["predictions"]):
                    if gene == true_gene:
                        true_gene_score = float(score)
                        logger.info("  Found true_gene %s with score %.6f", true_gene, true_gene_score)
                        break

                # Release objects and delete the temp file
                del chunk_preds, gChunk
                os.remove(edge_path)

            # Check if true gene was found
            if true_gene_score is None:
                rank = None
                classification = "MISSING_TRUE_GENE"
                logger.warning("True gene %s not found for disease %s", true_gene, disease)
                results.append((disease, "biolink:related_to", true_gene, None, rank, classification))
                continue

            # ---- PASS 2: Count genes with higher scores ----
            logger.info("[%d/%d] Ranking genes for disease %s - PASS 2: Counting higher-scoring genes",
                        d_idx, len(true_pairs), disease)

            better_count = 0  # Number of genes scoring strictly higher than true_gene

            for chunk_start in range(0, len(genes), CHUNK_SIZE):
                gene_chunk = genes[chunk_start: chunk_start + CHUNK_SIZE]

                # Temporary edge file: disease <-> each gene in this chunk
                edge_path = os.path.join(
                    tmp_dir, "chunk_%d_%d_pass2.tsv" % (d_idx, chunk_start)
                )
                with open(edge_path, "w", newline="") as fh:
                    writer = csv.writer(fh, delimiter="\t")
                    writer.writerow(["subject", "predicate", "object"])
                    writer.writerows(
                        (disease, "biolink:related_to", g) for g in gene_chunk
                    )

                gChunk = Graph.from_csv(
                    directed=False,
                    node_path=args.nodes,
                    edge_path=edge_path,
                    node_list_separator="\t",
                    edge_list_separator="\t",
                    verbose=False,
                    nodes_column="id",
                    node_list_node_types_column="category",
                    default_node_type="biolink:NamedThing",
                    sources_column="subject",
                    destinations_column="object",
                    edge_list_edge_types_column="predicate",
                    name="CandidateChunk",
                )

                chunk_preds = model.predict_proba(
                    graph=gChunk,
                    node_features=embedder,
                    support=gTrain,
                    return_predictions_dataframe=True,
                    return_node_names=True,
                )

                # Count genes with strictly higher scores (excluding true_gene itself)
                for gene, score in zip(chunk_preds["sources"],
                                       chunk_preds["predictions"]):
                    score = float(score)
                    if score > true_gene_score and gene != true_gene:
                        better_count += 1

                # Release objects and delete the temp file
                del chunk_preds, gChunk
                os.remove(edge_path)

            # Calculate final rank
            rank = better_count + 1  # Rank = 1 + number of genes scoring higher

            if rank <= 10:
                classification = "Top10-TP (rank=%d)" % rank
            elif rank <= 50:
                classification = "Top50-TP (rank=%d)" % rank
            else:
                classification = "TP-outside-Top50 (rank=%d)" % rank

            logger.info("  Final rank: %d (genes with higher scores: %d)", rank, better_count)

            results.append((
                disease,
                "biolink:related_to",
                true_gene,
                true_gene_score,
                rank,
                classification,
            ))

    # ------------------------------------------------------------------
    # Save gene ranking results
    # ------------------------------------------------------------------
    out_path = os.path.join(args.out, "gene_ranks.tsv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["disease", "predicate", "gene",
                         "score", "rank", "classification"])
        writer.writerows(results)

    logger.info("Gene ranking complete: results written to %s", out_path)
    logger.info("=" * 70)
    logger.info("Pipeline completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
