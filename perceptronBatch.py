import argparse
import tempfile
import csv
import os
import logging
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ensmallen import Graph
from sklearn.metrics import roc_auc_score, average_precision_score

from embiggen.embedders.ensmallen_embedders import FirstOrderLINEEnsmallen
from embiggen.edge_prediction import PerceptronEdgePrediction, RandomForestEdgePrediction, GradientBoostingEdgePrediction



# =========================================================
# Configuration
# =========================================================
CHUNK_SIZE = 50_000
TOPK = 50


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    filename='perceptron.log',
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================
# Graph loader
# =========================================================
def load_graph(nodesF, edgesF, name):
    return Graph.from_csv(
        directed=False,
        node_path=nodesF,
        edge_path=edgesF,
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


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--pos", default="TP_hgnc_mondo_edges.tsv")
    parser.add_argument("--neg", default="TN_hgnc_mondo_edges1.tsv")
    parser.add_argument("--candidates", default="geneCandidates.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -----------------------------------------------------
    # Load graphs
    # -----------------------------------------------------
    logger.warning("Loading graphs")
    gTrain = load_graph(args.nodes, args.train, "Train KG")
    gPos = load_graph(args.nodes, args.pos, "Positive Test KG")
    gNeg = load_graph(args.nodes, args.neg, "Negative Test KG")

    # -----------------------------------------------------
    # Load candidate genes
    # -----------------------------------------------------
    with open(args.candidates) as f:
        genes = [l.strip() for l in f if l.strip()]
    logger.warning("Loaded %d candidate genes", len(genes))

    # -----------------------------------------------------
    # Train embedder + model
    # -----------------------------------------------------
    logger.info("Training embeddings and edge prediction model")
    embedder = FirstOrderLINEEnsmallen(
        embedding_size=128,
        epochs=10,
    )

    #model = PerceptronEdgePrediction(edge_embeddings="Hadamard")
    #model = RandomForestEdgePrediction()
    model = GradientBoostingEdgePrediction()
    #gTrain = gTrain.remove_disconnected_nodes()

    model.fit(
        graph=gTrain,
        node_features=embedder,
    )

    # -----------------------------------------------------
    # Predict positives / negatives (once)
    # -----------------------------------------------------
    logger.warning("Scoring positive test edges")
    pos_pred = model.predict_proba(
        graph=gPos,
        node_features=embedder,
        support=gTrain,
        return_predictions_dataframe=True,
        return_node_names=True,
    )
    pos_pred["label"] = 1

    logger.warning("Scoring negative test edges")
    neg_pred = model.predict_proba(
        graph=gNeg,
        node_features=embedder,
        support=gTrain,
        return_predictions_dataframe=True,
        return_node_names=True,
    )
    neg_pred["label"] = 0

    preds = pd.concat([pos_pred, neg_pred], ignore_index=True)
    score_col = preds.columns[-2]
    preds = preds.rename(columns={score_col: "score"})

    preds.to_csv(os.path.join(args.out, "edge_predictions.tsv"), sep="\t", index=False)

    # -----------------------------------------------------
    # Confusion matrix + metrics
    # -----------------------------------------------------
    y_true = preds["label"].astype(int).values
    y_score = preds["prediction"].astype(float).values
    y_hat = (y_score >= args.threshold).astype(int)

    TP = int(((y_hat == 1) & (y_true == 1)).sum())
    FN = int(((y_hat == 0) & (y_true == 1)).sum())
    TN = int(((y_hat == 0) & (y_true == 0)).sum())
    FP = int(((y_hat == 1) & (y_true == 0)).sum())

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    with open(os.path.join(args.out, "confusion_summary.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["TP", "FN", "TN", "FP", "threshold", "auroc", "auprc"])
        writer.writerow([TP, FN, TN, FP, args.threshold, auroc, auprc])

    # -----------------------------------------------------
    # Score distributions
    # -----------------------------------------------------
    plt.hist(pos_pred["prediction"], bins=20, alpha=0.6, label="Positives")
    plt.hist(neg_pred["prediction"], bins=20, alpha=0.6, label="Negatives")
    plt.xlabel("Perceptron score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(args.out, "score_distributions.png"))
    plt.close()
    logger.warning("Starting one-pass disease→gene ranking")

    true_pairs = dict(
        zip(
            pos_pred.iloc[:, 2].values,  # disease
            pos_pred.iloc[:, 1].values,  # true gene
        )
    )

    genes = set(genes)
    true_genes = set(pos_pred.iloc[:,1].values)
    genes |= true_genes
    genes = list(genes)
    logger.info("true genes : %s", genes)

    results = []
    tmp_dir = tempfile.mkdtemp()

    for d_idx, (disease, true_gene) in enumerate(true_pairs.items(), 1):
        true_gene_score = None
        better_than_true = 0
        topk_heap = []

        for i in range(0, len(genes), CHUNK_SIZE):
            gene_chunk = genes[i:i + CHUNK_SIZE]

            edge_path = os.path.join(tmp_dir, f"{disease}_{i}.tsv")
            with open(edge_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
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

            preds = model.predict_proba(
                graph=gChunk,
                node_features=embedder,
                support=gTrain,
                return_predictions_dataframe=True,
                return_node_names=True,
            )

            gene_col = preds.columns[1]
            score_col = preds.columns[0]


            

            for _, row in preds.iterrows():
                gene = row[gene_col]
                score = float(row[score_col])

                # Maintain Top-K heap
                if len(topk_heap) < TOPK:
                    heapq.heappush(topk_heap, (score, gene))
                else:
                    heapq.heappushpop(topk_heap, (score, gene))

                # Ranking logic
                if gene == true_gene:
                    true_gene_score = score
                elif true_gene_score is None:
                    # True gene not seen yet, defer comparison
                    pass
                elif score > true_gene_score:
                    better_than_true += 1

            del preds, gChunk

        # -----------------------------------------
        # Final rank determination
        # -----------------------------------------
        if true_gene_score is None:
            rank = None
            classification = "MISSING_TRUE_GENE"
            logger.warning("True gene not scored for disease %s", disease)
        else:
            rank = better_than_true + 1

            if rank <= 10:
                classification = f"Top10-TP (rank={rank})"
            elif rank <= 50:
                classification = f"Top50-TP (rank={rank})"
            else:
                classification = f"TP-outside-Top50 (rank={rank})"

        results.append(
            (
                disease,
                "biolink:related_to",
                true_gene,
                true_gene_score,
                rank,
                classification,
            )
        )

    # -----------------------------------------
    # Save results
    # -----------------------------------------
    out_path = os.path.join(args.out, "gene_ranks.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["disease", "predicate", "gene", "score", "rank", "classification"]
        )
        writer.writerows(results)

    logger.info("One-pass disease→gene ranking completed")


    """with open(os.path.join(args.out, "top_genes.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["disease", "predicate", "gene", "score", "classification"])
        writer.writerows(top_gene_results)

    success_rate = (disease_hits / len(diseases)) * 100 if diseases else 0.0

    with open(os.path.join(args.out, "topk_summary.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "total_diseases",
                "diseases_with_true_in_top",
                "diseases_with_true_in_top10",
                "success_rate_percent",
            ]
        )
    writer.writerow([len(diseases), disease_hits, top10, success_rate])"""

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
