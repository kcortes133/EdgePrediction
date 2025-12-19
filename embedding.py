import csv, random, requests, time
import matplotlib.pyplot as plt
from ensmallen import Graph
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Tuple, List
import rareDiseaseSubsets, phenSim

from scipy.spatial.distance import cosine
from embiggen.edge_prediction import PerceptronEdgePrediction
from embiggen.edge_prediction.edge_prediction_sklearn import RandomForestEdgePrediction
from embiggen.embedders.ensmallen_embedders import FirstOrderLINEEnsmallen, TransEEnsmallen
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import ubergraphIC, ubergraphCached, pathInvestigation


# Train embeddings
###
def train_embeddings(g, outFolder):
    embedding = FirstOrderLINEEnsmallen().fit_transform(g)
    #embedding = TransEEnsmallen().fit_transform(g)
    embeddingDF = embedding.get_all_node_embedding()[0]
    embeddingDF.to_csv(outFolder+'FLOE_embeddings.csv')
    return embeddingDF

'''
def evaluate_embeddings(
    embedding_file: str,
    pos_edges_file: str,
    neg_edges_file: str,
    figure_path: str
    #threshold: float = 0.5
) -> Dict[str, int]:
    """
    Evaluate embeddings by classifying positive and negative edges
    based on cosine similarity.

    Parameters
    ----------
    embedding_file : str
        CSV file of embeddings (node_id, dim1, dim2, ...).
    pos_edges_file : str
        CSV file of positive test edges (removed edges).
    neg_edges_file : str
        CSV file of negative test edges (sampled).
    figure_path : str
        File path to save histogram of similarity scores.
    threshold : float, optional (default=0.5)
        Cosine similarity threshold for classification.

    Returns
    -------
    results : dict
        Dictionary with counts of TP, TN, FP, FN.
    """
    threshold = 0.5

    # --- Load embeddings ---
    with open(embedding_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    rows.pop(0)  # drop header
    embeddings = {row[0]: np.array(list(map(float, row[1:]))) for row in rows}

    # --- Helper to compute scores ---
    def compute_scores(edge_file: str) -> List[float]:
        with open(edge_file, "r") as f:
            reader = csv.reader(f)
            edges = list(reader)
        edges.pop(0)  # drop header
        scores = []
        for edge in edges:
            if len(edge) > 1:
                node1, node2 = edge[-2], edge[-1]  # subj, obj
                if node1 in embeddings and node2 in embeddings:
                    sim = cosine_similarity(
                        embeddings[node1].reshape(1, -1),
                        embeddings[node2].reshape(1, -1)
                    )[0, 0]
                    scores.append(sim)
        return scores

    # --- Scores for positives and negatives ---
    pos_scores = compute_scores(pos_edges_file)
    neg_scores = compute_scores(neg_edges_file)

    # --- Classification ---
    TP = sum(1 for s in pos_scores if s >= threshold)
    FN = sum(1 for s in pos_scores if s < threshold)
    TN = sum(1 for s in neg_scores if s < threshold)
    FP = sum(1 for s in neg_scores if s >= threshold)

    # --- Plot distributions ---
    plt.hist(pos_scores, bins=20, alpha=0.6, label="Positives")
    plt.hist(neg_scores, bins=20, alpha=0.6, label="Negatives")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

    return {"TP": TP, "FN": FN, "TN": TN, "FP": FP}
'''

def evaluate_embeddings(
    embedding_file: str,
    pos_edges_file: str,
    neg_edges_file: str,
    figure_path: str,
    results_csv: str,
    threshold: float = 0.5
) -> Tuple[Dict[str, int], List[Tuple[str, str, str, float, str]]]:
    """
    Evaluate embeddings by classifying positive and negative edges
    based on cosine similarity.

    Parameters
    ----------
    embedding_file : str
        CSV file of embeddings (node_id, dim1, dim2, ...).
    pos_edges_file : str
        CSV file of positive test edges (removed edges).
    neg_edges_file : str
        CSV file of negative test edges (sampled).
    figure_path : str
        File path to save histogram of similarity scores.
    results_csv : str
        File path to save detailed results (triples, score, classification).
    threshold : float, optional (default=0.5)
        Cosine similarity threshold for classification.

    Returns
    -------
    results : dict
        Dictionary with counts of TP, TN, FP, FN.
    triples : list of tuples
        Each entry is (subject, predicate, object, score, classification).
    """

    # --- Load embeddings ---
    embeddingDF = pd.read_csv(embedding_file, index_col=0)

    # --- Helper to compute scores + labels ---
    def compute_scores(edge_file: str, is_positive: bool) -> List[Tuple[str, str, str, float, str]]:
        with open(edge_file, "r") as f:
            if is_positive: reader = csv.reader(f, delimiter="\t")
            else: reader = csv.reader(f)
            edges = list(reader)
        edges.pop(0)  # drop header

        results = []
        for edge in edges:
            if len(edge) > 1:
                if is_positive:
                    subj, pred, obj = edge[-2], edge[2], edge[-1]
                else: subj, pred, obj = edge[0], edge[1], edge[2]
                score = cosine(embeddingDF.loc[subj],embeddingDF.loc[obj])
                # classify
                if is_positive:
                    classification = "TP" if score >= threshold else "FN"
                else:
                    classification = "FP" if score >= threshold else "TN"
                results.append((subj, pred, obj, score, classification))
        return results

    # --- Scores for positives and negatives ---
    pos_results = compute_scores(pos_edges_file, is_positive=True)
    neg_results = compute_scores(neg_edges_file, is_positive=False)

    # --- Classification counts ---
    TP = sum(1 for _, _, _, _, c in pos_results if c == "TP")
    FN = sum(1 for _, _, _, _, c in pos_results if c == "FN")
    TN = sum(1 for _, _, _, _, c in neg_results if c == "TN")
    FP = sum(1 for _, _, _, _, c in neg_results if c == "FP")

    # --- Save results to CSV ---
    all_results = pos_results + neg_results
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object", "score", "classification"])
        writer.writerows(all_results)

    # --- Plot distributions ---
    plt.hist([r[3] for r in pos_results], bins=20, alpha=0.6, label="Positives")
    plt.hist([r[3] for r in neg_results], bins=20, alpha=0.6, label="Negatives")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

    return {"TP": TP, "FN": FN, "TN": TN, "FP": FP}, all_results




def negative_sampling(
        kept_edges_file: str,
        removed_edges_file: str,
        output_folder: str,
        write: bool = True
) -> List[List[str]]:
    """
    Generate negative edges for link prediction using random negative sampling.

    Negative samples are generated by swapping objects/subjects of existing
    triples while ensuring the new triple does not exist in the training or test set.

    Parameters
    ----------
    kept_edges_file : str
        Path to the file containing kept/training edges (CSV format).
    removed_edges_file : str
        Path to the file containing removed/test edges (CSV format).
    output_folder : str
        Directory where the negative edge file will be saved if `write=True`.
    write : bool, optional (default=True)
        Whether to write the generated negative edges to `negEdges.csv`.

    Returns
    -------
    neg_edges : List[List[str]]
        List of generated negative triples in the form [subject, predicate, object].
    """
    predicates = [
        "biolink:has_phenotype",
        "biolink:gene_associated_with_condition",
        "biolink:contributes_to",
        "biolink:associated_with",
        "biolink:causes"
    ]

    # Load training edges
    with open(kept_edges_file, "r") as f:
        train_edges = list(csv.reader(f, delimiter=","))
        train_edges.pop(0)  # remove header

    # Load test edges
    with open(removed_edges_file, "r") as f:
        test_edges = list(csv.reader(f, delimiter=","))
        header = test_edges.pop(0)

    # Convert edges to triples
    train_triples = [[e[-2], e[2], e[-1]] for e in train_edges]
    test_triples = [[e[-2], e[2], e[-1]] for e in test_edges]

    # Store triple types for quick existence checks
    triple_types = [
        [subj.split(":")[0], pred, obj.split(":")[0]]
        for subj, pred, obj in test_triples
    ]

    neg_edges = []

    # Generate negative edges until we match test set size
    while len(neg_edges) < len(test_edges):
        rand_t1 = random.choice(test_triples)
        rand_t2 = random.choice(test_triples)

        subj1_prefix, pred1, obj1_prefix = rand_t1[0].split(":")[0], rand_t1[1], rand_t1[2].split(":")[0]
        subj2_prefix, pred2, obj2_prefix = rand_t2[0].split(":")[0], rand_t2[1], rand_t2[2].split(":")[0]

        # Case 1: replace object of rand_t1 with object of rand_t2
        if [subj1_prefix, pred1, obj2_prefix] in triple_types:
            candidate = [rand_t1[0], pred1, rand_t2[2]]
            if candidate not in train_triples and candidate not in test_triples:
                neg_edges.append(candidate)

        # Case 2: replace object of rand_t2 with object of rand_t1
        if [subj2_prefix, pred2, obj1_prefix] in triple_types:
            candidate = [rand_t2[0], pred2, rand_t1[2]]
            if candidate not in train_triples and candidate not in test_triples:
                neg_edges.append(candidate)

    # Optionally write negatives to CSV
    if write:
        out_file = output_folder + "negEdges.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "predicate", "object"])
            writer.writerows(neg_edges)

    return neg_edges


def negativeSampling(keptEdges, removedEdges, folder, write):
    negEdges = []
    with open(keptEdges, 'r') as f:
        trainEdges = list(csv.reader(f, delimiter='\t'))
        trainEdges.pop(0)
    with open(removedEdges, 'r') as f:
        testEdges = list(csv.reader(f, delimiter='\t'))
        header = testEdges.pop(0)
    testTriples = []
    trainTriples = []
    tripleTypes = []
    # pick random object s.t. (s,p,o)type exists in tripleTypes but specific edge does not exist in graph or removed edges
    for edge in testEdges:
        obj = edge[-1]
        subj = edge[-2]
        pred = edge[2]
        #predCount[pred]+=1
        testTriples.append([subj, pred, obj])
        tripleTypes.append([subj.split(':')[0], pred, obj.split(':')[0]])
    for edge in trainEdges:
        obj = edge[-1]
        subj = edge[-2]
        pred = edge[2]
        trainTriples.append([subj, pred, obj])

    while len(testEdges) > len(negEdges):
        randTriple1 = random.choice(testTriples)
        randTriple2 = random.choice(testTriples)

        sub1 = randTriple1[0].split(':')[0]
        sub2 = randTriple2[0].split(':')[0]

        pred1 = randTriple1[1]
        pred2 = randTriple2[1]

        obj1 = randTriple1[2].split(':')[0]
        obj2 = randTriple2[2].split(':')[0]

        #if sub1 == sub2 and pred1 == pred2 and obj1 == obj2:
        if [sub1, pred1, obj2] in tripleTypes:
            newNegTriple = [randTriple1[0], pred1, randTriple2[2]]
            if newNegTriple not in trainTriples and newNegTriple not in testTriples:
                #if negPredCounts[pred1] < predCount[pred1]+5:
                negEdges.append(newNegTriple)
                #negPredCounts[pred1]+=1
        if [sub2, pred2, obj1] in tripleTypes:
            newNegTriple = [randTriple2[0], pred2, randTriple1[2]]
            if newNegTriple not in trainTriples and newNegTriple not in testTriples:
                #if negPredCounts[pred1] < predCount[pred1] +5:
                negEdges.append(newNegTriple)
                #negPredCounts[pred1]+=1

    if write:
        with open(folder + 'negEdges.csv', 'w') as f:
            writer =csv.writer(f)
            writer.writerow(['subject', 'predicate', 'object'])
            writer.writerows(negEdges)

    return negEdges

import csv
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_embeddingsTOP10(
    graph: Graph,
    embedding_file: str,
    pos_edges_file: str,
    neg_edges_file: str,
    figure_path: str,
    results_csv: str,
    top_gene_csv: str,
    summary_csv: str,
    outFolder,
    threshold: float = 0.5
) -> Tuple[Dict[str, int], List[Tuple[str, str, str, float, str]], Dict[str, float]]:
    """
    Evaluate embeddings by classifying positive and negative edges,
    and extracting top 10 most similar HGNC genes for each MONDO disease.
    Marks as Top10-TP if the true disease-gene association is in the top 10.
    Also produces a summary of success rates across diseases.

    Parameters
    ----------
    embedding_file : str
        CSV file of embeddings (node_id, dim1, dim2, ...).
    pos_edges_file : str
        CSV file of positive test edges (removed edges).
    neg_edges_file : str
        CSV file of negative test edges (sampled).
    figure_path : str
        File path to save histogram of similarity scores.
    results_csv : str
        File path to save detailed edge classification results.
    top_gene_csv : str
        File path to save top-10 disease–gene results for each disease.
    summary_csv : str
        File path to save summary statistics about top-10 hits.
    threshold : float, optional (default=0.5)
        Cosine similarity threshold for classification.

    Returns
    -------
    results : dict
        Counts of TP, TN, FP, FN.
    triples : list of tuples
        Each entry is (subject, predicate, object, score, classification).
    summary : dict
        Summary of how many diseases had their true gene in top 10.
    """

    # --- Load embeddings ---
    embeddingDF = pd.read_csv(embedding_file, index_col=0)

    # --- Helper to compute scores + labels ---
    def compute_scores(edge_file: str, is_positive: bool) -> List[Tuple[str, str, str, float, str]]:
        with open(edge_file, "r") as f:
            reader = csv.reader(f, delimiter="\t" if is_positive else ",")
            edges = list(reader)
        edges.pop(0)  # drop header

        results = []
        for edge in edges:
            if len(edge) > 1:
                if is_positive:
                    subj, pred, obj = edge[-2], edge[2], edge[-1]
                else:
                    subj, pred, obj = edge[0], edge[1], edge[2]

                if subj in embeddingDF.index and obj in embeddingDF.index:
                    score = cosine_similarity(
                        embeddingDF.loc[subj].values.reshape(1, -1),
                        embeddingDF.loc[obj].values.reshape(1, -1),
                    )[0, 0]

                    if is_positive:
                        classification = "TP" if score >= threshold else "FN"
                    else:
                        classification = "FP" if score >= threshold else "TN"

                    results.append((subj, pred, obj, score, classification))
        return results

    # --- Scores for positives and negatives ---
    pos_results = compute_scores(pos_edges_file, is_positive=True)
    neg_results = compute_scores(neg_edges_file, is_positive=False)

    # --- Classification counts ---
    TP = sum(1 for _, _, _, _, c in pos_results if c == "TP")
    FN = sum(1 for _, _, _, _, c in pos_results if c == "FN")
    TN = sum(1 for _, _, _, _, c in neg_results if c == "TN")
    FP = sum(1 for _, _, _, _, c in neg_results if c == "FP")
    print('TP: ', TP, ' FN: ', FN, ' TN: ', TN, ' FP: ', FP)
    plt.hist([r[3] for r in pos_results], bins=20, alpha=0.6, label="Positives")
    plt.hist([r[3] for r in neg_results], bins=20, alpha=0.6, label="Negatives")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

    # --- Save edge-level classification ---
    all_results = pos_results + neg_results
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object", "score", "classification"])
        writer.writerows(all_results)

    # --- Top 10 HGNC genes per MONDO disease ---
    true_pairs = {obj: subj for subj, pred, obj, score, classification in pos_results}
    #true_pairs = {(obj, subj) for subj, _, obj, _, _ in pos_results if obj.startswith("MONDO:")}
    diseases = [obj for subj, _, obj, _, c in pos_results if obj.startswith("MONDO:") and c =='TP']
    genes1 = [node for node in embeddingDF.index if node.startswith("HGNC:")]
    genes = rareDiseaseSubsets.getGenesWODisConns('removedEdges.tsv', 'Genes Without Disease Conn.txt')
    print('Old Num Genes : ', len(genes1))

    print('Num genes: ', len(genes))
    # 44454
    print('Num true pairs: ', len(true_pairs))
    #12504
    print('Num diseases: ', len(diseases))
    # 7248
    # could filter out diseases that arent leaf nodes
    '''
    s = time.time()
    for disease in true_pairs:
        paths = pathInvestigation.get_n_shortest_paths(graph, true_pairs[disease] , disease, 4)
        print(disease, true_pairs[disease])
        print(paths)
        print(time.time() - s)'''




    top_gene_results = []
    disease_hits = 0
    total_diseases = 0
    top10 = 0
    #diseasePhens = phenSim.save_dis_phenotypes_to_file(diseases, 'disease_phenotypes.json')
    #candGenes = phenSim.load_gene_phenotypes_from_file('gene_phenotypes.json')

    start = time.time()
    for disease in diseases:
        disease_vec = embeddingDF.loc[disease].values.reshape(1, -1)

        similarities = []

        # get top 200 phenotyically similar genes
        #genesPhenTop = phenSim.get_top_similar_genes_from_file(disease, 'gene_phenotypes.json')
        #genesPhenTop = ubergraphIC.rank_genes_by_similarity(disease)
        #genesPhenTop = ubergraphCached.compute_gene_disease_similarity(candGenes, {disease: diseasePhens[disease] })

        #print(genesPhenTop)
        #phenSimGenes = [tup[0] for tup in genesPhenTop]

        for gene in genes:
            gene_vec = embeddingDF.loc[gene].values.reshape(1, -1)
            sim = cosine_similarity(gene_vec, disease_vec)[0, 0]
            similarities.append((disease, "biolink:related_to", gene, sim))

        topG = sorted(similarities, key=lambda x: x[3], reverse=True)[:50]

        #cosineSimGenes = [tup[2] for tup in top10]

        true_in_top10 = False

        for rank, (dis, pred, g, sim) in enumerate(topG):
            if true_pairs[dis] == g:
                if rank <10:
                    classification = f"Top10-TP (rank={rank})"
                    top10 += 1
                else:
                    classification = f"Top50-TP (rank={rank})"
                true_in_top10 = True
                top_gene_results.append((dis, pred, g, sim, classification))
                break  # optional: stop once found
            else:
                classification = "Not"

        total_diseases += 1
        if true_in_top10:
            disease_hits += 1

    # Save disease–gene ranking
    with open(top_gene_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "predicate", "gene", "score", "classification"])
        writer.writerows(top_gene_results)

    # --- Summary ---
    success_rate = (disease_hits / total_diseases) * 100 if total_diseases > 0 else 0.0
    summary = {
        "total_diseases": total_diseases,
        "diseases_with_true_in_top": disease_hits,
        "diseases_with_true_in_top10": top10,
        "success_rate_percent": success_rate,
    }

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    # --- Plot distributions ---
    plt.hist([r[3] for r in pos_results], bins=20, alpha=0.6, label="Positives")
    plt.hist([r[3] for r in neg_results], bins=20, alpha=0.6, label="Negatives")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

    return {"TP": TP, "FN": FN, "TN": TN, "FP": FP}, all_results

def topPhenSimGenes(disease):
    base_url = "https://api.monarchinitiative.org/v3/api/semsim/search"
    encoded_mondo = disease.replace(":", "%3A")
    url = f"{base_url}/{encoded_mondo}/Human%20Genes?metric=ancestor_information_content&directionality=bidirectional&limit=50"
    print(url)
    # Call API
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return []

    data = response.json()
    geneResults = []
    # Parse response
    results = []
    for item in data:
        subject = item.get("subject", {})
        results.append({
            "gene_id": subject.get("id", ""),
            "gene_name": subject.get("name", ""),
            "full_name": subject.get("full_name", ""),
            "score": item.get("score", None)
        })
        geneResults.append(subject.get("id", ""))

    return geneResults
