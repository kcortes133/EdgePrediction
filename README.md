# TRIM: Topology Refinement via Information-content-guided Model pruning

Code, data and results for the manuscript

> Cortes KG, Sundar S, Gehrke S, Korn DR, Caufield H, Schaper K, Reese J, Mungall C, Haendel M.
> **TRIM: Topology Refinement via Information-content-guided Model pruning.**
> Submitted to *Journal of Biomedical Semantics* (2026). DOI: _TBD_

## Summary

Biomedical knowledge graphs (KGs) are built on ontologies whose broad, general terms
(e.g. *Phenotypic abnormality*, *biological process*) collect very many edges and become
hubs. TRIM tests how these hubs affect embedding-based gene–disease prediction. Gene
Ontology (GO), Human Phenotype Ontology (HP) and Mondo (MONDO) terms in the Monarch KG
(September 2025 release) are scored by Ubergraph's normalized information content (nIC),
and terms below a threshold (nIC 40, 60, 80, 100) are removed with all their edges. Each
graph is embedded with first-order LINE, a perceptron is trained to predict held-out
gene–disease edges, and each held-out disease is used to rank candidate genes. A
random-node-removal control removes the same number of nodes as nIC80.

| Subset | Nodes removed | Edges removed | F1 | AUROC | Median rank of true gene | Top 10 | Top 50 |
|---|---|---|---|---|---|---|---|
| None | – | – | 0.690 | 0.791 | 1,549 | 10.8% | 18.6% |
| nIC40 | 100 | 341,045 (2.4%) | 0.688 | 0.799 | 1,280 | 12.0% | 19.7% |
| nIC60 | 747 | 1,424,046 (9.8%) | 0.697 | 0.824 | 434 | 14.9% | 26.4% |
| nIC80 | 6,448 | 2,763,428 (19.1%) | 0.691 | 0.826 | 302 | 15.1% | 29.3% |
| nIC100 | 33,548 | 4,167,304 (28.8%) | 0.666 | 0.775 | 294 | 9.3% | 24.5% |

Removing a small number of general, highly connected ontology terms improves the ranking
of the true gene (3,048 held-out gene–disease pairs; 2,591 of them involve a rare
disease), while binary classification changes little and stays within the variation of
the random-removal control (F1 0.716 ± 0.020). Full results: Tables 2–3 and Figures 3–5
in `analysis/results/paper_july/`.

## Pipeline

```
Monarch KG (Sept 2025, KGX TSV)
   │  src/preprocessing/editICNodes.py         nIC filtering per threshold      → Table 1, Fig 2
   │  src/preprocessing/testsetGenerator.py    hold out TP edges, sample TN     → Algorithm 1
   │  src/preprocessing/makeGeneCandidates.py  candidate genes for ranking
   │  src/preprocessing/randomNodeRemoval.py   random-node-removal control (6,448 nodes)
   ▼
src/ml/perceptronBatch.py                       LINE embedding, perceptron, scoring, gene ranking
   ▼
src/analysis/remove_tn_overlap.py               drop 4 TN pairs that are also TP pairs
src/analysis/paper_figures.py                   Tables 2–3, Figures 3–5
```

## Manuscript items

| Item | Script | Output / data |
|---|---|---|
| Fig 1 (FBN1 – Marfan syndrome hierarchy) | drawn by hand from Monarch association counts | — |
| Table 1 (nodes and edges removed) | `src/preprocessing/editICNodes.py` | `analysis/reports/kg_ic_filter_summary{40,60,80,100}.txt`, `analysis/reports/ic_filter_logs/` |
| Fig 2 (nIC distributions) | `src/preprocessing/editICNodes.py` (`plot_multi_ontology_ic_histogram`) | `fig2_nic_distribution.png` (written when the script runs) |
| Algorithm 1 (negative sampling) | `src/preprocessing/testsetGenerator.py` (`negativeSampling`) | `data/TP_hgnc_mondo_edges.tsv`, `data/TN_hgnc_mondo_edges1.tsv` |
| Table 2 (F1, AUROC, confusion counts) | `src/analysis/paper_figures.py` | `analysis/results/paper_july/table2_classification.tsv` |
| Table 3 (median rank, top 10/20/50) | `src/analysis/paper_figures.py` | `analysis/results/paper_july/ranking_summary.tsv` |
| Fig 3 (score histograms, confusion counts) | `src/analysis/paper_figures.py` | `analysis/results/paper_july/fig3_scores_confusion.{png,pdf}` |
| Fig 4 (top-k, all and rare diseases) | `src/analysis/paper_figures.py` | `analysis/results/paper_july/fig4_topk.{png,pdf}` |
| Fig 5 (F1 vs random-removal control) | `src/analysis/paper_figures.py` | `analysis/results/paper_july/fig5_f1_control.{png,pdf}` |
| Rare-disease annotation | `src/preprocessing/rareDiseaseSubsets.py` (Neo4j) | `analysis/data/Rare Disease Annotation.csv` |
| Discussion: GO "nucleoplasm" example | `src/analysis/termICLookup.py` | `analysis/results/term_ic/term_ic_lookup.json` |

## Repository layout

```
src/
  preprocessing/
    editICNodes.py         query Ubergraph nIC; drop GO/HP/MONDO terms below the threshold
    testsetGenerator.py    hold out TP edges (25%), build *.TestSet.tsv graphs, sample TN pairs
    makeGeneCandidates.py  candidate gene list (protein-coding genes without disease links + held-out genes)
    randomNodeRemoval.py   random-node-removal control graphs
    rareDiseaseSubsets.py  rare-disease annotation and gene queries (local Neo4j copy of the KG)
    editKG.py, queries.py, neo4jConfig.py, neo4jConnection.py   helpers
  ml/
    perceptronBatch.py     embed, train, score test edges, rank candidate genes (as run on the HPC)
  analysis/
    paper_figures.py       Tables 2–3 and Figures 3–5 (+ consistency check of the runs)
    remove_tn_overlap.py   remove the 4 TN pairs that also occur in the TP set; recompute metrics
    termICLookup.py        Ubergraph nIC and KG counts for single terms
    topRankSummary.py, topPredAnalysis.py, perceptronInvest.py, testSetAnalysis.py,
    rareDisesaseAnalysis.py, PerceptronAnalyisis.py, confSummary.py,
    f1_comparison_refresh.py   supplementary and exploratory analyses (not in the manuscript)
data/                              held-out TP / TN gene–disease test sets (see data/README.md)
analysis/
  data/                            candidate gene list, rare-disease annotations
  reports/                         nIC filter summaries and logs (Table 1)
  results/paper_july/              Tables 2–3, Figures 3–5
  results/results_*/               earlier runs (March 2026), not used in the manuscript
results_{none,40,60,80,100}_ranks/ perceptron runs reported in the manuscript (July 2026)
monarch_percep_rand{,7,13}_80/, results_13_Rand_80_ranks/   random-removal control runs
scripts/slurm/                     HPC job scripts (UNC Longleaf)
```

## Installation

Requires Python ≥ 3.12 (HPC runs used 3.12.1).

```bash
git clone https://github.com/kcortes133/EdgePrediction.git
cd EdgePrediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the experiments

Download the Monarch KG first (see [data/README.md](data/README.md)) and run all commands
from the repository root.

```bash
# 1. nIC filtering (set IC_THRESHOLD in the script; run for 40, 60, 80, 100)
python src/preprocessing/editICNodes.py

# 2. Held-out test set and negative sampling (writes the *.TestSet.tsv training graphs).
#    Reuses data/TP_hgnc_mondo_edges.tsv; add --resample to draw a new held-out set.
python src/preprocessing/testsetGenerator.py

# 3. Candidate genes for ranking (writes geneCandidates_regenerated.txt; the manuscript
#    used analysis/data/geneCandidates.txt, see data/README.md)
python src/preprocessing/makeGeneCandidates.py

# 4. Random-node-removal control on the training graph, one run per seed (42, 7, 13)
python src/preprocessing/randomNodeRemoval.py 7

# 5. Embedding + perceptron + gene ranking, one run per graph.
#    On SLURM, scripts/slurm/rerun_paper.sh runs all five subsets with a leakage check.
python src/ml/perceptronBatch.py \
    --nodes      monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_nodes.filtered_80.tsv \
    --train      monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_edges.filtered_80.TestSet.tsv \
    --pos        data/TP_hgnc_mondo_edges.tsv \
    --neg        data/TN_hgnc_mondo_edges1.tsv \
    --candidates analysis/data/geneCandidates.txt \
    --out        results_80_ranks

# 6. Remove the 4 TN pairs that also occur in the TP set and recompute the metrics
#    (originals are kept in _archive/pre_overlap_removal/)
python src/analysis/remove_tn_overlap.py results_{none,40,60,80,100}_ranks \
    monarch_percep_rand{,7,13}_80 results_13_Rand_80_ranks

# 7. Tables 2–3 and Figures 3–5
python src/analysis/paper_figures.py --runs . --suffix _ranks --out analysis/results/paper_july
```

## Methods details

- **nIC.** Ubergraph property `normalizedInformationContent`, queried from
  `https://ubergraph.apps.renci.org/sparql` on 7 January 2026 (see
  `analysis/reports/ic_filter_logs/`). Only GO, HP and MONDO nodes are scored; nodes with
  nIC below the threshold are removed with all their edges; nodes without a score are kept.
- **Test set.** 25% (random seed 42) of the HGNC → MONDO edges present in all five graph
  versions: 3,201 edges, 3,048 unique gene–disease pairs, 2,343 genes, 2,455 diseases
  (predicates `causes` 1,941, `gene_associated_with_condition` 1,164, `contributes_to` 96).
  These edges are removed from every graph before training.
- **Negatives.** 3,201 pairs made by recombining TP genes and diseases with the predicate
  type kept (Algorithm 1). Four of them link a gene and disease that are also a TP pair
  under another predicate and are removed before evaluation (3,197 remain;
  `data/TN_TP_overlap_pairs.tsv`).
- **Model.** `FirstOrderLINEEnsmallen(embedding_size=128, epochs=10)` and
  `PerceptronEdgePrediction(edge_embeddings="CosineSimilarity")` from GRAPE, otherwise
  defaults; pairs scoring ≥ 0.5 are classified as positive.
- **Gene ranking.** Each held-out disease is scored against `geneCandidates.txt` plus all
  TP genes (17,650 genes). Because the TP file is read as an undirected graph, the pool
  also contains the 2,455 test diseases and every pair is ranked in both orientations;
  `paper_figures.py` keeps the disease → gene rows only. Rank = 1 + number of candidates
  scoring higher than the true gene.
- **Random-removal control.** 6,448 nodes (the number removed at nIC80) sampled uniformly
  from all nodes except the test genes and diseases and removed from the training graph.
  This removes about 0.5% of edges (nIC80 removes 19.1%), so the control is matched on
  node count, not edge count.

`perceptronBatch.py` writes `edge_predictions.tsv`, `gene_ranks.tsv`,
`confusion_summary.tsv` and `score_distributions.png` to `--out`, and `perceptron.log` to
the working directory. `rareDiseaseSubsets.py` reads Neo4j credentials from `NEO4J_URI`,
`NEO4J_USER`, `NEO4J_PASSWORD` and `NEO4J_DB`.

## Notes

- Random-removal sampling uses `random.sample` over a set, so a seed reproduces the same
  nodes only with a fixed `PYTHONHASHSEED`.
- `_archive/` (not tracked) holds earlier and exploratory code (PrimeKG, ROBOKOP and
  model-organism experiments, earlier pipeline versions) that is not part of the manuscript.

## Citation

```bibtex
@article{trim2026,
  title   = {TRIM: Topology Refinement via Information-content-guided Model pruning},
  author  = {Cortes, Katherina G. and Sundar, Shilpa and Gehrke, Sarah and Korn, Daniel R.
             and Caufield, Harry and Schaper, Kevin and Reese, Justin and Mungall, Chris
             and Haendel, Melissa},
  journal = {Journal of Biomedical Semantics},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Code: [MIT](LICENSE). The Monarch KG is distributed under its own license; see
`data/README.md`.
