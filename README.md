# TRIM: Topology Refinement via Information-content-guided Model pruning

Code and results for the manuscript:

> **TRIM: Topology Refinement via Information-content-guided Model pruning.**
> _AUTHORS_. Submitted to _Journal of Biomedical Semantics_ (2026). DOI: _TBD_

We prune low-information-content (IC) Gene Ontology, Human Phenotype Ontology
and Mondo terms from the Monarch KG at four IC thresholds (40, 60, 80, 100),
embed each graph with first-order LINE (GRAPE), train a perceptron to classify
held-out gene–disease edges, and compare against the unpruned graph and a
size-matched random-node-removal control.

## Pipeline

```
Monarch KG (Sept 2025)
   │  src/preprocessing/editICNodes.py        IC filtering per threshold  → Table 1, Fig 3
   │  src/preprocessing/testsetGenerator.py   hold out TP edges, sample TN → Fig 2
   │  src/preprocessing/randomNodeRemoval.py  random-removal control (n = 6,448)
   ▼
src/ml/perceptronBatch.py                      LINE embeddings + perceptron, scoring, gene ranking
   ▼
src/analysis/*.py                              metrics, figures, rare-disease analysis
```

## Repository layout

```
src/
  preprocessing/
    editICNodes.py         query Ubergraph IC; drop GO/HPO/MONDO terms below threshold
    testsetGenerator.py    TP test-set extraction and negative (TN) sampling
    randomNodeRemoval.py   random-node-removal control graphs
    rareDiseaseSubsets.py  rare-disease annotation of test diseases (queries a local Neo4j)
    editKG.py              edge-removal helpers
    queries.py, neo4jConfig.py, neo4jConnection.py
  ml/
    perceptronBatch.py     main experiment: embed, train, score test edges, rank candidate genes
  analysis/
    paper_figures.py       Table 2, ranking numbers, Fig 4C, Fig 5 (+ stale-run check)
    f1_comparison_refresh.py  Table 2 metrics and Fig 6 (F1 vs random control)
    topRankSummary.py, topPredAnalysis.py, perceptronInvest.py   ranking analyses
    testSetAnalysis.py, rareDisesaseAnalysis.py, PerceptronAnalyisis.py   rare-disease analyses
    confSummary.py         metric summaries
analysis/
  reports/                 IC-filter summaries (Table 1) and other small outputs
  data/                    rare-disease annotations, candidate gene list, derived tables
  results/
    results_{none,40,60,80,100}/   perceptron runs reported in the manuscript
    comparison_figures/            cross-run metrics and figures
monarch_percep_rand{,7,13}_80/     random-removal control runs (3 seeds)
results_{none,40,60,80,100}_ranks/ later re-runs used by some ranking scripts (see Notes)
comparison_figures/                ranking summaries produced by src/analysis
data/                              held-out TP / TN gene–disease test sets
scripts/slurm/                     HPC job scripts (rerun_paper.sh reruns all five subsets)
```

## Manuscript figures and tables

| Item | Script | Output / data |
|---|---|---|
| Fig 1 (FBN1–Marfan hierarchy) | not scripted (drawn from Monarch association counts) | — |
| Fig 2 (negative-sampling algorithm) | `src/preprocessing/testsetGenerator.py` | `data/TP_hgnc_mondo_edges.tsv`, `data/TN_hgnc_mondo_edges1.tsv` |
| Fig 3 (IC distributions) | `src/preprocessing/editICNodes.py` (`plot_multi_ontology_ic_histogram`) | — |
| Table 1 (nodes/edges removed) | `src/preprocessing/editICNodes.py` | `analysis/reports/kg_ic_filter_summary{40,60,80,100}.txt` |
| Fig 4A–B (score distributions) | `src/ml/perceptronBatch.py` | `analysis/results/results_{none,80}/score_distributions.png` |
| Fig 4C (confusion counts) | `src/analysis/paper_figures.py` | `analysis/results/paper/fig4c_confusion_counts.png` |
| Table 2 (F1, AUROC, confusion) | `src/analysis/paper_figures.py`, `src/analysis/f1_comparison_refresh.py` | `analysis/results/paper/table2_classification.tsv` |
| Fig 5, ranking numbers in text | `src/analysis/paper_figures.py` | `analysis/results/paper/fig5_topk.png`, `ranking_summary.tsv` |
| Fig 6 (F1 vs random control) | `src/analysis/f1_comparison_refresh.py` | `analysis/results/comparison_figures/standard_refresh/f1.png` |
| Rare-disease analysis | `src/preprocessing/rareDiseaseSubsets.py`, `src/analysis/testSetAnalysis.py`, `rareDisesaseAnalysis.py`, `PerceptronAnalyisis.py` | `analysis/data/`, `analysis/results/annotation_subset_plots/` |

## Installation

Requires Python ≥ 3.12.

```bash
git clone https://github.com/kcortes133/EdgePrediction.git
cd EdgePrediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the experiments

Download the Monarch KG first (see [data/README.md](data/README.md)).
Run all commands from the repository root.

```bash
# 1. IC filtering (edit IC_THRESHOLD in the script; run for 40, 60, 80, 100)
python src/preprocessing/editICNodes.py

# 2. Held-out test set + negative sampling (writes *.TestSet.tsv training graphs)
python src/preprocessing/testsetGenerator.py

# 3. Random-removal control (edit seed in main(); run for each seed)
python src/preprocessing/randomNodeRemoval.py

# 4. Embedding + perceptron, one run per graph variant.
#    On SLURM, scripts/slurm/rerun_paper.sh runs all five (with a leakage check).
python src/ml/perceptronBatch.py \
    --nodes      monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_nodes.filtered_80.tsv \
    --train      monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_edges.filtered_80.TestSet.tsv \
    --pos        data/TP_hgnc_mondo_edges.tsv \
    --neg        data/TN_hgnc_mondo_edges1.tsv \
    --candidates analysis/data/geneCandidates.txt \
    --out        analysis/results/results_80

# 5. Metrics and figures
python src/analysis/paper_figures.py
python src/analysis/f1_comparison_refresh.py
```

Embeddings and the perceptron use GRAPE's default `random_state=42`.
`paper_figures.py` exits with an error if any subset's `gene_ranks.tsv`
contains pairs outside the TP test set or differs from the other subsets.

`perceptronBatch.py` writes `edge_predictions.tsv`, `gene_ranks.tsv`,
`confusion_summary.tsv` and `score_distributions.png` to `--out`.

`rareDiseaseSubsets.py` queries a local Neo4j copy of the Monarch KG and reads
credentials from `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DB`.

## Notes

- `results_*_ranks/` (repository root) are later re-runs on the current test
  set and are the inputs to `topRankSummary.py`, `PerceptronAnalyisis.py`,
  `perceptronInvest.py` and `rareDisesaseAnalysis.py`. The manuscript's
  Table 2 and Fig 6 use `analysis/results/results_*`.

## Citation

```bibtex
@article{trim2026,
  title   = {TRIM: Topology Refinement via Information-content-guided Model pruning},
  author  = {Cortes, Katherina and TBD},
  journal = {Journal of Biomedical Semantics},
  year    = {2026}
}
```

## License

Code: [MIT](LICENSE). The Monarch KG is distributed under its own license; see
`data/README.md`.
