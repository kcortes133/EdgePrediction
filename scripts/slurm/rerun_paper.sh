#!/bin/bash
# Rerun the five TRIM perceptron experiments (IC threshold none/40/60/80/100)
# on the current test set, then rebuild Table 2 and Figures 4-6.
#
# Submit from the repository root:
#     sbatch scripts/slurm/rerun_paper.sh
# When all five array tasks have finished:
#     python src/analysis/paper_figures.py
#     python src/analysis/f1_comparison_refresh.py
#
# Edit the file patterns below to match where the filtered graphs live
# on your cluster. Training graphs must be the *.TestSet.tsv files built by
# src/preprocessing/testsetGenerator.py from data/TP_hgnc_mondo_edges.tsv.
#
#SBATCH --job-name=trim_rerun
#SBATCH --array=0-4
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=60g --time=90:00:00
#SBATCH -p volta-gpu --qos=gpu_access --gres=gpu:2
#SBATCH --mail-type=all --mail-user=YOUR_EMAIL
#SBATCH --output=logs/trim_rerun_%a.out

module load python/3.12.1
module load cuda

SUBSETS=(none 40 60 80 100)
S=${SUBSETS[$SLURM_ARRAY_TASK_ID]}

if [ "$S" = "none" ]; then
    NODES=monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_nodes.tsv
    TRAIN=monarch-kg-Sept2025/monarch-kg-Sept2025/monarch-kg_edges.TestSet.tsv
else
    NODES=IC_removal_${S}/monarch-kg_nodes.filtered_${S}.tsv
    TRAIN=IC_removal_${S}/monarch-kg_edges.filtered_${S}.TestSet.tsv
fi

# Leakage check: no held-out TP edge may remain in the training graph.
LEAK=$(awk -F'\t' '
    { sub(/\r$/, "") }
    NR==FNR { if (FNR>1) tp[$1 FS $3]=1; next }
    FNR==1  { for (i=1;i<=NF;i++) { if ($i=="subject") a=i; if ($i=="object") b=i }; next }
    ($a FS $b) in tp || ($b FS $a) in tp { n++ }
    END { print n+0 }' data/TP_hgnc_mondo_edges.tsv "$TRAIN")
if [ "$LEAK" -gt 0 ]; then
    echo "ERROR: $LEAK test edges found in $TRAIN - rebuild it with testsetGenerator.py" >&2
    exit 1
fi

python src/ml/perceptronBatch.py \
    --nodes      "$NODES" \
    --train      "$TRAIN" \
    --pos        data/TP_hgnc_mondo_edges.tsv \
    --neg        data/TN_hgnc_mondo_edges1.tsv \
    --candidates analysis/data/geneCandidates.txt \
    --out        analysis/results/results_${S} \
    --threshold  0.5
