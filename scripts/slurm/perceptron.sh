#! /bin/bash
# Single perceptron run, as submitted on the UNC Longleaf cluster for the TRIM manuscript.
# The script on the cluster was named perceptronBatch_fixed.py; its code is src/ml/perceptronBatch.py.
# On the cluster the TP/TN/candidate files sat in the working directory (the script's
# defaults); here they are passed explicitly. The example below is the random-removal
# control; for an IC subset use IC_removal_<S>/monarch-kg_{nodes,edges}.filtered_<S>[.TestSet].tsv.
#SBATCH --ntasks=1 --time=90:00:00 -p volta-gpu --qos=gpu_access --gres=gpu:2 --mem=60g --cpus-per-task=8
#SBATCH --mail-type=all --mail-user=YOUR_EMAIL

module load python/3.12.1
module load cuda
python src/ml/perceptronBatch.py \
    --nodes      IC_removal_none/monarch-kg_nodes_Rand_80.tsv \
    --train      IC_removal_none/monarch-kg_edges_Rand_80.tsv \
    --pos        data/TP_hgnc_mondo_edges.tsv \
    --neg        data/TN_hgnc_mondo_edges1.tsv \
    --candidates analysis/data/geneCandidates.txt \
    --out        results_Rand_80_ranks
