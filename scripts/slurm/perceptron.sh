#! /bin/bash
#SBATCH --ntasks=1 --time=90:00:00 -p volta-gpu --qos=gpu_access --gres=gpu:2 --mem=60g --cpus-per-task=8
#SBATCH --mail-type=all --mail-user=YOUR_EMAIL

module load python/3.12.1
module load cuda
python src/ml/perceptronBatch.py --nodes IC_removal_80/monarch-kg_nodes.filtered_80.tsv --train IC_removal_80/monarch-kg_edges.filtered_80.TestSet.tsv --out Monarch_GB_80_128embed
