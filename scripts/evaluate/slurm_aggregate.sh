#!/usr/bin/env bash
# Aggregate partial results after the array job finishes.
#
# Usage (automatic dependency — run immediately after sbatch):
#   ARRAY_JOB_ID=$(sbatch --parsable scripts/evaluate/slurm_eval_array.sh)
#   sbatch --dependency=afterok:$ARRAY_JOB_ID scripts/evaluate/slurm_aggregate.sh
#
# Or interactively once all tasks are done:
#   bash scripts/evaluate/slurm_aggregate.sh

#SBATCH --job-name=cpomcp_agg
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:05:00
#SBATCH --output=logs/agg_%j.out
#SBATCH --account=a131
#SBATCH --partition=normal

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cyborg

LABEL="m1_c_pomcp_n50_s30"   # must match slurm_eval_array.sh

python -m scripts.evaluate.aggregate --label "$LABEL"
