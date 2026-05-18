#!/usr/bin/env bash
# SLURM array job: run all 9 C-POMCP eval combinations in parallel.

#SBATCH --job-name=cpomcp_eval
#SBATCH --array=0-8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --account=a131
#SBATCH --partition=normal

# ── Working directory & Python path ──────────────────────────────────────────
# SLURM_SUBMIT_DIR is always the directory from which sbatch was called.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cyborg

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# ── Combination mapping ───────────────────────────────────────────────────────
# Task IDs 0-8 → (red, max_steps)
# 0=bline/30  1=bline/50  2=bline/100
# 3=meander/30 4=meander/50 5=meander/100
# 6=sleep/30  7=sleep/50  8=sleep/100

REDS=(bline bline bline meander meander meander sleep sleep sleep)
STEPS=(30 50 100 30 50 100 30 50 100)

RED=${REDS[$SLURM_ARRAY_TASK_ID]}
MAX_STEPS=${STEPS[$SLURM_ARRAY_TASK_ID]}

# ── Hyperparameters ───────────────────────────────────────────────────────────
LABEL="c_pomcp_n50_s30"
N_PARTICLES=50
N_SIMULATIONS=30
DEPTH=4
C=0.5

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Task $SLURM_ARRAY_TASK_ID: red=$RED  max_steps=$MAX_STEPS"
echo "Working dir: $(pwd)"
echo "PYTHONPATH:  $PYTHONPATH"

python -m scripts.evaluate.evaluate \
  --algo c_pomcp \
  --red "$RED" \
  --max_steps "$MAX_STEPS" \
  --n_particles "$N_PARTICLES" \
  --n_simulations "$N_SIMULATIONS" \
  --depth "$DEPTH" \
  --c "$C" \
  --label "$LABEL" \
  --green behavioral
