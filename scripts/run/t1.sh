#!/usr/bin/env bash
# M1: train PPO (3 reds), eval the full lineup, and produce comparison plots.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
LOG="logs/m1_$(date +%Y%m%d_%H%M).log"
{
    python -m scripts.train.ppo_baseline
    python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_bline   --algo ppo
    python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_meander --algo ppo
    python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_mix     --algo ppo
    python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_mix     --algo simple_decoy --label m1_ppo_simple_decoy_mix
    python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_mix     --algo wide_decoy   --label m1_ppo_wide_decoy_mix
    python -m scripts.evaluate.plot --comparison all
} 2>&1 | tee "$LOG"
