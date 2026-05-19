"""
PPO baseline trainer.

Defaults to the M1 (green-off) scenario. Pass --scenario m2 to train under
Scenario2_M2.yaml (green-on); model filenames are prefixed accordingly.

Usage:
  python -m scripts.train.ppo_baseline                  # M1: m1_ppo_baseline_<red>.zip
  python -m scripts.train.ppo_baseline --scenario m2    # M2: m2_ppo_baseline_<red>_green.zip
"""
import argparse
import os
from stable_baselines3 import PPO
from scripts.config import MODELS_DIR, DEFAULT_MAX_STEPS, DEFAULT_TIMESTEPS, SEED
from scripts.utils import make_env

os.makedirs(MODELS_DIR, exist_ok=True)


def _model_name(red_label, scenario):
    if scenario == 'm1':
        return f"m1_ppo_baseline_{red_label}"
    if scenario == 'm2':
        return f"m2_ppo_baseline_{red_label}_green"
    raise ValueError(f"Unknown scenario={scenario!r}")


def train(red_label, scenario):
    name = _model_name(red_label, scenario)
    print(f"\n>>> Training: {name}  (scenario={scenario}, red={red_label})")
    env = make_env(red_label, DEFAULT_MAX_STEPS, seed=SEED, scenario=scenario)
    model = PPO('MlpPolicy', env, verbose=1, seed=SEED)
    model.learn(total_timesteps=DEFAULT_TIMESTEPS)
    model.save(f"{MODELS_DIR}/{name}")
    print(f"Saved: {name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='m1', choices=['m1', 'm2'],
                        help='m1 = Scenario2 (Green=Sleep); m2 = Scenario2_M2 (Green=BehavioralGreenAgent).')
    parser.add_argument('--reds', nargs='+', default=['bline', 'meander', 'mix'],
                        choices=['bline', 'meander', 'mix'],
                        help='Which training opponents to run (default: all three).')
    args = parser.parse_args()
    for label in args.reds:
        train(label, args.scenario)
