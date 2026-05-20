"""
PPO baseline / NetObs trainer.

Defaults: M1 scenario, 52-d base observation. Override via --scenario and --obs.

Usage:
  python -m scripts.train.ppo_baseline                                    # M1 baseline
  python -m scripts.train.ppo_baseline --scenario m2                      # M2 baseline (green-on)
  python -m scripts.train.ppo_baseline --scenario m2 --obs netobs         # M2 NetObs (52-d + 3 IDS bits)
"""
import argparse
import os
from stable_baselines3 import PPO
from scripts.config import MODELS_DIR, DEFAULT_MAX_STEPS, DEFAULT_TIMESTEPS, SEED
from scripts.utils import make_env

os.makedirs(MODELS_DIR, exist_ok=True)


def _model_name(red_label, scenario, obs):
    if obs == 'netobs':
        if scenario != 'm2':
            raise ValueError("--obs netobs only makes sense with --scenario m2 (green-on)")
        return f"m2_ppo_netobs_{red_label}_green"
    # obs == 'base'
    if scenario == 'm1':
        return f"m1_ppo_baseline_{red_label}"
    if scenario == 'm2':
        return f"m2_ppo_baseline_{red_label}_green"
    raise ValueError(f"Unknown scenario={scenario!r}")


def train(red_label, scenario, obs):
    name = _model_name(red_label, scenario, obs)
    print(f"\n>>> Training: {name}  (scenario={scenario}, obs={obs}, red={red_label})")
    env = make_env(red_label, DEFAULT_MAX_STEPS, seed=SEED, scenario=scenario, obs=obs)
    model = PPO('MlpPolicy', env, verbose=1, seed=SEED)
    model.learn(total_timesteps=DEFAULT_TIMESTEPS)
    model.save(f"{MODELS_DIR}/{name}")
    print(f"Saved: {name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='m1', choices=['m1', 'm2'],
                        help='m1 = Scenario2 (Green=Sleep); m2 = Scenario2_M2 (Green=BehavioralGreenAgent).')
    parser.add_argument('--obs', default='base', choices=['base', 'netobs'],
                        help='base = 52-d M1 obs. netobs = 55-d with 3 IDS subnet-recon bits.')
    parser.add_argument('--reds', nargs='+', default=['bline', 'meander', 'mix'],
                        choices=['bline', 'meander', 'mix'],
                        help='Which training opponents to run (default: all three).')
    args = parser.parse_args()
    for label in args.reds:
        train(label, args.scenario, args.obs)
