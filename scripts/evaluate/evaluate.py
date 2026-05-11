"""
CAGE 2 evaluation protocol: 3 reds x 3 episode lengths, score = sum of means.
Mirrors cage-challenge-2/CybORG/Evaluation/evaluation.py.

Every variant runs through the same loop. `load_policy` dispatches on `--algo`
and returns a uniform Policy (reset, predict).

Usage:
  python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_mix --algo ppo
  python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_mix --algo simple_decoy --label m1_ppo_simple_decoy_mix
  python -m scripts.evaluate.evaluate --model models/m1_ppo_baseline_mix --algo wide_decoy   --label m1_ppo_wide_decoy_mix
"""
import os, json, argparse
import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from scripts.config import SCENARIO_PATH, RESULTS_DIR, EVAL_EPISODES, EVAL_STEPS
from scripts.train.ppo_greedy_decoy import SimpleDecoyPolicy, WideDecoyPolicy

RED_AGENTS = {
    'bline':   B_lineAgent,
    'meander': RedMeanderAgent,
    'sleep':   SleepAgent,
}


class _PassthroughPolicy:
    """For SB3 PPO/MaskablePPO: reset is a no-op; predict is already stateless."""
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)


def load_policy(model_path, algo):
    if algo in ('ppo', 'masked'):
        from stable_baselines3 import PPO
        return _PassthroughPolicy(PPO.load(model_path))
    if algo == 'simple_decoy':
        from stable_baselines3 import PPO
        return SimpleDecoyPolicy(PPO.load(model_path))
    if algo == 'wide_decoy':
        from stable_baselines3 import PPO
        return WideDecoyPolicy(PPO.load(model_path))
    raise ValueError(f"Unknown algo: {algo}")


def run_episode(policy, env):
    policy.reset()
    obs, _ = env.reset()
    done, total = False, 0.0
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total += reward
    return total


def evaluate_combination(policy, red_cls, max_steps, n_episodes):
    cyborg = CybORG(SCENARIO_PATH, 'sim', agents={'Red': red_cls})
    env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=max_steps)
    rewards = [run_episode(policy, env) for _ in range(n_episodes)]
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model (no extension)')
    parser.add_argument('--algo',  required=True,
                        choices=['ppo', 'simple_decoy', 'wide_decoy', 'masked'])
    parser.add_argument('--label', default=None, help='Label for results file (default: model filename)')
    args = parser.parse_args()

    label = args.label or os.path.basename(args.model)
    policy = load_policy(args.model, args.algo)

    results = {'label': label, 'algo': args.algo, 'combinations': {}, 'total': 0.0}
    for red_name, red_cls in RED_AGENTS.items():
        for max_steps in EVAL_STEPS:
            key = f"{red_name}_{max_steps}"
            print(f"Evaluating: {key} ...")
            mean, std = evaluate_combination(policy, red_cls, max_steps, EVAL_EPISODES)
            results['combinations'][key] = {'mean': mean, 'std': std}
            results['total'] += mean
            print(f"  mean={mean:.2f}  std={std:.2f}")

    print(f"\nTotal score: {results['total']:.2f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/{label}_eval.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
