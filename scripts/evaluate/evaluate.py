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
from tqdm import tqdm
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from scripts.config import SCENARIO_PATH, RESULTS_DIR, EVAL_EPISODES, EVAL_STEPS
from scripts.train.ppo_greedy_decoy import SimpleDecoyPolicy, WideDecoyPolicy
from agents.green import BehavioralGreenAgent

RED_AGENTS = {
    'bline':   B_lineAgent,
    'meander': RedMeanderAgent,
    'sleep':   SleepAgent,
}


def _green_override(green):
    if green == 'off':
        return {}
    if green == 'behavioral':
        return {'Green': BehavioralGreenAgent}
    raise ValueError(f"Unknown green={green!r}")


class _PassthroughPolicy:
    """For SB3 PPO/MaskablePPO: reset is a no-op; predict is already stateless."""
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)


def load_policy(model_path, algo, **kwargs):
    if algo in ('ppo', 'masked'):
        from stable_baselines3 import PPO
        return _PassthroughPolicy(PPO.load(model_path))
    if algo == 'simple_decoy':
        from stable_baselines3 import PPO
        return SimpleDecoyPolicy(PPO.load(model_path))
    if algo == 'wide_decoy':
        from stable_baselines3 import PPO
        return WideDecoyPolicy(PPO.load(model_path))
    if algo == 'c_pomcp':
        from agents.c_pomcp_agent import CPOMCPAgent
        green_cls = None
        if kwargs.get('green', 'off') == 'behavioral':
            from agents.green import BehavioralGreenAgent
            green_cls = BehavioralGreenAgent
        return CPOMCPAgent(
            scenario_path=SCENARIO_PATH,
            green_agent_cls=green_cls,
            n_particles=kwargs.get('n_particles', 1000),
            search_time=kwargs.get('search_time', None),
            n_simulations=kwargs.get('n_simulations', 100),
            depth=kwargs.get('depth', 4),
            gamma=kwargs.get('gamma', 0.99),
            c=kwargs.get('c', 0.5),
        )
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


def evaluate_combination(policy, red_cls, max_steps, n_episodes, green='off', desc=''):
    agents = {'Red': red_cls, **_green_override(green)}
    rewards = []
    for _ in tqdm(range(n_episodes), desc=desc, unit='ep', dynamic_ncols=True):
        # Fresh CybORG per episode so green's per-session state stays clean.
        cyborg = CybORG(SCENARIO_PATH, 'sim', agents=agents)
        env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=max_steps)
        rewards.append(run_episode(policy, env))
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='Path to saved model (no extension); not required for c_pomcp')
    parser.add_argument('--algo',  required=True,
                        choices=['ppo', 'simple_decoy', 'wide_decoy', 'masked', 'c_pomcp'])
    parser.add_argument('--label', default=None, help='Label for results file (default: model filename or algo)')
    # Cluster parallelism: run a single combination instead of all 9
    parser.add_argument('--red',       default=None, choices=['bline', 'meander', 'sleep'],
                        help='Run only this red agent (for cluster array jobs)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Run only this episode length (for cluster array jobs)')
    # C-POMCP hyperparameters (paper Table 4 defaults)
    parser.add_argument('--n_particles',   type=int,   default=1000,  help='Particle filter size M (paper: 1000)')
    parser.add_argument('--search_time',   type=float, default=None,  help='Planning budget in seconds per step (paper sT=15); overrides --n_simulations')
    parser.add_argument('--n_simulations', type=int,   default=100,   help='MCTS rollouts per step; used when --search_time is not set')
    parser.add_argument('--depth',         type=int,   default=4,     help='Rollout depth (paper: 4)')
    parser.add_argument('--gamma',         type=float, default=0.99,  help='Discount factor γ (paper: 0.99)')
    parser.add_argument('--c',             type=float, default=0.5,   help='UCT exploration constant (paper: 0.5)')
    parser.add_argument('--green', default='off', choices=['off', 'behavioral'],
                        help='Green-agent mode (default: off, matches M1 baseline)')
    args = parser.parse_args()

    label = args.label or (os.path.basename(args.model) if args.model else args.algo)
    policy = load_policy(
        args.model, args.algo,
        n_particles=args.n_particles,
        search_time=args.search_time,
        n_simulations=args.n_simulations,
        depth=args.depth,
        gamma=args.gamma,
        c=args.c,
        green=args.green,
    )

    # Determine which combinations to run
    if args.red is not None and args.max_steps is not None:
        # Single-combination mode for cluster array jobs
        run_combos = [(args.red, RED_AGENTS[args.red], args.max_steps)]
        partial = True
    else:
        run_combos = [(n, c, s) for n, c in RED_AGENTS.items() for s in EVAL_STEPS]
        partial = False

    results = {'label': label, 'algo': args.algo, 'green': args.green,
               'combinations': {}, 'total': 0.0}
    for red_name, red_cls, max_steps in run_combos:
        key = f"{red_name}_{max_steps}"
        print(f"Evaluating: {key} ...")
        mean, std = evaluate_combination(policy, red_cls, max_steps, EVAL_EPISODES,
                                         green=args.green, desc=key)
        results['combinations'][key] = {'mean': mean, 'std': std}
        results['total'] += mean
        print(f"  mean={mean:.2f}  std={std:.2f}")

    print(f"\nTotal score: {results['total']:.2f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if partial:
        red_name, max_steps = args.red, args.max_steps
        out_path = f"{RESULTS_DIR}/{label}_{red_name}_{max_steps}_partial.json"
    else:
        out_path = f"{RESULTS_DIR}/{label}_eval.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
