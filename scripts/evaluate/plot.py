"""
Loads eval JSON files and produces three comparison plots:
- opponent: fix the algorithm (PPO), vary the training opponent (bline/meander/mix).
- algorithm: fix the training opponent (mix), vary the variant (PPO, +Simple, +Wide).
- sweep: heatmap of mean reward over max_steps x total_timesteps from sweep.py.

Usage:
  python -m scripts.evaluate.plot --comparison all
"""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

EVAL_KEYS = [
    'bline_30', 'bline_50', 'bline_100',
    'meander_30', 'meander_50', 'meander_100',
]
LABELS = [
    'B-line 30', 'B-line 50', 'B-line 100',
    'Meander 30', 'Meander 50', 'Meander 100',
]

def load_eval(path):
    with open(path) as f:
        return json.load(f)

def get_means(data):
    return [data['combinations'][k]['mean'] for k in EVAL_KEYS]

def get_stds(data):
    return [data['combinations'][k]['std'] for k in EVAL_KEYS]

def plot_opponent_comparison(results_dir, algo='ppo'):
    """Plain PPO trained on each of bline / meander / mix; evaluated on the full protocol."""
    configs = ['bline', 'meander', 'mix']
    files = {c: f"{results_dir}/m1_ppo_baseline_{c}_eval.json" for c in configs}
    files = {c: p for c, p in files.items() if os.path.exists(p)}

    if not files:
        print("No eval files found for opponent comparison.")
        return

    x = np.arange(len(EVAL_KEYS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (config, path) in enumerate(files.items()):
        data = load_eval(path)
        means = get_means(data)
        stds = get_stds(data)
        ax.bar(x + i * width, means, width, yerr=stds,
               label=f"Trained vs {config}", capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(LABELS, rotation=15)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title(f'Effect of Training Opponent ({algo.upper()})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plot_opponent_comparison_{algo}.png", dpi=150)
    plt.close()
    print(f"Saved: plot_opponent_comparison_{algo}.png")

def plot_algorithm_comparison(results_dir, training_config='mix'):
    """Same trained model, compare plain PPO vs PPO+SimpleDecoy vs PPO+WideDecoy."""
    algos = {
        'PPO':             f"{results_dir}/m1_ppo_baseline_{training_config}_eval.json",
        'PPO+SimpleDecoy': f"{results_dir}/m1_ppo_simple_decoy_{training_config}_eval.json",
        'PPO+WideDecoy':   f"{results_dir}/m1_ppo_wide_decoy_{training_config}_eval.json",
    }
    algos = {k: v for k, v in algos.items() if os.path.exists(v)}

    if not algos:
        print("No eval files found for algorithm comparison.")
        return

    x = np.arange(len(EVAL_KEYS))
    width = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (name, path) in enumerate(algos.items()):
        data = load_eval(path)
        means = get_means(data)
        stds = get_stds(data)
        ax.bar(x + i * width, means, width, yerr=stds,
               label=name, capsize=3)

    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels(LABELS, rotation=15)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title(f'Effect of Algorithm (trained vs {training_config})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plot_algorithm_comparison_{training_config}.png", dpi=150)
    plt.close()
    print(f"Saved: plot_algorithm_comparison_{training_config}.png")

def plot_sweep(results_dir):
    """Heatmap of total score over max_steps x total_timesteps."""
    sweep_path = f"{results_dir}/m1_sweep_results.json"
    if not os.path.exists(sweep_path):
        print("No sweep results found.")
        return

    with open(sweep_path) as f:
        sweep = json.load(f)

    red_agents = list({r['red_agent'] for r in sweep})
    for red_name in red_agents:
        rows = [r for r in sweep if r['red_agent'] == red_name]
        steps_vals = sorted({r['max_steps'] for r in rows})
        ts_vals    = sorted({r['total_timesteps'] for r in rows})

        grid = np.zeros((len(ts_vals), len(steps_vals)))
        for r in rows:
            i = ts_vals.index(r['total_timesteps'])
            j = steps_vals.index(r['max_steps'])
            grid[i, j] = r['mean_reward']

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(steps_vals)))
        ax.set_xticklabels([str(s) for s in steps_vals])
        ax.set_yticks(range(len(ts_vals)))
        ax.set_yticklabels([f"{t//1000}k" for t in ts_vals])
        ax.set_xlabel('Max Steps (episode length)')
        ax.set_ylabel('Total Timesteps')
        ax.set_title(f'Sweep: Mean Reward vs {red_name}')
        plt.colorbar(im, ax=ax, label='Mean Reward')

        for i in range(len(ts_vals)):
            for j in range(len(steps_vals)):
                ax.text(j, i, f"{grid[i,j]:.1f}", ha='center', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{results_dir}/plot_sweep_{red_name}.png", dpi=150)
        plt.close()
        print(f"Saved: plot_sweep_{red_name}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    parser.add_argument('--comparison',
                        choices=['opponent', 'algorithm', 'sweep', 'all'],
                        default='all')
    args = parser.parse_args()

    if args.comparison in ('opponent', 'all'):
        plot_opponent_comparison(args.results)
    if args.comparison in ('algorithm', 'all'):
        plot_algorithm_comparison(args.results)
    if args.comparison in ('sweep', 'all'):
        plot_sweep(args.results)

if __name__ == '__main__':
    main()