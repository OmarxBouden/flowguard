import os, json, itertools
from stable_baselines3 import PPO
from scripts.config import MODELS_DIR, RESULTS_DIR, EVAL_EPISODES, SEED
from scripts.utils import make_env

MAX_STEPS_OPTIONS       = [30, 50, 100]
TOTAL_TIMESTEPS_OPTIONS = [500_000, 1_000_000, 3_000_000]
RED_LABELS              = ['bline', 'meander', 'mix']

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def quick_eval(model, env, episodes=EVAL_EPISODES):
    total = 0
    obs, _ = env.reset()
    for _ in range(episodes):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total += reward
        obs, _ = env.reset()
    return total / episodes

def main():
    results = []

    for max_steps, total_ts, red_label in itertools.product(
        MAX_STEPS_OPTIONS, TOTAL_TIMESTEPS_OPTIONS, RED_LABELS
    ):
        run_id = f"m1_sweep_ppo_{red_label}_steps{max_steps}_ts{total_ts//1000}k"
        print(f"\n>>> {run_id}")

        env = make_env(red_label, max_steps, seed=SEED)
        model = PPO('MlpPolicy', env, verbose=0, seed=SEED)
        model.learn(total_timesteps=total_ts)
        model.save(f"{MODELS_DIR}/{run_id}")

        mean_reward = quick_eval(model, env)
        print(f"    mean reward ({EVAL_EPISODES} eps): {mean_reward:.2f}")

        results.append({
            'run_id':          run_id,
            'max_steps':       max_steps,
            'total_timesteps': total_ts,
            'red_agent':       red_label,
            'mean_reward':     mean_reward,
        })

    out_path = f"{RESULTS_DIR}/m1_sweep_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Sweep Results (sorted) ===")
    for r in sorted(results, key=lambda x: x['mean_reward'], reverse=True):
        print(f"  {r['run_id']:<55} {r['mean_reward']:>8.2f}")

    print(f"\nSaved: {out_path}")
    print("Update DEFAULT_MAX_STEPS and DEFAULT_TIMESTEPS in scripts/config.py accordingly.")

if __name__ == '__main__':
    main()