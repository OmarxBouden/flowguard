import os
from stable_baselines3 import PPO
from scripts.config import MODELS_DIR, DEFAULT_MAX_STEPS, DEFAULT_TIMESTEPS, SEED
from scripts.utils import make_env

os.makedirs(MODELS_DIR, exist_ok=True)

def train(red_label):
    print(f"\n>>> Training: m1_ppo_baseline_{red_label}")
    env = make_env(red_label, DEFAULT_MAX_STEPS, seed=SEED)
    model = PPO('MlpPolicy', env, verbose=1, seed=SEED)
    model.learn(total_timesteps=DEFAULT_TIMESTEPS)
    model.save(f"{MODELS_DIR}/m1_ppo_baseline_{red_label}")
    print(f"Saved: m1_ppo_baseline_{red_label}")

if __name__ == '__main__':
    for label in ['bline', 'meander', 'mix']:
        train(label)