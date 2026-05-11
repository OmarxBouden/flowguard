"""
M2 placeholder. Mask is currently all-ones, so this behaves as plain PPO.
At M2, replace `mask_fn` with one that disables inapplicable network-centric
actions for the current state.
"""
import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from scripts.config import MODELS_DIR, DEFAULT_MAX_STEPS, DEFAULT_TIMESTEPS, SEED
from scripts.utils import make_env

os.makedirs(MODELS_DIR, exist_ok=True)

def mask_fn(env):
    # TODO M2: mask network actions whose preconditions aren't met (e.g.
    # block-connection on hosts that don't have an active flow).
    return np.ones(env.action_space.n, dtype=bool)

def train(red_label):
    print(f"\n>>> Training: m1_ppo_masked_{red_label}")
    env = ActionMasker(make_env(red_label, DEFAULT_MAX_STEPS, seed=SEED), mask_fn)
    model = MaskablePPO('MlpPolicy', env, verbose=1, seed=SEED)
    model.learn(total_timesteps=DEFAULT_TIMESTEPS)
    model.save(f"{MODELS_DIR}/m1_ppo_masked_{red_label}")
    print(f"Saved: m1_ppo_masked_{red_label}")

if __name__ == '__main__':
    train('mix')
