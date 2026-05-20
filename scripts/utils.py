import random
import gymnasium as gym
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
# Side-effect: registers BehavioralGreenAgent in CybORG.Agents so
# Scenario2_M2.yaml can resolve it by name. Must run before any CybORG(...).
from agents.green import set_next_seed  # noqa: F401
from scripts.config import (
    SCENARIO_PATH, SCENARIO_PATH_M2, IDS_P_TP, IDS_P_FP,
)


def _resolve_scenario(scenario):
    """scenario='m1' → Scenario2.yaml; 'm2' → Scenario2_M2.yaml; else passthrough."""
    if scenario == 'm1':
        return SCENARIO_PATH
    if scenario == 'm2':
        return SCENARIO_PATH_M2
    return scenario


def _wrap_obs(env, obs, ids_seed=None):
    """Apply the observation wrappers selected by `obs`."""
    if obs == 'base':
        return env
    if obs == 'netobs':
        from wrappers.ids import IDSWrapper
        return IDSWrapper(env, p_tp=IDS_P_TP, p_fp=IDS_P_FP, seed=ids_seed)
    raise ValueError(f"Unknown obs={obs!r}")


class MixedEnv(gym.Env):
    """Recreates the env with a random red agent on each reset (B_line or Meander).

    Green-agent behavior is determined by the scenario file:
      scenario='m1' → Scenario2.yaml binds Green to SleepAgent (silent)
      scenario='m2' → Scenario2_M2.yaml binds Green to BehavioralGreenAgent (active)
    Observation wrapping:
      obs='base'   → 52-d M1 vector (default).
      obs='netobs' → 55-d, adds 3 IDS bits.
    """
    def __init__(self, max_steps, seed=None, scenario='m1', obs='base'):
        self.max_steps = max_steps
        self.scenario_path = _resolve_scenario(scenario)
        self.obs_mode = obs
        self._rng = random.Random(seed)
        self._make_env()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _make_env(self):
        red_cls = self._rng.choice([B_lineAgent, RedMeanderAgent])
        # Derive a per-episode green seed from the master RNG so episodes are
        # deterministic given the same `seed`.
        set_next_seed(self._rng.randint(0, 2**32 - 1))
        cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': red_cls})
        base = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=self.max_steps)
        # IDS gets its own deterministic seed derived from the master RNG too.
        ids_seed = self._rng.randint(0, 2**32 - 1)
        self._env = _wrap_obs(base, self.obs_mode, ids_seed=ids_seed)

    def reset(self, **kwargs):
        self._make_env()
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)


def make_env(red_label, max_steps, seed=None, scenario='m1', obs='base'):
    if red_label == 'mix':
        return MixedEnv(max_steps, seed=seed, scenario=scenario, obs=obs)
    red_cls = {'bline': B_lineAgent, 'meander': RedMeanderAgent}[red_label]
    if seed is not None:
        set_next_seed(seed)
    cyborg = CybORG(_resolve_scenario(scenario), 'sim', agents={'Red': red_cls})
    base = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=max_steps)
    return _wrap_obs(base, obs, ids_seed=seed)
