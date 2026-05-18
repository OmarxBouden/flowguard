import random
import gymnasium as gym
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from agents.green import BehavioralGreenAgent
from scripts.config import SCENARIO_PATH


def _agents(red_cls, green='off'):
    """Build the agents dict for CybORG().

    green='off'        — use scenario default (SleepAgent in Scenario2.yaml).
    green='behavioral' — multi-profile behavioral green agent (M2).
    """
    a = {'Red': red_cls}
    if green == 'behavioral':
        a['Green'] = BehavioralGreenAgent
    elif green != 'off':
        raise ValueError(f"Unknown green={green!r}")
    return a


class MixedEnv(gym.Env):
    """Recreates the env with a random red agent on each reset (B_line or Meander)."""
    def __init__(self, max_steps, seed=None, green='off'):
        self.max_steps = max_steps
        self.green = green
        self._rng = random.Random(seed)
        self._make_env()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _make_env(self):
        red_cls = self._rng.choice([B_lineAgent, RedMeanderAgent])
        cyborg = CybORG(SCENARIO_PATH, 'sim', agents=_agents(red_cls, self.green))
        self._env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=self.max_steps)

    def reset(self, **kwargs):
        self._make_env()
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)


def make_env(red_label, max_steps, seed=None, green='off'):
    if red_label == 'mix':
        return MixedEnv(max_steps, seed=seed, green=green)
    red_cls = {'bline': B_lineAgent, 'meander': RedMeanderAgent}[red_label]
    cyborg = CybORG(SCENARIO_PATH, 'sim', agents=_agents(red_cls, green))
    return ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=max_steps)
