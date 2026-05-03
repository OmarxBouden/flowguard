# scripts/utils.py
import random
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from scripts.config import SCENARIO_PATH


class MixedEnv(ChallengeWrapper):
    """Randomly picks a red agent on each reset for mixed training."""
    def reset(self, **kwargs):
        red_cls = random.choice([B_lineAgent, RedMeanderAgent])
        cyborg = CybORG(SCENARIO_PATH, 'sim', agents={'Red': red_cls()})
        self.env = cyborg
        return super().reset(**kwargs)


def make_env(red_label, max_steps):
    if red_label == 'mix':
        return MixedEnv(
            env=CybORG(SCENARIO_PATH, 'sim', agents={'Red': B_lineAgent()}),
            agent_name='Blue', max_steps=max_steps
        )
    red_cls = {'bline': B_lineAgent, 'meander': RedMeanderAgent}[red_label]
    cyborg = CybORG(SCENARIO_PATH, 'sim', agents={'Red': red_cls()})
    return ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=max_steps)