"""Behavioral green agents for M2.

Replaces the stock CybORG GreenAgent (which hardcodes ``session=0`` and picks
targets uniformly) with several distinct behavior profiles dispatched per
source host. The agent picks one session per env step, weighted by activity
level, then samples a profile-appropriate action.

Wiring: pass ``agents={'Green': BehavioralGreenAgent}`` to ``CybORG(...)``.
No scenario edits required.
"""
import random
from collections import defaultdict

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Actions import Sleep, GreenPortScan, GreenConnection


# Session id → source-host name. Order matches Scenario2.yaml
# starting_sessions; State.py:143-148 assigns ids in scenario order.
SESSION_HOST = {
    0: 'User0',
    1: 'User1',
    2: 'User2',
    3: 'User3',
    4: 'User4',
    5: 'Op_Host0',
    6: 'Op_Host1',
    7: 'Op_Host2',
}

# Per-host behavior profile.
#   weight:  prob of this session being picked on a given step
#   sleep:   prob of Sleep given this session is picked
#   targets: hostnames this session prefers to contact
PROFILES = {
    'User0': {  # red foothold — silent (real user is compromised, no traffic)
        'weight': 0.0, 'sleep': 1.0, 'targets': [],
    },
    'User1': {  # web user — browses Enterprise web servers
        'weight': 0.20, 'sleep': 0.25,
        'targets': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
    },
    'User2': {  # web user
        'weight': 0.20, 'sleep': 0.25,
        'targets': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
    },
    'User3': {  # admin / SSH — lower-frequency Enterprise traffic
        'weight': 0.10, 'sleep': 0.45,
        'targets': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
    },
    'User4': {  # admin / SSH
        'weight': 0.10, 'sleep': 0.45,
        'targets': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
    },
    'Op_Host0': {  # ops polling — internal monitoring to Op_Server
        'weight': 0.15, 'sleep': 0.35,
        'targets': ['Op_Server0'],
    },
    'Op_Host1': {  # ops polling
        'weight': 0.15, 'sleep': 0.35,
        'targets': ['Op_Server0'],
    },
    'Op_Host2': {  # ops user — mix of Op targets
        'weight': 0.10, 'sleep': 0.45,
        'targets': ['Op_Server0', 'Op_Host0', 'Op_Host1'],
    },
}


class BehavioralGreenAgent(BaseAgent):
    def __init__(self):
        self._rng = random.Random()
        self._sessions = list(SESSION_HOST.keys())
        self._weights = [PROFILES[SESSION_HOST[s]]['weight'] for s in self._sessions]
        # (session_id, target_host) tracker — scan once, then connect freely.
        # GreenConnection requires session.ports[target_ip] to be populated,
        # which only happens after a successful GreenPortScan against that ip.
        self._scanned = defaultdict(set)

    def get_action(self, observation, action_space):
        session_id = self._rng.choices(self._sessions, weights=self._weights, k=1)[0]
        host = SESSION_HOST[session_id]
        prof = PROFILES[host]

        if self._rng.random() < prof['sleep'] or not prof['targets']:
            return Sleep()

        target = self._rng.choice(prof['targets'])
        if target not in self._scanned[session_id]:
            self._scanned[session_id].add(target)
            return GreenPortScan(hostname=target, session=session_id, agent='Green')
        return GreenConnection(hostname=target, session=session_id, agent='Green')

    def train(self, results):
        pass

    def end_episode(self):
        self._scanned.clear()

    def set_initial_values(self, action_space, observation):
        pass
