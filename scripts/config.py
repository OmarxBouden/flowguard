import inspect
import os
from CybORG import CybORG

SCENARIO_PATH = str(inspect.getfile(CybORG))[:-10] + '/Shared/Scenarios/Scenario2.yaml'

# M2 scenario — identical to Scenario2 except the Green agent_type is
# BehavioralGreenAgent. Registered with CybORG.Agents on import of
# `agents.green` (side-effect in agents/green/__init__.py).
SCENARIO_PATH_M2 = os.path.join(os.path.dirname(__file__), '..', 'scenarios',
                                'Scenario2_M2.yaml')

MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Fixed by the CAGE 2 protocol — see cage-challenge-2/CybORG/Evaluation/evaluation.py
EVAL_EPISODES = 100
EVAL_STEPS = [30, 50, 100]

DEFAULT_MAX_STEPS = 50
DEFAULT_TIMESTEPS = 1_000_000
SEED = 42