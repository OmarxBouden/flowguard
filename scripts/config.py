# scripts/config.py
import inspect
from CybORG import CybORG

# Paths
SCENARIO_PATH = str(inspect.getfile(CybORG))[:-10] + '/Shared/Scenarios/Scenario2.yaml'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Eval config (fixed by CAGE 2)
EVAL_EPISODES = 100
EVAL_STEPS = [30, 50, 100]

# Training defaults
DEFAULT_MAX_STEPS = 50
DEFAULT_TIMESTEPS = 1_000_000