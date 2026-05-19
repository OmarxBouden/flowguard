from agents.green.behavioral_green import BehavioralGreenAgent, set_next_seed

# Register BehavioralGreenAgent in CybORG.Agents so the scenario YAML loader
# can resolve it via getattr(sys.modules['CybORG.Agents'], 'BehavioralGreenAgent').
# This fires the moment any code does `from agents.green import ...`.
import CybORG.Agents as _ca
_ca.BehavioralGreenAgent = BehavioralGreenAgent

__all__ = ['BehavioralGreenAgent', 'set_next_seed']
