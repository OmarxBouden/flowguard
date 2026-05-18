"""
C-POMCP Blue agent for CybORG CAGE-2.

Implements the Causal Partially Observable Monte-Carlo Planning algorithm
(arXiv 2407.11070) as a drop-in Blue defender compatible with the standard
ChallengeWrapper evaluation loop used in scripts/evaluate/evaluate.py.

Usage
-----
    from agents.c_pomcp_agent import CPOMCPAgent

    agent = CPOMCPAgent(scenario_path, red_agent_cls=B_lineAgent)

    # Gymnasium / evaluate.py style:
    agent.reset()
    action, _ = agent.predict(obs, deterministic=True)

    # Low-level CybORG style:
    action = agent.get_action(obs)
"""

import inspect
import numpy as np

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from agents.simulator_model import CybORGSandbox
from agents.causal_graph import get_causally_relevant_actions
from agents.particle_filter import BeliefParticleFilter
from agents.mcts_planner import MCTSPlanner


class CPOMCPAgent:
    """Causal POMCP Blue defender.

    Parameters
    ----------
    scenario_path:
        Path to Scenario2.yaml (or equivalent).  If None, auto-detected from
        the installed CybORG package.
    red_agent_cls:
        Red agent class used inside the sandbox (default: B_lineAgent).
        For mixed-opponent evaluation, pass B_lineAgent (conservative choice).
    n_particles:
        Number of belief particles M (paper: 1000; default 200).
    search_time:
        Wall-clock seconds per planning step (paper sT; default None).
        When set, the planner runs until the budget expires — exactly matching
        the paper's stopping criterion.  Overrides n_simulations when provided.
    n_simulations:
        Fixed MCTS rollout count per step; used when search_time is None
        (default 100).
    depth:
        Lookahead depth per rollout (paper: 4).
    gamma:
        Discount factor (paper: 0.99).
    c:
        UCT exploration constant (paper: 0.5).
    p_mutate:
        Per-host mutation probability during particle rejuvenation (default 0.05).
    """

    def __init__(
        self,
        scenario_path: str = None,
        red_agent_cls=None,
        n_particles: int = 200,
        search_time: float = None,
        n_simulations: int = 100,
        depth: int = 4,
        gamma: float = 0.99,
        c: float = 0.5,
        p_mutate: float = 0.05,
    ):
        if scenario_path is None:
            scenario_path = _default_scenario_path()
        if red_agent_cls is None:
            red_agent_cls = B_lineAgent

        # Step 1: build the simulation sandbox
        self._sandbox = CybORGSandbox(scenario_path, red_agent_cls)
        self._action_map = self._sandbox.build_action_map()

        # Step 3: particle filter
        self._pf = BeliefParticleFilter(
            sandbox=self._sandbox,
            M=n_particles,
            p_mutate=p_mutate,
        )

        # Step 4: MCTS planner
        self._planner = MCTSPlanner(
            sandbox=self._sandbox,
            search_time=search_time,
            n_simulations=n_simulations,
            depth=depth,
            gamma=gamma,
            c=c,
        )

        self._last_action: int = 0
        self._step: int = 0

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the agent at the start of a new episode."""
        self._pf.reset()
        self._planner.reset_tree()
        self._last_action = 0
        self._step = 0

    # ------------------------------------------------------------------
    # Core planning interface
    # ------------------------------------------------------------------

    def get_action(self, obs: np.ndarray) -> int:
        """Compute the best Blue action given the current observation.

        Parameters
        ----------
        obs:
            52-dim numpy array from ChallengeWrapper / BlueTableWrapper.

        Returns
        -------
        action_idx : int
        """
        obs = np.asarray(obs, dtype=np.float32)

        # Step 1: update particle filter with last action + current obs
        if self._step > 0:
            self._pf.update(self._last_action, obs)
            self._planner.advance_root(self._last_action, obs)

        # Step 2: derive causally relevant action mask
        action_mask = get_causally_relevant_actions(obs, self._action_map)

        # Step 3: run MCTS over current particle belief
        particles = self._pf._particles
        if len(particles) == 0:
            # Degenerate case: no particles – fall through to a safe default
            action = int(np.where(action_mask)[0][0])
        else:
            action = self._planner.plan(particles, action_mask)

        self._last_action = action
        self._step += 1
        return action

    # ------------------------------------------------------------------
    # evaluate.py-compatible predict interface
    # ------------------------------------------------------------------

    def predict(self, obs, deterministic: bool = True):
        """Mirror the SB3 policy.predict() signature expected by evaluate.py.

        Returns
        -------
        (action_idx, None)
        """
        return self.get_action(obs), None

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    def n_actions(self) -> int:
        return self._sandbox.n_actions

    def particle_count(self) -> int:
        return self._pf.particle_count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_scenario_path() -> str:
    return str(inspect.getfile(CybORG))[:-10] + "/Shared/Scenarios/Scenario2.yaml"
