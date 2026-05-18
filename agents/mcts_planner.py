"""
MCTS planner with UCT for C-POMCP.

Implements the online POMCP (Partially Observable Monte-Carlo Planning) tree
search over a particle belief.  At each planning call the planner runs
*n_simulations* rollouts, each starting from a randomly sampled particle,
then returns the action index with the highest estimated value.

Tree structure
--------------
The search tree is indexed by *history tuples* h = (a0, o0, a1, o1, …).
Each node stores:
  N[h]     – total visit count
  Q[h][a]  – accumulated return for action a at history h
  N_a[h][a] – visit count for action a at history h

UCT selection
-------------
  a* = argmax_a  Q(h,a)/N_a(h,a)  +  c * sqrt( ln(N(h)) / N_a(h,a) )

with c=0.5 (paper Table 4) and an exploration bonus of +∞ for unvisited
branches (forces expansion before exploitation).

Causal masking
--------------
The action mask passed in by the caller is applied at every tree node so that
only causally relevant actions are considered during both selection and
rollout.

Rollout policy
--------------
Uniform random over the causal action mask (baseline).
"""

import math
import random
import time
import numpy as np

from agents.simulator_model import CybORGSandbox


class MCTSPlanner:
    """Online POMCP tree search over a particle belief.

    Parameters
    ----------
    sandbox:
        CybORGSandbox instance used for all simulated transitions.
    search_time:
        Wall-clock seconds to spend planning per step (paper sT, default None).
        When set, overrides n_simulations and runs rollouts until the budget
        expires — exactly matching the paper's stopping criterion.
    n_simulations:
        Fixed rollout count per plan() call; used when search_time is None
        (default 100).
    depth:
        Maximum lookahead depth per simulation (paper: 4).
    gamma:
        Discount factor (paper: 0.99).
    c:
        UCT exploration constant (paper: 0.5).
    """

    def __init__(
        self,
        sandbox: CybORGSandbox,
        search_time: float = None,
        n_simulations: int = 100,
        depth: int = 4,
        gamma: float = 0.99,
        c: float = 0.5,
    ):
        self._sandbox = sandbox
        self.search_time = search_time      # sT in seconds (paper parameter)
        self.n_simulations = n_simulations  # fallback when search_time is None
        self.depth = depth
        self.gamma = gamma
        self.c = c

        # Tree storage – cleared at the start of each episode via reset_tree().
        # Keys are history tuples; values are dicts with 'N', 'Q', 'N_a'.
        self._tree: dict = {}

    # ------------------------------------------------------------------
    # Episode-level management
    # ------------------------------------------------------------------

    def reset_tree(self) -> None:
        """Clear the search tree (call at episode start)."""
        self._tree = {}

    def advance_root(self, action: int, obs_vec: np.ndarray) -> None:
        """Shift the root forward by (action, obs) after a real env step.

        Keeps the relevant subtree so statistics accumulated in prior steps
        are reused.  Nodes that are no longer reachable are garbage-collected
        by Python automatically.
        """
        self._root_history = getattr(self, "_root_history", ())
        self._root_history = self._root_history + (action, _obs_key(obs_vec))

    # ------------------------------------------------------------------
    # Main planning call
    # ------------------------------------------------------------------

    def plan(self, particles: list, action_mask: np.ndarray) -> int:
        """Run rollouts and return the best action index.

        Stopping criterion:
          - If search_time is set (paper sT): run until the wall-clock budget
            expires, regardless of how many simulations that yields.
          - Otherwise: run exactly n_simulations rollouts.

        Parameters
        ----------
        particles:
            List of state snapshots representing the current belief.
        action_mask:
            Boolean array of shape (n_actions,) from causal_graph.

        Returns
        -------
        best_action : int
        """
        root = getattr(self, "_root_history", ())

        if self.search_time is not None:
            deadline = time.monotonic() + self.search_time
            while time.monotonic() < deadline:
                particle = random.choice(particles)
                self._simulate(particle, root, action_mask, self.depth)
        else:
            for _ in range(self.n_simulations):
                particle = random.choice(particles)
                self._simulate(particle, root, action_mask, self.depth)

        return self._best_action(root, action_mask)

    # ------------------------------------------------------------------
    # Core POMCP recursion
    # ------------------------------------------------------------------

    def _simulate(
        self,
        particle: dict,
        history: tuple,
        action_mask: np.ndarray,
        depth: int,
    ) -> float:
        """Recursive POMCP simulation.  Returns discounted cumulative reward."""
        if depth == 0:
            return 0.0

        if history not in self._tree:
            # Leaf expansion: initialise node then do a random rollout.
            self._tree[history] = {"N": 0, "Q": {}, "N_a": {}}
            return self._rollout(particle, action_mask, depth)

        node = self._tree[history]

        # UCT action selection
        action = self._uct_select(node, action_mask)

        # Simulate one step
        try:
            next_particle, obs_vec, reward = self._sandbox.sample_transition(
                particle, action
            )
        except Exception:
            return 0.0

        obs_k = _obs_key(obs_vec)
        child_history = history + (action, obs_k)

        # Recurse
        future = self._simulate(next_particle, child_history, action_mask, depth - 1)
        total = reward + self.gamma * future

        # Back-propagate
        node["N"] += 1
        if action not in node["Q"]:
            node["Q"][action] = 0.0
            node["N_a"][action] = 0
        node["Q"][action] += total
        node["N_a"][action] += 1

        return total

    # ------------------------------------------------------------------
    # UCT selection
    # ------------------------------------------------------------------

    def _uct_select(self, node: dict, action_mask: np.ndarray) -> int:
        """Select action using UCT; unvisited masked actions tried first."""
        valid = np.where(action_mask)[0].tolist()

        # Prefer unvisited actions (infinite exploration bonus)
        unvisited = [a for a in valid if a not in node["N_a"]]
        if unvisited:
            return random.choice(unvisited)

        # UCT formula
        log_n = math.log(node["N"]) if node["N"] > 0 else 0.0

        best_val = -float("inf")
        best_a = valid[0]
        for a in valid:
            q_avg = node["Q"][a] / node["N_a"][a]
            bonus = self.c * math.sqrt(log_n / node["N_a"][a])
            val = q_avg + bonus
            if val > best_val:
                best_val = val
                best_a = a

        return best_a

    # ------------------------------------------------------------------
    # Rollout (simulation) policy – uniform random over mask
    # ------------------------------------------------------------------

    def _rollout(
        self,
        particle: dict,
        action_mask: np.ndarray,
        depth: int,
    ) -> float:
        """Random rollout from *particle* for *depth* steps."""
        if depth == 0:
            return 0.0

        valid = np.where(action_mask)[0]
        if len(valid) == 0:
            return 0.0

        action = int(np.random.choice(valid))
        try:
            next_particle, _, reward = self._sandbox.sample_transition(
                particle, action
            )
        except Exception:
            return 0.0

        return reward + self.gamma * self._rollout(next_particle, action_mask, depth - 1)

    # ------------------------------------------------------------------
    # Best action extraction
    # ------------------------------------------------------------------

    def _best_action(self, root: tuple, action_mask: np.ndarray) -> int:
        """Return action with highest mean Q at *root*, masked by action_mask."""
        node = self._tree.get(root)
        if node is None or not node["N_a"]:
            # No tree statistics yet – fall back to first valid action
            valid = np.where(action_mask)[0]
            return int(valid[0]) if len(valid) > 0 else 0

        valid_set = set(np.where(action_mask)[0].tolist())
        best_val = -float("inf")
        best_a = None

        for a, n_a in node["N_a"].items():
            if a not in valid_set:
                continue
            mean_q = node["Q"][a] / n_a
            if mean_q > best_val:
                best_val = mean_q
                best_a = a

        if best_a is None:
            valid = np.where(action_mask)[0]
            return int(valid[0]) if len(valid) > 0 else 0

        return best_a


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _obs_key(obs_vec: np.ndarray) -> tuple:
    """Convert an observation vector to a hashable tuple key."""
    return tuple(obs_vec.astype(int).tolist())
