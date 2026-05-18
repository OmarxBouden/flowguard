"""
CybORGSandbox: an internal simulation clone used for C-POMCP particle rollouts.

Wraps a private CybORG instance (same scenario + Red heuristic as the real env)
and exposes save/restore/step primitives so the planner can branch from any
serialised particle state without touching the true game instance.
"""
import copy
import numpy as np

from CybORG import CybORG
from CybORG.Agents.Wrappers.BlueTableWrapper import BlueTableWrapper
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper


class CybORGSandbox:
    """Self-contained CybORG simulation for offline particle rollouts.

    The sandbox owns its own CybORG instance.  The caller never interacts with
    the real evaluation environment through this object.
    """

    def __init__(self, scenario_path: str, red_agent_cls, green_agent_cls=None):
        self._scenario_path = scenario_path
        self._red_agent_cls = red_agent_cls
        self._green_agent_cls = green_agent_cls
        self._build_env()

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _build_env(self):
        agents = {"Red": self._red_agent_cls}
        if self._green_agent_cls is not None:
            agents["Green"] = self._green_agent_cls
        cyborg = CybORG(self._scenario_path, "sim", agents=agents)
        self._cyborg = cyborg
        self._sim_ctrl = cyborg.environment_controller

        # Wrap only through BlueTable + EnumAction so we can intercept the
        # observation vector and reward directly.
        self._btw = BlueTableWrapper(cyborg, output_mode="vector")
        self._enum = EnumActionWrapper(self._btw)

        # Warm up: one reset so possible_actions is populated and blue_info
        # baseline is established.
        self._enum.reset("Blue")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset to a new episode and return the initial 52-dim obs vector."""
        result = self._enum.reset("Blue")
        return np.array(result.observation, dtype=np.float32)

    def save_state(self) -> dict:
        """Serialise the full simulation state as a deep-copy snapshot."""
        return {
            "sim_state": copy.deepcopy(self._sim_ctrl.state),
            "action": copy.deepcopy(self._sim_ctrl.action),
            "observation": copy.deepcopy(self._sim_ctrl.observation),
            "reward": copy.deepcopy(self._sim_ctrl.reward),
            "done": self._sim_ctrl.done,
            "steps": self._sim_ctrl.steps,
            "blue_info": copy.deepcopy(self._btw.blue_info),
            "baseline": copy.deepcopy(self._btw.baseline),
        }

    def restore_state(self, snapshot: dict) -> None:
        """Restore the simulation to a previously saved snapshot."""
        self._sim_ctrl.state = copy.deepcopy(snapshot["sim_state"])
        self._sim_ctrl.action = copy.deepcopy(snapshot["action"])
        self._sim_ctrl.observation = copy.deepcopy(snapshot["observation"])
        self._sim_ctrl.reward = copy.deepcopy(snapshot["reward"])
        self._sim_ctrl.done = snapshot["done"]
        self._sim_ctrl.steps = snapshot["steps"]
        self._btw.blue_info = copy.deepcopy(snapshot["blue_info"])
        self._btw.baseline = copy.deepcopy(snapshot["baseline"])

    def sample_transition(
        self, state_snapshot: dict, action_idx: int
    ) -> tuple:
        """Restore *state_snapshot*, step with *action_idx*, return results.

        Returns
        -------
        next_snapshot : dict
            Serialised state after the transition.
        obs_vec : np.ndarray
            52-dim Blue observation produced by BlueTableWrapper.
        reward : float
            Scalar reward for the Blue agent.
        """
        self.restore_state(state_snapshot)
        result = self._enum.step("Blue", action_idx)
        obs_vec = np.array(result.observation, dtype=np.float32)
        reward = float(result.reward) if result.reward is not None else 0.0
        next_snapshot = self.save_state()
        return next_snapshot, obs_vec, reward

    # ------------------------------------------------------------------
    # Accessors used by other modules
    # ------------------------------------------------------------------

    @property
    def possible_actions(self) -> list:
        """The enumerated Action objects (indexed 0 … n_actions-1)."""
        return self._enum.possible_actions

    @property
    def n_actions(self) -> int:
        return len(self._enum.possible_actions)

    def build_action_map(self) -> dict:
        """Return {action_idx: {'type': str, 'hostname': str | None}}."""
        mapping = {}
        for i, action in enumerate(self.possible_actions):
            params = action.get_params() if hasattr(action, "get_params") else {}
            mapping[i] = {
                "type": type(action).__name__,
                "hostname": params.get("hostname", None),
            }
        return mapping
