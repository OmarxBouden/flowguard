"""
Particle-filter belief tracker for C-POMCP.

Each *particle* is a serialised CybORG state snapshot (produced by
CybORGSandbox.save_state).  The filter maintains M such particles and updates
them after every Blue action / real observation pair.

Update procedure
----------------
1. For each particle, step it through the sandbox with the Blue action taken.
2. Keep particles whose simulated observation matches the real observation.
3. If the surviving count falls below 10 % of M, run particle rejuvenation:
   - Upsample surviving particles back to M by resampling with replacement.
   - Perturb each new particle's compromise flags with probability p_mutate.
4. If zero particles survive, reinitialise from the current sandbox reset.
"""

import copy
import random
import numpy as np

from agents.simulator_model import CybORGSandbox
from agents.causal_graph import HOSTS_ORDERED, HOST_INDEX

# Observation fields per host (4 floats: act0, act1, comp0, comp1)
_OBS_PER_HOST = 4


class BeliefParticleFilter:
    """Approximate belief state as a set of M CybORG state snapshots.

    Parameters
    ----------
    sandbox:
        A CybORGSandbox instance used for simulating transitions.
    M:
        Target number of particles (default 200).
    p_mutate:
        Per-host compromise-flag mutation probability during rejuvenation
        (default 0.05 as specified in C-POMCP paper).
    rejuv_threshold:
        Fraction of M below which rejuvenation is triggered (default 0.10).
    """

    def __init__(
        self,
        sandbox: CybORGSandbox,
        M: int = 200,
        p_mutate: float = 0.05,
        rejuv_threshold: float = 0.10,
    ):
        self._sandbox = sandbox
        self.M = M
        self.p_mutate = p_mutate
        self._rejuv_threshold = rejuv_threshold
        self._particles: list = []  # list of state snapshots

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the sandbox and initialise all particles from the start state.

        Returns
        -------
        obs : np.ndarray
            The initial 52-dim observation from the sandbox reset.
        """
        obs = self._sandbox.reset()
        initial_snap = self._sandbox.save_state()
        self._particles = [copy.deepcopy(initial_snap) for _ in range(self.M)]
        return obs

    # ------------------------------------------------------------------
    # Belief update
    # ------------------------------------------------------------------

    def update(self, action_taken: int, real_observation: np.ndarray) -> None:
        """Propagate particles and weight by observation consistency.

        Parameters
        ----------
        action_taken:
            Integer index of the Blue action just executed in the real env.
        real_observation:
            The 52-dim observation vector returned by the real ChallengeWrapper
            after the step.
        """
        surviving: list = []

        for snap in self._particles:
            try:
                next_snap, sim_obs, _ = self._sandbox.sample_transition(
                    snap, action_taken
                )
            except Exception:
                # Skip corrupted particles rather than crashing.
                continue

            if self._obs_match(sim_obs, real_observation):
                surviving.append(next_snap)

        # Rejuvenation when surviving count is too low
        min_count = max(1, int(self._rejuv_threshold * self.M))
        if len(surviving) < min_count:
            surviving = self._rejuvenate(surviving, real_observation)

        self._particles = surviving

    # ------------------------------------------------------------------
    # Particle access
    # ------------------------------------------------------------------

    def sample_particle(self) -> dict:
        """Return a uniformly sampled particle snapshot."""
        return random.choice(self._particles)

    def particle_count(self) -> int:
        return len(self._particles)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_match(a: np.ndarray, b: np.ndarray) -> bool:
        """True if the two observation vectors are identical (binary)."""
        return np.array_equal(a.astype(int), b.astype(int))

    def _rejuvenate(
        self, survivors: list, real_observation: np.ndarray
    ) -> list:
        """Resample survivors to M particles, adding state perturbations.

        If no survivors exist, generate fresh particles by resetting the
        sandbox (all from the initial state).
        """
        if len(survivors) == 0:
            # Hard reset: rebuild pool from current reset state
            self._sandbox.reset()
            base = self._sandbox.save_state()
            survivors = [copy.deepcopy(base) for _ in range(self.M)]
            return survivors

        # Resample with replacement from surviving pool
        resampled = random.choices(survivors, k=self.M)

        # Apply minor compromise-state mutations to diversify
        mutated = []
        for snap in resampled:
            new_snap = copy.deepcopy(snap)
            self._mutate_blue_info(new_snap)
            mutated.append(new_snap)

        return mutated

    def _mutate_blue_info(self, snapshot: dict) -> None:
        """In-place perturbation of blue_info compromise flags in the snapshot.

        With probability p_mutate per host, flip the compromise status between
        adjacent levels ('No'↔'Unknown', 'User'↔'Privileged') to introduce
        state diversity without generating wildly implausible states.
        """
        blue_info = snapshot.get("blue_info", {})
        # blue_info structure: {hostname: [subnet, ip, hostname, activity, compromised]}
        LEVELS = ["No", "Unknown", "User", "Privileged"]

        for hostname, row in blue_info.items():
            if random.random() < self.p_mutate:
                current = row[-1]  # compromise status is last element
                if current in LEVELS:
                    idx = LEVELS.index(current)
                    # Flip to adjacent level
                    new_idx = idx + random.choice([-1, 1])
                    new_idx = max(0, min(len(LEVELS) - 1, new_idx))
                    row[-1] = LEVELS[new_idx]
