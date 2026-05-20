"""Probabilistic IDS wrapper — adds 3 per-subnet recon-detection bits to obs.

After every env step, peeks at the last action taken by Red (and Green) via
the underlying CybORG and emits a 1-bit detection flag per subnet:

    bit_i = Bernoulli(p_tp)   if red's last action targets subnet i
            Bernoulli(p_fp)   if green's last action targets subnet i
            Bernoulli(p_fp/5) otherwise (small background false-positive rate)

Subnets, in order: User, Enterprise, Operational. The 3 bits are appended
to whatever the inner env's observation already had (52-d for ChallengeWrapper),
yielding a 55-d observation space.

Why action-level inspection: at our abstraction CybORG doesn't emit packets,
so a real IDS signature (multiple SYN to one host, ICMP broadcast to a /28,
etc.) has no direct analog. The action type is its abstract equivalent —
``DiscoverRemoteSystems`` *is* the pingsweep, ``DiscoverNetworkServices`` *is*
the portscan. Reading them via ``cyborg.get_last_action(agent)`` is the
analog of a SOC tool reading off NetFlow / EDR.
"""
import random

import numpy as np
import gymnasium as gym


SUBNETS = ['User', 'Enterprise', 'Operational']
EXTRA_DIMS = len(SUBNETS)  # 3 bits, one per subnet


class IDSWrapper(gym.Wrapper):
    def __init__(self, env, p_tp=0.95, p_fp=0.05, seed=None):
        super().__init__(env)
        self.p_tp = float(p_tp)
        self.p_fp = float(p_fp)
        self._rng = random.Random(seed)

        # Extend the observation space.
        low = np.concatenate([env.observation_space.low,
                              np.zeros(EXTRA_DIMS, dtype=env.observation_space.dtype)])
        high = np.concatenate([env.observation_space.high,
                               np.ones(EXTRA_DIMS, dtype=env.observation_space.dtype)])
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=env.observation_space.dtype)

        # Lookup tables rebuilt on every reset() since IPs are randomised
        # per-episode by CybORG. Subnet names are stable.
        self._cidr_to_subnet = None
        self._host_to_subnet = None
        self._ip_to_subnet = None

    # ------------------------------------------------------------------
    # CybORG access helpers
    # ------------------------------------------------------------------
    def _cyborg(self):
        """Walk down the wrapper chain to the raw CybORG instance."""
        env = self.env
        while not hasattr(env, 'environment_controller'):
            env = env.env
        return env

    def _build_maps(self):
        # NOTE: state.subnets is keyed by IPv4Network, not by name — the human
        # name (e.g. 'User', 'Enterprise', 'Operational') lives on s.name.
        state = self._cyborg().environment_controller.state
        self._cidr_to_subnet = {str(s.cidr): s.name for s in state.subnets.values()}
        self._host_to_subnet = {}
        for hostname, host in state.hosts.items():
            for iface in host.interfaces:
                cidr = str(iface.subnet) if iface.subnet is not None else None
                if cidr in self._cidr_to_subnet:
                    self._host_to_subnet[hostname] = self._cidr_to_subnet[cidr]
                    break
        self._ip_to_subnet = {ip: self._host_to_subnet[h]
                              for ip, h in state.ip_addresses.items()
                              if h in self._host_to_subnet}

    # ------------------------------------------------------------------
    # Action classification
    # ------------------------------------------------------------------
    def _classify(self, action):
        """Return the *name* of the subnet this action targets, or None."""
        if action is None:
            return None
        # Lazy imports — these modules are heavy to load and we only need
        # the classes for isinstance checks.
        from CybORG.Shared.Actions.AbstractActions.DiscoverRemoteSystems import DiscoverRemoteSystems
        from CybORG.Shared.Actions.AbstractActions.DiscoverNetworkServices import DiscoverNetworkServices
        from CybORG.Shared.Actions.AbstractActions.ExploitRemoteService import ExploitRemoteService
        from CybORG.Shared.Actions import GreenPingSweep, GreenPortScan, GreenConnection

        if isinstance(action, (DiscoverRemoteSystems, GreenPingSweep)):
            cidr = str(action.subnet) if getattr(action, 'subnet', None) is not None else None
            return self._cidr_to_subnet.get(cidr)
        if isinstance(action, (DiscoverNetworkServices, ExploitRemoteService)):
            ip = getattr(action, 'ip_address', None)
            return self._ip_to_subnet.get(ip)
        if isinstance(action, (GreenPortScan, GreenConnection)):
            hostname = getattr(action, 'hostname', None)
            return self._host_to_subnet.get(hostname)
        return None

    # ------------------------------------------------------------------
    # IDS bit computation
    # ------------------------------------------------------------------
    def _ids_bits(self):
        cyborg = self._cyborg()
        red_target = self._classify(cyborg.get_last_action('Red'))
        green_target = self._classify(cyborg.get_last_action('Green'))

        bits = np.zeros(EXTRA_DIMS, dtype=self.observation_space.dtype)
        for i, name in enumerate(SUBNETS):
            if red_target == name:
                p = self.p_tp
            elif green_target == name:
                p = self.p_fp
            else:
                p = self.p_fp / 5.0
            bits[i] = 1.0 if self._rng.random() < p else 0.0
        return bits

    def _extend(self, obs):
        if self._host_to_subnet is None:
            self._build_maps()
        return np.concatenate([np.asarray(obs, dtype=self.observation_space.dtype),
                               self._ids_bits()])

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        # IPs change per episode — rebuild the maps lazily on first call.
        self._host_to_subnet = None
        self._cidr_to_subnet = None
        self._ip_to_subnet = None
        return self._extend(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extend(obs), reward, terminated, truncated, info
