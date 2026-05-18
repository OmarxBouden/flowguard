"""
Causal structure engine for CAGE-2 / C-POMCP.

Implements the POMIS-based action pruning described in §VI-B of arXiv 2407.11070.

Three types of Blue interventions are NEVER included in a POMIS (paper §VI-B,
Remark 4 follow-on paragraph):

  (i)  Decoy actions on hosts where a decoy is already running.
       NOTE: determining this precisely requires agent state tracking beyond
       what the obs vector alone provides; rule (i) is therefore approximated
       as "always allow decoy actions" (conservative fallback).

  (ii) Defensive interventions (Remove, Restore) on hosts that are NOT
       believed compromised according to the belief b̂.

  (iii) Forensic / deceptive interventions (Analyse, all Decoy variants) on
        hosts that ARE believed compromised according to b̂.

Belief b̂ is approximated from the 52-dim obs vector:
  believed_compromised(H) = True  iff  compromise field of H ∈ {'User', 'Privileged'}
  believed_compromised(H) = False iff  compromise field of H ∈ {'No', 'Unknown'}

Sleep and Monitor are always included (global actions with no host target).
If the resulting mask is all-False, fall back to all actions enabled.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Static CAGE-2 host list
# ---------------------------------------------------------------------------

# Alphabetical ordering matches BlueTableWrapper's sorted table (sortby='Hostname').
HOSTS_ORDERED = [
    "Defender",       # 0
    "Enterprise0",    # 1
    "Enterprise1",    # 2
    "Enterprise2",    # 3
    "Op_Host0",       # 4
    "Op_Host1",       # 5
    "Op_Host2",       # 6
    "Op_Server0",     # 7
    "User0",          # 8
    "User1",          # 9
    "User2",          # 10
    "User3",          # 11
    "User4",          # 12
]

HOST_INDEX = {h: i for i, h in enumerate(HOSTS_ORDERED)}

# ---------------------------------------------------------------------------
# POMIS action-type classification (paper §VI-B)
# ---------------------------------------------------------------------------

# Rule (ii): include ONLY for believed-compromised hosts
_DEFENSIVE_ACTIONS = frozenset({"Remove", "Restore"})

# Rule (iii): include ONLY for NOT believed-compromised hosts
_FORENSIC_DECEPTIVE_ACTIONS = frozenset({
    "Analyse",
    "DeployDecoy",
    "DecoyApache", "DecoyFemitter", "DecoyHarakaSMPT", "DecoySmss",
    "DecoySSHD",   "DecoySvchost",  "DecoyTomcat",     "DecoyVsftpd",
})

# ---------------------------------------------------------------------------
# Obs-vector parsing helpers
# ---------------------------------------------------------------------------

# Each host occupies 4 floats: [act0, act1, comp0, comp1]
_OBS_PER_HOST = 4


def _parse_obs(obs: np.ndarray) -> list:
    """Return per-host status dicts from the 52-dim Blue obs vector.

    Returns a list of 13 dicts with keys:
      activity   : 'None' | 'Scan' | 'Exploit'
      compromised: 'No'   | 'Unknown' | 'User' | 'Privileged'
    """
    statuses = []
    for i, host in enumerate(HOSTS_ORDERED):
        base = i * _OBS_PER_HOST
        a0, a1 = int(obs[base]), int(obs[base + 1])
        c0, c1 = int(obs[base + 2]), int(obs[base + 3])

        if a0 == 0 and a1 == 0:
            activity = "None"
        elif a0 == 1 and a1 == 0:
            activity = "Scan"
        else:
            activity = "Exploit"

        if c0 == 0 and c1 == 0:
            comp = "No"
        elif c0 == 1 and c1 == 0:
            comp = "Unknown"
        elif c0 == 0 and c1 == 1:
            comp = "User"
        else:
            comp = "Privileged"

        statuses.append({"host": host, "activity": activity, "compromised": comp})
    return statuses


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def get_causally_relevant_actions(
    obs: np.ndarray,
    action_map: dict,
) -> np.ndarray:
    """Compute a boolean action-mask of shape (n_actions,) using POMIS rules.

    Implements §VI-B of arXiv 2407.11070:
      - Rule (ii): Remove/Restore included only for believed-compromised hosts.
      - Rule (iii): Analyse/Decoy included only for NOT believed-compromised hosts.
      - Sleep/Monitor: always included.
      - Unknown action types: conservatively included.

    Parameters
    ----------
    obs:
        52-dim Blue observation vector from ChallengeWrapper/BlueTableWrapper.
    action_map:
        Dict mapping action index → {'type': str, 'hostname': str | None},
        produced by CybORGSandbox.build_action_map().

    Returns
    -------
    mask : np.ndarray[bool], shape (n_actions,)
    """
    n = len(action_map)
    mask = np.zeros(n, dtype=bool)

    statuses = _parse_obs(obs)
    believed_compromised = {
        s["host"]: s["compromised"] in ("User", "Privileged")
        for s in statuses
    }

    for idx, info in action_map.items():
        atype = info["type"]
        hostname = info["hostname"]

        if atype in ("Sleep", "Monitor"):
            mask[idx] = True
            continue

        if hostname is None:
            mask[idx] = True
            continue

        is_comp = believed_compromised.get(hostname, False)

        if atype in _DEFENSIVE_ACTIONS:
            # Rule (ii): only useful on compromised hosts
            mask[idx] = is_comp
        elif atype in _FORENSIC_DECEPTIVE_ACTIONS:
            # Rule (iii): only useful on non-compromised hosts
            mask[idx] = not is_comp
        else:
            mask[idx] = True

    # Safety fallback: never return an all-False mask
    if not mask.any():
        mask[:] = True

    return mask
