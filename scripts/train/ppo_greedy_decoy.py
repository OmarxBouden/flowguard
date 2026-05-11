"""
Decoy prologues applied at inference on top of a plain PPO model — no separate
training, since the saved weights are identical to ppo_baseline. Action indices
are raw indices into the 145-action ChallengeWrapper space.
"""


class SimpleDecoyPolicy:
    """4-action prologue: DecoyApache on Enterprise0/1/2 and Op_Server0, then defer to PPO."""
    PRIORITY = [29, 30, 31, 35]

    def __init__(self, model):
        self.model = model
        self._queue = list(self.PRIORITY)

    def reset(self):
        self._queue = list(self.PRIORITY)

    def predict(self, obs, deterministic=True):
        if self._queue:
            return self._queue.pop(0), None
        return self.model.predict(obs, deterministic=deterministic)


class WideDecoyPolicy:
    """
    17-action prologue that tries multiple valid decoy types on every high-value
    host, then defers to PPO. Only the per-host action indices are reused from
    the Cardiff CAGE 2 submission (john-cardiff/-cyborg-cage-2); the rest of
    their algorithm (reduced action space, obs augmentation, red fingerprinting,
    per-red models) is not used here.
    """
    HOST_QUEUES = [
        ('Enterprise0', [55, 107, 120, 29]),  # HarakaSMPT, Tomcat, Vsftpd, Apache
        ('Enterprise1', [43]),                # Femitter
        ('Enterprise2', [44]),                # Femitter
        ('Op_Server0',  [61, 35, 113, 126]),  # HarakaSMPT, Apache, Tomcat, Vsftpd
        ('User1', [51, 116, 38, 90]),         # Femitter, Tomcat, Apache, SSHD
        ('User2', [130, 91]),                 # Vsftpd, SSHD
        ('User3', [131]),                     # Vsftpd
    ]

    def __init__(self, model):
        self.model = model
        self._queue = self._build_queue()

    @classmethod
    def _build_queue(cls):
        q = []
        for _, actions in cls.HOST_QUEUES:
            q.extend(actions)
        return q

    def reset(self):
        self._queue = self._build_queue()

    def predict(self, obs, deterministic=True):
        if self._queue:
            return self._queue.pop(0), None
        return self.model.predict(obs, deterministic=deterministic)
