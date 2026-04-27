"""Baseline MAC protocols for WiFi v4 evaluation."""

from dataclasses import dataclass

import numpy as np

CW_MIN = 16
CW_MAX = 1024
RETRY_LIMIT = 6


@dataclass
class BackoffState:
    cw: int = CW_MIN
    backoff: int = 0
    retry: int = 0


class MLDBackoffMAC:
    """BEB-style MAC baseline driven by WiFi v4 access opportunities."""

    def __init__(
        self,
        num_agents: int,
        agent_to_mld_link,
        cw_min: int = CW_MIN,
        cw_max: int = CW_MAX,
        retry_limit: int = RETRY_LIMIT,
        rng=None,
    ):
        self.num_agents = num_agents
        self.agent_to_mld_link = list(agent_to_mld_link)
        self.cw_min = cw_min
        self.cw_max = cw_max
        self.retry_limit = retry_limit
        self.rng = np.random.default_rng() if rng is None else rng
        self.states = [BackoffState() for _ in range(num_agents)]

    def _draw_backoff(self, cw: int) -> int:
        return int(self.rng.integers(0, cw))

    def _pending(self, env, aid: int) -> bool:
        mld_id, _ = self.agent_to_mld_link[aid]
        return bool(env.D[mld_id] > env.S[mld_id])

    def reset_round(self, env):
        for aid, state in enumerate(self.states):
            state.cw = self.cw_min
            state.retry = 0
            state.backoff = self._draw_backoff(state.cw) if self._pending(env, aid) else 0

    def act(self, env):
        actions = np.zeros((self.num_agents, 1), dtype=np.int32)
        pending_mask = np.zeros(self.num_agents, dtype=bool)
        for aid, state in enumerate(self.states):
            pending = self._pending(env, aid)
            pending_mask[aid] = pending
            if pending and state.backoff == 0:
                actions[aid, 0] = 1
        return actions, pending_mask

    def update(self, env, actions, infos, pending_mask):
        actions_flat = actions.reshape(-1)
        for aid, state in enumerate(self.states):
            pending_before = bool(pending_mask[aid])
            pending_after = self._pending(env, aid)

            if not pending_before:
                state.cw = self.cw_min
                state.retry = 0
                state.backoff = 0
                continue

            result = infos[aid].get("txop_result", "")
            transmitted = actions_flat[aid] == 1

            if transmitted:
                if result == "success":
                    state.cw = self.cw_min
                    state.retry = 0
                    state.backoff = self._draw_backoff(state.cw) if pending_after else 0
                elif result == "collision":
                    state.retry += 1
                    if state.retry > self.retry_limit:
                        state.cw = self.cw_min
                        state.retry = 0
                    else:
                        state.cw = min(state.cw * 2, self.cw_max)
                    state.backoff = self._draw_backoff(state.cw) if pending_after else 0
                elif not pending_after:
                    state.cw = self.cw_min
                    state.retry = 0
                    state.backoff = 0
            elif not pending_after:
                state.cw = self.cw_min
                state.retry = 0
                state.backoff = 0
            elif result == "idle" and state.backoff > 0:
                state.backoff -= 1
