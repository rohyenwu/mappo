"""Baseline MAC protocols for WiFi v2 evaluation."""

from dataclasses import dataclass

import numpy as np

CW_MIN = 16
CW_MAX = 1024
RETRY_LIMIT = 6
DIFS_SLOTS = 2


@dataclass
class BackoffState:
    cw: int = CW_MIN
    backoff: int = 0
    retry: int = 0


class MLDBackoffMAC:
    """BEB-style MAC baseline for MLD agents on both links."""

    def __init__(
        self,
        num_agents: int,
        agent_to_mld_link,
        cw_min: int = CW_MIN,
        cw_max: int = CW_MAX,
        retry_limit: int = RETRY_LIMIT,
        difs_slots: int = DIFS_SLOTS,
        rng=None,
    ):
        self.num_agents = num_agents
        self.agent_to_mld_link = list(agent_to_mld_link)
        self.cw_min = cw_min
        self.cw_max = cw_max
        self.retry_limit = retry_limit
        self.difs_slots = difs_slots
        self.rng = np.random.default_rng() if rng is None else rng
        self.states = [BackoffState() for _ in range(num_agents)]
        self.num_links = max(link_id for _, link_id in self.agent_to_mld_link) + 1
        self.idle_slots_by_link = np.zeros(self.num_links, dtype=np.int32)

    def _draw_backoff(self, cw: int) -> int:
        return int(self.rng.integers(0, cw))

    def reset_round(self, env):
        self.idle_slots_by_link.fill(0)
        for aid, state in enumerate(self.states):
            state.cw = self.cw_min
            state.retry = 0
            mld_id, link_id = self.agent_to_mld_link[aid]
            pending = env.D[mld_id, link_id] > env.S[mld_id, link_id]
            state.backoff = self._draw_backoff(state.cw) if pending else 0

    def act(self, env):
        actions = np.zeros((self.num_agents, 1), dtype=np.int32)
        pending_mask = np.zeros(self.num_agents, dtype=bool)

        for aid, state in enumerate(self.states):
            mld_id, link_id = self.agent_to_mld_link[aid]
            pending = env.D[mld_id, link_id] > env.S[mld_id, link_id]
            pending_mask[aid] = pending
            if pending and state.backoff == 0:
                actions[aid, 0] = 1

        return actions, pending_mask

    def update(self, env, actions, infos, pending_mask):
        actions_flat = actions.reshape(-1)
        link_results = [""] * self.num_links

        for aid, info in enumerate(infos):
            _, link_id = self.agent_to_mld_link[aid]
            result = info.get("txop_result", "")
            if result:
                link_results[link_id] = result

        for link_id, result in enumerate(link_results):
            if result == "idle":
                self.idle_slots_by_link[link_id] += 1
            elif result:
                self.idle_slots_by_link[link_id] = 0

        for aid, state in enumerate(self.states):
            pending_before = bool(pending_mask[aid])
            _, link_id = self.agent_to_mld_link[aid]
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
                    mld_id, link_id = self.agent_to_mld_link[aid]
                    pending_after = env.D[mld_id, link_id] > env.S[mld_id, link_id]
                    state.backoff = self._draw_backoff(state.cw) if pending_after else 0
                elif result == "collision":
                    state.retry += 1
                    if state.retry > self.retry_limit:
                        state.cw = self.cw_min
                        state.retry = 0
                    else:
                        state.cw = min(state.cw * 2, self.cw_max)
                    state.backoff = self._draw_backoff(state.cw)
            else:
                if (
                    result == "idle"
                    and state.backoff > 0
                    and self.idle_slots_by_link[link_id] > self.difs_slots
                ):
                    state.backoff -= 1
