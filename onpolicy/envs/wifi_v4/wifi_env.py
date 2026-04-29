"""WiFi v4 environment with slot-time access opportunities."""

import numpy as np
from gym import spaces

SLD_CW_MIN = 16
SLD_CW_MAX = 1024
SLD_RETRY_LIMIT = 6
DIFS_SLOTS = 2
TX_BUSY_SLOTS = 1


class WiFiEnvV4:
    """
    WiFi coexistence environment with slot-time channel evolution.

    - 1 env step = 1 access opportunity for the learning agents
    - internal channel time advances in slot units
    - MLD actions are binary decisions at each access opportunity
    - SLD stations follow slot-based DCF-style backoff on 2.4 GHz
    """

    def __init__(
        self,
        num_mld: int = 3,
        num_sld: int = 3,
        round_length: int = 50,
        mu_range: tuple = (0.2, 0.8),
        mu_profile=None,
        sld_mu: float = 0.3,
        f_func: str = "fulfillment",
        g_func: str = "fulfillment",
        eta: float = 1.0,
        zeta: float = 1.0,
        r_sld: float = 0.3,
        c_idle: float = 0.3,
        theta_scale: float = 1.0,
        gamma: float = 0.99,
    ):
        del sld_mu, f_func, g_func, gamma

        self.num_mld = num_mld
        self.num_sld = num_sld
        self.round_length = round_length
        self.mu_range = mu_range
        self.mu_profile = None if mu_profile is None else np.asarray(mu_profile, dtype=np.float32)
        self.eta = eta
        self.zeta = zeta
        self.r_sld = r_sld
        self.c_idle = c_idle
        self.theta_scale = theta_scale

        self.num_links = 2
        self.n_sld_per_link = [num_sld, 0]
        self.num_agents = num_mld * 2

        self.agent_to_mld_link = []
        for mld_id in range(num_mld):
            self.agent_to_mld_link.append((mld_id, 0))
            self.agent_to_mld_link.append((mld_id, 1))

        self.link_agents = {0: [], 1: []}
        for aid, (_, link_id) in enumerate(self.agent_to_mld_link):
            self.link_agents[link_id].append(aid)

        self.mu = np.zeros((num_mld, 2), dtype=np.float32)

        self.obs_dim = 10
        obs_low = np.zeros(self.obs_dim, dtype=np.float32)
        obs_high = np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = [
            spaces.Box(obs_low, obs_high, dtype=np.float32)
        ] * self.num_agents

        self.share_obs_dim = self.obs_dim * self.num_mld + 2 + self.num_mld + 2
        self.share_observation_space = [
            spaces.Box(
                low=np.zeros(self.share_obs_dim, dtype=np.float32),
                high=np.ones(self.share_obs_dim, dtype=np.float32),
                dtype=np.float32,
            )
        ] * self.num_agents

        self.action_space = [spaces.Discrete(2)] * self.num_agents

        self._init_state()

    def _init_state(self):
        self.t = 0
        self.total_slots_elapsed = 0
        self.last_step_slots = 0

        self.D = np.zeros(self.num_mld, dtype=np.int32)
        self.S = np.zeros(self.num_mld, dtype=np.int32)
        self.P = np.zeros(self.num_mld, dtype=np.int32)

        self.sld_state = []
        for _ in range(self.num_sld):
            self.sld_state.append(
                {
                    "cw": SLD_CW_MIN,
                    "backoff": int(np.random.randint(0, SLD_CW_MIN)),
                    "retry": 0,
                }
            )

        self.round_sld_success = 0

        self.last_round_S = np.zeros(self.num_mld, dtype=np.int32)
        self.last_round_D = np.zeros(self.num_mld, dtype=np.int32)
        self.last_round_sld_success = 0
        self.last_round_total_slots = 0

        self.link_successes = np.zeros((self.num_mld, 2), dtype=np.int32)
        self.last_round_link_successes = np.zeros((self.num_mld, 2), dtype=np.int32)
        self.link_attempts = np.zeros((self.num_mld, 2), dtype=np.int32)
        self.last_round_link_attempts = np.zeros((self.num_mld, 2), dtype=np.int32)
        self.prev_actions = np.zeros(self.num_agents, dtype=np.int32)

        self.round_collisions = np.zeros(2, dtype=np.int32)
        self.round_mld_transmissions = np.zeros(2, dtype=np.int32)
        self.last_round_collisions = np.zeros(2, dtype=np.int32)
        self.last_round_mld_transmissions = np.zeros(2, dtype=np.int32)

        self.link_busy_slots = np.zeros(self.num_links, dtype=np.int32)
        self.link_idle_slots = np.zeros(self.num_links, dtype=np.int32)

    def _reset_round(self):
        self.t = 0
        self.total_slots_elapsed = 0
        self.last_step_slots = 0
        self.S[:] = 0
        self.P[:] = 0
        self.D[:] = 0
        self.link_successes[:] = 0
        self.link_attempts[:] = 0
        self.prev_actions[:] = 0
        self.round_sld_success = 0
        self.round_collisions[:] = 0
        self.round_mld_transmissions[:] = 0
        self.link_busy_slots[:] = 0
        self.link_idle_slots[:] = 0

        for sld in self.sld_state:
            sld["cw"] = SLD_CW_MIN
            sld["backoff"] = int(np.random.randint(0, sld["cw"]))
            sld["retry"] = 0

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _generate_mu(self):
        if self.mu_profile is not None:
            if self.mu_profile.shape != (self.num_mld,):
                raise ValueError(
                    f"mu_profile must have length {self.num_mld}, got shape {self.mu_profile.shape}"
                )
            profile = np.clip(self.mu_profile.astype(np.float32), 0.0, 1.0)
            self.mu[:, 0] = profile
            self.mu[:, 1] = profile
            return

        mu_min, mu_max = self.mu_range
        if mu_min > mu_max:
            mu_min, mu_max = mu_max, mu_min
        mu_min = float(np.clip(mu_min, 0.0, 1.0))
        mu_max = float(np.clip(mu_max, 0.0, 1.0))
        fixed_mu = np.linspace(mu_min, mu_max, self.num_mld, dtype=np.float32)
        self.mu[:, 0] = fixed_mu
        self.mu[:, 1] = fixed_mu

    def _generate_packets(self):
        for mld_id in range(self.num_mld):
            self.D[mld_id] = sum(
                np.random.binomial(self.round_length, self.mu[mld_id, link_id])
                for link_id in range(self.num_links)
            )

    def _get_fulfillment(self, mld_id, link_id):
        del link_id
        demand = self.D[mld_id]
        if demand == 0:
            return 1.0
        return self.S[mld_id] / demand

    def _get_link_urgencies(self, link_id):
        lacks = {}
        total_lack = 0.0
        for aid in self.link_agents[link_id]:
            mld_id, _ = self.agent_to_mld_link[aid]
            lack = max(0.0, 1.0 - self._get_fulfillment(mld_id, link_id))
            lacks[aid] = lack
            total_lack += lack

        if total_lack <= 1e-9:
            return {aid: 0.0 for aid in self.link_agents[link_id]}
        return {aid: lack / total_lack for aid, lack in lacks.items()}

    def _get_link_top_urgency_agent(self, link_id, urgencies=None):
        if urgencies is None:
            urgencies = self._get_link_urgencies(link_id)
        if not urgencies:
            return None

        max_urgency = max(urgencies.values())
        if max_urgency <= 1e-9:
            return None

        top_candidates = [
            aid for aid, urgency in urgencies.items()
            if abs(urgency - max_urgency) < 1e-9
        ]
        return min(top_candidates)

    def _build_obs(self):
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        slots_norm = max(self.last_round_total_slots, 1)
        for aid in range(self.num_agents):
            mld_id, link_id = self.agent_to_mld_link[aid]
            shared_load = self.D[mld_id] / max(self.round_length * self.num_links, 1)
            n_sld_norm = self.n_sld_per_link[link_id] / max(self.num_sld, 1)
            fulfillment = self._get_fulfillment(mld_id, link_id)
            prev_action_self = float(self.prev_actions[aid])
            peer_aid = aid + 1 if (aid % 2 == 0) else aid - 1
            prev_action_peer = float(self.prev_actions[peer_aid])
            busy_flag = float(self.link_busy_slots[link_id] > 0)
            idle_progress = min(self.link_idle_slots[link_id], DIFS_SLOTS) / float(DIFS_SLOTS)
            step_slots_norm = min(self.last_step_slots, slots_norm) / float(slots_norm)
            link_onehot = [1.0, 0.0] if link_id == 0 else [0.0, 1.0]
            obs[aid] = [
                shared_load,
                n_sld_norm,
                fulfillment,
                prev_action_self,
                prev_action_peer,
                busy_flag,
                idle_progress,
                step_slots_norm,
                *link_onehot,
            ]
        return obs

    def _build_available_actions(self):
        available_actions = np.ones((self.num_agents, 2), dtype=np.float32)
        for aid in range(self.num_agents):
            mld_id, _ = self.agent_to_mld_link[aid]
            if self.S[mld_id] >= self.D[mld_id]:
                available_actions[aid] = [1.0, 0.0]
        return available_actions

    def _build_share_obs(self, obs):
        share_obs = np.zeros((self.num_agents, self.share_obs_dim), dtype=np.float32)
        curr_avg_sld = self.round_sld_success / max(self.total_slots_elapsed, 1)
        for aid in range(self.num_agents):
            _, link_id = self.agent_to_mld_link[aid]
            link_onehot = [1.0, 0.0] if link_id == 0 else [0.0, 1.0]
            link_aids = self.link_agents[link_id]
            link_obs_flat = obs[link_aids].flatten()
            link_shared_fulfillment = obs[link_aids, 2]
            link_avg_sld = curr_avg_sld if link_id == 0 else 0.0
            share_obs[aid] = np.concatenate(
                [
                    link_obs_flat,
                    [link_avg_sld, obs[aid, 7]],
                    link_shared_fulfillment,
                    link_onehot,
                ]
            )
        return share_obs

    def _decrement_sld_backoff(self):
        for sld in self.sld_state:
            if sld["backoff"] > 0:
                sld["backoff"] -= 1

    def _advance_until_access_opportunity(self):
        slots_advanced = 0
        while True:
            slots_advanced += 1
            self.total_slots_elapsed += 1
            ready_links = np.zeros(self.num_links, dtype=bool)

            for link_id in range(self.num_links):
                if self.link_busy_slots[link_id] > 0:
                    self.link_busy_slots[link_id] -= 1
                    if self.link_busy_slots[link_id] == 0:
                        self.link_idle_slots[link_id] = 0
                    continue

                self.link_idle_slots[link_id] += 1
                if self.link_idle_slots[link_id] == DIFS_SLOTS:
                    ready_links[link_id] = True
                elif self.link_idle_slots[link_id] > DIFS_SLOTS:
                    if link_id == 0:
                        self._decrement_sld_backoff()
                    ready_links[link_id] = True

            if np.all(ready_links):
                self.last_step_slots = slots_advanced
                return

    def _get_sld_transmitters(self, link_id):
        if link_id != 0:
            return []
        return [idx for idx, sld in enumerate(self.sld_state) if sld["backoff"] == 0]

    def _update_sld_after_event(self, result, sld_txers):
        for idx, sld in enumerate(self.sld_state):
            if idx not in sld_txers:
                continue
            if result == "success":
                sld["cw"] = SLD_CW_MIN
                sld["retry"] = 0
                sld["backoff"] = int(np.random.randint(0, sld["cw"]))
                self.round_sld_success += 1
            elif result == "collision":
                sld["retry"] += 1
                if sld["retry"] > SLD_RETRY_LIMIT:
                    sld["cw"] = SLD_CW_MIN
                    sld["retry"] = 0
                else:
                    sld["cw"] = min(sld["cw"] * 2, SLD_CW_MAX)
                sld["backoff"] = int(np.random.randint(0, sld["cw"]))

    def reset(self):
        if np.any(self.S > 0) or self.round_sld_success > 0:
            self.last_round_D = self.D.copy()
            self.last_round_S = self.S.copy()
            self.last_round_link_successes = self.link_successes.copy()
            self.last_round_link_attempts = self.link_attempts.copy()
            self.last_round_sld_success = self.round_sld_success
            self.last_round_collisions = self.round_collisions.copy()
            self.last_round_mld_transmissions = self.round_mld_transmissions.copy()
            self.last_round_total_slots = self.total_slots_elapsed

        self._generate_mu()
        self._reset_round()
        self._generate_packets()

        obs = self._build_obs()
        share_obs = self._build_share_obs(obs)
        available_actions = self._build_available_actions()
        return obs, share_obs, available_actions

    def step(self, actions):
        self._advance_until_access_opportunity()

        actions_flat = actions.flatten().astype(int)
        step_available_actions = self._build_available_actions()
        actions_flat[step_available_actions[:, 1] < 0.5] = 0

        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        reward_global = np.zeros(self.num_agents, dtype=np.float32)
        reward_local = np.zeros(self.num_agents, dtype=np.float32)
        reward_sparse = np.zeros(self.num_agents, dtype=np.float32)
        reward_dense = np.zeros(self.num_agents, dtype=np.float32)
        link_result = np.full(self.num_agents, "", dtype=object)

        for link_id in range(self.num_links):
            link_aids = self.link_agents[link_id]
            mld_txers = [aid for aid in link_aids if actions_flat[aid] == 1]
            sld_txers = self._get_sld_transmitters(link_id)
            total_tx = len(mld_txers) + len(sld_txers)

            if total_tx == 0:
                result = "idle"
            elif total_tx == 1:
                result = "success"
            else:
                result = "collision"

            success_aid = None
            success_is_sld = False
            if result == "success":
                if len(mld_txers) == 1 and len(sld_txers) == 0:
                    success_aid = mld_txers[0]
                elif len(sld_txers) == 1 and len(mld_txers) == 0:
                    success_is_sld = True

            urgencies = self._get_link_urgencies(link_id)
            top_aid = self._get_link_top_urgency_agent(link_id, urgencies)

            if result == "success" and not success_is_sld:
                r_global = 1.0 if success_aid == top_aid else urgencies.get(success_aid, 0.0)
            elif result == "success" and success_is_sld:
                r_global = 0.0
            elif result == "collision":
                r_global = -1.0
            else:
                r_global = -self.c_idle

            for aid in link_aids:
                mld_id, _ = self.agent_to_mld_link[aid]
                transmitted = actions_flat[aid] == 1
                urgency = urgencies.get(aid, 0.0)
                remaining_demand = self.D[mld_id] - self.S[mld_id]

                if remaining_demand <= 0:
                    r_local = -1.0 if transmitted else 0.0
                elif top_aid is None:
                    r_local = -1.0 if transmitted else 0.0
                elif aid == top_aid:
                    r_local = 1.0 if transmitted else -(1.0 + urgency)
                else:
                    r_local = -1.0 if transmitted else 1.0

                rewards[aid, 0] = r_global + r_local
                reward_global[aid] = r_global
                reward_local[aid] = r_local
                reward_dense[aid] = r_global + r_local
                link_result[aid] = result

            if success_aid is not None:
                mld_id, _ = self.agent_to_mld_link[success_aid]
                if self.D[mld_id] > self.S[mld_id]:
                    self.S[mld_id] += 1
                    self.link_successes[mld_id, link_id] += 1

            for aid in mld_txers:
                mld_id, _ = self.agent_to_mld_link[aid]
                self.P[mld_id] += 1
                self.link_attempts[mld_id, link_id] += 1

            self.round_mld_transmissions[link_id] += len(mld_txers)
            if result == "collision":
                self.round_collisions[link_id] += 1

            self._update_sld_after_event(result, sld_txers)

            if result != "idle":
                self.link_busy_slots[link_id] = TX_BUSY_SLOTS
                self.link_idle_slots[link_id] = 0

        self.t += 1
        done = self.t >= self.round_length

        if done:
            reward_sparse = self._apply_sparse_reward_with_trace(rewards)

        self.prev_actions = actions_flat.copy()

        obs = self._build_obs()
        share_obs = self._build_share_obs(obs)
        dones = np.full(self.num_agents, done, dtype=bool)
        available_actions = self._build_available_actions()

        infos = []
        for aid in range(self.num_agents):
            mld_id, link_id = self.agent_to_mld_link[aid]
            infos.append(
                {
                    "fulfillment": self._get_fulfillment(mld_id, link_id),
                    "round_done": done,
                    "reward/global": float(reward_global[aid]),
                    "reward/local": float(reward_local[aid]),
                    "reward/dense": float(reward_dense[aid]),
                    "reward/sparse": float(reward_sparse[aid]),
                    "reward/total": float(rewards[aid, 0]),
                    f"reward/link_{link_id}/global": float(reward_global[aid]),
                    f"reward/link_{link_id}/local": float(reward_local[aid]),
                    f"reward/link_{link_id}/dense": float(reward_dense[aid]),
                    f"reward/link_{link_id}/sparse": float(reward_sparse[aid]),
                    f"reward/link_{link_id}/total": float(rewards[aid, 0]),
                    "txop_result": link_result[aid],
                    "step_slots": float(self.last_step_slots),
                    "total_slots_elapsed": float(self.total_slots_elapsed),
                }
            )

        if done:
            self.last_round_D = self.D.copy()
            self.last_round_S = self.S.copy()
            self.last_round_link_successes = self.link_successes.copy()
            self.last_round_link_attempts = self.link_attempts.copy()
            self.last_round_sld_success = self.round_sld_success
            self.last_round_collisions = self.round_collisions.copy()
            self.last_round_mld_transmissions = self.round_mld_transmissions.copy()
            self.last_round_total_slots = self.total_slots_elapsed

        return obs, share_obs, rewards, dones, infos, available_actions

    def _apply_sparse_reward_with_trace(self, rewards):
        sparse_rewards = np.zeros(self.num_agents, dtype=np.float32)

        n_mld_24 = len(self.link_agents[0])
        base_theta = self.num_sld / max(self.num_sld + n_mld_24, 1)
        theta = self.theta_scale * base_theta
        target_sld_success = theta * self.round_length
        actual_sld_success = float(self.round_sld_success)

        link_0_aids = self.link_agents[0]
        participations = []
        skips = []
        for aid in link_0_aids:
            mld_id, _ = self.agent_to_mld_link[aid]
            p_i = self.link_attempts[mld_id, 0]
            participations.append(p_i)
            skips.append(self.t - p_i)

        p_avg = np.mean(participations) if participations else 0
        skip_avg = np.mean(skips) if skips else 0

        for idx, aid in enumerate(link_0_aids):
            if actual_sld_success < target_sld_success:
                penalty = self.eta * max(0, participations[idx] - p_avg)
                rewards[aid, 0] -= penalty
                sparse_rewards[aid] -= penalty
            else:
                bonus = self.zeta * max(0, skips[idx] - skip_avg)
                rewards[aid, 0] += bonus
                sparse_rewards[aid] += bonus

        return sparse_rewards

    def close(self):
        pass

    def render(self, mode="human"):
        del mode

    def get_throughput(self):
        successes = self.last_round_link_successes
        sld_success = self.last_round_sld_success
        round_steps = max(self.round_length, 1)
        total_slots = max(self.last_round_total_slots, 1)

        result = {}
        for link_id, link_name in enumerate(["2_4GHz", "5GHz"]):
            mld_success = sum(successes[mld_id, link_id] for mld_id in range(self.num_mld))
            sld_link_success = sld_success if link_id == 0 else 0
            total_success = mld_success + sld_link_success
            result[f"throughput/{link_name}/mld"] = mld_success / round_steps
            result[f"throughput/{link_name}/sld"] = sld_link_success / round_steps
            result[f"throughput/{link_name}/total"] = total_success / round_steps
            result[f"throughput_slot/{link_name}/mld"] = mld_success / total_slots
            result[f"throughput_slot/{link_name}/sld"] = sld_link_success / total_slots
            result[f"throughput_slot/{link_name}/total"] = total_success / total_slots

        result["throughput/mld_total"] = successes.sum() / round_steps
        result["throughput/sld_total"] = sld_success / round_steps
        result["throughput/system"] = result["throughput/mld_total"] + result["throughput/sld_total"]
        result["throughput_slot/mld_total"] = successes.sum() / total_slots
        result["throughput_slot/sld_total"] = sld_success / total_slots
        result["throughput_slot/system"] = (
            result["throughput_slot/mld_total"] + result["throughput_slot/sld_total"]
        )
        result["throughput/slots_elapsed"] = float(self.last_round_total_slots)
        return result

    def get_collision_rate(self):
        opportunities = max(self.round_length, 1)
        total_slots = max(self.last_round_total_slots, 1)
        result = {}
        for link_id, link_name in enumerate(["2_4GHz", "5GHz"]):
            collisions = self.last_round_collisions[link_id]
            tx = self.last_round_mld_transmissions[link_id]
            result[f"collision_rate/{link_name}/per_opportunity"] = collisions / opportunities
            result[f"collision_rate/{link_name}/per_txop"] = result[
                f"collision_rate/{link_name}/per_opportunity"
            ]
            result[f"collision_rate/{link_name}/per_tx"] = collisions / max(tx, 1)
            result[f"collision_rate/{link_name}/per_slot"] = collisions / total_slots

        total_collisions = self.last_round_collisions.sum()
        total_tx = self.last_round_mld_transmissions.sum()
        result["collision_rate/system_per_opportunity"] = total_collisions / (opportunities * 2)
        result["collision_rate/system_per_txop"] = result["collision_rate/system_per_opportunity"]
        result["collision_rate/system_per_tx"] = total_collisions / max(total_tx, 1)
        result["collision_rate/system_per_slot"] = total_collisions / total_slots
        return result

    def get_allocation_metrics(self):
        demand = self.last_round_D.astype(np.float32)
        success = self.last_round_S.astype(np.float32)

        total_demand = float(demand.sum())
        total_success = float(success.sum())

        if total_demand > 0.0:
            expected_share = demand / total_demand
        else:
            expected_share = np.zeros(self.num_mld, dtype=np.float32)

        if total_success > 0.0:
            actual_share = success / total_success
        else:
            actual_share = np.zeros(self.num_mld, dtype=np.float32)

        gap = actual_share - expected_share
        abs_gap = np.abs(gap)

        result = {
            "allocation/expected_total_demand": total_demand,
            "allocation/actual_total_success": total_success,
            "allocation/abs_gap_mean": float(np.mean(abs_gap)),
            "allocation/match": float(1.0 - np.mean(abs_gap)),
        }

        for mld_id in range(self.num_mld):
            result[f"allocation/mld_{mld_id}/expected_share"] = float(expected_share[mld_id])
            result[f"allocation/mld_{mld_id}/actual_share"] = float(actual_share[mld_id])
            result[f"allocation/mld_{mld_id}/gap"] = float(gap[mld_id])
            result[f"allocation/mld_{mld_id}/abs_gap"] = float(abs_gap[mld_id])

        return result
