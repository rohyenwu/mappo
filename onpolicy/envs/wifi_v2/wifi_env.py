"""
WiFi SLD/MLD 공존 환경 v2 — 동기 TXOP 기반, per-TXOP binary action.

설계서: docs/project_wifi_redesign_v4.md
"""
import numpy as np
from gym import spaces

# ── SLD CSMA/CA 파라미터 ──────────────────────────────────────────────────────
SLD_CW_MIN = 16
SLD_CW_MAX = 1024
SLD_RETRY_LIMIT = 6


class WiFiEnvV2:
    """
    동기 TXOP 기반 WiFi 다중링크 공존 환경.

    - 링크: 2개 (2.4GHz, 5GHz)
    - MLD: RL agent, 매 TXOP binary action (transmit/skip)
    - SLD: 2.4GHz에만 존재, CSMA/CA
    - 패킷: 라운드 시작 시 Binomial(T, μ) 일괄 생성
    - Reward: r_global + r_local (dense) + r_sparse (라운드 끝)

    에이전트 인덱싱
    ---------------
    MLD i는 2개 agent slot을 가짐:
      agent 2*i     = MLD i의 2.4GHz
      agent 2*i + 1 = MLD i의 5GHz
    총 num_agents = num_mld * 2
    """

    def __init__(
        self,
        num_mld: int = 3,
        num_sld: int = 3,
        round_length: int = 50,
        mu_range: tuple = (0.2, 0.8),
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
        self.num_mld = num_mld
        self.num_sld = num_sld
        self.round_length = round_length  # T
        self.mu_range = mu_range
        self.sld_mu = sld_mu
        self.f_func = f_func
        self.g_func = g_func
        self.eta = eta
        self.zeta = zeta
        self.r_sld = r_sld
        self.c_idle = c_idle
        self.theta_scale = theta_scale

        # 링크 설정
        self.num_links = 2  # 0=2.4GHz, 1=5GHz
        self.n_sld_per_link = [num_sld, 0]  # SLD는 2.4GHz만

        # 에이전트 설정: MLD i → agent 2*i (2.4), agent 2*i+1 (5)
        self.num_agents = num_mld * 2

        # 에이전트별 (mld_id, link_id) 매핑
        self.agent_to_mld_link = []
        for i in range(num_mld):
            self.agent_to_mld_link.append((i, 0))  # 2.4GHz
            self.agent_to_mld_link.append((i, 1))  # 5GHz

        # 링크별 agent 인덱스
        self.link_agents = {0: [], 1: []}
        for aid, (_, link) in enumerate(self.agent_to_mld_link):
            self.link_agents[link].append(aid)

        # MLD별 도착률 (링크별)
        self.mu = np.zeros((num_mld, 2), dtype=np.float32)

        # ── Gym spaces ────────────────────────────────────────────────────────
        # obs: [μ, N_SLD, fulfillment, link_id(2)] = 5차원
        self.obs_dim = 5
        obs_low = np.zeros(self.obs_dim, dtype=np.float32)
        obs_high = np.ones(self.obs_dim, dtype=np.float32)

        self.observation_space = [
            spaces.Box(obs_low, obs_high, dtype=np.float32)
        ] * self.num_agents

        # share_obs: same-link agent obs concat + current avg_SLD + same-link current satisfaction + link_id
        # = 5*num_mld + 1 + num_mld + 2
        self.share_obs_dim = self.obs_dim * self.num_mld + 1 + self.num_mld + 2
        self.share_observation_space = [
            spaces.Box(
                low=np.zeros(self.share_obs_dim, dtype=np.float32),
                high=np.ones(self.share_obs_dim, dtype=np.float32),
                dtype=np.float32,
            )
        ] * self.num_agents

        # action: binary (0=skip, 1=transmit)
        self.action_space = [spaces.Discrete(2)] * self.num_agents

        # ── 상태 변수 ─────────────────────────────────────────────────────────
        self._init_state()

    def _init_state(self):
        """최초 환경 생성 시 전체 상태 초기화."""
        self.t = 0
        self.D = np.zeros((self.num_mld, 2), dtype=np.int32)
        self.S = np.zeros((self.num_mld, 2), dtype=np.int32)
        self.P = np.zeros((self.num_mld, 2), dtype=np.int32)

        self.sld_state = []
        for _ in range(self.num_sld):
            cw = SLD_CW_MIN
            self.sld_state.append({
                'cw': cw,
                'backoff': int(np.random.randint(0, cw)),
                'retry': 0,
            })

        self.round_sld_success = 0

        self.last_round_S = np.zeros((self.num_mld, 2), dtype=np.int32)
        self.last_round_sld_success = 0

        # 충돌 추적
        self.round_collisions = np.zeros(2, dtype=np.int32)      # 링크별 충돌 TXOP 수
        self.round_mld_transmissions = np.zeros(2, dtype=np.int32)  # 링크별 MLD 전송 시도 수
        self.last_round_collisions = np.zeros(2, dtype=np.int32)
        self.last_round_mld_transmissions = np.zeros(2, dtype=np.int32)

    def _reset_round(self):
        """라운드 전환 시 라운드 내 상태만 리셋. 이전 라운드 통계는 유지."""
        self.t = 0
        self.S[:] = 0
        self.P[:] = 0
        self.D[:] = 0
        self.round_sld_success = 0
        self.round_collisions[:] = 0
        self.round_mld_transmissions[:] = 0

        for sld in self.sld_state:
            sld['cw'] = SLD_CW_MIN
            sld['backoff'] = int(np.random.randint(0, sld['cw']))
            sld['retry'] = 0

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _generate_mu(self):
        """MLD? ??? ???? mu_range? ?? ??."""
        mu_min, mu_max = self.mu_range
        if mu_min > mu_max:
            mu_min, mu_max = mu_max, mu_min
        mu_min = float(np.clip(mu_min, 0.0, 1.0))
        mu_max = float(np.clip(mu_max, 0.0, 1.0))
        fixed_mu = np.linspace(mu_min, mu_max, self.num_mld, dtype=np.float32)
        self.mu[:, 0] = fixed_mu
        self.mu[:, 1] = fixed_mu

    def _generate_packets(self):
        """라운드 시작 시 패킷 일괄 생성."""
        for i in range(self.num_mld):
            for l in range(2):
                self.D[i, l] = np.random.binomial(self.round_length, self.mu[i, l])

    def _get_fulfillment(self, mld_id, link_id):
        """현재 충족률 S/D. D=0이면 1.0 (수요 없음 = 완전 충족)."""
        d = self.D[mld_id, link_id]
        if d == 0:
            return 1.0
        return self.S[mld_id, link_id] / d

    def _get_link_min_fulfillment_agents(self, link_id):
        """해당 링크에서 fulfillment가 가장 낮은 agent 인덱스들 반환 (동률 포함)."""
        min_f = float('inf')
        min_aids = []
        for aid in self.link_agents[link_id]:
            mld_id, _ = self.agent_to_mld_link[aid]
            f = self._get_fulfillment(mld_id, link_id)
            if f < min_f - 1e-9:
                min_f = f
                min_aids = [aid]
            elif abs(f - min_f) < 1e-9:
                min_aids.append(aid)
        return set(min_aids)

    def _get_link_urgencies(self, link_id):
        """Return each agent's relative urgency on the link."""
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
        """Return a single top-urgency agent on the link."""
        if urgencies is None:
            urgencies = self._get_link_urgencies(link_id)

        if not urgencies:
            return None

        max_urgency = max(urgencies.values())
        top_candidates = [
            aid for aid, urgency in urgencies.items()
            if abs(urgency - max_urgency) < 1e-9
        ]
        return min(top_candidates)

    def _build_obs(self):
        """모든 agent의 observation 생성."""
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        for aid in range(self.num_agents):
            mld_id, link_id = self.agent_to_mld_link[aid]
            mu_norm = self.mu[mld_id, link_id]
            n_sld_norm = self.n_sld_per_link[link_id] / max(self.num_sld, 1)
            fulfillment = self._get_fulfillment(mld_id, link_id)
            link_onehot = [1.0, 0.0] if link_id == 0 else [0.0, 1.0]
            obs[aid] = [mu_norm, n_sld_norm, fulfillment, *link_onehot]
        return obs

    def _build_share_obs(self, obs):
        """Critic? ??? state ??."""
        share_obs = np.zeros((self.num_agents, self.share_obs_dim), dtype=np.float32)
        curr_avg_sld = self.round_sld_success / max(self.round_length, 1)
        for aid in range(self.num_agents):
            _, link_id = self.agent_to_mld_link[aid]
            link_onehot = [1.0, 0.0] if link_id == 0 else [0.0, 1.0]
            link_aids = self.link_agents[link_id]
            link_obs_flat = obs[link_aids].flatten()
            link_curr_satisfaction = obs[link_aids, 2]
            link_avg_sld = curr_avg_sld if link_id == 0 else 0.0
            share_obs[aid] = np.concatenate([
                link_obs_flat,
                [link_avg_sld],
                link_curr_satisfaction,
                link_onehot,
            ])
        return share_obs

    def _compute_f(self, success_agent_id):
        """최저충족 아닌 agent가 성공했을 때 r_global 값. 추후 공식 확정."""
        mld_id, link_id = self.agent_to_mld_link[success_agent_id]
        # 기본: 성공한 agent의 잔여 수요 비율
        f = self._get_fulfillment(mld_id, link_id)
        return max(0.0, 1.0 - f)  # 충족률 낮을수록 큰 보상

    def _compute_g(self, agent_id):
        """최저충족 agent가 대기했을 때 r_local 페널티. 추후 공식 확정."""
        mld_id, link_id = self.agent_to_mld_link[agent_id]
        f = self._get_fulfillment(mld_id, link_id)
        return max(0.0, 1.0 - f)  # 충족률 낮을수록 큰 페널티

    def _step_sld(self, link_id):
        """SLD CSMA/CA 처리. 전송할 SLD 인덱스 리스트 반환."""
        if link_id != 0:
            return []

        for sld in self.sld_state:
            if sld['backoff'] > 0:
                sld['backoff'] -= 1

        transmitting = []
        for idx, sld in enumerate(self.sld_state):
            if sld['backoff'] == 0:
                transmitting.append(idx)

        return transmitting

    def _update_sld_after_txop(self, link_id, result, sld_txers):
        """TXOP 결과에 따른 SLD 상태 업데이트."""
        if link_id != 0:
            return

        for idx, sld in enumerate(self.sld_state):
            if idx in sld_txers:
                if result == "success":
                    sld['cw'] = SLD_CW_MIN
                    sld['retry'] = 0
                    sld['backoff'] = int(np.random.randint(0, sld['cw']))
                    self.round_sld_success += 1
                elif result == "collision":
                    sld['retry'] += 1
                    if sld['retry'] > SLD_RETRY_LIMIT:
                        sld['cw'] = SLD_CW_MIN
                        sld['retry'] = 0
                    else:
                        sld['cw'] = min(sld['cw'] * 2, SLD_CW_MAX)
                    sld['backoff'] = int(np.random.randint(0, sld['cw']))
            else:
                # TXOP 시작 시 countdown을 진행하므로 idle 결과만으로는 backoff를 줄이지 않음.
                pass

    def reset(self):
        """
        환경 초기화. 새 라운드 시작.
        ShareDummyVecEnv가 done=True 시 자동 호출하므로,
        라운드 전환 로직은 여기서만 수행.

        Returns: obs, share_obs, available_actions
        """
        # 이전 라운드 통계 저장 (첫 reset이 아닌 경우)
        if np.any(self.S > 0) or self.round_sld_success > 0:
            self.last_round_S = self.S.copy()
            self.last_round_sld_success = self.round_sld_success
        self._generate_mu()
        self._reset_round()
        self._generate_packets()

        obs = self._build_obs()
        share_obs = self._build_share_obs(obs)
        available_actions = np.ones((self.num_agents, 2), dtype=np.float32)

        return obs, share_obs, available_actions

    def step(self, actions):
        """
        1 TXOP 진행.

        Parameters
        ----------
        actions : np.ndarray shape (num_agents, 1), 값 0 or 1

        Returns
        -------
        obs, share_obs, rewards, dones, infos, available_actions
        """
        actions_flat = actions.flatten().astype(int)

        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        reward_global = np.zeros(self.num_agents, dtype=np.float32)
        reward_local = np.zeros(self.num_agents, dtype=np.float32)
        reward_sparse = np.zeros(self.num_agents, dtype=np.float32)
        reward_dense = np.zeros(self.num_agents, dtype=np.float32)
        link_result = np.full(self.num_agents, "", dtype=object)

        # ── 링크별 TXOP 처리 ──────────────────────────────────────────────────
        for link_id in range(self.num_links):
            link_aids = self.link_agents[link_id]

            # MLD 전송자
            mld_txers = [aid for aid in link_aids if actions_flat[aid] == 1]

            # SLD 전송자
            sld_txers = self._step_sld(link_id)

            # 총 전송자 수
            total_tx = len(mld_txers) + len(sld_txers)

            # 결과 판정
            if total_tx == 0:
                result = "idle"
            elif total_tx == 1:
                result = "success"
            else:
                result = "collision"

            # 성공한 agent 식별
            success_aid = None
            success_is_sld = False
            if result == "success":
                if len(mld_txers) == 1 and len(sld_txers) == 0:
                    success_aid = mld_txers[0]
                elif len(sld_txers) == 1 and len(mld_txers) == 0:
                    success_is_sld = True

            # ── reward 계산 (S 증가 전에 수행) ────────────────────────────────
            # 최저충족 판정과 f(·)/g(·) 계산은 현재 fulfillment 기준이어야 함
            urgencies = self._get_link_urgencies(link_id)
            top_aid = self._get_link_top_urgency_agent(link_id, urgencies)

            # r_global
            r_global = 0.0
            if result == "success" and not success_is_sld:
                if success_aid == top_aid:
                    r_global = 1.0
                else:
                    r_global = urgencies.get(success_aid, 0.0)
            elif result == "success" and success_is_sld:
                r_global = 0.0
            elif result == "collision":
                r_global = -1.0
            elif result == "idle":
                r_global = -self.c_idle

            # r_local (agent별)
            for aid in link_aids:
                mld_id, _ = self.agent_to_mld_link[aid]
                transmitted = (actions_flat[aid] == 1)
                urgency = urgencies.get(aid, 0.0)

                # D=0 (수요 없음): 전송하면 페널티, skip하면 중립
                if self.D[mld_id, link_id] == 0:
                    r_local = -1.0 if transmitted else 0.0
                elif aid == top_aid:
                    if transmitted:
                        r_local = 1.0
                    else:
                        r_local = -(1.0 + urgency)
                else:
                    if transmitted:
                        r_local = -2.0
                    else:
                        r_local = 1.0

                rewards[aid, 0] = r_global + r_local
                reward_global[aid] = r_global
                reward_local[aid] = r_local
                reward_dense[aid] = r_global + r_local
                link_result[aid] = result

            # ── S, P, 충돌 카운터 업데이트 (reward 계산 후) ─────────────────
            if success_aid is not None:
                mld_id, _ = self.agent_to_mld_link[success_aid]
                if self.D[mld_id, link_id] > self.S[mld_id, link_id]:
                    self.S[mld_id, link_id] += 1

            for aid in mld_txers:
                mld_id, _ = self.agent_to_mld_link[aid]
                self.P[mld_id, link_id] += 1

            self.round_mld_transmissions[link_id] += len(mld_txers)
            if result == "collision":
                self.round_collisions[link_id] += 1

            # ── SLD 상태 업데이트 ─────────────────────────────────────────────
            self._update_sld_after_txop(link_id, result, sld_txers)

        # ── TXOP 진행 ─────────────────────────────────────────────────────────
        self.t += 1
        done = (self.t >= self.round_length)

        # ── Sparse reward (라운드 끝, 2.4GHz만) ──────────────────────────────
        if done:
            reward_sparse = self._apply_sparse_reward_with_trace(rewards)

        # ── 결과 구성 ─────────────────────────────────────────────────────────
        obs = self._build_obs()
        share_obs = self._build_share_obs(obs)
        dones = np.full(self.num_agents, done, dtype=bool)
        available_actions = np.ones((self.num_agents, 2), dtype=np.float32)

        infos = []
        for aid in range(self.num_agents):
            mld_id, link_id = self.agent_to_mld_link[aid]
            infos.append({
                'fulfillment': self._get_fulfillment(mld_id, link_id),
                'round_done': done,
                'reward/global': float(reward_global[aid]),
                'reward/local': float(reward_local[aid]),
                'reward/dense': float(reward_dense[aid]),
                'reward/sparse': float(reward_sparse[aid]),
                'reward/total': float(rewards[aid, 0]),
                f'reward/link_{link_id}/global': float(reward_global[aid]),
                f'reward/link_{link_id}/local': float(reward_local[aid]),
                f'reward/link_{link_id}/dense': float(reward_dense[aid]),
                f'reward/link_{link_id}/sparse': float(reward_sparse[aid]),
                f'reward/link_{link_id}/total': float(rewards[aid, 0]),
                'txop_result': link_result[aid],
            })

        # 라운드 끝이면 throughput/collision 캐시 저장
        # (reset()에서 _reset_round() 전에 이전 통계를 저장함)
        if done:
            self.last_round_S = self.S.copy()
            self.last_round_sld_success = self.round_sld_success
            self.last_round_collisions = self.round_collisions.copy()
            self.last_round_mld_transmissions = self.round_mld_transmissions.copy()

        return obs, share_obs, rewards, dones, infos, available_actions

    def _apply_sparse_reward(self, rewards):
        """라운드 끝에서 SLD 최저보장 sparse reward 적용 (2.4GHz agent만)."""
        # SLD 평균 throughput (이번 라운드)
        avg_sld = self.round_sld_success / max(self.round_length, 1)

        # SLD 최저기준: SLD가 공정하게 받아야 할 비율
        # 간단한 기준: SLD 수 / (SLD 수 + 2.4GHz MLD 수)
        n_mld_24 = len(self.link_agents[0])
        base_theta = self.num_sld / max(self.num_sld + n_mld_24, 1)
        theta = self.theta_scale * base_theta

        # 2.4GHz agent들의 참여/skip 통계
        link_0_aids = self.link_agents[0]
        participations = []
        skips = []
        for aid in link_0_aids:
            mld_id, _ = self.agent_to_mld_link[aid]
            p_i = self.P[mld_id, 0]
            participations.append(p_i)
            skips.append(self.round_length - p_i)

        p_avg = np.mean(participations) if participations else 0
        skip_avg = np.mean(skips) if skips else 0

        for idx, aid in enumerate(link_0_aids):
            if avg_sld < theta:
                # 미달: 과점유 MLD 페널티
                penalty = self.eta * max(0, participations[idx] - p_avg)
                rewards[aid, 0] -= penalty
            else:
                # 달성: 양보 MLD 보상
                bonus = self.zeta * max(0, skips[idx] - skip_avg)
                rewards[aid, 0] += bonus

    def _apply_sparse_reward_with_trace(self, rewards):
        """Apply sparse reward and return the per-agent sparse component."""
        sparse_rewards = np.zeros(self.num_agents, dtype=np.float32)

        avg_sld = self.round_sld_success / max(self.round_length, 1)
        n_mld_24 = len(self.link_agents[0])
        base_theta = self.num_sld / max(self.num_sld + n_mld_24, 1)
        theta = self.theta_scale * base_theta

        link_0_aids = self.link_agents[0]
        participations = []
        skips = []
        for aid in link_0_aids:
            mld_id, _ = self.agent_to_mld_link[aid]
            p_i = self.P[mld_id, 0]
            participations.append(p_i)
            skips.append(self.round_length - p_i)

        p_avg = np.mean(participations) if participations else 0
        skip_avg = np.mean(skips) if skips else 0

        for idx, aid in enumerate(link_0_aids):
            if avg_sld < theta:
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

    def render(self, mode='human'):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # 통계 인터페이스
    # ──────────────────────────────────────────────────────────────────────────

    def get_throughput(self):
        """이전 라운드의 throughput 통계 반환 (라운드 끝 후 리셋되므로 캐시 사용)."""
        S = self.last_round_S
        sld_success = self.last_round_sld_success
        T = max(self.round_length, 1)

        result = {}
        for link_id, link_name in enumerate(['2_4GHz', '5GHz']):
            mld_success = sum(S[mld_id, link_id] for mld_id in range(self.num_mld))
            sld_s = sld_success if link_id == 0 else 0
            result[f'throughput/{link_name}/mld'] = mld_success / T
            result[f'throughput/{link_name}/sld'] = sld_s / T
            result[f'throughput/{link_name}/total'] = (mld_success + sld_s) / T

        result['throughput/mld_total'] = sum(
            S[mld_id, link_id]
            for mld_id in range(self.num_mld)
            for link_id in range(2)
        ) / T
        result['throughput/sld_total'] = sld_success / T
        result['throughput/system'] = result['throughput/mld_total'] + result['throughput/sld_total']
        return result

    def get_collision_rate(self):
        """이전 라운드의 충돌률 통계 반환."""
        T = max(self.round_length, 1)
        result = {}
        for link_id, link_name in enumerate(['2_4GHz', '5GHz']):
            col = self.last_round_collisions[link_id]
            result[f'collision_rate/{link_name}/per_txop'] = col / T
            tx = self.last_round_mld_transmissions[link_id]
            result[f'collision_rate/{link_name}/per_tx'] = col / max(tx, 1)

        total_col = self.last_round_collisions.sum()
        total_tx = self.last_round_mld_transmissions.sum()
        result['collision_rate/system_per_txop'] = total_col / (T * 2)
        result['collision_rate/system_per_tx'] = total_col / max(total_tx, 1)
        return result
