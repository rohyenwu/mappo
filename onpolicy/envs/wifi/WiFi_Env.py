import numpy as np
from gym import spaces

# ── 채널 상수 ──────────────────────────────────────────────────────────────────
DIFS = 2                    # idle 슬롯 2번 이후 AO 허용
CW_TABLE = {1: 4, 2: 8, 3: 16, 4: 32, 5: 64}   # action → CW 크기
LINK_VALS = [2.4, 5.0, 6.0]                       # obs에 넣는 링크 식별값
A_TABLE = np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)  # action → 공격성

CW_MIN = 16
CW_MAX = 1024
RETRY_LIMIT = 6


class WiFiEnv:
    """
    WiFi 다중링크 공존 환경 (ShareDummyVecEnv 호환).

    시스템 구성
    -----------
    SLD   : CSMA/CA (DIFS + BEB) — 링크별 num_sld 개
    MLD-A : 2개 STA, 링크 {2.4GHz, 5GHz}
    MLD-B : 2개 STA, 링크 {2.4GHz, 5GHz, 6GHz}

    에이전트 = 각 (MLD STA, 링크) 쌍  →  총 10개
        agent 0-1  : MLD-A STA0 × {2.4, 5}
        agent 2-3  : MLD-A STA1 × {2.4, 5}
        agent 4-6  : MLD-B STA0 × {2.4, 5, 6}
        agent 7-9  : MLD-B STA1 × {2.4, 5, 6}

    1 step = 1 contention 슬롯 (idle / success / collision)

    Observation (에이전트당)
    -----------------------
    [z_{i,j},  h_{i,j},  link_val]
        z       : squash(W_{i,j} − W̄_j)  — 링크 평균 대비 대기시간 편차
        h       : 직전 결과 (−1=충돌, 0=idle/타인성공, 1=내 성공)
        link_val: 2.4 / 5.0 / 6.0

    Action  Discrete(6)
    -------------------
    0 = 전송 안 함
    1~5 = CW level  (1: 가장 공격적, 5: 가장 보수적)

    Reward
    ------
    A*_{i,j}   = 0.5 × (1 + z × h_prev)        ← 목표 공격성
    r_local    = −(A(action) − A*)^2             ← AO 슬롯에서만
    r_global   = {success:1, idle:0, collision:−1}
    r_total    = r_local + r_global
    """

    def __init__(self, num_mld_a: int = 2, num_mld_b: int = 2,
                 num_sld_per_link: int = 2):
        self.num_mld_a = num_mld_a
        self.num_mld_b = num_mld_b
        self.num_sld   = num_sld_per_link
        self.num_links = 3
        self.total_mld = num_mld_a + num_mld_b

        # ── 에이전트 목록: (sta_id, link_id) ──────────────────────────────
        # MLD-A: sta 0..num_mld_a-1, 링크 [0, 1]
        # MLD-B: sta num_mld_a..total_mld-1, 링크 [0, 1, 2]
        self.agent_sta_link = []
        for sta in range(num_mld_a):
            for link in [0, 1]:
                self.agent_sta_link.append((sta, link))
        for sta in range(num_mld_a, self.total_mld):
            for link in [0, 1, 2]:
                self.agent_sta_link.append((sta, link))
        self.num_agents = len(self.agent_sta_link)   # 4 + 6 = 10

        # 링크 j 위의 에이전트 인덱스 집합 N_j
        self.link_agents = {j: [] for j in range(self.num_links)}
        for aid, (_, link) in enumerate(self.agent_sta_link):
            self.link_agents[link].append(aid)

        # ── Gym 공간 정의 ──────────────────────────────────────────────────
        obs_low  = np.array([-1.0, -1.0,  0.0], dtype=np.float32)
        obs_high = np.array([ 1.0,  1.0, 10.0], dtype=np.float32)

        self.observation_space = [
            spaces.Box(obs_low, obs_high, dtype=np.float32)
        ] * self.num_agents

        self.share_observation_space = [
            spaces.Box(
                low=np.tile(obs_low, self.num_agents),
                high=np.tile(obs_high, self.num_agents),
                dtype=np.float32,
            )
        ] * self.num_agents

        self.action_space = [spaces.Discrete(6)] * self.num_agents

        # 내부 상태는 reset()에서 초기화
        self._init_state()

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────────────────────

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        """
        환경 초기화.
        Returns: obs (num_agents,3), share_obs (num_agents, num_agents*3),
                 available_actions (num_agents,6)
        """
        self._init_state()
        obs, share_obs = self._build_obs()
        available_actions = np.ones((self.num_agents, 6), dtype=np.float32)
        return obs, share_obs, available_actions

    def step(self, actions):
        """
        1 contention 슬롯 진행.

        Parameters
        ----------
        actions : np.ndarray  shape (num_agents, 1)  정수 0~5

        Returns
        -------
        obs, share_obs, rewards, dones, infos, available_actions
        """
        actions = actions.flatten().astype(int)   # (num_agents,)

        # ── step 시작 시점의 z, h 저장 (reward 계산용) ─────────────────────
        z_prev = self._compute_z()
        h_prev = self.h.copy()

        # ── AO 여부 판단 (DIFS 충족된 에이전트) ───────────────────────────
        is_ao = (self.difs >= DIFS)   # (num_agents,) bool

        # ── 링크별 의도 수집 및 판정 ───────────────────────────────────────
        link_results = {}
        for j in range(self.num_links):

            # MLD 전송 의도
            mld_txers = [
                aid for aid in self.link_agents[j]
                if is_ao[aid] and actions[aid] > 0
            ]

            # SLD 전송 의도 (CSMA/CA: DIFS 충족 + backoff==0)
            sld_txers = [
                idx for idx, sld in enumerate(self.sld_state[j])
                if sld['difs'] >= DIFS and sld['backoff'] == 0
            ]

            total_tx = len(mld_txers) + len(sld_txers)
            if total_tx == 0:
                result = "idle"
            elif total_tx == 1:
                result = "success"
            else:
                result = "collision"

            link_results[j] = (result, mld_txers, sld_txers)

        # ── MLD 상태 업데이트 ──────────────────────────────────────────────
        new_h = np.zeros(self.num_agents, dtype=np.float32)
        for aid, (sta, link) in enumerate(self.agent_sta_link):
            result, mld_txers, _ = link_results[link]

            if result in ("success", "collision"):
                # 채널 busy → DIFS 리셋
                self.difs[aid] = 0
                if aid in mld_txers:
                    new_h[aid] = 1.0 if result == "success" else -1.0
                    if result == "success":
                        self.last_success[sta, link] = self.t
                # 타인 성공/충돌: new_h = 0 (기본값)
            else:
                # idle
                if is_ao[aid]:
                    self.difs[aid] = 0   # AO 사용 후 DIFS 리셋
                else:
                    self.difs[aid] = min(self.difs[aid] + 1, DIFS)
                # new_h = 0 (기본값)

        self.h = new_h

        # ── SLD 상태 업데이트 (channel.py / sta.py의 sld_process 로직) ────
        for j in range(self.num_links):
            result, _, sld_txers = link_results[j]
            for idx, sld in enumerate(self.sld_state[j]):
                if result == "idle":
                    sld['difs'] = min(sld['difs'] + 1, DIFS)
                    if sld['difs'] >= DIFS and sld['backoff'] > 0:
                        sld['backoff'] -= 1
                elif result == "success":
                    if idx in sld_txers:
                        sld['cw']      = CW_MIN
                        sld['retry']   = 0
                        sld['backoff'] = int(np.random.randint(0, sld['cw']))
                    sld['difs'] = 0
                else:  # collision
                    if idx in sld_txers:
                        sld['retry'] += 1
                        if sld['retry'] > RETRY_LIMIT:
                            sld['cw']    = CW_MIN
                            sld['retry'] = 0
                        else:
                            sld['cw'] = min(sld['cw'] * 2, CW_MAX)
                        sld['backoff'] = int(np.random.randint(0, sld['cw']))
                    sld['difs'] = 0

        self.t += 1

        # ── Reward 계산 ────────────────────────────────────────────────────
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        for aid, (_, link) in enumerate(self.agent_sta_link):
            result, mld_txers, _ = link_results[link]

            # r_global: 링크 결과 기반
            r_global = {"success": 1.0, "idle": 0.0, "collision": -1.0}[result]

            # r_local: AO 슬롯에서만, z/h_prev 기준
            if is_ao[aid]:
                a_star  = 0.5 * (1.0 + z_prev[aid] * h_prev[aid])
                r_local = -float((A_TABLE[actions[aid]] - a_star) ** 2)
            else:
                r_local = 0.0

            rewards[aid, 0] = r_local + r_global

        # ── 다음 Observation 구성 ──────────────────────────────────────────
        obs, share_obs = self._build_obs()
        available_actions = np.ones((self.num_agents, 6), dtype=np.float32)
        dones   = np.zeros(self.num_agents, dtype=bool)
        infos   = [
            {
                'link_result':   link_results[self.agent_sta_link[aid][1]][0],
                'bad_transition': False,
            }
            for aid in range(self.num_agents)
        ]

        return obs, share_obs, rewards, dones, infos, available_actions

    def close(self):
        pass

    def render(self, mode='human'):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _init_state(self):
        """내부 상태 초기화."""
        self.t = 0

        # last_success[sta_id, link_id]: 마지막 성공 시각
        self.last_success = np.zeros(
            (self.total_mld, self.num_links), dtype=np.float64
        )

        # h[agent_id]: 직전 결과 (−1 / 0 / 1)
        self.h = np.zeros(self.num_agents, dtype=np.float32)

        # difs[agent_id]: 연속 idle 슬롯 카운터
        self.difs = np.zeros(self.num_agents, dtype=np.int32)

        # SLD CSMA/CA 상태: sld_state[link][sld_idx]
        self.sld_state = []
        for _ in range(self.num_links):
            link_slds = []
            for _ in range(self.num_sld):
                cw = CW_MIN
                link_slds.append({
                    'cw':      cw,
                    'backoff': int(np.random.randint(0, cw)),
                    'retry':   0,
                    'difs':    0,
                })
            self.sld_state.append(link_slds)

    def _compute_z(self) -> np.ndarray:
        """
        z_{i,j} = ΔW_{i,j} / (1 + |ΔW_{i,j}|)
        ΔW_{i,j} = W_{i,j} − W̄_j
        W_{i,j}  = t − last_success[i,j]
        """
        z = np.zeros(self.num_agents, dtype=np.float32)
        for j in range(self.num_links):
            agents = self.link_agents[j]
            if not agents:
                continue
            W = np.array(
                [self.t - self.last_success[self.agent_sta_link[a][0], j]
                 for a in agents],
                dtype=np.float64,
            )
            W_mean = W.mean()
            for idx, aid in enumerate(agents):
                dW = W[idx] - W_mean
                z[aid] = float(dW / (1.0 + abs(dW)))
        return z

    def _build_obs(self):
        """
        obs        : (num_agents, 3)          — [z, h, link_val]
        share_obs  : (num_agents, num_agents*3) — 전체 obs를 모든 에이전트에 복사
        """
        z   = self._compute_z()
        obs = np.zeros((self.num_agents, 3), dtype=np.float32)
        for aid, (_, link) in enumerate(self.agent_sta_link):
            obs[aid] = [z[aid], self.h[aid], LINK_VALS[link]]
        share_obs = np.tile(obs.flatten(), (self.num_agents, 1))
        return obs, share_obs
