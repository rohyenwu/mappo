import numpy as np
from gym import spaces

# ── 채널 상수 ──────────────────────────────────────────────────────────────────
DIFS = 2
CW_TABLE = {0: (0, 4), 1: (4, 8), 2: (8, 16), 3: (16, 32), 4: (32, 64)}
LINK_VALS = [2.4, 5.0, 6.0]
A_TABLE = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)  # index = action (0~4)

CW_MIN = 16
CW_MAX = 1024
RETRY_LIMIT = 6

# ── 학습 하이퍼파라미터 ────────────────────────────────────────────────────────
LAMBDA = 0.2   # h EMA 감쇠율
BETA   = 1.0   # r_ind 스케일
ALPHA  = 1.0   # r_global 스케일

# ── throughput 파라미터 ────────────────────────────────────────────────────────
PKT_PER_SUCCESS = [1, 2, 3]  # 링크별 성공당 패킷 수 (2.4GHz, 5GHz, 6GHz)


class WiFiEnv:
    """
    WiFi 다중링크 공존 환경 — 결과 기반 이벤트 샘플링 (ShareDummyVecEnv 호환).

    시스템 구성
    -----------
    SLD   : CSMA/CA (DIFS + BEB) — 링크별 num_sld 개, 학습 안 함
    MLD-A : num_mld_a 개 STA, 링크 {2.4GHz, 5GHz}
    MLD-B : num_mld_b 개 STA, 링크 {2.4GHz, 5GHz, 6GHz}

    에이전트 = 각 (MLD STA, 링크) 쌍  →  총 10개 (기본값)

    Decision point
    --------------
    success/collision 결과 발생 시점.
    결과를 겪은 에이전트(was_tx=True)만 새 CW 결정 필요.
    결과 직후 z, h를 저장 → 다음 결과 발생 시 reward 계산에 사용.

    Sample
    ------
    s  = 결과 발생 직후 상태 (새 action 선택 직전)
    a  = 선택한 CW 레벨 (1~5)
    r  = 다음 결과 발생 시 계산 (ao_z, ao_h, ao_action 사용)
    s' = 다음 결과 발생 직후 상태

    Action  Discrete(5)
    -------------------
    0~4 = CW level  (0: 가장 공격적 A=1.0, 4: 가장 보수적 A=0.2)

    Reward
    ------
    A*(h)    = (h + 1) / 2
    r_ind    = −β × |A(action) − A*(h_prev)|
    r_global = {success:+1, collision:−1, idle:0}
    r_total  = r_ind + α × r_global
    """

    def __init__(self, num_mld_a: int = 2, num_mld_b: int = 2,
                 num_sld_per_link: int = 2):
        self.num_mld_a = num_mld_a
        self.num_mld_b = num_mld_b
        self.num_sld   = num_sld_per_link
        self.num_links = 3
        self.total_mld = num_mld_a + num_mld_b

        # ── 에이전트 목록: (sta_id, link_id) ──────────────────────────────────
        self.agent_sta_link = []
        for sta in range(num_mld_a):
            for link in [0, 1]:
                self.agent_sta_link.append((sta, link))
        for sta in range(num_mld_a, self.total_mld):
            for link in [0, 1, 2]:
                self.agent_sta_link.append((sta, link))
        self.num_agents = len(self.agent_sta_link)

        self.link_agents = {j: [] for j in range(self.num_links)}
        for aid, (_, link) in enumerate(self.agent_sta_link):
            self.link_agents[link].append(aid)

        # ── Gym 공간 정의 ──────────────────────────────────────────────────────
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

        self.action_space = [spaces.Discrete(5)] * self.num_agents  # 0~4

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
        첫 번째 결과(success/collision) 발생까지 내부 슬롯 진행.
        결과를 겪은 에이전트가 첫 decision을 내릴 준비가 된 상태로 반환.

        초기 backoff는 CW_TABLE[3] (중간값)으로 설정.

        Returns
        -------
        obs              : (num_agents, 3)
        share_obs        : (num_agents, num_agents*3)
        available_actions: (num_agents, 5)
        """
        self._init_state()

        # 모든 에이전트 초기 backoff 설정 (action=2, 중간 CW)
        for aid in range(self.num_agents):
            cw_min, cw_max = CW_TABLE[2]
            self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))

        # 첫 결과 발생까지 슬롯 진행
        dummy_rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        while not np.any(self.need_decision):
            self._advance_one_slot(dummy_rewards)
            self.t += 1

        obs, share_obs = self._build_obs()
        return obs, share_obs, self._make_available_actions()

    def step(self, actions):
        """
        결과 기반 이벤트 1 step.

        Parameters
        ----------
        actions : np.ndarray  shape (num_agents, 1)  정수 1~5
            need_decision=True인 에이전트의 action만 반영.

        Returns
        -------
        obs, share_obs, rewards, dones, infos, available_actions
        """
        actions = actions.flatten().astype(int)
        actions = np.clip(actions, 0, 4)

        # ── Phase 1: decision 에이전트 action 적용 ────────────────────────────
        decided = self.need_decision.copy()

        for aid in range(self.num_agents):
            if not self.need_decision[aid]:
                continue
            act = actions[aid]
            self.ao_action[aid] = act
            cw_min, cw_max = CW_TABLE[act]
            self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))

        self.need_decision[:] = False

        # ── Phase 2: 다음 결과 발생까지 내부 슬롯 진행 ────────────────────────
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        while not np.any(self.need_decision):
            self._advance_one_slot(rewards)
            self.t += 1

        obs, share_obs = self._build_obs()
        dones = np.zeros(self.num_agents, dtype=bool)
        infos = [
            {
                'bad_transition': False,
                'decided': bool(decided[aid]),  # 이번 step에서 decision했는지
            }
            for aid in range(self.num_agents)
        ]

        return obs, share_obs, rewards, dones, infos, self._make_available_actions()

    def close(self):
        pass

    def render(self, mode='human'):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _init_state(self):
        self.t = 0

        self.last_success = np.zeros(
            (self.total_mld, self.num_links), dtype=np.float64
        )
        self.h = np.zeros(self.num_agents, dtype=np.float32)
        self.difs = np.zeros(self.num_agents, dtype=np.int32)

        # mld_backoff:  -1=DIFS 카운팅, 0=전송준비, >0=카운트다운
        self.mld_backoff = np.full(self.num_agents, -1, dtype=np.int32)

        # need_decision: 결과를 겪어 새 CW 결정이 필요한 에이전트
        self.need_decision = np.zeros(self.num_agents, dtype=bool)

        # 새 action 선택 직전 저장값 (delayed reward 계산용)
        self.ao_h      = np.zeros(self.num_agents, dtype=np.float32)
        self.ao_action = np.ones(self.num_agents, dtype=np.int32) * 2  # 초기 기본값

        # throughput 카운터 (링크별)
        self.mld_success_count = np.zeros(self.num_links, dtype=np.int64)
        self.sld_success_count = np.zeros(self.num_links, dtype=np.int64)

        # SLD 상태
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

    def _advance_one_slot(self, rewards: np.ndarray):
        """
        슬롯 1개 진행.

        was_tx=True 에이전트에 대해:
          1. reward 계산 (ao_z, ao_h, ao_action 사용)
          2. 상태 업데이트 (h, last_success)
          3. z 재계산
          4. ao_z, ao_h 저장 (새 action 선택 직전 상태)
          5. need_decision = True
        """
        # ── 링크별 전송 결과 ──────────────────────────────────────────────────
        link_results = {}
        for j in range(self.num_links):
            mld_txers = [
                aid for aid in self.link_agents[j]
                if self.difs[aid] >= DIFS and self.mld_backoff[aid] == 0
            ]
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

        # ── MLD 상태 업데이트 ─────────────────────────────────────────────────
        new_h = self.h.copy()
        was_tx_agents = []

        for aid, (sta, link) in enumerate(self.agent_sta_link):
            result, mld_txers, _ = link_results[link]
            was_tx = (self.mld_backoff[aid] == 0 and self.difs[aid] >= DIFS)

            if result in ("success", "collision"):
                self.difs[aid] = 0
                if was_tx:
                    x = 1.0 if result == "success" else -1.0
                    new_h[aid] = (1.0 - LAMBDA) * self.h[aid] + LAMBDA * x
                    if result == "success":
                        self.last_success[sta, link] = self.t
                        self.mld_success_count[link] += 1
                    self.mld_backoff[aid] = -1

                    # ── reward 계산 (이전 ao_h, ao_action 기준) ───────────────
                    a_star   = (self.ao_h[aid] + 1.0) / 2.0
                    r_ind    = float(-BETA * abs(A_TABLE[self.ao_action[aid]] - a_star))
                    r_global = 1.0 if result == "success" else -1.0
                    rewards[aid, 0] = r_ind + ALPHA * r_global

                    was_tx_agents.append(aid)
                # was_tx=False: backoff freeze
            else:
                # idle
                if self.difs[aid] < DIFS:
                    self.difs[aid] += 1
                elif self.mld_backoff[aid] > 0:
                    self.mld_backoff[aid] -= 1

        self.h = new_h

        # ── was_tx 에이전트: z 재계산 후 ao_z, ao_h 저장 ─────────────────────
        if was_tx_agents:
            for aid in was_tx_agents:
                self.ao_h[aid] = self.h[aid]   # 새 action 선택 직전 h (방금 결과)
                self.need_decision[aid] = True

        # ── SLD 상태 업데이트 ─────────────────────────────────────────────────
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
                        self.sld_success_count[j] += 1
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

    def get_throughput(self) -> dict:
        """
        링크별 MLD/SLD 성공 카운트로 throughput 지표 반환.

        throughput = success_count × PKT_PER_SUCCESS / total_slots

        Returns
        -------
        dict with keys:
            system, mld_total, sld_total,
            link0~2 (total/mld/sld)
        """
        t = max(self.t, 1)
        pkt = PKT_PER_SUCCESS
        link_names = ['2.4GHz', '5GHz', '6GHz']

        result = {}

        # per-link
        for j in range(self.num_links):
            name = link_names[j]
            mld = self.mld_success_count[j] * pkt[j] / t
            sld = self.sld_success_count[j] * pkt[j] / t
            result[f'throughput/{name}/total'] = mld + sld
            result[f'throughput/{name}/mld']   = mld
            result[f'throughput/{name}/sld']   = sld

        # aggregate
        result['throughput/mld_total'] = sum(
            self.mld_success_count[j] * pkt[j] for j in range(self.num_links)
        ) / t
        result['throughput/sld_total'] = sum(
            self.sld_success_count[j] * pkt[j] for j in range(self.num_links)
        ) / t
        result['throughput/system'] = result['throughput/mld_total'] + result['throughput/sld_total']

        return result

    def _make_available_actions(self) -> np.ndarray:
        """
        need_decision=True : 모든 action(0~4, 즉 CW 1~5) 가능
        need_decision=False: action 0만 (softmax NaN 방지용, 실제로는 무시됨)
        """
        avail = np.zeros((self.num_agents, 5), dtype=np.float32)
        for aid in range(self.num_agents):
            if self.need_decision[aid]:
                avail[aid] = 1.0
            else:
                avail[aid, 0] = 1.0
        return avail

    def _compute_z(self) -> np.ndarray:
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
        z   = self._compute_z()
        obs = np.zeros((self.num_agents, 3), dtype=np.float32)
        for aid, (_, link) in enumerate(self.agent_sta_link):
            obs[aid] = [z[aid], self.h[aid], LINK_VALS[link]]
        share_obs = np.tile(obs.flatten(), (self.num_agents, 1))
        return obs, share_obs
