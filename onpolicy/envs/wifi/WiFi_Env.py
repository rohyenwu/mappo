import numpy as np
from gym import spaces

# ── 채널 상수 ──────────────────────────────────────────────────────────────────
DIFS = 2
CW_TABLE = {
    0: (2,  6),   # 2~5
    1: (6,  12),  # 6~11
    2: (12, 24),  # 12~23
    3: (24, 48),  # 24~47
    4: (48, 96),  # 48~95
    5: (96, 128), # 96~127
}
LINK_VALS = [2.4, 5.0, 6.0]
A_TABLE = np.array([0.85, 0.7, 0.5, 0.3, 0.15, 0.05], dtype=np.float32)  # index = action (0~5)

CW_MIN = 16
CW_MAX = 1024
RETRY_LIMIT = 6
W_MAX = 1000   # W clip 상한 (한 번도 성공 못 한 경우 포함)

# ── 학습 하이퍼파라미터 ────────────────────────────────────────────────────────
LAMBDA = 0.2   # h EMA 감쇠율
BETA   = 0.1   # r_ind 스케일
ALPHA  = 1.0   # r_global 스케일

# ── 목표 공격성 sigmoid 스케일 ────────────────────────────────────────────────
W_SCALE = 0.005  # W cap 200 → 기여 0~1.0
H_SCALE = 1.0    # h 범위 -1~1 → 기여 -1~1
R_SCALE = 0.8    # retry cap 6 → 기여 0~4.8

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
    결과 직후 W, h를 저장 → 다음 결과 발생 시 reward 계산에 사용.

    Sample
    ------
    s  = 결과 발생 직후 상태 (새 action 선택 직전)
    a  = 선택한 CW 레벨 (0~4)
    r  = 다음 결과 발생 시 계산 (ao_h, ao_action 사용)
    s' = 다음 결과 발생 직후 상태

    State  (6 dim)
    --------------
    [W, h, retry, one_hot_link(3)]
    W         : 마지막 성공 이후 경과 슬롯 수 (raw, feature_norm으로 처리)
    h         : 채널 품질 EMA [-1, 1]
    retry     : 연속 충돌 횟수 (성공 시 0 리셋)
    one_hot   : 링크 ID one-hot {2.4GHz, 5GHz, 6GHz}

    Action  Discrete(6)
    -------------------
    0~5 = CW level  (0: 가장 공격적 A=0.85, 5: 가장 보수적 A=0.05)

    Reward  (use_ind_reward=True 일 때)
    ------------------------------------
    A*(W,h,retry) = sigmoid(w_scale*W + h_scale*h - r_scale*retry)
    e = (A(action) − A*)²
    success : r = +(1 − e)    →  [0, +1]
    collision: r = −e          →  [−1, 0]

    Reward  (use_ind_reward=False 일 때)
    -------------------------------------
    success : r = +1
    collision: r = −1
    """

    def __init__(self, num_mld_a: int = 2, num_mld_b: int = 2,
                 num_sld_per_link: int = 2,
                 use_ind_reward: bool = True,
                 use_w_in_astar: bool = True):
        self.num_mld_a     = num_mld_a
        self.num_mld_b     = num_mld_b
        self.num_sld       = num_sld_per_link
        self.use_ind_reward = use_ind_reward
        self.use_w_in_astar = use_w_in_astar
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

        # 링크별 최대 에이전트 수 (share_obs 차원 결정)
        self.max_link_agents = max(len(aids) for aids in self.link_agents.values())

        # ── Gym 공간 정의 ──────────────────────────────────────────────────────
        # obs: [W, h, retry, one_hot_link(3)] = 6 dim
        obs_low  = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = [
            spaces.Box(obs_low, obs_high, dtype=np.float32)
        ] * self.num_agents

        # share_obs: 같은 링크 에이전트만 (max_link_agents × 6, 패딩 포함)
        self.share_observation_space = [
            spaces.Box(
                low=np.tile(obs_low, self.max_link_agents),
                high=np.tile(obs_high, self.max_link_agents),
                dtype=np.float32,
            )
        ] * self.num_agents

        self.action_space = [spaces.Discrete(6)] * self.num_agents  # 0~5

        self._init_state()

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────────────────────

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def reset(self, warmup_decisions: int = None):
        """
        환경 초기화.
        첫 번째 결과(success/collision) 발생까지 내부 슬롯 진행.
        이후 warmup_decisions 번 추가로 result를 발생시켜 W를 안정화.

        초기 backoff는 CW_TABLE[2] (중간값)으로 설정.

        Parameters
        ----------
        warmup_decisions : int
            초기 W 안정화를 위해 추가로 진행할 decision 횟수.
            기본값은 num_agents.

        Returns
        -------
        obs              : (num_agents, 6)
        share_obs        : (num_agents, max_link_agents*6)
        available_actions: (num_agents, 6)
        """
        if warmup_decisions is None:
            warmup_decisions = self.num_agents

        self._init_state()

        # 모든 에이전트 초기 backoff 설정 (action=2, 중간 CW)
        for aid in range(self.num_agents):
            cw_min, cw_max = CW_TABLE[2]
            self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))

        dummy_rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        # 첫 결과 발생까지 슬롯 진행
        while not np.any(self.need_decision):
            self._advance_one_slot(dummy_rewards)
            self.t += 1

        # warmup: N번 추가 decision cycle 진행 (결과 버림)
        for _ in range(warmup_decisions):
            # need_decision인 에이전트만 중간 CW로 action 적용
            for aid in range(self.num_agents):
                if self.need_decision[aid]:
                    cw_min, cw_max = CW_TABLE[2]
                    self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))
            self.need_decision[:] = False
            while not np.any(self.need_decision):
                self._advance_one_slot(dummy_rewards)
                self.t += 1

        # throughput / 충돌 카운터 및 기준 시점 초기화 (warmup 슬롯 제외)
        self.mld_success_count[:]   = 0
        self.sld_success_count[:]   = 0
        self.mld_collision_count[:] = 0
        self.sld_collision_count[:] = 0
        self.t_train_start = self.t

        obs, share_obs = self._build_obs()
        return obs, share_obs, self._make_available_actions()

    def step(self, actions):
        """
        결과 기반 이벤트 1 step.

        Parameters
        ----------
        actions : np.ndarray  shape (num_agents, 1)  정수 0~5
            need_decision=True인 에이전트의 action만 반영.

        Returns
        -------
        obs, share_obs, rewards, dones, infos, available_actions
        """
        actions = actions.flatten().astype(int)
        actions = np.clip(actions, 0, 5)

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
        self._last_e = np.zeros(self.num_agents, dtype=np.float32)

        while not np.any(self.need_decision):
            self._advance_one_slot(rewards)
            self.t += 1

        obs, share_obs = self._build_obs()
        dones = np.zeros(self.num_agents, dtype=bool)
        infos = [
            {
                'bad_transition': False,
                'decided': bool(decided[aid]),
                'e': float(self._last_e[aid]),
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
        self.t_train_start = 0

        self.last_success = np.full(
            (self.total_mld, self.num_links), -W_MAX, dtype=np.float64
        )
        self.h = np.zeros(self.num_agents, dtype=np.float32)
        self.difs = np.zeros(self.num_agents, dtype=np.int32)

        # mld_backoff:  -1=DIFS 카운팅, 0=전송준비, >0=카운트다운
        self.mld_backoff = np.full(self.num_agents, -1, dtype=np.int32)

        # need_decision: 결과를 겪어 새 CW 결정이 필요한 에이전트
        self.need_decision = np.zeros(self.num_agents, dtype=bool)

        # 새 action 선택 직전 저장값 (delayed reward 계산용)
        self.ao_h      = np.zeros(self.num_agents, dtype=np.float32)
        self.ao_w      = np.zeros(self.num_agents, dtype=np.float32)
        self.ao_retry  = np.zeros(self.num_agents, dtype=np.int32)
        self.ao_action = np.ones(self.num_agents, dtype=np.int32) * 2  # 초기 기본값

        # 연속 충돌 횟수 (성공 시 0 리셋)
        self.retry = np.zeros(self.num_agents, dtype=np.int32)

        # throughput 카운터 (링크별)
        self.mld_success_count   = np.zeros(self.num_links, dtype=np.int64)
        self.sld_success_count   = np.zeros(self.num_links, dtype=np.int64)

        # 충돌 카운터 (링크별)
        self.mld_collision_count = np.zeros(self.num_links, dtype=np.int64)
        self.sld_collision_count = np.zeros(self.num_links, dtype=np.int64)

        # SLD 상태 (2.4GHz 링크에만 존재)
        self.sld_state = []
        for j in range(self.num_links):
            link_slds = []
            if j == 0:  # 2.4GHz only
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
          1. reward 계산 (ao_h, ao_action 사용)
          2. 상태 업데이트 (h, last_success)
          3. ao_h 저장 (새 action 선택 직전 상태)
          4. need_decision = True
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
                        self.retry[aid] = 0
                    else:
                        self.mld_collision_count[link] += 1
                        self.retry[aid] += 1
                    self.mld_backoff[aid] = -1

                    # ── reward 계산 (이전 ao 상태 기준) ───────────────────────
                    if self.use_ind_reward:
                        retry_clip = min(self.ao_retry[aid], 6)
                        sig_input = (H_SCALE * self.ao_h[aid]
                                     - R_SCALE * retry_clip)
                        if self.use_w_in_astar:
                            w_clip = min(self.ao_w[aid], 200.0)
                            sig_input += W_SCALE * w_clip
                        a_star = 1.0 / (1.0 + np.exp(-sig_input))
                        e = (float(A_TABLE[self.ao_action[aid]]) - a_star) ** 2
                        self._last_e[aid] = e
                        if result == "success":
                            rewards[aid, 0] = 1.0 - e
                        else:
                            rewards[aid, 0] = -e
                    else:
                        rewards[aid, 0] = 1.0 if result == "success" else -1.0

                    was_tx_agents.append(aid)
                # was_tx=False: backoff freeze
            else:
                # idle
                if self.difs[aid] < DIFS:
                    self.difs[aid] += 1
                if self.difs[aid] >= DIFS and self.mld_backoff[aid] > 0:
                    self.mld_backoff[aid] -= 1

        self.h = new_h

        # ── was_tx 에이전트: ao 상태 저장 (새 action 선택 직전) ────────────
        if was_tx_agents:
            w = self._compute_w()
            for aid in was_tx_agents:
                self.ao_h[aid] = self.h[aid]
                self.ao_w[aid] = w[aid]
                self.ao_retry[aid] = self.retry[aid]
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
                        self.sld_collision_count[j] += 1
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
        t = max(self.t - self.t_train_start, 1)
        pkt = PKT_PER_SUCCESS
        link_names = ['2_4GHz', '5GHz', '6GHz']

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

    def get_collision_rate(self) -> dict:
        """
        링크별 MLD/SLD 충돌률 반환.

        collision_rate = collision_count / (success_count + collision_count)

        Returns
        -------
        dict with keys:
            system, mld_total, sld_total,
            link0~2 (total/mld/sld)
        """
        link_names = ['2_4GHz', '5GHz', '6GHz']
        result = {}

        for j in range(self.num_links):
            name    = link_names[j]
            mld_tx  = self.mld_success_count[j] + self.mld_collision_count[j]
            sld_tx  = self.sld_success_count[j] + self.sld_collision_count[j]
            total_tx  = mld_tx + sld_tx
            total_col = self.mld_collision_count[j] + self.sld_collision_count[j]

            result[f'collision_rate/{name}/total'] = total_col / total_tx if total_tx > 0 else 0.0
            result[f'collision_rate/{name}/mld']   = self.mld_collision_count[j] / mld_tx if mld_tx > 0 else 0.0
            result[f'collision_rate/{name}/sld']   = self.sld_collision_count[j] / sld_tx if sld_tx > 0 else 0.0

        total_mld_tx  = sum(self.mld_success_count[j] + self.mld_collision_count[j] for j in range(self.num_links))
        total_mld_col = sum(self.mld_collision_count[j] for j in range(self.num_links))
        total_sld_tx  = sum(self.sld_success_count[j] + self.sld_collision_count[j] for j in range(self.num_links))
        total_sld_col = sum(self.sld_collision_count[j] for j in range(self.num_links))
        total_tx      = total_mld_tx + total_sld_tx
        total_col     = total_mld_col + total_sld_col

        result['collision_rate/mld_total'] = total_mld_col / total_mld_tx if total_mld_tx > 0 else 0.0
        result['collision_rate/sld_total'] = total_sld_col / total_sld_tx if total_sld_tx > 0 else 0.0
        result['collision_rate/system']    = total_col / total_tx if total_tx > 0 else 0.0

        return result

    def _make_available_actions(self) -> np.ndarray:
        """
        need_decision=True : 모든 action(0~5) 가능
        need_decision=False: action 0만 (softmax NaN 방지용, 실제로는 무시됨)
        """
        avail = np.zeros((self.num_agents, 6), dtype=np.float32)
        for aid in range(self.num_agents):
            if self.need_decision[aid]:
                avail[aid] = 1.0
            else:
                avail[aid, 0] = 1.0
        return avail

    def _compute_w(self) -> np.ndarray:
        w = np.zeros(self.num_agents, dtype=np.float32)
        for aid, (sta, link) in enumerate(self.agent_sta_link):
            w[aid] = float(min(self.t - self.last_success[sta, link], W_MAX))
        return w

    def _build_obs(self):
        w   = self._compute_w()
        obs = np.zeros((self.num_agents, 6), dtype=np.float32)
        for aid, (_, link) in enumerate(self.agent_sta_link):
            one_hot = np.zeros(3, dtype=np.float32)
            one_hot[link] = 1.0
            w_norm = min(w[aid], 200.0) / 200.0
            r_norm = min(float(self.retry[aid]), float(RETRY_LIMIT)) / float(RETRY_LIMIT)
            obs[aid] = [w_norm, self.h[aid], r_norm, *one_hot]

        # share_obs: 같은 링크 에이전트 obs만 (max_link_agents × 6, 나머지 0 패딩)
        share_obs = np.zeros((self.num_agents, self.max_link_agents * 6), dtype=np.float32)
        for aid, (_, link) in enumerate(self.agent_sta_link):
            for i, la in enumerate(self.link_agents[link]):
                share_obs[aid, i*6:(i+1)*6] = obs[la]

        return obs, share_obs
