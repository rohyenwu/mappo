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

# ── throughput 파라미터 ────────────────────────────────────────────────────────
PKT_PER_SUCCESS = [1, 2, 3]  # 링크별 성공당 패킷 수 (2.4GHz, 5GHz, 6GHz)


class WiFiEnv:
    """
    WiFi 다중링크 공존 환경 — 배경 MLD/SLD 랜덤화 지원.

    학습 에이전트 (고정)
    --------------------
    MLD-A num_mld_a개 × {2.4, 5} + MLD-B num_mld_b개 × {2.4, 5, 6}

    배경 에이전트 (에피소드마다 랜덤)
    ---------------------------------
    MLD-A: num_mld_a ~ max_mld_a개  (초과분이 배경)
    MLD-B: num_mld_b ~ max_mld_b개  (초과분이 배경)
    SLD  : 1 ~ max_sld개            (2.4GHz only)

    배경 MLD는 학습 policy로 행동하지만 buffer sample에는 포함되지 않음.
    share_obs에는 배경 MLD obs도 포함 (max_link_agents 기준 zero-padding).
    priority 계산에도 배경 MLD의 W가 포함됨.

    에이전트 슬롯 배치 (고정, 변하지 않음)
    ---------------------------------------
    [0, num_agents)              : 학습 에이전트
    [bg_a_start, bg_a_end)       : 배경 MLD-A 슬롯
    [bg_b_start, bg_b_end)       : 배경 MLD-B 슬롯
    active 마스크로 현재 활성 여부 관리.
    """

    def __init__(self, num_mld_a: int = 2, num_mld_b: int = 2,
                 num_sld_per_link: int = 2,
                 max_mld_a: int = 6, max_mld_b: int = 6,
                 max_sld_per_link: int = 4):

        # ── 학습 에이전트 설정 (고정) ─────────────────────────────────────────
        self.num_mld_a = num_mld_a
        self.num_mld_b = num_mld_b
        self.num_sld_base = num_sld_per_link

        # ── 최대 설정 (배경 포함) ─────────────────────────────────────────────
        self.max_mld_a = max_mld_a
        self.max_mld_b = max_mld_b
        self.max_sld = max_sld_per_link
        self.num_links = 3

        # ── 고정 에이전트 슬롯 배치 ──────────────────────────────────────────
        # STA ID 규칙:
        #   MLD-A: sta 0 .. max_mld_a-1
        #   MLD-B: sta max_mld_a .. max_mld_a+max_mld_b-1
        self.all_sta_link = []  # agent_id -> (sta_id, link_id), 고정

        # 학습 MLD-A
        for sta in range(num_mld_a):
            for link in [0, 1]:
                self.all_sta_link.append((sta, link))
        # 학습 MLD-B
        for i in range(num_mld_b):
            sta = max_mld_a + i
            for link in [0, 1, 2]:
                self.all_sta_link.append((sta, link))

        self.num_agents = len(self.all_sta_link)  # 학습 에이전트 수 (외부 노출)

        # 배경 MLD-A 슬롯
        self.bg_a_start = self.num_agents
        for sta in range(num_mld_a, max_mld_a):
            for link in [0, 1]:
                self.all_sta_link.append((sta, link))
        self.bg_a_end = len(self.all_sta_link)

        # 배경 MLD-B 슬롯
        self.bg_b_start = self.bg_a_end
        for i in range(num_mld_b, max_mld_b):
            sta = max_mld_a + i
            for link in [0, 1, 2]:
                self.all_sta_link.append((sta, link))
        self.bg_b_end = len(self.all_sta_link)

        self.max_all_agents = len(self.all_sta_link)
        self.max_total_mld = max_mld_a + max_mld_b

        # 링크별 최대 에이전트 수 (share_obs 차원)
        # Link 0,1: max_mld_a + max_mld_b
        # Link 2  : max_mld_b
        self.max_link_agents = max_mld_a + max_mld_b

        # ── Gym 공간 (학습 에이전트 기준) ─────────────────────────────────────
        obs_low  = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0,  1.0, 1.0, 1.0, 1.0, 1.0,  1.0], dtype=np.float32)

        self.observation_space = [
            spaces.Box(obs_low, obs_high, dtype=np.float32)
        ] * self.num_agents

        self.obs_dim = len(obs_low)
        self.share_observation_space = [
            spaces.Box(
                low=np.tile(obs_low, self.max_link_agents),
                high=np.tile(obs_high, self.max_link_agents),
                dtype=np.float32,
            )
        ] * self.num_agents

        self.action_space = [spaces.Discrete(6)] * self.num_agents

        # ── 현재 배경 설정 ────────────────────────────────────────────────────
        self.cur_mld_a = num_mld_a
        self.cur_mld_b = num_mld_b
        self.cur_sld = num_sld_per_link

        # 배경 에이전트 action 저장 (기본 CW level 2)
        self._stored_bg_actions = np.ones(self.max_all_agents, dtype=np.int32) * 2

        # active 마스크
        self.active = np.zeros(self.max_all_agents, dtype=bool)
        self._update_active_mask()

        # 배경 obs 캐시
        self._bg_obs_cache = np.zeros((self.max_all_agents, 7), dtype=np.float32)

        self._init_state()

    # ──────────────────────────────────────────────────────────────────────────
    # active 마스크 관리
    # ──────────────────────────────────────────────────────────────────────────

    def _update_active_mask(self):
        """현재 cur_mld_a/b에 맞춰 active 마스크 및 link_agents 갱신."""
        self.active[:] = False
        self.active[:self.num_agents] = True  # 학습 에이전트 항상 활성

        # 배경 MLD-A
        n_bg_a = (self.cur_mld_a - self.num_mld_a) * 2
        if n_bg_a > 0:
            self.active[self.bg_a_start:self.bg_a_start + n_bg_a] = True

        # 배경 MLD-B
        n_bg_b = (self.cur_mld_b - self.num_mld_b) * 3
        if n_bg_b > 0:
            self.active[self.bg_b_start:self.bg_b_start + n_bg_b] = True

        # link_agents 재구축
        self.link_agents = {j: [] for j in range(self.num_links)}
        for aid in range(self.max_all_agents):
            if self.active[aid]:
                _, link = self.all_sta_link[aid]
                self.link_agents[link].append(aid)

        # 배경 에이전트 인덱스
        bg_mask = self.active.copy()
        bg_mask[:self.num_agents] = False
        self.bg_agent_indices = np.where(bg_mask)[0]
        self.num_bg_agents = len(self.bg_agent_indices)

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────────────────────

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def reset(self, warmup_decisions: int = None):
        """
        환경 초기화. 배경 MLD/SLD 수를 랜덤 설정.

        Returns
        -------
        obs              : (num_agents, 6)          학습 에이전트만
        share_obs        : (num_agents, max_link_agents*6)  배경 포함
        available_actions: (num_agents, 6)           학습 에이전트만
        """
        if warmup_decisions is None:
            warmup_decisions = self.num_agents

        # 배경 수 랜덤화
        self.cur_mld_a = np.random.randint(self.num_mld_a, self.max_mld_a + 1)
        self.cur_mld_b = np.random.randint(self.num_mld_b, self.max_mld_b + 1)
        self.cur_sld   = np.random.randint(1, self.max_sld + 1)
        self._update_active_mask()

        self._init_state()

        # 모든 활성 에이전트 초기 backoff (CW level 2)
        for aid in range(self.max_all_agents):
            if self.active[aid]:
                cw_min, cw_max = CW_TABLE[2]
                self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))

        dummy = np.zeros((self.max_all_agents, 1), dtype=np.float32)

        # 첫 결과 발생까지
        while not np.any(self.need_decision & self.active):
            self._advance_one_slot(dummy)
            self.t += 1

        # warmup
        for _ in range(warmup_decisions):
            for aid in range(self.max_all_agents):
                if self.active[aid] and self.need_decision[aid]:
                    cw_min, cw_max = CW_TABLE[2]
                    self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))
            self.need_decision[:] = False
            while not np.any(self.need_decision & self.active):
                self._advance_one_slot(dummy)
                self.t += 1

        # 카운터 초기화
        self.mld_success_count[:]   = 0
        self.sld_success_count[:]   = 0
        self.mld_collision_count[:] = 0
        self.sld_collision_count[:] = 0
        self.t_train_start = self.t
        self.pending_reward[:] = 0.0

        all_obs = self._build_all_obs()
        self._bg_obs_cache = all_obs
        obs       = all_obs[:self.num_agents]
        share_obs = self._build_share_obs(all_obs)
        return obs, share_obs, self._make_available_actions()

    def step(self, actions):
        """
        Parameters
        ----------
        actions : np.ndarray  shape (num_agents, 1)  학습 에이전트만

        Returns
        -------
        obs, share_obs, rewards, dones, infos, available_actions
            모두 학습 에이전트 기준. share_obs만 배경 포함.
        """
        actions_flat = np.clip(actions.flatten().astype(int), 0, 5)

        # 학습 + 배경 action 합치기
        all_actions = np.zeros(self.max_all_agents, dtype=int)
        all_actions[:self.num_agents] = actions_flat
        for aid in self.bg_agent_indices:
            all_actions[aid] = self._stored_bg_actions[aid]

        # ── Phase 0: 학습 에이전트 pending reward 수거 ────────────────────────
        decided = self.need_decision[:self.num_agents].copy()
        pending_rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        for aid in range(self.num_agents):
            if decided[aid]:
                pending_rewards[aid, 0] = self.pending_reward[aid]
                self.pending_reward[aid] = 0.0

        # 배경 에이전트 pending도 클리어 (버퍼에는 안 감)
        for aid in self.bg_agent_indices:
            if self.need_decision[aid]:
                self.pending_reward[aid] = 0.0

        # ── Phase 1: 모든 decided 에이전트 action 적용 + priority ─────────────
        w = self._compute_w()
        for aid in range(self.max_all_agents):
            if not self.active[aid] or not self.need_decision[aid]:
                continue
            act = all_actions[aid]
            self.ao_action[aid] = act
            cw_min, cw_max = CW_TABLE[act]
            self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))

            # priority: 같은 링크 내 W log 스케일 min-max
            _, link = self.all_sta_link[aid]
            link_w = np.array([w[a] for a in self.link_agents[link]], dtype=np.float32)
            log_link_w = np.log1p(link_w)
            log_wi  = np.log1p(w[aid])
            log_min = np.min(log_link_w)
            log_max = np.max(log_link_w)
            if log_max - log_min < 1e-6:
                self.ao_priority[aid] = 0.5
            else:
                self.ao_priority[aid] = 0.1 + 0.8 * (log_wi - log_min) / (log_max - log_min)

        self.need_decision[:] = False

        # ── Phase 2: 다음 결과 발생까지 슬롯 진행 ────────────────────────────
        dummy = np.zeros((self.max_all_agents, 1), dtype=np.float32)
        while not np.any(self.need_decision & self.active):
            self._advance_one_slot(dummy)
            self.t += 1

        # ── 결과 구성 (학습 에이전트만) ──────────────────────────────────────
        all_obs = self._build_all_obs()
        self._bg_obs_cache = all_obs
        obs       = all_obs[:self.num_agents]
        share_obs = self._build_share_obs(all_obs)

        dones = np.zeros(self.num_agents, dtype=bool)
        infos = []
        for aid in range(self.num_agents):
            info = {
                'bad_transition': False,
                'decided':        bool(decided[aid]),
                'priority':       float(self.ao_priority[aid]),
                'pending_reward': float(pending_rewards[aid, 0]),
                'w_norm':         float(min(w[aid], 200.0) / 200.0),
            }
            if decided[aid]:
                info['result_type']     = str(self.pending_result_type[aid])
                info['result_priority'] = float(self.pending_result_priority[aid])
                info['action']          = int(actions_flat[aid])
            infos.append(info)

        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        return obs, share_obs, rewards, dones, infos, self._make_available_actions()

    def close(self):
        pass

    def render(self, mode='human'):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # 배경 에이전트 인터페이스 (Runner가 호출)
    # ──────────────────────────────────────────────────────────────────────────

    def get_bg_obs(self):
        """배경 에이전트 관측 + available_actions 반환.

        Returns (None, None) if 배경 에이전트 없음.
        """
        if self.num_bg_agents == 0:
            return None, None
        bg_obs = self._bg_obs_cache[self.bg_agent_indices].copy()
        bg_avail = np.zeros((self.num_bg_agents, 6), dtype=np.float32)
        for i, aid in enumerate(self.bg_agent_indices):
            if self.need_decision[aid]:
                bg_avail[i] = 1.0
            else:
                bg_avail[i, 0] = 1.0
        return bg_obs, bg_avail

    def set_bg_actions(self, actions):
        """Runner가 policy forward 결과를 전달."""
        if actions is not None and self.num_bg_agents > 0:
            actions_flat = np.array(actions).flatten().astype(int)
            for i, aid in enumerate(self.bg_agent_indices):
                if i < len(actions_flat):
                    self._stored_bg_actions[aid] = actions_flat[i]

    def randomize_background(self):
        """배경 MLD/SLD 수 랜덤화. 에피소드 사이에 Runner가 호출."""
        self.cur_mld_a = np.random.randint(self.num_mld_a, self.max_mld_a + 1)
        self.cur_mld_b = np.random.randint(self.num_mld_b, self.max_mld_b + 1)
        self.cur_sld   = np.random.randint(1, self.max_sld + 1)

        old_active = self.active.copy()
        self._update_active_mask()

        # 새로 활성화된 배경 에이전트 초기화
        newly_active = self.active & ~old_active
        for aid in np.where(newly_active)[0]:
            sta, link = self.all_sta_link[aid]
            cw_min, cw_max = CW_TABLE[2]
            self.mld_backoff[aid] = int(np.random.randint(cw_min, cw_max))
            self.need_decision[aid] = False
            self.h[aid]     = 0.0
            self.difs[aid]  = 0
            self.retry[aid] = 0
            self.pending_reward[aid] = 0.0
            self.last_success[sta, link] = self.t  # W=0으로 시작

        # SLD 재구성
        self.sld_state = []
        for j in range(self.num_links):
            link_slds = []
            if j == 0:
                for _ in range(self.cur_sld):
                    cw = CW_MIN
                    link_slds.append({
                        'cw':      cw,
                        'backoff': int(np.random.randint(0, cw)),
                        'retry':   0,
                        'difs':    0,
                    })
            self.sld_state.append(link_slds)

        # obs 캐시 갱신
        self._bg_obs_cache = self._build_all_obs()

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _init_state(self):
        self.t = 0
        self.t_train_start = 0
        self.pending_reward = np.zeros(self.max_all_agents, dtype=np.float32)

        self.last_success = np.full(
            (self.max_total_mld, self.num_links), -W_MAX, dtype=np.float64
        )
        self.link_last_success = np.full(self.num_links, -W_MAX, dtype=np.float64)
        self.h    = np.zeros(self.max_all_agents, dtype=np.float32)
        self.difs = np.zeros(self.max_all_agents, dtype=np.int32)

        self.mld_backoff = np.full(self.max_all_agents, -1, dtype=np.int32)
        self.need_decision = np.zeros(self.max_all_agents, dtype=bool)

        self.ao_priority = np.full(self.max_all_agents, 0.5, dtype=np.float32)
        self.ao_action   = np.ones(self.max_all_agents, dtype=np.int32) * 2

        self.pending_result_priority = np.zeros(self.max_all_agents, dtype=np.float32)
        self.pending_result_type     = np.full(self.max_all_agents, '', dtype=object)

        self.retry = np.zeros(self.max_all_agents, dtype=np.int32)

        # throughput / 충돌 카운터
        self.mld_success_count   = np.zeros(self.num_links, dtype=np.int64)
        self.sld_success_count   = np.zeros(self.num_links, dtype=np.int64)
        self.mld_collision_count = np.zeros(self.num_links, dtype=np.int64)
        self.sld_collision_count = np.zeros(self.num_links, dtype=np.int64)

        # SLD 상태
        self.sld_state = []
        for j in range(self.num_links):
            link_slds = []
            if j == 0:
                for _ in range(self.cur_sld):
                    cw = CW_MIN
                    link_slds.append({
                        'cw':      cw,
                        'backoff': int(np.random.randint(0, cw)),
                        'retry':   0,
                        'difs':    0,
                    })
            self.sld_state.append(link_slds)

    def _advance_one_slot(self, rewards: np.ndarray):
        """슬롯 1개 진행. 모든 활성 에이전트 처리."""

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

        for aid in range(self.max_all_agents):
            if not self.active[aid]:
                continue
            sta, link = self.all_sta_link[aid]
            result, mld_txers, _ = link_results[link]
            was_tx = (self.mld_backoff[aid] == 0 and self.difs[aid] >= DIFS)

            if result in ("success", "collision"):
                self.difs[aid] = 0
                if was_tx:
                    x = 1.0 if result == "success" else -1.0
                    new_h[aid] = (1.0 - LAMBDA) * self.h[aid] + LAMBDA * x
                    if result == "success":
                        self.last_success[sta, link] = self.t
                        self.link_last_success[link] = self.t
                        self.mld_success_count[link] += 1
                        self.retry[aid] = 0
                    else:
                        self.mld_collision_count[link] += 1
                        self.retry[aid] += 1
                    self.mld_backoff[aid] = -1

                    self.pending_result_priority[aid] = self.ao_priority[aid]
                    self.pending_result_type[aid]     = result
                    if result == "success":
                        self.pending_reward[aid] = 1.0 * self.ao_priority[aid]
                    else:
                        self.pending_reward[aid] = -1.0

                    was_tx_agents.append(aid)
            else:
                # idle
                if self.difs[aid] < DIFS:
                    self.difs[aid] += 1
                if self.difs[aid] >= DIFS and self.mld_backoff[aid] > 0:
                    self.mld_backoff[aid] -= 1

        self.h = new_h

        for aid in was_tx_agents:
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
                        self.link_last_success[j] = self.t
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

    def _compute_w(self) -> np.ndarray:
        w = np.zeros(self.max_all_agents, dtype=np.float32)
        for aid in range(self.max_all_agents):
            if self.active[aid]:
                sta, link = self.all_sta_link[aid]
                w[aid] = float(min(self.t - self.last_success[sta, link], W_MAX))
        return w

    def _build_all_obs(self) -> np.ndarray:
        """모든 활성 에이전트의 관측 생성."""
        w = self._compute_w()
        link_w = np.array([
            min(self.t - self.link_last_success[j], W_MAX) for j in range(self.num_links)
        ], dtype=np.float32)
        all_obs = np.zeros((self.max_all_agents, 7), dtype=np.float32)
        for aid in range(self.max_all_agents):
            if not self.active[aid]:
                continue
            _, link = self.all_sta_link[aid]
            one_hot = np.zeros(3, dtype=np.float32)
            one_hot[link] = 1.0
            w_norm = min(w[aid], 200.0) / 200.0
            r_norm = min(float(self.retry[aid]), float(RETRY_LIMIT)) / float(RETRY_LIMIT)
            link_w_norm = min(link_w[link], 200.0) / 200.0
            relative_w = link_w_norm - w_norm  # -1 ~ +1
            all_obs[aid] = [w_norm, self.h[aid], r_norm, *one_hot, relative_w]
        return all_obs

    def _build_share_obs(self, all_obs: np.ndarray) -> np.ndarray:
        """학습 에이전트의 share_obs 생성 (배경 포함, zero-padding)."""
        obs_dim = all_obs.shape[1]
        share_obs = np.zeros(
            (self.num_agents, self.max_link_agents * obs_dim), dtype=np.float32
        )
        for aid in range(self.num_agents):
            _, link = self.all_sta_link[aid]
            for i, la in enumerate(self.link_agents[link]):
                if i >= self.max_link_agents:
                    break
                share_obs[aid, i*obs_dim:(i+1)*obs_dim] = all_obs[la]
        return share_obs

    def _make_available_actions(self) -> np.ndarray:
        """학습 에이전트의 available_actions."""
        avail = np.zeros((self.num_agents, 6), dtype=np.float32)
        for aid in range(self.num_agents):
            if self.need_decision[aid]:
                avail[aid] = 1.0
            else:
                avail[aid, 0] = 1.0
        return avail

    # ──────────────────────────────────────────────────────────────────────────
    # Throughput / Collision rate
    # ──────────────────────────────────────────────────────────────────────────

    def get_throughput(self) -> dict:
        t = max(self.t - self.t_train_start, 1)
        pkt = PKT_PER_SUCCESS
        link_names = ['2_4GHz', '5GHz', '6GHz']
        result = {}

        for j in range(self.num_links):
            name = link_names[j]
            mld = self.mld_success_count[j] * pkt[j] / t
            sld = self.sld_success_count[j] * pkt[j] / t
            result[f'throughput/{name}/total'] = mld + sld
            result[f'throughput/{name}/mld']   = mld
            result[f'throughput/{name}/sld']   = sld

        result['throughput/mld_total'] = sum(
            self.mld_success_count[j] * pkt[j] for j in range(self.num_links)
        ) / t
        result['throughput/sld_total'] = sum(
            self.sld_success_count[j] * pkt[j] for j in range(self.num_links)
        ) / t
        result['throughput/system'] = (
            result['throughput/mld_total'] + result['throughput/sld_total']
        )
        return result

    def get_collision_rate(self) -> dict:
        link_names = ['2_4GHz', '5GHz', '6GHz']
        result = {}

        for j in range(self.num_links):
            name    = link_names[j]
            mld_tx  = self.mld_success_count[j] + self.mld_collision_count[j]
            sld_tx  = self.sld_success_count[j] + self.sld_collision_count[j]
            total_tx  = mld_tx + sld_tx
            total_col = self.mld_collision_count[j] + self.sld_collision_count[j]

            result[f'collision_rate/{name}/total'] = (
                total_col / total_tx if total_tx > 0 else 0.0
            )
            result[f'collision_rate/{name}/mld'] = (
                self.mld_collision_count[j] / mld_tx if mld_tx > 0 else 0.0
            )
            result[f'collision_rate/{name}/sld'] = (
                self.sld_collision_count[j] / sld_tx if sld_tx > 0 else 0.0
            )

        total_mld_tx  = sum(
            self.mld_success_count[j] + self.mld_collision_count[j]
            for j in range(self.num_links)
        )
        total_mld_col = sum(self.mld_collision_count[j] for j in range(self.num_links))
        total_sld_tx  = sum(
            self.sld_success_count[j] + self.sld_collision_count[j]
            for j in range(self.num_links)
        )
        total_sld_col = sum(self.sld_collision_count[j] for j in range(self.num_links))
        total_tx  = total_mld_tx + total_sld_tx
        total_col = total_mld_col + total_sld_col

        result['collision_rate/mld_total'] = (
            total_mld_col / total_mld_tx if total_mld_tx > 0 else 0.0
        )
        result['collision_rate/sld_total'] = (
            total_sld_col / total_sld_tx if total_sld_tx > 0 else 0.0
        )
        result['collision_rate/system'] = (
            total_col / total_tx if total_tx > 0 else 0.0
        )
        return result
