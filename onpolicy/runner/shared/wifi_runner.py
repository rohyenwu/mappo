import time
import csv
import os
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class WiFiRunner(Runner):
    """
    MAPPO 학습 루프 — WiFi 다중링크 공존 환경 전용.

    SMACRunner 구조를 따르며 (ShareDummyVecEnv 호환),
    SMAC 고유 로직(battle win rate 등)은 제거하고
    WiFi 환경에 맞는 로깅만 유지.

    Reward는 WiFiEnv.step() 내부에서 계산되어 반환되므로
    Runner는 별도 reward 계산 없이 buffer에 그대로 삽입.
    """

    def __init__(self, config):
        super(WiFiRunner, self).__init__(config)

        # use_wandb=True 이면 base_runner가 log_dir을 설정하지 않으므로 save_dir로 대체
        if not hasattr(self, 'log_dir'):
            self.log_dir = self.save_dir

        # CSV 파일 초기화
        self.train_csv = os.path.join(str(self.log_dir), 'train_throughput.csv')
        self.eval_csv  = os.path.join(str(self.log_dir), 'eval_throughput.csv')
        self._csv_initialized = False
        self._eval_csv_initialized = False

        self.train_collision_csv = os.path.join(str(self.log_dir), 'train_collision_rate.csv')
        self.eval_collision_csv  = os.path.join(str(self.log_dir), 'eval_collision_rate.csv')
        self._collision_csv_initialized = False
        self._eval_collision_csv_initialized = False

        self.train_action_dist_csv = os.path.join(str(self.log_dir), 'train_action_dist.csv')
        self._action_dist_csv_initialized = False

        # priority값 수집용 버퍼
        self._episode_priority_values = []
        self._episode_success_priorities = []
        self._episode_collision_priorities = []

        # W구간별 action 수집용 버퍼
        self._episode_w_action = []  # list of (w_norm, action)

        # 소급 보상: 각 에이전트의 마지막 decision buffer step 추적
        # key: (thread_idx, agent_idx), value: buffer step index
        self._last_decision_step = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 메인 학습 루프
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        self.warmup()

        start    = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if self.all_args.entropy_coef_min is not None:
                self.trainer.entropy_decay(episode, episodes)

            # ── 에피소드 롤아웃 수집 ─────────────────────────────────────────
            self._last_decision_step.clear()  # 새 에피소드마다 초기화

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = \
                    self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = \
                    self.envs.step(actions)

                data = (obs, share_obs, rewards, dones, infos, available_actions,
                        values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)
                self.insert(data)

            # 에피소드 끝: 아직 reward가 배치되지 않은 마지막 action에
            # 현재 env의 pending_reward를 소급 배치
            self._flush_pending_rewards()

            # ── GAE 계산 + PPO 업데이트 ──────────────────────────────────────
            self.compute()
            train_infos = self.train()

            # ── 저장 / 로깅 ──────────────────────────────────────────────────
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    f"\n[WiFi] Algo {self.algorithm_name} | "
                    f"Exp {self.experiment_name} | "
                    f"Episode {episode}/{episodes} | "
                    f"Steps {total_num_steps}/{self.num_env_steps} | "
                    f"FPS {int(total_num_steps / (end - start))}"
                )
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)

                # deciding 에이전트만의 평균 reward
                active = self.buffer.active_masks[:-1]  # (episode_length, n_threads, n_agents, 1)
                rewards = self.buffer.rewards             # (episode_length, n_threads, n_agents, 1)
                decided_mask = active.flatten() > 0
                if decided_mask.any():
                    decided_rewards = rewards.flatten()[decided_mask]
                    avg_decided = float(np.mean(decided_rewards))
                    avg_decided_pos = float(np.mean(decided_rewards[decided_rewards > 0])) if (decided_rewards > 0).any() else 0.0
                    avg_decided_neg = float(np.mean(decided_rewards[decided_rewards < 0])) if (decided_rewards < 0).any() else 0.0
                    success_ratio = float((decided_rewards > 0).sum()) / float(len(decided_rewards))
                else:
                    avg_decided = avg_decided_pos = avg_decided_neg = success_ratio = 0.0

                train_infos["decided_avg_reward"] = avg_decided
                train_infos["decided_avg_reward_success"] = avg_decided_pos
                train_infos["decided_avg_reward_collision"] = avg_decided_neg
                train_infos["decided_success_ratio"] = success_ratio

                # 평균 priority
                if self._episode_priority_values:
                    avg_priority = float(np.mean(self._episode_priority_values))
                else:
                    avg_priority = 0.0
                train_infos["decided_avg_priority"] = avg_priority
                self._episode_priority_values = []

                # 결과별 priority
                if self._episode_success_priorities:
                    avg_success_priority = float(np.mean(self._episode_success_priorities))
                else:
                    avg_success_priority = 0.0
                if self._episode_collision_priorities:
                    avg_collision_priority = float(np.mean(self._episode_collision_priorities))
                else:
                    avg_collision_priority = 0.0
                train_infos["priority_at_success"] = avg_success_priority
                train_infos["priority_at_collision"] = avg_collision_priority
                self._episode_success_priorities = []
                self._episode_collision_priorities = []

                print(f"  average step reward: {train_infos['average_step_rewards']:.4f}")
                print(f"  decided avg reward:  {avg_decided:.4f}  "
                      f"(success: {avg_decided_pos:.4f}, collision: {avg_decided_neg:.4f}, "
                      f"success_ratio: {success_ratio:.4f})")
                print(f"  decided avg priority: {avg_priority:.4f}")
                print(f"  priority at success: {avg_success_priority:.4f}  "
                      f"priority at collision: {avg_collision_priority:.4f}")

                # W구간별 평균 action
                if self._episode_w_action:
                    wa = self._episode_w_action
                    low_w  = [a for w, a in wa if w < 0.33]
                    mid_w  = [a for w, a in wa if 0.33 <= w < 0.66]
                    high_w = [a for w, a in wa if w >= 0.66]
                    avg_act_low  = float(np.mean(low_w))  if low_w  else -1
                    avg_act_mid  = float(np.mean(mid_w))  if mid_w  else -1
                    avg_act_high = float(np.mean(high_w)) if high_w else -1
                    train_infos["action_by_w/low_w_avg_action"]  = avg_act_low
                    train_infos["action_by_w/mid_w_avg_action"]  = avg_act_mid
                    train_infos["action_by_w/high_w_avg_action"] = avg_act_high
                    print(f"  action by W:  low(<0.33)={avg_act_low:.2f}  "
                          f"mid(0.33~0.66)={avg_act_mid:.2f}  "
                          f"high(>=0.66)={avg_act_high:.2f}")
                self._episode_w_action = []

                print(f"  entropy_coef:        {self.trainer.entropy_coef:.4f}")
                train_infos["entropy_coef"] = self.trainer.entropy_coef
                self.log_train(train_infos, total_num_steps)

                # throughput 측정 및 CSV 저장
                tp = self.envs.get_throughput()
                self._log_throughput(tp, total_num_steps, tag='train')

                # 충돌률 측정 및 CSV 저장
                cr = self.envs.get_collision_rate()
                self._log_collision_rate(cr, total_num_steps, tag='train')

                # action 분포 측정 및 CSV 저장
                self._log_action_dist(total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    # ──────────────────────────────────────────────────────────────────────────
    # warmup / collect / insert
    # ──────────────────────────────────────────────────────────────────────────

    def warmup(self):
        """환경 reset 후 buffer 첫 슬롯 초기화."""
        obs, share_obs, available_actions = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0]        = share_obs.copy()
        self.buffer.obs[0]              = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

        # 모든 action 가능(sum=7) → need_decision=True → active
        active_masks = (available_actions.sum(axis=-1, keepdims=True) > 1).astype(np.float32)
        self.buffer.active_masks[0] = active_masks

    @torch.no_grad()
    def collect(self, step):
        """
        현재 buffer 슬롯에서 policy를 실행해 action, value 등을 샘플링.

        Returns
        -------
        values, actions, action_log_probs, rnn_states, rnn_states_critic
            shape: (n_rollout_threads, num_agents, dim)
        """
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
                np.concatenate(self.buffer.available_actions[step]),
            )

        values           = np.array(np.split(_t2n(value),            self.n_rollout_threads))
        actions          = np.array(np.split(_t2n(action),           self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob),  self.n_rollout_threads))
        rnn_states       = np.array(np.split(_t2n(rnn_state),        self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """
        환경 step 결과를 SharedReplayBuffer에 삽입.

        Notes
        -----
        - dones_env  : 에피소드 종료 여부 (모든 에이전트가 done)
          WiFiEnv는 에피소드가 없으므로 항상 False.
        - bad_masks  : 타임리밋 패널티용 — 모두 1.0 (해당 없음).
        - active_masks: 에이전트 생사 — 모두 1.0 (해당 없음).
        """
        (obs, share_obs, rewards, dones, infos, available_actions,
         values, actions, action_log_probs, rnn_states, rnn_states_critic) = data

        dones_env = np.all(dones, axis=1)  # (n_rollout_threads,)

        # done 에피소드의 RNN 상태 초기화
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents,
             self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents,
             *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # AO였던 에이전트만 actor 학습에 포함
        active_masks = np.array(
            [[[1.0] if info[aid]['decided'] else [0.0]
              for aid in range(self.num_agents)]
             for info in infos],
            dtype=np.float32,
        )  # shape: (n_rollout_threads, num_agents, 1)

        # priority값 수집 (deciding 에이전트만)
        for t_idx, info in enumerate(infos):
            for aid in range(self.num_agents):
                if info[aid]['decided']:
                    self._episode_priority_values.append(info[aid]['priority'])
                    self._episode_w_action.append((info[aid]['w_norm'], info[aid]['action']))
                    if info[aid].get('result_type') == 'success':
                        self._episode_success_priorities.append(info[aid]['result_priority'])
                    elif info[aid].get('result_type') == 'collision':
                        self._episode_collision_priorities.append(info[aid]['result_priority'])

        bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        # ── 소급 보상 배치 ────────────────────────────────────────────────────
        # 현재 step에서 decided=True인 에이전트는 이전 action의 결과(pending_reward)를
        # 갖고 있다. 이 reward를 이전 action이 저장된 buffer step에 소급 배치한다.
        current_step = self.buffer.step

        for t_idx, info in enumerate(infos):
            for aid in range(self.num_agents):
                if info[aid]['decided']:
                    pending_r = info[aid]['pending_reward']
                    key = (t_idx, aid)

                    # 이전 decision step이 있으면 그곳에 reward 소급 배치
                    if key in self._last_decision_step:
                        prev_step = self._last_decision_step[key]
                        self.buffer.rewards[prev_step, t_idx, aid, 0] = pending_r

                    # 현재 step을 새 decision step으로 기록
                    self._last_decision_step[key] = current_step

        self.buffer.insert(
            share_obs, obs,
            rnn_states, rnn_states_critic,
            actions, action_log_probs, values,
            rewards, masks, bad_masks, None,
            available_actions,
        )
        self.buffer.active_masks[current_step] = active_masks

    def _flush_pending_rewards(self):
        """에피소드 끝에서 아직 소급 배치되지 않은 마지막 action의 reward를 처리.

        env 내부의 pending_reward (이전 action의 결과로 아직 수거 안 된 것)를
        마지막 decision step에 배치한다.
        """
        for (t_idx, aid), prev_step in self._last_decision_step.items():
            env = self.envs.envs[t_idx]
            pending_r = env.pending_reward[aid]
            if pending_r != 0.0:
                self.buffer.rewards[prev_step, t_idx, aid, 0] = pending_r

    # ──────────────────────────────────────────────────────────────────────────
    # 평가
    # ──────────────────────────────────────────────────────────────────────────

    def _log_throughput(self, tp: dict, total_num_steps: int, tag: str):
        """throughput 출력, wandb 로깅, CSV 저장."""
        print(f"  [{tag}] throughput/system:    {tp['throughput/system']:.4f}")
        print(f"  [{tag}] throughput/mld_total: {tp['throughput/mld_total']:.4f}")
        print(f"  [{tag}] throughput/sld_total: {tp['throughput/sld_total']:.4f}")
        for link in ['2_4GHz', '5GHz', '6GHz']:
            print(
                f"  [{tag}] {link}: "
                f"total={tp[f'throughput/{link}/total']:.4f}  "
                f"mld={tp[f'throughput/{link}/mld']:.4f}  "
                f"sld={tp[f'throughput/{link}/sld']:.4f}"
            )

        # wandb 로깅 (train/ 또는 eval/ 접두사로 구분)
        if self.use_wandb:
            import wandb
            wandb_log = {f"{tag}/{k}": v for k, v in tp.items()}
            wandb.log(wandb_log, step=total_num_steps)
        else:
            for k, v in tp.items():
                self.writter.add_scalars(f"{tag}/{k}", {f"{tag}/{k}": v}, total_num_steps)

        # CSV 저장
        csv_path = self.train_csv if tag == 'train' else self.eval_csv
        init_flag = '_csv_initialized' if tag == 'train' else '_eval_csv_initialized'
        fieldnames = ['total_num_steps'] + list(tp.keys())

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not getattr(self, init_flag):
                writer.writeheader()
                setattr(self, init_flag, True)
            row = {'total_num_steps': total_num_steps, **tp}
            writer.writerow(row)

    def _log_action_dist(self, total_num_steps: int):
        """action 분포(CW level 0~5 비율) 출력, wandb 로깅, CSV 저장.
        decided=True인 step의 action만 카운트.
        """
        # Bug3 수정 후: active_masks[t] = decided at step t → actions[t]와 직접 대응
        decided = self.buffer.active_masks[:-1].flatten() > 0
        flat_actions = self.buffer.actions.flatten()[decided].astype(int)

        total = len(flat_actions) if len(flat_actions) > 0 else 1
        dist = {f'action_dist/cw_level_{a}': (flat_actions == a).sum() / total for a in range(6)}

        cw_ranges = {0: '2~5', 1: '6~11', 2: '12~23', 3: '24~47', 4: '48~95', 5: '96~127'}
        a_vals    = {0: 0.85, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.15, 5: 0.05}
        print("  [train] action distribution:")
        for a in range(6):
            print(f"    CW level {a} (CW={cw_ranges[a]}, A={a_vals[a]}): {dist[f'action_dist/cw_level_{a}']:.4f}")

        if self.use_wandb:
            import wandb
            wandb.log({f"train/{k}": v for k, v in dist.items()}, step=total_num_steps)
        else:
            for k, v in dist.items():
                self.writter.add_scalars(f"train/{k}", {f"train/{k}": v}, total_num_steps)

        fieldnames = ['total_num_steps'] + list(dist.keys())
        with open(self.train_action_dist_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._action_dist_csv_initialized:
                writer.writeheader()
                self._action_dist_csv_initialized = True
            writer.writerow({'total_num_steps': total_num_steps, **dist})

    def _log_collision_rate(self, cr: dict, total_num_steps: int, tag: str):
        """충돌률 출력, wandb 로깅, CSV 저장."""
        print(f"  [{tag}] collision_rate/system:    {cr['collision_rate/system']:.4f}")
        print(f"  [{tag}] collision_rate/mld_total: {cr['collision_rate/mld_total']:.4f}")
        print(f"  [{tag}] collision_rate/sld_total: {cr['collision_rate/sld_total']:.4f}")
        for link in ['2_4GHz', '5GHz', '6GHz']:
            print(
                f"  [{tag}] {link}: "
                f"total={cr[f'collision_rate/{link}/total']:.4f}  "
                f"mld={cr[f'collision_rate/{link}/mld']:.4f}  "
                f"sld={cr[f'collision_rate/{link}/sld']:.4f}"
            )

        if self.use_wandb:
            import wandb
            wandb_log = {f"{tag}/{k}": v for k, v in cr.items()}
            wandb.log(wandb_log, step=total_num_steps)
        else:
            for k, v in cr.items():
                self.writter.add_scalars(f"{tag}/{k}", {f"{tag}/{k}": v}, total_num_steps)

        csv_path  = self.train_collision_csv if tag == 'train' else self.eval_collision_csv
        init_flag = '_collision_csv_initialized' if tag == 'train' else '_eval_collision_csv_initialized'
        fieldnames = ['total_num_steps'] + list(cr.keys())

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not getattr(self, init_flag):
                writer.writeheader()
                setattr(self, init_flag, True)
            row = {'total_num_steps': total_num_steps, **cr}
            writer.writerow(row)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents,
             self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        eval_episode_rewards = []

        for _ in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=True,
            )
            eval_actions     = np.array(np.split(_t2n(eval_actions),     self.n_eval_rollout_threads))
            eval_rnn_states  = np.array(np.split(_t2n(eval_rnn_states),  self.n_eval_rollout_threads))

            eval_obs, eval_share_obs, eval_rewards, eval_dones, _, eval_available_actions = \
                self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents,
                 self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        avg_reward = np.mean(np.sum(eval_episode_rewards, axis=0))
        print(f"  [eval] average episode reward: {avg_reward:.4f}")
        self.log_env(
            {'eval_average_episode_rewards': np.sum(eval_episode_rewards, axis=0)},
            total_num_steps,
        )

        # eval throughput 측정 및 CSV 저장
        tp = self.eval_envs.get_throughput()
        self._log_throughput(tp, total_num_steps, tag='eval')

        # eval 충돌률 측정 및 CSV 저장
        cr = self.eval_envs.get_collision_rate()
        self._log_collision_rate(cr, total_num_steps, tag='eval')
