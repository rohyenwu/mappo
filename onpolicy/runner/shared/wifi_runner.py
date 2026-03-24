import time
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

            # ── 에피소드 롤아웃 수집 ─────────────────────────────────────────
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = \
                    self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = \
                    self.envs.step(actions)

                data = (obs, share_obs, rewards, dones, infos, available_actions,
                        values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)
                self.insert(data)

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
                print(f"  average step reward: {train_infos['average_step_rewards']:.4f}")
                self.log_train(train_infos, total_num_steps)

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

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs, obs,
            rnn_states, rnn_states_critic,
            actions, action_log_probs, values,
            rewards, masks, bad_masks, active_masks,
            available_actions,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 평가
    # ──────────────────────────────────────────────────────────────────────────

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
