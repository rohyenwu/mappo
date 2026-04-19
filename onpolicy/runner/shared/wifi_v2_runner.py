"""
WiFi v2 environment runner for MAPPO.

Design reference: docs/project_wifi_redesign_v4.md
"""
import csv
import os
import time

import numpy as np
import torch

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class WiFiV2Runner(Runner):
    """
    Training loop for the TXOP-synchronous WiFi v2 environment.

    - 1 episode = 1 round = T TXOP steps
    - Per-step binary action (transmit / skip)
    - Dense reward is computed in the environment
    - Sparse SLD coexistence reward is applied at round end
    """

    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self, "log_dir"):
            self.log_dir = self.save_dir

        self.train_csv = os.path.join(str(self.log_dir), "train_metrics.csv")
        self._csv_initialized = False

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            infos = None
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = \
                    self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = \
                    self.envs.step(actions)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / (end - start)) if (end - start) > 0 else 0

                avg_reward = float(np.mean(self.buffer.rewards))
                avg_reward_pos = float(np.mean(self.buffer.rewards[self.buffer.rewards > 0])) \
                    if (self.buffer.rewards > 0).any() else 0.0
                avg_reward_neg = float(np.mean(self.buffer.rewards[self.buffer.rewards < 0])) \
                    if (self.buffer.rewards < 0).any() else 0.0

                avg_fulfillment = 0.0
                if infos is not None:
                    fulfillments = []
                    for env_infos in infos:
                        for info in env_infos:
                            fulfillments.append(info.get("fulfillment", 0.0))
                    avg_fulfillment = float(np.mean(fulfillments)) if fulfillments else 0.0

                flat_actions = self.buffer.actions.flatten().astype(int)
                n_total = max(len(flat_actions), 1)
                transmit_ratio = (flat_actions == 1).sum() / n_total

                print(
                    f"\n[WiFi-v2] Episode {episode}/{episodes} | "
                    f"Steps {total_num_steps}/{self.num_env_steps} | FPS {fps}"
                )
                print(
                    f"  avg reward:      {avg_reward:.4f} "
                    f"(pos: {avg_reward_pos:.4f}, neg: {avg_reward_neg:.4f})"
                )
                print(f"  transmit ratio:  {transmit_ratio:.4f}")
                print(f"  avg fulfillment: {avg_fulfillment:.4f}")

                train_infos["average_step_rewards"] = avg_reward
                train_infos["transmit_ratio"] = transmit_ratio
                train_infos["avg_fulfillment"] = avg_fulfillment

                env0 = self.envs.envs[0]
                tp = env0.get_throughput()
                for k, v in tp.items():
                    print(f"  {k}: {v:.4f}")
                    train_infos[k] = v

                cr = env0.get_collision_rate()
                for k, v in cr.items():
                    print(f"  {k}: {v:.4f}")
                    train_infos[k] = v

                self.log_train(train_infos, total_num_steps)
                self._save_csv(total_num_steps, train_infos)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        # When the round ends, the bootstrap value should be zero.
        next_values = next_values * self.buffer.masks[-1]
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
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

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env] = np.zeros(
            (dones_env.sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env] = np.zeros(
            (dones_env.sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = np.zeros((dones_env.sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        if self.eval_envs is None:
            return

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
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
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_obs, eval_share_obs, eval_rewards, eval_dones, _, eval_available_actions = \
                self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env] = np.zeros(
                (eval_dones_env.sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones_env] = 0.0

        eval_episode_rewards = np.array(eval_episode_rewards)
        avg_reward = np.mean(np.sum(eval_episode_rewards, axis=0))
        print(f"  [eval] average episode reward: {avg_reward:.4f}")

    def _save_csv(self, total_num_steps, metrics):
        fieldnames = ["total_num_steps"] + [
            k for k in sorted(metrics.keys()) if isinstance(metrics[k], (int, float, np.floating))
        ]
        row = {"total_num_steps": total_num_steps}
        for k in fieldnames[1:]:
            row[k] = metrics[k]

        with open(self.train_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)
