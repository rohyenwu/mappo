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
    if isinstance(x, np.ndarray):
        return x
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

    def _log_episode_reward(self, episode_idx):
        """Log episode-level reward so convergence can be viewed on an episode axis."""
        episode_reward_total = float(np.sum(self.buffer.rewards))
        episode_reward_mean = float(np.mean(self.buffer.rewards))
        episode_reward_per_agent = episode_reward_total / max(
            self.n_rollout_threads * self.num_agents, 1
        )

        if self.use_wandb:
            import wandb

            wandb.log(
                {
                    "episode": episode_idx + 1,
                    "episode_reward/total": episode_reward_total,
                    "episode_reward/mean_step_agent": episode_reward_mean,
                    "episode_reward/per_agent": episode_reward_per_agent,
                }
            )
        else:
            self.writter.add_scalar("episode_reward/total", episode_reward_total, episode_idx + 1)
            self.writter.add_scalar(
                "episode_reward/mean_step_agent", episode_reward_mean, episode_idx + 1
            )
            self.writter.add_scalar(
                "episode_reward/per_agent", episode_reward_per_agent, episode_idx + 1
            )

    def _log_episode_fulfillment(self, episode_idx, infos):
        """Log per-episode final fulfillment for each agent/MLD on an episode axis."""
        if infos is None:
            return

        # `infos` is from the last step of the episode, so fulfillment here is the
        # final fulfillment for the just-finished round.
        agent_fulfillments = {aid: [] for aid in range(self.num_agents)}
        for env_infos in infos:
            for aid, info in enumerate(env_infos):
                agent_fulfillments[aid].append(info.get("fulfillment", 0.0))

        if self.use_wandb:
            import wandb

            payload = {}
            for aid, values in agent_fulfillments.items():
                if values:
                    payload[f"episode_fulfillment/agent_{aid}"] = float(np.mean(values))

            num_mld = self.num_agents // 2
            for mld_id in range(num_mld):
                aid_24 = 2 * mld_id
                aid_5 = 2 * mld_id + 1
                per_mld_values = []

                if agent_fulfillments[aid_24]:
                    payload[f"episode_fulfillment/mld_{mld_id}/2_4GHz"] = float(
                        np.mean(agent_fulfillments[aid_24])
                    )
                    per_mld_values.extend(agent_fulfillments[aid_24])
                if agent_fulfillments[aid_5]:
                    payload[f"episode_fulfillment/mld_{mld_id}/5GHz"] = float(
                        np.mean(agent_fulfillments[aid_5])
                    )
                    per_mld_values.extend(agent_fulfillments[aid_5])
                if per_mld_values:
                    payload[f"episode_fulfillment/mld_{mld_id}/avg"] = float(
                        np.mean(per_mld_values)
                    )

            if payload:
                wandb.log(payload)
        else:
            for aid, values in agent_fulfillments.items():
                if values:
                    self.writter.add_scalar(
                        f"episode_fulfillment/agent_{aid}",
                        float(np.mean(values)),
                        episode_idx + 1,
                    )

            num_mld = self.num_agents // 2
            for mld_id in range(num_mld):
                aid_24 = 2 * mld_id
                aid_5 = 2 * mld_id + 1
                per_mld_values = []

                if agent_fulfillments[aid_24]:
                    val_24 = float(np.mean(agent_fulfillments[aid_24]))
                    self.writter.add_scalar(
                        f"episode_fulfillment/mld_{mld_id}/2_4GHz", val_24, episode_idx + 1
                    )
                    per_mld_values.extend(agent_fulfillments[aid_24])
                if agent_fulfillments[aid_5]:
                    val_5 = float(np.mean(agent_fulfillments[aid_5]))
                    self.writter.add_scalar(
                        f"episode_fulfillment/mld_{mld_id}/5GHz", val_5, episode_idx + 1
                    )
                    per_mld_values.extend(agent_fulfillments[aid_5])
                if per_mld_values:
                    self.writter.add_scalar(
                        f"episode_fulfillment/mld_{mld_id}/avg",
                        float(np.mean(per_mld_values)),
                        episode_idx + 1,
                    )

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
            self._log_episode_reward(episode)
            self._log_episode_fulfillment(episode, infos)

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

                if infos is not None:
                    # Agent-wise fulfillment so we can inspect who is starved.
                    agent_fulfillments = {aid: [] for aid in range(self.num_agents)}
                    for env_infos in infos:
                        for aid, info in enumerate(env_infos):
                            agent_fulfillments[aid].append(info.get("fulfillment", 0.0))

                    for aid, values in agent_fulfillments.items():
                        if values:
                            train_infos[f"fulfillment/agent_{aid}"] = float(np.mean(values))

                    # MLD-wise fulfillment averaged across the two links.
                    num_mld = self.num_agents // 2
                    for mld_id in range(num_mld):
                        per_mld_values = []
                        aid_24 = 2 * mld_id
                        aid_5 = 2 * mld_id + 1
                        if agent_fulfillments[aid_24]:
                            train_infos[f"fulfillment/mld_{mld_id}/2_4GHz"] = float(
                                np.mean(agent_fulfillments[aid_24])
                            )
                            per_mld_values.extend(agent_fulfillments[aid_24])
                        if agent_fulfillments[aid_5]:
                            train_infos[f"fulfillment/mld_{mld_id}/5GHz"] = float(
                                np.mean(agent_fulfillments[aid_5])
                            )
                            per_mld_values.extend(agent_fulfillments[aid_5])
                        if per_mld_values:
                            train_infos[f"fulfillment/mld_{mld_id}/avg"] = float(
                                np.mean(per_mld_values)
                            )

                    reward_keys = [
                        "reward/global",
                        "reward/local",
                        "reward/dense",
                        "reward/sparse",
                        "reward/total",
                        "reward/link_0/global",
                        "reward/link_0/local",
                        "reward/link_0/dense",
                        "reward/link_0/sparse",
                        "reward/link_0/total",
                        "reward/link_1/global",
                        "reward/link_1/local",
                        "reward/link_1/dense",
                        "reward/link_1/sparse",
                        "reward/link_1/total",
                    ]
                    for key in reward_keys:
                        values = []
                        for env_infos in infos:
                            for info in env_infos:
                                if key in info:
                                    values.append(info[key])
                        if values:
                            train_infos[key] = float(np.mean(values))

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
