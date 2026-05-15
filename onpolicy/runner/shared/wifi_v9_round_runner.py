"""WiFi v9 runner that updates by completed simulation rounds."""

import time

import numpy as np

from onpolicy.runner.shared.wifi_v3_runner import WiFiV2Runner


class WiFiV9RoundRunner(WiFiV2Runner):
    """Collect a fixed number of 50 ms WiFi rounds before each PPO update.

    The replay buffer still has a fixed maximum length, but collection stops
    once every rollout environment has completed ``rounds_per_update`` rounds.
    Any unused buffer tail is masked out of policy/value losses.
    """

    def _mask_unused_buffer_tail(self, used_steps: int):
        if used_steps >= self.episode_length:
            return

        last_idx = max(used_steps, 0)
        self.buffer.share_obs[last_idx + 1 :] = self.buffer.share_obs[last_idx]
        self.buffer.obs[last_idx + 1 :] = self.buffer.obs[last_idx]
        self.buffer.rnn_states[last_idx + 1 :] = self.buffer.rnn_states[last_idx]
        self.buffer.rnn_states_critic[last_idx + 1 :] = self.buffer.rnn_states_critic[last_idx]
        self.buffer.masks[last_idx + 1 :] = 0.0
        self.buffer.bad_masks[last_idx + 1 :] = 0.0
        self.buffer.active_masks[last_idx:] = 0.0
        if self.buffer.available_actions is not None:
            self.buffer.available_actions[last_idx + 1 :] = self.buffer.available_actions[last_idx]

        self.buffer.actions[last_idx:] = 0.0
        self.buffer.action_log_probs[last_idx:] = 0.0
        self.buffer.value_preds[last_idx:] = 0.0
        self.buffer.rewards[last_idx:] = 0.0

    def run(self):
        self.warmup()

        start = time.time()
        max_rollout_steps = self.episode_length
        target_rounds = int(getattr(self.all_args, "rounds_per_update", 1))
        updates = int(self.num_env_steps) // max(max_rollout_steps * self.n_rollout_threads, 1)

        for update_idx in range(updates):
            self._maybe_update_scenario(update_idx)

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(update_idx, updates)

            infos = None
            completed_rounds = np.zeros(self.n_rollout_threads, dtype=np.int32)
            used_steps = max_rollout_steps

            for step in range(max_rollout_steps):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

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

                dones_env = np.all(dones, axis=1)
                completed_rounds += dones_env.astype(np.int32)
                if np.all(completed_rounds >= target_rounds):
                    used_steps = step + 1
                    break

            self._mask_unused_buffer_tail(used_steps)
            self.compute()
            train_infos = self.train()
            self.buffer.step = 0

            self._log_episode_reward(update_idx)
            self._log_episode_fulfillment(update_idx, infos)

            total_num_steps = (update_idx + 1) * max_rollout_steps * self.n_rollout_threads

            if update_idx % self.save_interval == 0 or update_idx == updates - 1:
                self.save()

            if update_idx % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / (end - start)) if (end - start) > 0 else 0

                active_rewards = self.buffer.rewards[self.buffer.active_masks[:-1] > 0.0]
                if active_rewards.size > 0:
                    avg_reward = float(np.mean(active_rewards))
                    avg_reward_pos = (
                        float(np.mean(active_rewards[active_rewards > 0]))
                        if (active_rewards > 0).any()
                        else 0.0
                    )
                    avg_reward_neg = (
                        float(np.mean(active_rewards[active_rewards < 0]))
                        if (active_rewards < 0).any()
                        else 0.0
                    )
                else:
                    avg_reward = avg_reward_pos = avg_reward_neg = 0.0

                flat_actions = self.buffer.actions[:used_steps].flatten().astype(int)
                transmit_ratio = (
                    float((flat_actions == 1).sum()) / float(max(len(flat_actions), 1))
                )

                avg_fulfillment = 0.0
                if infos is not None:
                    fulfillments = [
                        info.get("fulfillment", 0.0)
                        for env_infos in infos
                        for info in env_infos
                        if info.get("active", True)
                    ]
                    avg_fulfillment = float(np.mean(fulfillments)) if fulfillments else 0.0

                print(
                    f"\n[WiFi-v9] Update {update_idx}/{updates} | "
                    f"Steps {total_num_steps}/{self.num_env_steps} | FPS {fps}"
                )
                print(
                    f"  collected rounds/env: {completed_rounds.tolist()} "
                    f"(target={target_rounds}, used_steps={used_steps}/{max_rollout_steps})"
                )
                print(
                    f"  avg reward:      {avg_reward:.4f} "
                    f"(pos: {avg_reward_pos:.4f}, neg: {avg_reward_neg:.4f})"
                )
                print(f"  transmit ratio:  {transmit_ratio:.4f}")
                print(f"  avg fulfillment: {avg_fulfillment:.4f}")

                train_infos["average_step_rewards"] = avg_reward
                train_infos["average_step_rewards_pos"] = avg_reward_pos
                train_infos["average_step_rewards_neg"] = avg_reward_neg
                train_infos["transmit_ratio"] = transmit_ratio
                train_infos["avg_fulfillment"] = avg_fulfillment
                train_infos["rollout_reward/total_active"] = float(np.sum(active_rewards))
                train_infos["rollout_reward/mean_active"] = avg_reward
                train_infos["rounds/target_per_update"] = float(target_rounds)
                train_infos["rounds/used_steps"] = float(used_steps)
                train_infos["rounds/min_completed"] = float(np.min(completed_rounds))
                train_infos["rounds/max_completed"] = float(np.max(completed_rounds))

                tp = self._mean_env_metric(self.envs, "get_throughput")
                for k, v in tp.items():
                    train_infos[f"throughput/{k}"] = v

                cr = self._mean_env_metric(self.envs, "get_collision_rate")
                for k, v in cr.items():
                    train_infos[f"collision/{k}"] = v

                alloc = self._mean_env_metric(self.envs, "get_allocation_metrics")
                for k, v in alloc.items():
                    train_infos[f"allocation/{k}"] = v

                self.log_train(train_infos, total_num_steps)
                self._save_csv(total_num_steps, train_infos)

            if self.use_eval and update_idx % self.eval_interval == 0:
                self.eval(total_num_steps)
