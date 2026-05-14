"""
WiFi v2 environment runner for MAPPO.

Design reference: docs/project_wifi_redesign_v4.md
"""
import csv
import os
import time
from datetime import datetime
from pathlib import Path

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

    def _write_model_metadata(self):
        metadata_path = Path(self.save_dir) / "MODEL_METADATA.md"
        saved_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        content = f"""# WiFi_v3 Model Metadata

Saved at: {saved_at}

## Run
- env_name: {self.all_args.env_name}
- algorithm_name: {self.all_args.algorithm_name}
- experiment_name: {self.all_args.experiment_name}
- seed: {self.all_args.seed}
- num_mld: {self.all_args.num_mld}
- num_sld: {self.all_args.num_sld}
- round_length: {self.all_args.round_length}
- rounds_per_update: {getattr(self.all_args, "rounds_per_update", 1)}
- rollout_length: {self.all_args.episode_length}

## Local Observation (Actor Input)
- shared_load: current MLD shared demand normalized by `round_length * num_links`
- n_sld_norm: normalized SLD count on the current link
- shared_fulfillment: current shared fulfillment of the MLD
- prev_action_self: this agent's previous action (`0=skip`, `1=transmit`)
- prev_action_peer: paired same-MLD agent's previous action
- link_onehot: link identity (`[1,0]` for 2.4 GHz, `[0,1]` for 5 GHz)

## Shared Observation (Critic Input)
- flattened local observations of all agents on the same link
- current average SLD success ratio on 2.4 GHz (`0` on 5 GHz critic input)
- same-link shared fulfillment vector
- link one-hot

## Reward
### Global Reward
- MLD success by top-urgency agent: `+1`
- MLD success by non-top agent: `urgency(success_agent)`
- SLD success: `0`
- collision: `-1`
- idle: `-c_idle`

### Local Reward
- shared queue already satisfied and transmit: `-2`
- shared queue already satisfied and skip: `0`
- no urgent agent on the link and transmit: `-2`
- no urgent agent on the link and skip: `0`
- top-urgency agent transmit: `+1`
- top-urgency agent skip: `-(1 + urgency)`
- non-top agent transmit: `-2`
- non-top agent skip: `+1`

### Sparse Reward
- applied only on the final step of the round
- only affects 2.4 GHz agents
- if SLD throughput is below threshold: penalize above-average 2.4 GHz participation
- otherwise: reward above-average skipping / yielding on 2.4 GHz

## Allocation Metric
- expected share: each MLD's round demand divided by total round demand
- actual share: each MLD's round success count divided by total round success
- allocation/match: `1 - mean(abs(expected_share - actual_share))`
"""
        metadata_path.write_text(content, encoding="utf-8")

    def save(self):
        super().save()
        self._write_model_metadata()

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

    def _mean_env_metric(self, env_collection, method_name):
        """Average a dict metric returned by each underlying environment."""
        if env_collection is None:
            return {}

        if hasattr(env_collection, "get_env_metrics"):
            metric_dicts = env_collection.get_env_metrics(method_name)
        elif hasattr(env_collection, "envs"):
            metric_dicts = [
                getattr(env, method_name)()
                for env in env_collection.envs
                if hasattr(env, method_name)
            ]
        else:
            metric_dicts = []

        aggregated = {}
        for metrics in metric_dicts:
            for key, value in metrics.items():
                aggregated.setdefault(key, []).append(float(value))

        if not aggregated:
            return {}

        return {key: float(np.mean(values)) for key, values in aggregated.items()}

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            self._maybe_update_scenario(episode)

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

                tp = self._mean_env_metric(self.envs, "get_throughput")
                for k, v in tp.items():
                    print(f"  {k}: {v:.4f}")
                    train_infos[k] = v

                cr = self._mean_env_metric(self.envs, "get_collision_rate")
                for k, v in cr.items():
                    print(f"  {k}: {v:.4f}")
                    train_infos[k] = v

                alloc = self._mean_env_metric(self.envs, "get_allocation_metrics")
                for k, v in alloc.items():
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
        self.buffer.active_masks[0] = self._get_env_active_masks()

    def _get_env_active_masks(self):
        if hasattr(self.envs, "get_active_masks"):
            active_masks = self.envs.get_active_masks()
            if active_masks is not None:
                return active_masks.copy()
        return np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

    def _set_buffer_start(self, obs, share_obs, available_actions):
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()
        self.buffer.active_masks[0] = self._get_env_active_masks()
        self.buffer.rnn_states[0] = np.zeros_like(self.buffer.rnn_states[0])
        self.buffer.rnn_states_critic[0] = np.zeros_like(self.buffer.rnn_states_critic[0])

    def _maybe_update_scenario(self, episode):
        if getattr(self.all_args, "scenario_order", "sequential") == "parallel":
            return
        if not hasattr(self.envs, "set_scenario_by_episode"):
            return
        interval = getattr(self.all_args, "scenario_interval_episodes", None)
        if interval is None:
            return
        result = self.envs.set_scenario_by_episode(episode, interval)
        if result is None:
            return
        obs, share_obs, available_actions = result
        self._set_buffer_start(obs, share_obs, available_actions)

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

        active_masks = self._get_env_active_masks()
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

        eval_alloc = self._mean_env_metric(self.eval_envs, "get_allocation_metrics")
        for k, v in eval_alloc.items():
            print(f"  [eval] {k}: {v:.4f}")

        if eval_alloc:
            self.log_train(
                {f"eval/{k}": v for k, v in eval_alloc.items()},
                total_num_steps,
            )

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
