"""SETL-DQN(MA) baseline components for WiFi v9.

This module reimplements the core algorithmic pieces from Ke and Astuti's
SETL-DQN(MA): per-station DQN selection of a CW threshold, SETL backoff
updates, collision-history observations, and cooperative throughput reward.
The WiFi v9 scripts attach these pieces to the local STR MLD/SLD environment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CW_MIN = 16
CW_MAX = 1024
SETL_LINEAR_STEP = 32


@dataclass
class SETLBackoffState:
    cw: int = CW_MIN
    backoff: int = 0
    cw_threshold: int = 512
    success_count: int = 0
    collision_count: int = 0
    collision_prev: float = 0.0
    collision_curr: float = 0.0
    last_obs: np.ndarray | None = None
    last_action: int = 0


class SETLMLDBackoffMAC:
    """SETL backoff controller with per-agent CW-threshold actions."""

    def __init__(
        self,
        num_agents: int,
        agent_to_mld_link,
        thresholds: Iterable[int] = (128, 256, 384, 512, 640, 768, 896, 1024),
        cw_min: int = CW_MIN,
        cw_max: int = CW_MAX,
        linear_step: int = SETL_LINEAR_STEP,
        rng=None,
    ):
        self.num_agents = int(num_agents)
        self.agent_to_mld_link = list(agent_to_mld_link)
        self.thresholds = [int(value) for value in thresholds]
        if not self.thresholds:
            raise ValueError("thresholds must contain at least one CW threshold")
        self.cw_min = int(cw_min)
        self.cw_max = int(cw_max)
        self.linear_step = int(linear_step)
        self.rng = np.random.default_rng() if rng is None else rng
        self.states = [SETLBackoffState(cw=self.cw_min) for _ in range(self.num_agents)]

    @property
    def obs_dim(self) -> int:
        return 2

    @property
    def act_dim(self) -> int:
        return len(self.thresholds)

    def _draw_backoff(self, cw: int) -> int:
        return int(self.rng.integers(0, max(int(cw), 1)))

    def _pending(self, env, aid: int) -> bool:
        mld_id, _ = self.agent_to_mld_link[aid]
        return bool(env.D[mld_id] > env.S[mld_id])

    def _active(self, env, aid: int) -> bool:
        if hasattr(env, "_is_active_agent"):
            return bool(env._is_active_agent(aid))
        return True

    def reset_round(self, env):
        for aid, state in enumerate(self.states):
            state.cw = self.cw_min
            state.backoff = self._draw_backoff(state.cw) if self._pending(env, aid) else 0
            state.cw_threshold = self.thresholds[min(state.last_action, self.act_dim - 1)]
            state.success_count = 0
            state.collision_count = 0
            state.collision_prev = 0.0
            state.collision_curr = 0.0
            state.last_obs = None

    def observations(self, env) -> np.ndarray:
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        for aid, state in enumerate(self.states):
            if self._active(env, aid):
                obs[aid] = [state.collision_curr, state.collision_prev]
        return obs

    def act(self, env, threshold_actions=None):
        actions = np.zeros((self.num_agents, 1), dtype=np.int32)
        pending_mask = np.zeros(self.num_agents, dtype=bool)
        if threshold_actions is not None:
            threshold_actions = np.asarray(threshold_actions).reshape(-1)

        for aid, state in enumerate(self.states):
            pending = self._pending(env, aid)
            pending_mask[aid] = pending

            if threshold_actions is not None and self._active(env, aid):
                action_idx = int(np.clip(threshold_actions[aid], 0, self.act_dim - 1))
                state.last_action = action_idx
                state.cw_threshold = self.thresholds[action_idx]

            if pending and state.backoff == 0:
                actions[aid, 0] = 1
        return actions, pending_mask

    def update(self, env, actions, infos, pending_mask):
        actions_flat = actions.reshape(-1)
        attempted = np.zeros(self.num_agents, dtype=bool)
        collided = np.zeros(self.num_agents, dtype=bool)

        for aid, state in enumerate(self.states):
            pending_before = bool(pending_mask[aid])
            pending_after = self._pending(env, aid)

            if not pending_before:
                state.cw = self.cw_min
                state.backoff = 0
                continue

            result = infos[aid].get("txop_result", "")
            transmitted = actions_flat[aid] == 1
            attempted[aid] = transmitted
            collided[aid] = transmitted and result == "collision"

            if transmitted:
                if result == "success":
                    state.success_count += 1
                    state.cw = max(self.cw_min, int(np.ceil(state.cw / 2.0)))
                    state.backoff = self._draw_backoff(state.cw) if pending_after else 0
                elif result == "collision":
                    state.collision_count += 1
                    if state.cw < state.cw_threshold:
                        state.cw = min(state.cw * 2, self.cw_max)
                    else:
                        state.cw = min(state.cw + self.linear_step, self.cw_max)
                    state.backoff = self._draw_backoff(state.cw) if pending_after else 0
                elif not pending_after:
                    state.cw = self.cw_min
                    state.backoff = 0
            elif not pending_after:
                state.cw = self.cw_min
                state.backoff = 0
            elif result == "idle" and state.backoff > 0:
                state.backoff -= 1

        for aid, state in enumerate(self.states):
            state.collision_prev = state.collision_curr
            denom = state.collision_count + state.success_count
            state.collision_curr = (
                float(state.collision_count) / float(denom) if denom > 0 else 0.0
            )

        return attempted, collided


class ReplayMemory:
    def __init__(self, capacity: int, rng=None):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng() if rng is None else rng
        self.buffer = []
        self.position = 0

    def append(self, obs, action, reward, next_obs, done):
        item = (
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = self.rng.choice(len(self.buffer), size=int(batch_size), replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        hidden_layers: int = 3,
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class SharedDQNAgent:
    """Parameter-shared DQN used by all SETL-DQN(MA) station agents."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 32,
        memory_size: int = 20000,
        min_replay_size: int = 100,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.0,
        epsilon_decay: float = 1e-6,
        device: str | torch.device = "cpu",
        seed: int = 1,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update_interval = int(target_update_interval)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.policy_net = DQNNetwork(obs_dim, act_dim, hidden_size, hidden_layers).to(self.device)
        self.target_net = DQNNetwork(obs_dim, act_dim, hidden_size, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=float(lr))
        self.memory = ReplayMemory(memory_size, rng=self.rng)
        self.learn_steps = 0

    def select_actions(self, obs: np.ndarray, active_mask=None, explore: bool = True):
        obs = np.asarray(obs, dtype=np.float32)
        if active_mask is None:
            active_mask = np.ones(obs.shape[0], dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            greedy = self.policy_net(obs_t).argmax(dim=1).cpu().numpy()

        actions = greedy.astype(np.int64)
        if explore and self.epsilon > 0.0:
            random_mask = self.rng.random(obs.shape[0]) < self.epsilon
            random_actions = self.rng.integers(0, self.act_dim, size=obs.shape[0])
            actions = np.where(random_mask & active_mask, random_actions, actions)
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        actions[~active_mask] = 0
        return actions

    def remember_batch(self, obs, actions, reward, next_obs, done, active_mask):
        active_mask = np.asarray(active_mask, dtype=bool)
        for aid in np.flatnonzero(active_mask):
            self.memory.append(obs[aid], actions[aid], reward, next_obs[aid], done)

    def learn(self):
        if len(self.memory) < max(self.min_replay_size, self.batch_size):
            return None

        obs, actions, rewards, next_obs, dones = self.memory.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=1).values
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, extra=None):
        payload = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str | Path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        target_state = checkpoint.get("target_state_dict", checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(target_state)
        self.target_net.eval()
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.learn_steps = int(checkpoint.get("learn_steps", self.learn_steps))
        return checkpoint
