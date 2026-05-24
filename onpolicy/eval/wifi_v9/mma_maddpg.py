"""WiFi v9 adapter for the official MMA MADDPG model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from onpolicy.eval.wifi_v9.official_mma_model import (
    MADDPG as OfficialMMAMADDPG,
    gumbel_softmax,
    onehot_from_logits,
)


class ReplayBuffer:
    def __init__(self, capacity: int, rng=None):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng() if rng is None else rng
        self.buffer = []
        self.position = 0

    def add(self, states, actions, rewards, next_states, active_mask):
        item = (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(active_mask, dtype=np.float32),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = self.rng.choice(len(self.buffer), size=int(batch_size), replace=False)
        states, actions, rewards, next_states, masks = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(masks, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class WiFiV9MMAMADDPG:
    """Thin v9 wrapper around the official MMA MADDPG model.

    The official code stores transitions as per-agent lists of tensors. WiFi v9
    stores them batch-first for convenience, then converts back before calling
    the official ``MADDPG.update`` method.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int = 2,
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
        actor_lr: float = 5e-4,
        critic_lr: float = 5e-4,
        gamma: float = 0.95,
        tau: float = 1e-2,
        batch_size: int = 64,
        buffer_size: int = 100000,
        minimal_size: int = 4000,
        learning_interval: int = 100,
        update_interval: int = 200,
        device: str | torch.device = "cpu",
        seed: int = 1,
    ):
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.critic_input_dim = self.num_agents * (self.state_dim + self.action_dim)
        self.batch_size = int(batch_size)
        self.minimal_size = int(minimal_size)
        self.learning_interval = int(learning_interval)
        self.update_interval = int(update_interval)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.model = OfficialMMAMADDPG(
            num_mld=self.num_agents,
            device=self.device,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            hidden_dim_a=actor_hidden_dim,
            hidden_dim_c=critic_hidden_dim,
            state_dims=[self.state_dim] * self.num_agents,
            action_dims=[self.action_dim] * self.num_agents,
            critic_input_dim=self.critic_input_dim,
            gamma=gamma,
            tau=tau,
        )
        self.replay_buffer = ReplayBuffer(buffer_size, rng=self.rng)
        self.total_steps = 0
        self.total_updates = 0

    @property
    def agents(self):
        return self.model.agents

    def select_actions(self, states, active_mask=None, available_actions=None, explore=True):
        states = np.asarray(states, dtype=np.float32)
        if active_mask is None:
            active_mask = np.ones(self.num_agents, dtype=bool)
        else:
            active_mask = np.asarray(active_mask).reshape(-1).astype(bool)
        if available_actions is None:
            available_actions = np.ones((self.num_agents, self.action_dim), dtype=np.float32)
        else:
            available_actions = np.asarray(available_actions, dtype=np.float32)

        actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        actions[:, 0] = 1.0
        with torch.no_grad():
            for aid, agent in enumerate(self.model.agents):
                if not active_mask[aid]:
                    continue
                legal = available_actions[aid] > 0.0
                if legal.sum() <= 1:
                    chosen = int(np.argmax(legal))
                    actions[aid] = 0.0
                    actions[aid, chosen] = 1.0
                    continue
                state_t = torch.as_tensor(states[aid : aid + 1], dtype=torch.float32, device=self.device)
                logits = agent.actor(state_t)
                illegal = torch.as_tensor(~legal, dtype=torch.bool, device=self.device).view(1, -1)
                logits = logits.masked_fill(illegal, -1e9)
                action_t = gumbel_softmax(logits) if explore else onehot_from_logits(logits, eps=0.0)
                actions[aid] = action_t.cpu().numpy()[0]
        return actions

    def _stack_sample_for_official_update(self, sample):
        states, actions, rewards, next_states, _ = sample
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        obs = [states_t[:, aid, :] for aid in range(self.num_agents)]
        act = [actions_t[:, aid, :] for aid in range(self.num_agents)]
        rew = [rewards_t[:, aid] for aid in range(self.num_agents)]
        next_obs = [next_states_t[:, aid, :] for aid in range(self.num_agents)]
        return obs, act, rew, next_obs

    def learn(self):
        self.total_steps += 1
        if len(self.replay_buffer) < self.minimal_size:
            return None
        if self.total_steps % self.learning_interval != 0:
            return None

        raw_sample = self.replay_buffer.sample(self.batch_size)
        masks = raw_sample[4]
        update_agent_indices = np.where(np.asarray(masks).sum(axis=0) > 0.0)[0]
        sample = self._stack_sample_for_official_update(raw_sample)
        losses = []
        for aid in update_agent_indices:
            update_loss = self.model.update(sample, aid)
            if update_loss is not None:
                critic_loss, actor_loss = update_loss
                losses.append(critic_loss + actor_loss)
        self.total_updates += 1
        if self.total_updates % self.update_interval == 0:
            self.model.update_all_targets()
        return float(np.mean(losses)) if losses else 0.0

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "actor_state_dicts": [agent.actor.state_dict() for agent in self.model.agents],
            "critic_state_dicts": [agent.critic.state_dict() for agent in self.model.agents],
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path, map_location=None, allow_agent_expand=False):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        checkpoint_agents = int(checkpoint.get("num_agents", len(checkpoint["actor_state_dicts"])))
        if checkpoint_agents != self.num_agents:
            if not allow_agent_expand:
                raise ValueError(
                    f"MMA checkpoint has {checkpoint_agents} agents, "
                    f"but this paper-aligned model expects {self.num_agents}. "
                    "Retrain MMA-MADDPG with the current MLD-level formulation, "
                    "or pass --allow_agent_expand for evaluation-only actor reuse."
                )
            if checkpoint_agents < 1:
                raise ValueError("MMA checkpoint does not contain any agents to expand.")
            print(
                f"[MMA-MADDPG] Expanding checkpoint agents "
                f"{checkpoint_agents} -> {self.num_agents} by cyclic actor reuse."
            )
        actor_state_dicts = checkpoint["actor_state_dicts"]
        for aid, agent in enumerate(self.model.agents):
            state_dict = actor_state_dicts[aid % checkpoint_agents]
            agent.actor.load_state_dict(state_dict)
            agent.target_actor.load_state_dict(state_dict)
        if "critic_state_dicts" in checkpoint and checkpoint_agents == self.num_agents:
            critic_state_dicts = checkpoint["critic_state_dicts"]
            for aid, agent in enumerate(self.model.agents):
                state_dict = critic_state_dicts[aid % len(critic_state_dicts)]
                agent.critic.load_state_dict(state_dict)
                agent.target_critic.load_state_dict(state_dict)
        elif "critic_state_dicts" in checkpoint:
            print(
                "[MMA-MADDPG] Skipping critic load during agent expansion; "
                "evaluation uses actors only."
            )
        return checkpoint


@dataclass
class MMALinkStateTracker:
    """Track paper-style MMA states per MLD and per link.

    The paper trains one actor/critic pair per MLD in a single-link Dec-POMDP,
    then reuses each MLD actor on every link during multi-link execution.  This
    tracker therefore stores a separate action-observation history for each
    (MLD, link), while the MADDPG model itself has only ``num_mld`` agents.
    """

    num_mld: int
    num_links: int = 2
    history_length: int = 10

    def __post_init__(self):
        self.step_dim = 8
        self.state_dim = self.step_dim * int(self.history_length)
        self.states = np.zeros((self.num_mld, self.num_links, self.state_dim), dtype=np.float32)
        self.wait_self = np.zeros((self.num_mld, self.num_links), dtype=np.float32)
        self.wait_other = np.ones((self.num_mld, self.num_links), dtype=np.float32)

    def reset(self):
        self.states.fill(0.0)
        self.wait_self.fill(0.0)
        self.wait_other.fill(1.0)

    def _link_peers(self, active_mask):
        return [idx for idx, active in enumerate(active_mask) if active]

    def build_rewards(self, env, actions_onehot, infos, active_mask, link_id: int, alpha: float):
        active_mask = np.asarray(active_mask).reshape(-1).astype(bool)
        actions_onehot = np.asarray(actions_onehot, dtype=np.float32)
        actions_binary = np.argmax(actions_onehot, axis=1)
        rewards_raw = np.zeros(self.num_mld, dtype=np.float32)
        collisions = np.zeros(self.num_mld, dtype=np.float32)
        link_obs = np.zeros((self.num_mld, 4), dtype=np.float32)
        aid_by_mld = {
            int(mld_id): aid
            for aid, (mld_id, aid_link_id) in enumerate(env.agent_to_mld_link)
            if int(aid_link_id) == int(link_id)
        }

        for mld_id in range(self.num_mld):
            if not active_mask[mld_id]:
                continue
            info = infos[aid_by_mld[mld_id]]
            result = info.get("txop_result", "")
            if result == "success":
                rewards_raw[mld_id] = 1.0 if actions_binary[mld_id] == 1 else 0.0
                link_obs[mld_id] = [1, 0, 0, 0] if actions_binary[mld_id] == 1 else [0, 1, 0, 0]
            elif result == "collision":
                collisions[mld_id] = 1.0 if actions_binary[mld_id] == 1 else 0.0
                link_obs[mld_id] = [0, 0, 1, 0]
            elif result == "idle":
                link_obs[mld_id] = [0, 0, 0, 1]
            else:
                link_obs[mld_id] = [0, 0, 0, 1]

        rewards = np.zeros(self.num_mld, dtype=np.float32)
        peers = self._link_peers(active_mask)
        wait_sum = float(np.sum(self.wait_self[peers, link_id])) if peers else 0.0
        if wait_sum > 0.0 and peers:
            peer_phi = {p: self.wait_self[p, link_id] / wait_sum for p in peers}
            max_phi = max(peer_phi.values())
            top_candidates = [
                p for p, value in peer_phi.items()
                if abs(value - max_phi) < 1e-9
            ]
            top_mld = min(top_candidates)
        else:
            top_mld = min(peers) if peers else 0

        for mld_id in range(self.num_mld):
            if not active_mask[mld_id]:
                continue
            phi = self.wait_self[mld_id, link_id] / wait_sum if wait_sum > 0.0 else 0.0
            is_max = mld_id == top_mld

            global_reward = 0.0
            if rewards_raw[mld_id] > 0.0:
                global_reward = 1.0 if is_max else phi
            elif collisions[mld_id] > 0.0:
                global_reward = -1.0

            if is_max and actions_binary[mld_id] == 1:
                individual_reward = 1.0
            elif is_max and actions_binary[mld_id] == 0:
                individual_reward = -1.0 / max(1.0 - phi, 1e-6)
            elif (not is_max) and actions_binary[mld_id] == 1:
                individual_reward = -1.0
            else:
                individual_reward = 1.0
            rewards[mld_id] = float(alpha) * global_reward + (1.0 - float(alpha)) * individual_reward

        return rewards, link_obs, collisions, rewards_raw

    def update_link(self, link_id: int, actions_onehot, link_obs, success_raw, active_mask):
        active_mask = np.asarray(active_mask).reshape(-1).astype(bool)
        self.wait_self[active_mask, link_id] += 1.0
        peers = self._link_peers(active_mask)
        for mld_id in range(self.num_mld):
            if not active_mask[mld_id]:
                continue
            other_waits = [self.wait_self[p, link_id] for p in peers if p != mld_id]
            self.wait_other[mld_id, link_id] = float(np.mean(other_waits)) if other_waits else 1.0
        for mld_id, success in enumerate(success_raw):
            if active_mask[mld_id] and success > 0.0:
                self.wait_self[mld_id, link_id] = 0.0

        wait_self = self.wait_self[:, link_id]
        wait_other = self.wait_other[:, link_id]
        denom = wait_self + wait_other
        ci = np.divide(wait_self, denom, out=np.zeros_like(wait_self), where=denom > 0)
        cminus = np.divide(wait_other, denom, out=np.zeros_like(wait_other), where=denom > 0)
        step_features = np.concatenate(
            [
                np.asarray(actions_onehot, dtype=np.float32),
                np.asarray(link_obs, dtype=np.float32),
                ci[:, None].astype(np.float32),
                cminus[:, None].astype(np.float32),
            ],
            axis=1,
        )
        self.states[:, link_id, :] = np.concatenate(
            [self.states[:, link_id, self.step_dim :], step_features],
            axis=1,
        )
        return self.states[:, link_id, :].copy()


MMAStateTracker = MMALinkStateTracker
