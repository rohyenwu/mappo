import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
import numpy as np


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic_24 = R_Critic(args, self.share_obs_space, self.device)
        self.critic_5 = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        critic_params = list(self.critic_24.parameters()) + list(self.critic_5.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params,
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def _route_critics(self, cent_obs, rnn_states_critic, masks):
        cent_obs_np = np.asarray(cent_obs)
        link_bits_np = cent_obs_np[:, -2:]
        idx_24_np = np.flatnonzero(link_bits_np[:, 0] >= link_bits_np[:, 1])
        idx_5_np = np.flatnonzero(link_bits_np[:, 0] < link_bits_np[:, 1])

        cent_obs_t = torch.as_tensor(cent_obs, dtype=torch.float32, device=self.device)
        rnn_states_t = torch.as_tensor(rnn_states_critic, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)

        total_rows = cent_obs_t.shape[0]
        state_rows = rnn_states_t.shape[0]

        # Feed-forward / rollout path: one critic state per sample row.
        if state_rows == total_rows:
            values = torch.zeros((total_rows, 1), dtype=torch.float32, device=self.device)
            next_rnn_states = rnn_states_t.clone()

            if idx_24_np.size > 0:
                idx_24 = torch.as_tensor(idx_24_np, dtype=torch.long, device=self.device)
                values_24, next_states_24 = self.critic_24(
                    cent_obs_t[idx_24], rnn_states_t[idx_24], masks_t[idx_24]
                )
                values[idx_24] = values_24
                next_rnn_states[idx_24] = next_states_24

            if idx_5_np.size > 0:
                idx_5 = torch.as_tensor(idx_5_np, dtype=torch.long, device=self.device)
                values_5, next_states_5 = self.critic_5(
                    cent_obs_t[idx_5], rnn_states_t[idx_5], masks_t[idx_5]
                )
                values[idx_5] = values_5
                next_rnn_states[idx_5] = next_states_5

            return values, next_rnn_states

        # Recurrent training path: one critic state per sequence chunk, cent_obs is flattened L*N.
        if total_rows % state_rows != 0:
            raise ValueError(
                f"Unexpected critic routing shapes: cent_obs rows={total_rows}, "
                f"rnn_state rows={state_rows}"
            )

        seq_len = total_rows // state_rows
        cent_obs_seq = cent_obs_t.view(seq_len, state_rows, -1)
        masks_seq = masks_t.view(seq_len, state_rows, -1)

        first_rows = np.arange(state_rows, dtype=np.int64)
        idx_24_np = first_rows[link_bits_np[:state_rows, 0] >= link_bits_np[:state_rows, 1]]
        idx_5_np = first_rows[link_bits_np[:state_rows, 0] < link_bits_np[:state_rows, 1]]

        values_seq = torch.zeros((seq_len, state_rows, 1), dtype=torch.float32, device=self.device)
        next_rnn_states = rnn_states_t.clone()

        if idx_24_np.size > 0:
            idx_24 = torch.as_tensor(idx_24_np, dtype=torch.long, device=self.device)
            cent_obs_24 = cent_obs_seq[:, idx_24, :].reshape(seq_len * idx_24_np.size, -1)
            masks_24 = masks_seq[:, idx_24, :].reshape(seq_len * idx_24_np.size, -1)
            values_24, next_states_24 = self.critic_24(
                cent_obs_24, rnn_states_t[idx_24], masks_24
            )
            values_seq[:, idx_24, :] = values_24.view(seq_len, idx_24_np.size, -1)
            next_rnn_states[idx_24] = next_states_24

        if idx_5_np.size > 0:
            idx_5 = torch.as_tensor(idx_5_np, dtype=torch.long, device=self.device)
            cent_obs_5 = cent_obs_seq[:, idx_5, :].reshape(seq_len * idx_5_np.size, -1)
            masks_5 = masks_seq[:, idx_5, :].reshape(seq_len * idx_5_np.size, -1)
            values_5, next_states_5 = self.critic_5(
                cent_obs_5, rnn_states_t[idx_5], masks_5
            )
            values_seq[:, idx_5, :] = values_5.view(seq_len, idx_5_np.size, -1)
            next_rnn_states[idx_5] = next_states_5

        return values_seq.reshape(total_rows, -1), next_rnn_states

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self._route_critics(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self._route_critics(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self._route_critics(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def get_action_probs(self, obs, rnn_states_actor, masks, available_actions=None):
        """
        Compute action probabilities for the actor policy.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent.
        """
        return self.actor.get_probs(obs, rnn_states_actor, masks, available_actions)
