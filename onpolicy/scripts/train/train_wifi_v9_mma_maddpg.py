#!/usr/bin/env python
"""Train an MMA/MADDPG WiFi v9 baseline."""

import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.wifi_v9.wifi_env import WiFiEnvV9
from onpolicy.eval.wifi_v5.utils import parse_mu_profile
from onpolicy.eval.wifi_v9.mma_maddpg import MMALinkStateTracker, WiFiV9MMAMADDPG


def parse_scenario_profile(profile_text):
    scenarios = []
    for raw_part in str(profile_text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        mld_text, sld_text = part.split(":", 1)
        scenarios.append((int(mld_text), int(sld_text)))
    if not scenarios:
        raise ValueError("--scenario_profile must contain at least one MLD:SLD pair")
    return scenarios


def make_wifi_env(args, seed: int, scenario):
    env = WiFiEnvV9(
        max_mld=args.max_mld,
        max_sld=args.max_sld,
        scenario_profile=[scenario],
        round_length=args.round_length,
        mu_range=(args.mu_min, args.mu_max),
        mu_profile=parse_mu_profile(getattr(args, "mu_profile", None)),
        eta=args.eta,
        zeta=args.zeta,
        r_sld=args.r_sld,
        c_idle=args.c_idle,
        theta_scale=args.theta_scale,
        sld_target_low_scale=args.sld_target_low_scale,
        sld_target_high_scale=args.sld_target_high_scale,
        sld_target_bonus=args.sld_target_bonus,
        mld_success_reward=args.mld_success_reward,
        collision_penalty=args.collision_penalty,
        non_top_tx_penalty=args.non_top_tx_penalty,
        slot_time_sec=args.slot_time_sec,
        episode_duration_sec=args.episode_duration_sec,
    )
    env.seed(seed)
    return env


def select_device(args):
    if args.cuda and torch.cuda.is_available():
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return torch.device("cuda:0")
    torch.set_num_threads(args.n_training_threads)
    return torch.device("cpu")


def build_run_dir(args):
    repo_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]).parents[1]
    run_dir = repo_dir / "model" / args.env_name / "mma_maddpg" / f"{args.experiment_name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def init_wandb(args, run_dir):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(
        config=vars(args),
        project=getattr(args, "wandb_project", "wifi_v9_mma_maddpg"),
        entity=getattr(args, "wandb_entity", None) or args.user_name,
        notes=socket.gethostname(),
        name=getattr(args, "wandb_run_name", None) or f"mma_maddpg_{args.experiment_name}_seed{args.seed}",
        group=getattr(args, "wandb_group", None) or "mma_maddpg",
        dir=str(run_dir),
        job_type="train",
        reinit=True,
    )


def parse_args(args, parser):
    parser.add_argument("--max_mld", type=int, default=30)
    parser.add_argument("--max_sld", type=int, default=10)
    parser.add_argument("--scenario_profile", type=str, default="15:3,15:5,25:3,25:5")
    parser.add_argument("--round_length", type=int, default=500)
    parser.add_argument("--mu_min", type=float, default=0.01)
    parser.add_argument("--mu_max", type=float, default=0.12)
    parser.add_argument("--mu_profile", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--zeta", type=float, default=0.2)
    parser.add_argument("--r_sld", type=float, default=0.3)
    parser.add_argument("--c_idle", type=float, default=0.3)
    parser.add_argument("--theta_scale", type=float, default=1.0)
    parser.add_argument("--sld_target_low_scale", type=float, default=0.7)
    parser.add_argument("--sld_target_high_scale", type=float, default=1.0)
    parser.add_argument("--sld_target_bonus", type=float, default=0.5)
    parser.add_argument("--mld_success_reward", type=float, default=1.0)
    parser.add_argument("--collision_penalty", type=float, default=1.0)
    parser.add_argument("--non_top_tx_penalty", type=float, default=1.0)
    parser.add_argument("--slot_time_sec", type=float, default=9e-6)
    parser.add_argument("--episode_duration_sec", type=float, default=0.05)
    parser.add_argument("--mma_train_episodes", type=int, default=50000)
    parser.add_argument("--mma_history_length", type=int, default=10)
    parser.add_argument("--mma_actor_hidden_dim", type=int, default=64)
    parser.add_argument("--mma_critic_hidden_dim", type=int, default=128)
    parser.add_argument("--mma_actor_lr", type=float, default=5e-4)
    parser.add_argument("--mma_critic_lr", type=float, default=5e-4)
    parser.add_argument("--mma_gamma", type=float, default=0.95)
    parser.add_argument("--mma_tau", type=float, default=1e-2)
    parser.add_argument("--mma_alpha", type=float, default=0.3)
    parser.add_argument("--mma_batch_size", type=int, default=64)
    parser.add_argument("--mma_buffer_size", type=int, default=100000)
    parser.add_argument("--mma_minimal_size", type=int, default=4000)
    parser.add_argument("--mma_learning_interval", type=int, default=100)
    parser.add_argument("--mma_update_interval", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="wifi_v9_mma_maddpg")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_known_args(args)[0]


def ready_active_mld_mask(env, link_id: int):
    active = np.zeros(env.max_mld, dtype=bool)
    active[: int(env.active_mld)] = bool(env.last_ready_links[link_id])
    return active


def onehot_to_env_actions(actions_onehot):
    return np.argmax(actions_onehot, axis=1).astype(np.int32).reshape(-1, 1)


def build_env_actions(env, actions_by_link):
    env_actions = np.zeros(env.num_agents, dtype=np.int32)
    for aid, (mld_id, link_id) in enumerate(env.agent_to_mld_link):
        if mld_id < env.active_mld and env.last_ready_links[link_id]:
            env_actions[aid] = int(np.argmax(actions_by_link[mld_id, link_id]))
    return env_actions.reshape(-1, 1)


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)

    scenarios = parse_scenario_profile(all_args.scenario_profile)
    device = select_device(all_args)
    run_dir = Path(all_args.save_dir) if all_args.save_dir else build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    run = init_wandb(all_args, run_dir)

    template_env = make_wifi_env(all_args, all_args.seed, scenarios[0])
    tracker = MMALinkStateTracker(
        num_mld=template_env.max_mld,
        num_links=template_env.num_links,
        history_length=all_args.mma_history_length,
    )
    maddpg = WiFiV9MMAMADDPG(
        num_agents=template_env.max_mld,
        state_dim=tracker.state_dim,
        actor_hidden_dim=all_args.mma_actor_hidden_dim,
        critic_hidden_dim=all_args.mma_critic_hidden_dim,
        actor_lr=all_args.mma_actor_lr,
        critic_lr=all_args.mma_critic_lr,
        gamma=all_args.mma_gamma,
        tau=all_args.mma_tau,
        batch_size=all_args.mma_batch_size,
        buffer_size=all_args.mma_buffer_size,
        minimal_size=all_args.mma_minimal_size,
        learning_interval=all_args.mma_learning_interval,
        update_interval=all_args.mma_update_interval,
        device=device,
        seed=all_args.seed,
    )
    template_env.close()

    total_episodes = int(all_args.mma_train_episodes)
    log_interval = max(int(all_args.log_interval), 1)
    save_interval = max(int(all_args.save_interval), 1)

    for episode in range(total_episodes):
        scenario = scenarios[episode % len(scenarios)]
        env = make_wifi_env(all_args, all_args.seed + episode, scenario)
        env.reset()
        tracker.reset()
        episode_reward = 0.0
        losses = []
        transmit_count = 0
        action_count = 0
        next_arrival_step = int(all_args.round_length)

        while True:
            states = tracker.states.copy()
            masks_by_link = [
                ready_active_mld_mask(env, link_id)
                for link_id in range(env.num_links)
            ]
            actions_by_link = np.zeros((env.max_mld, env.num_links, 2), dtype=np.float32)
            actions_by_link[:, :, 0] = 1.0
            for link_id, mask in enumerate(masks_by_link):
                if not np.any(mask):
                    continue
                actions_by_link[:, link_id, :] = maddpg.select_actions(
                    states[:, link_id, :],
                    active_mask=mask,
                    explore=True,
                )
            env_actions = build_env_actions(env, actions_by_link)
            transmit_count += int(env_actions.sum())
            action_count += int(sum(mask.sum() for mask in masks_by_link))
            _, _, _, dones, infos, _ = env.step(env_actions)

            for link_id, mask in enumerate(masks_by_link):
                if not np.any(mask):
                    continue
                rewards, link_obs, _, success_raw = tracker.build_rewards(
                    env,
                    actions_by_link[:, link_id, :],
                    infos,
                    mask,
                    link_id,
                    all_args.mma_alpha,
                )
                next_states = tracker.update_link(
                    link_id,
                    actions_by_link[:, link_id, :],
                    link_obs,
                    success_raw,
                    mask,
                )
                maddpg.replay_buffer.add(
                    states[:, link_id, :],
                    actions_by_link[:, link_id, :],
                    rewards,
                    next_states,
                    mask,
                )
                loss = maddpg.learn()
                if loss is not None:
                    losses.append(loss)
                episode_reward += float(np.sum(rewards[mask]))

            while env.t >= next_arrival_step and not bool(np.all(dones)):
                env.add_packet_arrivals(all_args.round_length)
                next_arrival_step += int(all_args.round_length)
            if bool(np.all(dones)):
                break

        metrics = {
            "episode": float(episode + 1),
            "reward/episode": float(episode_reward),
            "loss/maddpg": float(np.mean(losses)) if losses else 0.0,
            "replay/size": float(len(maddpg.replay_buffer)),
            "action/transmit_ratio": float(transmit_count / max(action_count, 1)),
            "scenario/active_mld": float(scenario[0]),
            "scenario/active_sld": float(scenario[1]),
        }
        if run is not None:
            import wandb

            wandb.log(metrics, step=episode + 1)
        if (episode + 1) % log_interval == 0 or episode == 0:
            print(
                f"[MMA-MADDPG] Episode {episode + 1}/{total_episodes} | "
                f"reward={episode_reward:.3f} | "
                f"loss={metrics['loss/maddpg']:.6f} | "
                f"replay={len(maddpg.replay_buffer)} | "
                f"tx_ratio={metrics['action/transmit_ratio']:.4f}"
            )
        if (episode + 1) % save_interval == 0:
            maddpg.save(run_dir / "mma_maddpg.pt", extra={"args": vars(all_args)})
        env.close()

    maddpg.save(run_dir / "mma_maddpg.pt", extra={"args": vars(all_args)})
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
