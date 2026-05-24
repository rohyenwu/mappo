#!/usr/bin/env python
"""Evaluate a trained MMA/MADDPG WiFi v9 baseline on Mbps metrics."""

import sys

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.wifi_v9.wifi_env import WiFiEnvV9
from onpolicy.eval.wifi_common.mbps_metrics import (
    MbpsAccumulator,
    MbpsTimeModel,
    add_mu_representative_metrics,
    infer_link_events,
    save_mbps_bar_chart,
)
from onpolicy.eval.wifi_v5.utils import (
    build_eval_run_dir,
    finalize_wandb,
    init_wandb,
    log_episode_metrics,
    log_wandb_image,
    parse_mu_profile,
    save_summary,
    summarize_metrics,
)
from onpolicy.eval.wifi_v9.mma_maddpg import MMALinkStateTracker, WiFiV9MMAMADDPG


def make_wifi_env(args, seed: int):
    env = WiFiEnvV9(
        max_mld=args.max_mld,
        max_sld=args.max_sld,
        scenario_profile=[(args.num_mld, args.num_sld)],
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
        episode_duration_sec=args.eval_duration_sec * 100.0,
    )
    env.seed(seed)
    return env


def build_time_model(args):
    return MbpsTimeModel(
        eval_duration_sec=args.eval_duration_sec,
        slot_time_sec=args.slot_time_sec,
        phy_preamble_sec=args.phy_preamble_sec,
        sifs_sec=args.sifs_sec,
        difs_sec=args.difs_sec,
        ack_bits=args.ack_bits,
        payload_bits=args.payload_bits,
        mac_header_bits=args.mac_header_bits,
        basic_rate_bps=args.basic_rate_bps,
        data_rate_24_bps=args.data_rate_24_bps,
        data_rate_5_bps=args.data_rate_5_bps,
    )


def parse_args(args, parser):
    parser.add_argument("--num_mld", type=int, default=10)
    parser.add_argument("--num_sld", type=int, default=2)
    parser.add_argument("--max_mld", type=int, default=30)
    parser.add_argument("--max_sld", type=int, default=10)
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
    parser.add_argument("--eval_duration_sec", type=float, default=30.0)
    parser.add_argument("--slot_time_sec", type=float, default=9e-6)
    parser.add_argument("--phy_preamble_sec", type=float, default=20e-6)
    parser.add_argument("--sifs_sec", type=float, default=16e-6)
    parser.add_argument("--difs_sec", type=float, default=34e-6)
    parser.add_argument("--ack_bits", type=float, default=112.0)
    parser.add_argument("--payload_bits", type=float, default=131072.0)
    parser.add_argument("--mac_header_bits", type=float, default=288.0)
    parser.add_argument("--basic_rate_bps", type=float, default=24e6)
    parser.add_argument("--data_rate_24_bps", type=float, default=24e6)
    parser.add_argument("--data_rate_5_bps", type=float, default=48e6)
    parser.add_argument("--mma_history_length", type=int, default=10)
    parser.add_argument("--mma_actor_hidden_dim", type=int, default=64)
    parser.add_argument("--mma_critic_hidden_dim", type=int, default=128)
    parser.add_argument("--mma_actor_lr", type=float, default=5e-4)
    parser.add_argument("--mma_critic_lr", type=float, default=5e-4)
    parser.add_argument("--mma_gamma", type=float, default=0.95)
    parser.add_argument("--mma_tau", type=float, default=1e-2)
    parser.add_argument("--mma_batch_size", type=int, default=64)
    parser.add_argument("--mma_buffer_size", type=int, default=100000)
    parser.add_argument("--mma_minimal_size", type=int, default=4000)
    parser.add_argument("--mma_learning_interval", type=int, default=100)
    parser.add_argument("--mma_update_interval", type=int, default=200)
    parser.add_argument("--mma_checkpoint", type=str, required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--allow_agent_expand", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="WiFi_v9_mma_maddpg_eval_mbps")
    parser.add_argument("--wandb_group", type=str, default="compare_wifi_v9_mma_maddpg_mbps")
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
    if all_args.wandb_entity:
        all_args.user_name = all_args.wandb_entity

    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")

    run_dir = build_eval_run_dir(all_args, "wifi_v9_mma_maddpg_mbps")
    run = init_wandb(all_args, run_dir, "wifi_v9_mma_maddpg_mbps")

    env = make_wifi_env(all_args, all_args.seed)
    tracker = MMALinkStateTracker(
        num_mld=env.max_mld,
        num_links=env.num_links,
        history_length=all_args.mma_history_length,
    )
    maddpg = WiFiV9MMAMADDPG(
        num_agents=env.max_mld,
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
    maddpg.load(
        all_args.mma_checkpoint,
        map_location=device,
        allow_agent_expand=all_args.allow_agent_expand,
    )
    time_model = build_time_model(all_args)

    episode_metrics = []
    for episode in range(all_args.eval_episodes):
        env.seed(all_args.seed + episode)
        env.reset()
        tracker.reset()
        accumulator = MbpsAccumulator(time_model)
        episode_reward_total = 0.0
        transmit_count = 0
        action_count = 0
        last_infos = None
        prev_link_successes = env.link_successes.copy()
        prev_link_packet_successes = env.link_packet_successes.copy()
        prev_sld_success = int(env.round_sld_success)
        next_arrival_step = int(all_args.round_length)
        built_rewards = np.zeros(env.max_mld, dtype=np.float32)

        while not accumulator.done():
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
                    explore=all_args.stochastic,
                )
            env_actions = build_env_actions(env, actions_by_link)
            transmit_count += int(env_actions.sum())
            action_count += int(sum(mask.sum() for mask in masks_by_link))
            _, _, rewards, dones, infos, _ = env.step(env_actions)
            for link_id, mask in enumerate(masks_by_link):
                if not np.any(mask):
                    continue
                built_rewards, link_obs, _, success_raw = tracker.build_rewards(
                    env,
                    actions_by_link[:, link_id, :],
                    infos,
                    mask,
                    link_id,
                    alpha=0.3,
                )
                tracker.update_link(
                    link_id,
                    actions_by_link[:, link_id, :],
                    link_obs,
                    success_raw,
                    mask,
                )
            episode_reward_total += float(np.sum(rewards))
            last_infos = infos

            link_events, prev_link_successes, prev_sld_success, prev_link_packet_successes = infer_link_events(
                env, infos, prev_link_successes, prev_sld_success, prev_link_packet_successes
            )
            step_slots = infos[0].get("step_slots", env.last_step_slots) if infos else env.last_step_slots
            accumulator.add_step(link_events, step_slots=step_slots)

            while env.t >= next_arrival_step:
                env.add_packet_arrivals(all_args.round_length)
                next_arrival_step += int(all_args.round_length)

            if bool(np.all(dones)):
                if accumulator.done():
                    break
                raise RuntimeError(
                    "WiFi v9 MMA-MADDPG Mbps eval reached env done before the "
                    "fixed-duration accumulator finished; increase the eval guard."
                )

        del built_rewards
        metrics = accumulator.as_metrics()
        add_mu_representative_metrics(metrics, env)
        metrics["episode_reward/total"] = float(episode_reward_total)
        metrics["policy_type"] = 3.0
        metrics["action/transmit_ratio"] = float(transmit_count / max(action_count, 1))
        metrics["avg_fulfillment"] = float(
            np.mean([info.get("fulfillment", 0.0) for info in last_infos if info.get("active", True)])
        ) if last_infos else 0.0
        metrics["scenario/active_mld"] = float(env.active_mld)
        metrics["scenario/active_sld"] = float(env.active_sld)
        metrics["scenario/max_mld"] = float(env.max_mld)
        metrics["scenario/max_sld"] = float(env.max_sld)
        episode_metrics.append(metrics)
        log_episode_metrics(run, episode, metrics)
        print(
            f"[MMA-MADDPG Mbps Eval] Episode {episode + 1}/{all_args.eval_episodes} | "
            f"mbps/system={metrics['mbps/system']:.4f} | "
            f"mbps/mld_total={metrics['mbps/mld_total']:.4f} | "
            f"mbps/sld_total={metrics['mbps/sld_total']:.4f} | "
            f"tx_ratio={metrics['action/transmit_ratio']:.4f}"
        )

    summary = summarize_metrics(episode_metrics)
    save_summary(run_dir, "mma_maddpg_mbps_summary.json", summary)
    chart_path = save_mbps_bar_chart(run_dir, "mbps_bar_chart.png", summary)
    print("\n[MMA-MADDPG Mbps Summary]")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.6f}")

    log_wandb_image(run, "summary/mbps_bar_chart", chart_path)
    finalize_wandb(run)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
