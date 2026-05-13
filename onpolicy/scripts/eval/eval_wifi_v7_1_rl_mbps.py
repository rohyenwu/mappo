#!/usr/bin/env python
"""Evaluate a trained WiFi v7.1 RL checkpoint with fixed-duration Mbps metrics."""

import sys

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.wifi_v7_1.wifi_env import WiFiEnvV7_1
from onpolicy.eval.wifi_common.mbps_metrics import (
    MbpsAccumulator,
    MbpsTimeModel,
    infer_link_events,
    save_mbps_bar_chart,
)
from onpolicy.eval.wifi_v5.utils import (
    build_eval_run_dir,
    finalize_wandb,
    init_wandb,
    load_policy,
    log_episode_metrics,
    log_wandb_image,
    parse_mu_profile,
    save_summary,
    select_device,
    summarize_metrics,
)


def make_wifi_env(args, seed: int):
    mu_profile = parse_mu_profile(getattr(args, "mu_profile", None))
    env = WiFiEnvV7_1(
        max_mld=args.max_mld,
        max_sld=args.max_sld,
        scenario_profile=[(args.num_mld, args.num_sld)],
        round_length=args.round_length,
        mu_range=(args.mu_min, args.mu_max),
        mu_profile=mu_profile,
        eta=args.eta,
        zeta=args.zeta,
        r_sld=args.r_sld,
        c_idle=args.c_idle,
        theta_scale=args.theta_scale,
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
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--r_sld", type=float, default=0.3)
    parser.add_argument("--c_idle", type=float, default=0.3)
    parser.add_argument("--theta_scale", type=float, default=1.0)
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
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="WiFi_v7_1_eval_mbps")
    parser.add_argument("--wandb_group", type=str, default="compare_wifi_v7_1_mbps")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--debug_prob_steps", type=int, default=5)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.num_mld < 1 or all_args.num_mld > all_args.max_mld:
        raise ValueError("--num_mld must be in [1, max_mld]")
    if all_args.num_sld < 0 or all_args.num_sld > all_args.max_sld:
        raise ValueError("--num_sld must be in [0, max_sld]")
    if all_args.wandb_entity:
        all_args.user_name = all_args.wandb_entity

    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    device = select_device(all_args)
    run_dir = build_eval_run_dir(all_args, "wifi_v7_1_rl_mbps")
    run = init_wandb(all_args, run_dir, "wifi_v7_1_rl_mbps")

    env = make_wifi_env(all_args, all_args.seed)
    policy = load_policy(all_args, env, device)
    time_model = build_time_model(all_args)

    episode_metrics = []
    for episode in range(all_args.eval_episodes):
        env.seed(all_args.seed + episode)
        obs, share_obs, available_actions = env.reset()
        del share_obs

        rnn_states = np.zeros(
            (env.num_agents, all_args.recurrent_N, all_args.hidden_size),
            dtype=np.float32,
        )
        masks = env.get_active_masks()

        accumulator = MbpsAccumulator(time_model)
        episode_reward_total = 0.0
        transmit_count = 0
        action_count = 0
        round_count = 1
        last_infos = None
        prev_link_successes = env.link_successes.copy()
        prev_link_packet_successes = env.link_packet_successes.copy()
        prev_sld_success = int(env.round_sld_success)

        while not accumulator.done():
            if episode == 0 and env.t < all_args.debug_prob_steps:
                action_probs = policy.get_action_probs(
                    obs,
                    rnn_states,
                    masks,
                    available_actions,
                )
                if torch.is_tensor(action_probs):
                    action_probs = action_probs.detach().cpu().numpy()
                active_masks = env.get_active_masks().reshape(-1) > 0.5
                active_probs = action_probs[active_masks]
                avg_skip = float(np.mean(active_probs[:, 0]))
                avg_transmit = float(np.mean(active_probs[:, 1]))
                print(
                    f"[DebugProb] ep=1 round={round_count} step={env.t + 1} "
                    f"avg_p0={avg_skip:.4f} avg_p1={avg_transmit:.4f}"
                )

            actions, rnn_states = policy.act(
                obs,
                rnn_states,
                masks,
                available_actions,
                deterministic=all_args.deterministic,
            )
            if torch.is_tensor(actions):
                actions = actions.detach().cpu().numpy()
            if torch.is_tensor(rnn_states):
                rnn_states = rnn_states.detach().cpu().numpy()

            active_masks = env.get_active_masks().reshape(-1) > 0.5
            transmit_count += int(actions.reshape(-1)[active_masks].sum())
            action_count += int(active_masks.sum())

            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
            del share_obs
            episode_reward_total += float(np.sum(rewards))
            last_infos = infos

            link_events, prev_link_successes, prev_sld_success, prev_link_packet_successes = infer_link_events(
                env, infos, prev_link_successes, prev_sld_success, prev_link_packet_successes
            )
            step_slots = infos[0].get("step_slots", env.last_step_slots) if infos else env.last_step_slots
            accumulator.add_step(link_events, step_slots=step_slots)

            if bool(np.all(dones)):
                if accumulator.done():
                    break
                round_count += 1
                obs, share_obs, available_actions = env.reset()
                del share_obs
                rnn_states[:] = 0.0
                masks = env.get_active_masks()
                prev_link_successes = env.link_successes.copy()
                prev_link_packet_successes = env.link_packet_successes.copy()
                prev_sld_success = int(env.round_sld_success)
            else:
                masks = env.get_active_masks()

        metrics = accumulator.as_metrics()
        metrics["episode_reward/total"] = float(episode_reward_total)
        metrics["policy_type"] = 1.0
        metrics["action/transmit_ratio"] = (
            float(transmit_count) / float(action_count) if action_count > 0 else 0.0
        )
        metrics["avg_fulfillment"] = float(
            np.mean([info.get("fulfillment", 0.0) for info in last_infos if info.get("active", True)])
        ) if last_infos else 0.0
        metrics["scenario/active_mld"] = float(env.active_mld)
        metrics["scenario/active_sld"] = float(env.active_sld)
        metrics["scenario/max_mld"] = float(env.max_mld)
        metrics["scenario/max_sld"] = float(env.max_sld)
        metrics["timing/rounds_completed"] = float(round_count)
        episode_metrics.append(metrics)
        log_episode_metrics(run, episode, metrics)
        print(
            f"[RL Mbps Eval] Episode {episode + 1}/{all_args.eval_episodes} | "
            f"mbps/system={metrics['mbps/system']:.4f} | "
            f"mbps/mld_total={metrics['mbps/mld_total']:.4f} | "
            f"mbps/sld_total={metrics['mbps/sld_total']:.4f} | "
            f"tx_ratio={metrics['action/transmit_ratio']:.4f}"
        )

    summary = summarize_metrics(episode_metrics)
    save_summary(run_dir, "rl_mbps_summary.json", summary)
    chart_path = save_mbps_bar_chart(run_dir, "mbps_bar_chart.png", summary)
    print("\n[RL Mbps Summary]")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.6f}")

    log_wandb_image(run, "summary/mbps_bar_chart", chart_path)
    finalize_wandb(run)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
