#!/usr/bin/env python
"""Evaluate a trained WiFi v7.1 RL policy checkpoint on one active scenario."""

import sys

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.wifi_v7_1.wifi_env import WiFiEnvV7_1
from onpolicy.eval.wifi_v5.utils import (
    build_eval_run_dir,
    compute_episode_metrics,
    finalize_wandb,
    init_wandb,
    load_policy,
    log_episode_metrics,
    log_wandb_image,
    parse_mu_profile,
    print_episode_metrics,
    save_summary,
    save_throughput_bar_chart,
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


def compute_v7_1_episode_metrics(env, infos, episode_reward_total: float):
    metrics = compute_episode_metrics(env, infos, episode_reward_total)
    active_infos = [info for info in infos if info.get("active", True)]
    active_fulfillments = [info.get("fulfillment", 0.0) for info in active_infos]
    metrics["avg_fulfillment"] = (
        float(np.mean(active_fulfillments)) if active_fulfillments else 0.0
    )
    metrics["scenario/active_mld"] = float(env.active_mld)
    metrics["scenario/active_sld"] = float(env.active_sld)
    metrics["scenario/max_mld"] = float(env.max_mld)
    metrics["scenario/max_sld"] = float(env.max_sld)
    return metrics


def parse_args(args, parser):
    parser.add_argument("--num_mld", type=int, default=10)
    parser.add_argument("--num_sld", type=int, default=2)
    parser.add_argument("--max_mld", type=int, default=30)
    parser.add_argument("--max_sld", type=int, default=10)
    parser.add_argument("--round_length", type=int, default=500)
    parser.add_argument("--mu_min", type=float, default=0.01)
    parser.add_argument("--mu_max", type=float, default=0.12)
    parser.add_argument(
        "--mu_profile",
        type=str,
        default=None,
        help="Comma-separated per-MLD demand rates. Overrides mu_min/mu_max when set.",
    )
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--r_sld", type=float, default=0.3)
    parser.add_argument("--c_idle", type=float, default=0.3)
    parser.add_argument("--theta_scale", type=float, default=1.0)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="WiFi_v7_1_eval")
    parser.add_argument("--wandb_group", type=str, default="compare_wifi_v7_1")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--debug_prob_steps",
        type=int,
        default=5,
        help="Number of initial env steps to print action probabilities for.",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use greedy actions during evaluation (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample actions from the policy distribution during evaluation.",
    )
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
    run_dir = build_eval_run_dir(all_args, "wifi_v7_1_rl")
    run = init_wandb(all_args, run_dir, "wifi_v7_1_rl")

    env = make_wifi_env(all_args, all_args.seed)
    policy = load_policy(all_args, env, device)

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

        done = False
        episode_reward_total = 0.0
        last_infos = None
        transmit_count = 0
        action_count = 0

        while not done:
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
                sample_agent_ids = np.where(active_masks)[0][:4]
                sample_prob_text = ", ".join(
                    [
                        f"a{aid}: p0={action_probs[aid, 0]:.4f}, p1={action_probs[aid, 1]:.4f}"
                        for aid in sample_agent_ids
                    ]
                )
                print(
                    f"[DebugProb] ep=1 step={env.t + 1} "
                    f"avg_p0={avg_skip:.4f} avg_p1={avg_transmit:.4f} | "
                    f"{sample_prob_text}"
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
            done = bool(np.all(dones))
            if done:
                masks = np.zeros((env.num_agents, 1), dtype=np.float32)
                rnn_states[:] = 0.0
            else:
                masks = env.get_active_masks()

        metrics = compute_v7_1_episode_metrics(env, last_infos, episode_reward_total)
        metrics["policy_type"] = 1.0
        metrics["action/transmit_ratio"] = (
            float(transmit_count) / float(action_count) if action_count > 0 else 0.0
        )
        episode_metrics.append(metrics)
        log_episode_metrics(run, episode, metrics)
        print_episode_metrics(episode, all_args.eval_episodes, metrics)
        print(
            f"        scenario=m{env.active_mld}_s{env.active_sld} "
            f"action/transmit_ratio={metrics['action/transmit_ratio']:.4f} "
            f"(deterministic={all_args.deterministic})"
        )

    summary = summarize_metrics(episode_metrics)
    save_summary(run_dir, "rl_summary.json", summary)
    chart_path = save_throughput_bar_chart(run_dir, "throughput_bar_chart.png", summary)
    print("\n[RL Summary]")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.6f}")

    log_wandb_image(run, "summary/throughput_bar_chart", chart_path)

    finalize_wandb(run)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
