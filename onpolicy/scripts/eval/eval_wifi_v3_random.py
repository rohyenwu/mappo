#!/usr/bin/env python
"""Evaluate a fixed Bernoulli random baseline on WiFi v3."""

import sys

import numpy as np

from onpolicy.config import get_config
from onpolicy.eval.wifi_v3.utils import (
    build_eval_run_dir,
    compute_episode_metrics,
    finalize_wandb,
    init_wandb,
    log_episode_metrics,
    log_wandb_image,
    make_wifi_env,
    print_episode_metrics,
    save_summary,
    save_throughput_bar_chart,
    summarize_metrics,
)


def parse_args(args, parser):
    parser.add_argument("--num_mld", type=int, default=3)
    parser.add_argument("--num_sld", type=int, default=3)
    parser.add_argument("--round_length", type=int, default=50)
    parser.add_argument("--mu_min", type=float, default=0.01)
    parser.add_argument("--mu_max", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--r_sld", type=float, default=0.3)
    parser.add_argument("--c_idle", type=float, default=0.3)
    parser.add_argument("--theta_scale", type=float, default=1.0)
    parser.add_argument(
        "--transmit_prob",
        type=float,
        default=0.5,
        help="Probability of action 1 (transmit). Action 0 (skip) is 1 - transmit_prob.",
    )
    parser.add_argument("--wandb_project", type=str, default="WiFi_v3_eval")
    parser.add_argument("--wandb_group", type=str, default="compare_wifi_v3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_known_args(args)[0]


def sample_random_actions(rng, env, available_actions, transmit_prob):
    probs = np.full((env.num_agents,), transmit_prob, dtype=np.float32)

    if available_actions is not None:
        can_skip = available_actions[:, 0] > 0.5
        can_transmit = available_actions[:, 1] > 0.5

        probs = np.where(~can_transmit & can_skip, 0.0, probs)
        probs = np.where(can_transmit & ~can_skip, 1.0, probs)

    actions = rng.binomial(1, probs, size=env.num_agents).astype(np.int64)
    return actions.reshape(env.num_agents, 1)


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if not 0.0 <= all_args.transmit_prob <= 1.0:
        raise ValueError("--transmit_prob must be in [0, 1].")

    np.random.seed(all_args.seed)
    rng = np.random.default_rng(all_args.seed)

    run_dir = build_eval_run_dir(all_args, "wifi_v3_random")
    run = init_wandb(all_args, run_dir, "wifi_v3_random")

    env = make_wifi_env(all_args, all_args.seed)

    episode_metrics = []
    for episode in range(all_args.eval_episodes):
        env.seed(all_args.seed + episode)
        obs, share_obs, available_actions = env.reset()
        del obs, share_obs

        done = False
        episode_reward_total = 0.0
        last_infos = None
        transmit_count = 0
        action_count = 0

        while not done:
            actions = sample_random_actions(
                rng=rng,
                env=env,
                available_actions=available_actions,
                transmit_prob=all_args.transmit_prob,
            )

            transmit_count += int(actions.sum())
            action_count += int(actions.size)

            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
            del obs, share_obs

            episode_reward_total += float(np.sum(rewards))
            last_infos = infos
            done = bool(np.all(dones))

        metrics = compute_episode_metrics(env, last_infos, episode_reward_total)
        metrics["policy_type"] = -1.0
        metrics["action/transmit_ratio"] = (
            float(transmit_count) / float(action_count) if action_count > 0 else 0.0
        )
        metrics["action/configured_transmit_prob"] = float(all_args.transmit_prob)

        episode_metrics.append(metrics)
        log_episode_metrics(run, episode, metrics)
        print_episode_metrics(episode, all_args.eval_episodes, metrics)
        print(
            f"        action/transmit_ratio={metrics['action/transmit_ratio']:.4f} "
            f"(configured_p1={all_args.transmit_prob:.4f})"
        )

    summary = summarize_metrics(episode_metrics)
    save_summary(run_dir, "random_summary.json", summary)
    chart_path = save_throughput_bar_chart(run_dir, "throughput_bar_chart.png", summary)
    print("\n[Random Summary]")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.6f}")

    log_wandb_image(run, "summary/throughput_bar_chart", chart_path)
    finalize_wandb(run)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
