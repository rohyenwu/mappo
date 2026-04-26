#!/usr/bin/env python
"""Evaluate a CSMA/CA-style BEB baseline on WiFi v2."""

import sys
import numpy as np

from onpolicy.config import get_config
from onpolicy.eval.wifi_v2.mac import MLDBackoffMAC
from onpolicy.eval.wifi_v2.utils import (
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
    parser.add_argument("--num_mld", type=int, default=10)
    parser.add_argument("--num_sld", type=int, default=5)
    parser.add_argument("--round_length", type=int, default=500)
    parser.add_argument("--mu_min", type=float, default=0.2)
    parser.add_argument("--mu_max", type=float, default=0.8)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--r_sld", type=float, default=0.3)
    parser.add_argument("--c_idle", type=float, default=0.3)
    parser.add_argument("--theta_scale", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default="WiFi_v2_eval")
    parser.add_argument("--wandb_group", type=str, default="compare_wifi_v2")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    np.random.seed(all_args.seed)

    run_dir = build_eval_run_dir(all_args, "wifi_v2_beb")
    run = init_wandb(all_args, run_dir, "wifi_v2_beb")

    env = make_wifi_env(all_args, all_args.seed)
    mac = MLDBackoffMAC(
        env.num_agents,
        env.agent_to_mld_link,
        rng=np.random.default_rng(all_args.seed),
    )

    episode_metrics = []
    for episode in range(all_args.eval_episodes):
        env.seed(all_args.seed + episode)
        obs, share_obs, available_actions = env.reset()
        del obs, share_obs, available_actions
        mac.reset_round(env)

        done = False
        episode_reward_total = 0.0
        last_infos = None

        while not done:
            actions, pending_mask = mac.act(env)
            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
            del obs, share_obs, available_actions
            mac.update(env, actions, infos, pending_mask)
            episode_reward_total += float(np.sum(rewards))
            last_infos = infos
            done = bool(np.all(dones))

        metrics = compute_episode_metrics(env, last_infos, episode_reward_total)
        metrics["policy_type"] = 0.0
        episode_metrics.append(metrics)
        log_episode_metrics(run, episode, metrics)
        print_episode_metrics(episode, all_args.eval_episodes, metrics)

    summary = summarize_metrics(episode_metrics)
    save_summary(run_dir, "beb_summary.json", summary)
    chart_path = save_throughput_bar_chart(run_dir, "throughput_bar_chart.png", summary)
    print("\n[BEB Summary]")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.6f}")

    log_wandb_image(run, "summary/throughput_bar_chart", chart_path)
    finalize_wandb(run)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
