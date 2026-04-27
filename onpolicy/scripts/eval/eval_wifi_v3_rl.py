#!/usr/bin/env python
"""Evaluate a trained WiFi v3 RL policy checkpoint."""

import sys
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.eval.wifi_v3.utils import (
    build_eval_run_dir,
    compute_episode_metrics,
    finalize_wandb,
    init_wandb,
    load_policy,
    log_episode_metrics,
    log_wandb_image,
    make_wifi_env,
    print_episode_metrics,
    save_summary,
    save_throughput_bar_chart,
    select_device,
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
    parser.add_argument("--wandb_project", type=str, default="WiFi_v3_eval")
    parser.add_argument("--wandb_group", type=str, default="compare_wifi_v3")
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

    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    device = select_device(all_args)
    run_dir = build_eval_run_dir(all_args, "wifi_v3_rl")
    run = init_wandb(all_args, run_dir, "wifi_v3_rl")

    env = make_wifi_env(all_args, all_args.seed)
    policy = load_policy(all_args, env, device)

    episode_metrics = []
    for episode in range(all_args.eval_episodes):
        env.seed(all_args.seed + episode)
        obs, share_obs, available_actions = env.reset()

        rnn_states = np.zeros(
            (env.num_agents, all_args.recurrent_N, all_args.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((env.num_agents, 1), dtype=np.float32)

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

                avg_skip = float(np.mean(action_probs[:, 0]))
                avg_transmit = float(np.mean(action_probs[:, 1]))
                sample_agents = min(4, env.num_agents)
                sample_prob_text = ", ".join(
                    [
                        f"a{aid}: p0={action_probs[aid, 0]:.4f}, p1={action_probs[aid, 1]:.4f}"
                        for aid in range(sample_agents)
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

            transmit_count += int(actions.sum())
            action_count += int(actions.size)

            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
            del share_obs

            episode_reward_total += float(np.sum(rewards))
            last_infos = infos
            done = bool(np.all(dones))
            if done:
                masks = np.zeros((env.num_agents, 1), dtype=np.float32)
                rnn_states[:] = 0.0
            else:
                masks = np.ones((env.num_agents, 1), dtype=np.float32)

        metrics = compute_episode_metrics(env, last_infos, episode_reward_total)
        metrics["policy_type"] = 1.0
        metrics["action/transmit_ratio"] = (
            float(transmit_count) / float(action_count) if action_count > 0 else 0.0
        )
        episode_metrics.append(metrics)
        log_episode_metrics(run, episode, metrics)
        print_episode_metrics(episode, all_args.eval_episodes, metrics)
        print(
            f"        action/transmit_ratio={metrics['action/transmit_ratio']:.4f} "
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
