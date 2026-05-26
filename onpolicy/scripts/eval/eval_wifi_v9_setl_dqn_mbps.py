#!/usr/bin/env python
"""Evaluate a trained SETL-DQN(MA) baseline on WiFi v9 Mbps metrics."""

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
from onpolicy.eval.wifi_v9.setl_dqn import SETLMLDBackoffMAC, SharedDQNAgent


def make_wifi_env(args, seed: int):
    mu_profile = parse_mu_profile(getattr(args, "mu_profile", None))
    env = WiFiEnvV9(
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


def parse_thresholds(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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
    parser.add_argument("--sld_target_low_scale", type=float, default=0.5)
    parser.add_argument("--sld_target_high_scale", type=float, default=0.7)
    parser.add_argument("--sld_target_bonus", type=float, default=0.0)
    parser.add_argument("--mld_success_reward", type=float, default=1.0)
    parser.add_argument("--collision_penalty", type=float, default=1.0)
    parser.add_argument("--non_top_tx_penalty", type=float, default=0.5)
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
    parser.add_argument("--setl_thresholds", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--setl_linear_step", type=int, default=32)
    parser.add_argument(
        "--setl_feedback_report_bits",
        type=float,
        default=128.0,
        help=(
            "Conservative per-agent AP feedback report size in bits for "
            "SETL-DQN effective-throughput accounting."
        ),
    )
    parser.add_argument(
        "--setl_feedback_broadcast_bits",
        type=float,
        default=128.0,
        help=(
            "Conservative AP aggregate-feedback broadcast size in bits for "
            "SETL-DQN effective-throughput accounting."
        ),
    )
    parser.add_argument(
        "--setl_feedback_interval_sec",
        type=float,
        default=0.1024,
        help=(
            "AP-assisted feedback interval in seconds. Set to 0 to disable "
            "SETL-DQN control-overhead accounting."
        ),
    )
    parser.add_argument("--dqn_hidden_size", type=int, default=128)
    parser.add_argument("--dqn_hidden_layers", type=int, default=3)
    parser.add_argument("--dqn_checkpoint", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="WiFi_v9_setl_dqn_eval_mbps")
    parser.add_argument("--wandb_group", type=str, default="compare_wifi_v9_setl_dqn_mbps")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_known_args(args)[0]


def add_setl_feedback_overhead_metrics(metrics, env, args):
    interval_sec = float(args.setl_feedback_interval_sec)
    report_bits = float(args.setl_feedback_report_bits)
    broadcast_bits = float(args.setl_feedback_broadcast_bits)

    active_agents = int(env.active_mld * env.num_links)
    active_mld = int(env.active_mld)
    data_rates = [
        float(args.data_rate_24_bps),
        float(args.data_rate_5_bps),
    ]
    while len(data_rates) < int(env.num_links):
        data_rates.append(float(args.data_rate_5_bps))

    link_names = ["2_4GHz", "5GHz"]
    while len(link_names) < int(env.num_links):
        link_names.append(f"link_{len(link_names)}")

    report_airtimes = []
    broadcast_airtimes = []
    feedback_airtimes = []
    airtime_fractions = []
    for link_id in range(int(env.num_links)):
        data_rate = max(data_rates[link_id], 1.0)
        report_airtime_sec = active_mld * report_bits / data_rate
        broadcast_airtime_sec = (
            broadcast_bits / float(args.basic_rate_bps)
            if float(args.basic_rate_bps) > 0.0
            else 0.0
        )
        feedback_airtime_sec = report_airtime_sec + broadcast_airtime_sec
        airtime_fraction = 0.0
        if interval_sec > 0.0:
            airtime_fraction = min(feedback_airtime_sec / interval_sec, 1.0)

        report_airtimes.append(report_airtime_sec)
        broadcast_airtimes.append(broadcast_airtime_sec)
        feedback_airtimes.append(feedback_airtime_sec)
        airtime_fractions.append(airtime_fraction)

    raw_24_mld = float(metrics.get("mbps/2_4GHz/mld", 0.0))
    raw_24_sld = float(metrics.get("mbps/2_4GHz/sld", 0.0))
    raw_5_mld = float(metrics.get("mbps/5GHz/mld", 0.0))
    raw_5_sld = float(metrics.get("mbps/5GHz/sld", 0.0))
    frac_24 = airtime_fractions[0] if airtime_fractions else 0.0
    frac_5 = airtime_fractions[1] if len(airtime_fractions) > 1 else 0.0

    effective_24_mld = raw_24_mld * (1.0 - frac_24)
    effective_24_sld = raw_24_sld * (1.0 - frac_24)
    effective_5_mld = raw_5_mld * (1.0 - frac_5)
    effective_5_sld = raw_5_sld * (1.0 - frac_5)
    effective_24_total = effective_24_mld + effective_24_sld
    effective_5_total = effective_5_mld + effective_5_sld
    effective_mld_total = effective_24_mld + effective_5_mld
    effective_sld_total = effective_24_sld + effective_5_sld
    effective_system = effective_mld_total + effective_sld_total

    system_raw = float(metrics.get("mbps/system", 0.0))
    overhead_mbps = max(system_raw - effective_system, 0.0)

    # Also expose the payload-rate accounting used for sensitivity checks.
    payload_overhead_mbps = 0.0
    if interval_sec > 0.0:
        payload_overhead_bits = active_agents * report_bits + broadcast_bits
        payload_overhead_mbps = payload_overhead_bits / interval_sec / 1e6

    metrics["setl_feedback/active_agents"] = float(active_agents)
    metrics["setl_feedback/report_bits_per_agent"] = report_bits
    metrics["setl_feedback/broadcast_bits"] = broadcast_bits
    metrics["setl_feedback/interval_sec"] = interval_sec
    for link_id, link_name in enumerate(link_names[: int(env.num_links)]):
        metrics[f"setl_feedback/{link_name}/report_airtime_sec"] = report_airtimes[link_id]
        metrics[f"setl_feedback/{link_name}/broadcast_airtime_sec"] = broadcast_airtimes[link_id]
        metrics[f"setl_feedback/{link_name}/airtime_sec_per_interval"] = feedback_airtimes[link_id]
        metrics[f"setl_feedback/{link_name}/airtime_fraction"] = airtime_fractions[link_id]
    metrics["setl_feedback/payload_rate_overhead_mbps"] = payload_overhead_mbps
    metrics["setl_feedback/overhead_mbps"] = overhead_mbps
    metrics["mbps/2_4GHz/mld_effective_after_setl_feedback"] = effective_24_mld
    metrics["mbps/2_4GHz/sld_effective_after_setl_feedback"] = effective_24_sld
    metrics["mbps/2_4GHz/total_effective_after_setl_feedback"] = effective_24_total
    metrics["mbps/5GHz/mld_effective_after_setl_feedback"] = effective_5_mld
    metrics["mbps/5GHz/sld_effective_after_setl_feedback"] = effective_5_sld
    metrics["mbps/5GHz/total_effective_after_setl_feedback"] = effective_5_total
    metrics["mbps/mld_total_effective_after_setl_feedback"] = effective_mld_total
    metrics["mbps/sld_total_effective_after_setl_feedback"] = effective_sld_total
    metrics["mbps/system_effective_after_setl_feedback"] = effective_system


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    if all_args.wandb_entity:
        all_args.user_name = all_args.wandb_entity

    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")

    run_dir = build_eval_run_dir(all_args, "wifi_v9_setl_dqn_mbps")
    run = init_wandb(all_args, run_dir, "wifi_v9_setl_dqn_mbps")

    env = make_wifi_env(all_args, all_args.seed)
    thresholds = parse_thresholds(all_args.setl_thresholds)
    mac = SETLMLDBackoffMAC(
        env.num_agents,
        env.agent_to_mld_link,
        thresholds=thresholds,
        linear_step=all_args.setl_linear_step,
        rng=np.random.default_rng(all_args.seed),
    )
    agent = SharedDQNAgent(
        obs_dim=mac.obs_dim,
        act_dim=mac.act_dim,
        hidden_size=all_args.dqn_hidden_size,
        hidden_layers=all_args.dqn_hidden_layers,
        device=device,
        seed=all_args.seed,
        epsilon_start=0.0,
        epsilon_end=0.0,
    )
    checkpoint = agent.load(all_args.dqn_checkpoint, map_location=device)
    if "thresholds" in checkpoint:
        mac.thresholds = [int(value) for value in checkpoint["thresholds"]]
    time_model = build_time_model(all_args)

    episode_metrics = []
    for episode in range(all_args.eval_episodes):
        env.seed(all_args.seed + episode)
        env.reset()
        mac.reset_round(env)

        accumulator = MbpsAccumulator(time_model)
        episode_reward_total = 0.0
        transmit_count = 0
        action_count = 0
        last_infos = None
        prev_link_successes = env.link_successes.copy()
        prev_link_packet_successes = env.link_packet_successes.copy()
        prev_sld_success = int(env.round_sld_success)
        next_arrival_step = int(all_args.round_length)

        while not accumulator.done():
            dqn_obs = mac.observations(env)
            active_mask = env.get_active_masks().reshape(-1).astype(bool)
            threshold_actions = agent.select_actions(dqn_obs, active_mask=active_mask, explore=False)
            actions, pending_mask = mac.act(env, threshold_actions)
            transmit_count += int(actions.sum())
            action_count += int(actions.size)

            _, _, rewards, dones, infos, _ = env.step(actions)
            mac.update(env, actions, infos, pending_mask)
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
                    "WiFi v9 SETL-DQN Mbps eval reached env done before the "
                    "fixed-duration accumulator finished; increase the eval "
                    "episode duration guard."
                )

        metrics = accumulator.as_metrics()
        add_mu_representative_metrics(metrics, env)
        metrics["episode_reward/total"] = float(episode_reward_total)
        metrics["policy_type"] = 2.0
        metrics["action/transmit_ratio"] = float(transmit_count / max(action_count, 1))
        metrics["avg_fulfillment"] = float(
            np.mean([info.get("fulfillment", 0.0) for info in last_infos if info.get("active", True)])
        ) if last_infos else 0.0
        metrics["scenario/active_mld"] = float(env.active_mld)
        metrics["scenario/active_sld"] = float(env.active_sld)
        metrics["scenario/max_mld"] = float(env.max_mld)
        metrics["scenario/max_sld"] = float(env.max_sld)
        add_setl_feedback_overhead_metrics(metrics, env, all_args)
        episode_metrics.append(metrics)
        log_episode_metrics(run, episode, metrics)
        print(
            f"[SETL-DQN Mbps Eval] Episode {episode + 1}/{all_args.eval_episodes} | "
            f"mbps/system={metrics['mbps/system']:.4f} | "
            f"mbps/system_effective={metrics['mbps/system_effective_after_setl_feedback']:.4f} | "
            f"overhead={metrics['setl_feedback/overhead_mbps']:.4f} | "
            f"mbps/mld_total={metrics['mbps/mld_total']:.4f} | "
            f"mbps/sld_total={metrics['mbps/sld_total']:.4f} | "
            f"tx_ratio={metrics['action/transmit_ratio']:.4f}"
        )

    summary = summarize_metrics(episode_metrics)
    save_summary(run_dir, "setl_dqn_mbps_summary.json", summary)
    chart_path = save_mbps_bar_chart(run_dir, "mbps_bar_chart.png", summary)
    print("\n[SETL-DQN Mbps Summary]")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.6f}")

    log_wandb_image(run, "summary/mbps_bar_chart", chart_path)
    finalize_wandb(run)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
