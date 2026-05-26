#!/usr/bin/env python
"""Train a SETL-DQN(MA) CW-threshold baseline on WiFi v9."""

import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.wifi_v9.wifi_env import WiFiEnvV9
from onpolicy.eval.wifi_common.mbps_metrics import infer_link_events
from onpolicy.eval.wifi_v5.utils import parse_mu_profile
from onpolicy.eval.wifi_v9.setl_dqn import SETLMLDBackoffMAC, SharedDQNAgent


def parse_scenario_profile(profile_text):
    if profile_text is None:
        return None
    scenarios = []
    for raw_part in profile_text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid scenario '{part}'. Expected MLD:SLD, e.g. '10:5'."
            )
        mld_text, sld_text = part.split(":", 1)
        scenarios.append((int(mld_text.strip()), int(sld_text.strip())))
    if not scenarios:
        raise ValueError("--scenario_profile was provided but no scenarios were found.")
    return scenarios


def make_wifi_env(args, seed: int, scenario):
    mu_profile = parse_mu_profile(getattr(args, "mu_profile", None))
    env = WiFiEnvV9(
        max_mld=args.max_mld,
        max_sld=args.max_sld,
        scenario_profile=[scenario],
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
        episode_duration_sec=args.episode_duration_sec,
    )
    env.seed(seed)
    return env


def parse_thresholds(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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
    run_dir = (
        repo_dir
        / "model"
        / args.env_name
        / "setl_dqn_ma"
        / f"{args.experiment_name}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def init_wandb(args, run_dir):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(
        config=vars(args),
        project=getattr(args, "wandb_project", "wifi_v9_setl_dqn"),
        entity=getattr(args, "wandb_entity", None) or args.user_name,
        notes=socket.gethostname(),
        name=getattr(args, "wandb_run_name", None)
        or f"setl_dqn_ma_{args.experiment_name}_seed{args.seed}",
        group=getattr(args, "wandb_group", None) or "setl_dqn_ma",
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
    parser.add_argument("--setl_thresholds", type=str, default="128,256,384,512,640,768,896,1024")
    parser.add_argument("--setl_linear_step", type=int, default=32)
    parser.add_argument("--dqn_hidden_size", type=int, default=128)
    parser.add_argument("--dqn_hidden_layers", type=int, default=3)
    parser.add_argument("--dqn_lr", type=float, default=1e-3)
    parser.add_argument("--dqn_gamma", type=float, default=0.99)
    parser.add_argument("--dqn_batch_size", type=int, default=32)
    parser.add_argument("--dqn_memory_size", type=int, default=20000)
    parser.add_argument("--dqn_min_replay_size", type=int, default=100)
    parser.add_argument("--dqn_target_update_interval", type=int, default=200)
    parser.add_argument("--dqn_epsilon_start", type=float, default=0.1)
    parser.add_argument("--dqn_epsilon_end", type=float, default=0.01)
    parser.add_argument("--dqn_epsilon_decay", type=float, default=1e-6)
    parser.add_argument(
        "--setl_train_episodes",
        type=int,
        default=50000,
        help="Number of WiFi episodes used to train the SETL-DQN(MA) baseline.",
    )
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="wifi_v9_setl_dqn")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_known_args(args)[0]


def cooperative_step_reward(link_events, step_slots: float, slot_time_sec: float, max_links: int = 2):
    delivered_packets = 0.0
    for link_id in (0, 1):
        result = link_events[link_id]["result"]
        if result == "success":
            delivered_packets += float(link_events[link_id].get("packet_count", 1.0) or 1.0)
    del step_slots, slot_time_sec
    return delivered_packets / max(float(max_links), 1.0)


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)

    scenarios = parse_scenario_profile(all_args.scenario_profile)
    thresholds = parse_thresholds(all_args.setl_thresholds)
    device = select_device(all_args)
    run_dir = Path(all_args.save_dir) if all_args.save_dir else build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    run = init_wandb(all_args, run_dir)

    template_env = make_wifi_env(all_args, all_args.seed, scenarios[0])
    template_mac = SETLMLDBackoffMAC(
        template_env.num_agents,
        template_env.agent_to_mld_link,
        thresholds=thresholds,
        linear_step=all_args.setl_linear_step,
        rng=np.random.default_rng(all_args.seed),
    )
    agent = SharedDQNAgent(
        obs_dim=template_mac.obs_dim,
        act_dim=template_mac.act_dim,
        hidden_size=all_args.dqn_hidden_size,
        hidden_layers=all_args.dqn_hidden_layers,
        lr=all_args.dqn_lr,
        gamma=all_args.dqn_gamma,
        batch_size=all_args.dqn_batch_size,
        memory_size=all_args.dqn_memory_size,
        min_replay_size=all_args.dqn_min_replay_size,
        target_update_interval=all_args.dqn_target_update_interval,
        epsilon_start=all_args.dqn_epsilon_start,
        epsilon_end=all_args.dqn_epsilon_end,
        epsilon_decay=all_args.dqn_epsilon_decay,
        device=device,
        seed=all_args.seed,
    )
    template_env.close()

    total_episodes = int(all_args.setl_train_episodes)
    log_interval = max(int(all_args.log_interval), 1)
    save_interval = max(int(all_args.save_interval), 1)

    for episode in range(total_episodes):
        scenario = scenarios[episode % len(scenarios)]
        env = make_wifi_env(all_args, all_args.seed + episode, scenario)
        mac = SETLMLDBackoffMAC(
            env.num_agents,
            env.agent_to_mld_link,
            thresholds=thresholds,
            linear_step=all_args.setl_linear_step,
            rng=np.random.default_rng(all_args.seed + episode),
        )
        env.reset()
        mac.reset_round(env)

        episode_reward = 0.0
        episode_loss = []
        transmit_count = 0
        action_count = 0
        prev_link_successes = env.link_successes.copy()
        prev_link_packet_successes = env.link_packet_successes.copy()
        prev_sld_success = int(env.round_sld_success)
        next_arrival_step = int(all_args.round_length)

        while True:
            dqn_obs = mac.observations(env)
            active_mask = env.get_active_masks().reshape(-1).astype(bool)
            threshold_actions = agent.select_actions(dqn_obs, active_mask=active_mask, explore=True)
            actions, pending_mask = mac.act(env, threshold_actions)
            transmit_count += int(actions.sum())
            action_count += int(actions.size)

            _, _, _, dones, infos, _ = env.step(actions)
            mac.update(env, actions, infos, pending_mask)
            next_obs = mac.observations(env)

            link_events, prev_link_successes, prev_sld_success, prev_link_packet_successes = infer_link_events(
                env, infos, prev_link_successes, prev_sld_success, prev_link_packet_successes
            )
            step_slots = infos[0].get("step_slots", env.last_step_slots) if infos else env.last_step_slots
            reward = cooperative_step_reward(link_events, step_slots, all_args.slot_time_sec, env.num_links)
            done = bool(np.all(dones))
            agent.remember_batch(dqn_obs, threshold_actions, reward, next_obs, done, active_mask)
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            episode_reward += reward

            while env.t >= next_arrival_step and not done:
                env.add_packet_arrivals(all_args.round_length)
                next_arrival_step += int(all_args.round_length)

            if done:
                break

        metrics = {
            "episode": float(episode + 1),
            "reward/episode": float(episode_reward),
            "loss/dqn": float(np.mean(episode_loss)) if episode_loss else 0.0,
            "epsilon": float(agent.epsilon),
            "replay/size": float(len(agent.memory)),
            "action/transmit_ratio": float(transmit_count / max(action_count, 1)),
            "scenario/active_mld": float(scenario[0]),
            "scenario/active_sld": float(scenario[1]),
        }
        if run is not None:
            import wandb

            wandb.log(metrics, step=episode + 1)
        if (episode + 1) % log_interval == 0:
            print(
                f"[SETL-DQN(MA)] Episode {episode + 1}/{total_episodes} | "
                f"reward={metrics['reward/episode']:.3f} | "
                f"loss={metrics['loss/dqn']:.6f} | "
                f"eps={metrics['epsilon']:.4f} | "
                f"replay={int(metrics['replay/size'])}"
            )
        if (episode + 1) % save_interval == 0:
            agent.save(
                run_dir / "setl_dqn_ma.pt",
                extra={"thresholds": thresholds, "args": vars(all_args)},
            )
        env.close()

    agent.save(run_dir / "setl_dqn_ma.pt", extra={"thresholds": thresholds, "args": vars(all_args)})
    if run is not None:
        run.finish()
    print(f"Saved SETL-DQN(MA) checkpoint: {run_dir / 'setl_dqn_ma.pt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
