"""Common evaluation utilities for WiFi v3."""

import json
import os
import socket
from pathlib import Path

import numpy as np
import torch

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.envs.wifi_v3.wifi_env import WiFiEnvV2


def configure_algorithm_flags(args):
    if args.algorithm_name == "rmappo":
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "mappo":
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError(f"Unsupported algorithm for WiFi v2 evaluation: {args.algorithm_name}")


def make_wifi_env(args, seed: int):
    env = WiFiEnvV2(
        num_mld=args.num_mld,
        num_sld=args.num_sld,
        round_length=args.round_length,
        mu_range=(args.mu_min, args.mu_max),
        eta=args.eta,
        zeta=args.zeta,
        r_sld=args.r_sld,
        c_idle=args.c_idle,
        theta_scale=args.theta_scale,
    )
    env.seed(seed)
    return env


def select_device(args):
    if args.cuda and torch.cuda.is_available():
        print("GPU 사용")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return device

    print("CPU 사용")
    device = torch.device("cpu")
    torch.set_num_threads(args.n_training_threads)
    return device


def build_eval_run_dir(args, eval_name: str):
    base_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]).parents[2]
        / "scripts"
        / "eval_results"
        / args.env_name
        / eval_name
        / args.experiment_name
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def init_wandb(args, run_dir: Path, eval_name: str):
    if not args.use_wandb:
        return None

    import wandb

    group_name = getattr(args, "wandb_group", None) or eval_name
    run_name = getattr(args, "wandb_run_name", None) or f"{eval_name}_{args.experiment_name}_seed{args.seed}"

    return wandb.init(
        config=vars(args),
        project=getattr(args, "wandb_project", "WiFi_v3_eval"),
        entity=args.user_name,
        notes=socket.gethostname(),
        name=run_name,
        group=group_name,
        dir=str(run_dir),
        job_type="evaluation",
        reinit=True,
    )


def finalize_wandb(run):
    if run is not None:
        run.finish()


def save_summary(run_dir: Path, filename: str, summary: dict):
    out_path = run_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_throughput_bar_chart(run_dir: Path, filename: str, summary: dict):
    """Save a compact bar chart for the key throughput metrics."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[Eval] matplotlib is not installed; skipping throughput bar chart generation.")
        return None

    metric_keys = [
        "throughput/mld_total",
        "throughput/sld_total",
        "throughput/system",
    ]
    labels = ["MLD", "SLD", "System"]
    values = [float(summary.get(key, 0.0)) for key in metric_keys]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_ylabel("Throughput")
    ax.set_title("WiFi v3 Evaluation Throughput")
    ymax = max(values) if values else 0.0
    ax.set_ylim(0.0, max(0.1, ymax * 1.2))
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    out_path = run_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_episode_metrics(env, infos, episode_reward_total: float):
    metrics = {}
    metrics["episode_reward/total"] = float(episode_reward_total)
    metrics.update(env.get_throughput())
    metrics.update(env.get_collision_rate())
    if hasattr(env, "get_allocation_metrics"):
        metrics.update(env.get_allocation_metrics())

    fulfillments = [info.get("fulfillment", 0.0) for info in infos]
    metrics["avg_fulfillment"] = float(np.mean(fulfillments)) if fulfillments else 0.0

    for aid, info in enumerate(infos):
        metrics[f"episode_fulfillment/agent_{aid}"] = float(info.get("fulfillment", 0.0))

    num_mld = env.num_agents // 2
    for mld_id in range(num_mld):
        aid_24 = 2 * mld_id
        aid_5 = 2 * mld_id + 1
        f_24 = float(infos[aid_24].get("fulfillment", 0.0))
        f_5 = float(infos[aid_5].get("fulfillment", 0.0))
        metrics[f"episode_fulfillment/mld_{mld_id}/2_4GHz"] = f_24
        metrics[f"episode_fulfillment/mld_{mld_id}/5GHz"] = f_5
        metrics[f"episode_fulfillment/mld_{mld_id}/avg"] = (f_24 + f_5) / 2.0

    return metrics


def print_episode_metrics(episode_idx: int, total_episodes: int, metrics: dict):
    print(
        f"[Eval] Episode {episode_idx + 1}/{total_episodes} | "
        f"reward={metrics['episode_reward/total']:.4f} | "
        f"throughput/system={metrics['throughput/system']:.4f} | "
        f"collision/system_per_txop={metrics['collision_rate/system_per_txop']:.4f} | "
        f"fulfillment={metrics['avg_fulfillment']:.4f}"
    )


def log_episode_metrics(run, episode_idx: int, metrics: dict):
    if run is None:
        return
    import wandb

    wandb.log(metrics, step=episode_idx + 1)


def log_wandb_image(run, key: str, image_path: Path):
    if run is None or image_path is None:
        return
    import wandb

    wandb.log({key: wandb.Image(str(image_path))})


def summarize_metrics(all_episode_metrics):
    summary = {}
    if not all_episode_metrics:
        return summary

    keys = all_episode_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_episode_metrics if key in m]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def load_policy(args, env, device):
    configure_algorithm_flags(args)
    policy = R_MAPPOPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
    )

    model_dir = Path(args.model_dir)
    actor_path = model_dir / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"actor checkpoint not found: {actor_path}")

    actor_state_dict = torch.load(actor_path, map_location=device)
    policy.actor.load_state_dict(actor_state_dict)
    policy.actor.eval()
    return policy
