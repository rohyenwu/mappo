#!/usr/bin/env python
"""Train MAPPO on the WiFi v7.1 scenario-rotation environment."""

import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from onpolicy.envs.wifi_v7_1.wifi_env import WiFiEnvV7_1
from onpolicy.eval.wifi_v5.utils import parse_mu_profile


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


def make_env(all_args, seed_offset: int):
    mu_profile = parse_mu_profile(getattr(all_args, "mu_profile", None))
    scenario_profile = parse_scenario_profile(getattr(all_args, "scenario_profile", None))

    def get_env_fn(rank: int):
        def init_env():
            env = WiFiEnvV7_1(
                max_mld=all_args.max_mld,
                max_sld=all_args.max_sld,
                scenario_profile=scenario_profile,
                round_length=all_args.round_length,
                mu_range=(all_args.mu_min, all_args.mu_max),
                mu_profile=mu_profile,
                eta=all_args.eta,
                zeta=all_args.zeta,
                r_sld=all_args.r_sld,
                c_idle=all_args.c_idle,
                theta_scale=all_args.theta_scale,
            )
            env.seed(seed_offset + rank * 1000)
            return env

        return init_env

    return get_env_fn


def make_train_env(all_args):
    fns = [make_env(all_args, all_args.seed)(i) for i in range(all_args.n_rollout_threads)]
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv(fns)
    return ShareSubprocVecEnv(fns)


def make_eval_env(all_args):
    fns = [
        make_env(all_args, all_args.seed * 50000)(i)
        for i in range(all_args.n_eval_rollout_threads)
    ]
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv(fns)
    return ShareSubprocVecEnv(fns)


def parse_args(args, parser):
    parser.add_argument("--max_mld", type=int, default=30, help="Maximum number of MLD stations")
    parser.add_argument("--max_sld", type=int, default=10, help="Maximum number of SLD stations")
    parser.add_argument(
        "--scenario_profile",
        type=str,
        default="10:2,10:10,20:5,30:2,30:10",
        help="Comma-separated active MLD:SLD scenarios, e.g. '10:5,15:5,30:5'.",
    )
    parser.add_argument(
        "--scenario_order",
        type=str,
        default="sequential",
        choices=["sequential"],
        help="Scenario order. v7.1 initially supports sequential order only.",
    )
    parser.add_argument(
        "--scenario_interval_episodes",
        type=int,
        default=60,
        help="Number of PPO update episodes to keep each scenario.",
    )
    parser.add_argument(
        "--round_length",
        type=int,
        default=50,
        help="Number of access opportunities per round",
    )
    parser.add_argument("--mu_min", type=float, default=0.01, help="Minimum demand rate")
    parser.add_argument("--mu_max", type=float, default=0.12, help="Maximum demand rate")
    parser.add_argument(
        "--mu_profile",
        type=str,
        default=None,
        help="Comma-separated per-MLD demand rates. Overrides mu_min/mu_max when set.",
    )
    parser.add_argument("--eta", type=float, default=1.0, help="SLD deficit penalty scale")
    parser.add_argument("--zeta", type=float, default=1.0, help="SLD protection bonus scale")
    parser.add_argument("--r_sld", type=float, default=0.3, help="Reserved for compatibility")
    parser.add_argument("--c_idle", type=float, default=0.3, help="Idle opportunity penalty")
    parser.add_argument(
        "--theta_scale",
        type=float,
        default=1.0,
        help="Scale factor for the SLD protection threshold",
    )
    parser.add_argument(
        "--rounds_per_update",
        type=int,
        default=1,
        help="Number of WiFi rounds to collect before each policy update",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="wifi_v7_1",
        help="Weights & Biases project name for WiFi v7.1 runs",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity. Overrides --user_name when set.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional Weights & Biases group name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.rounds_per_update < 1:
        raise ValueError("--rounds_per_update must be >= 1")
    if all_args.scenario_interval_episodes < 1:
        raise ValueError("--scenario_interval_episodes must be >= 1")

    scenarios = parse_scenario_profile(all_args.scenario_profile)
    max_profile_mld = max(active_mld for active_mld, _ in scenarios)
    max_profile_sld = max(active_sld for _, active_sld in scenarios)
    if max_profile_mld > all_args.max_mld:
        raise ValueError(
            f"--max_mld {all_args.max_mld} is smaller than scenario MLD {max_profile_mld}"
        )
    if max_profile_sld > all_args.max_sld:
        raise ValueError(
            f"--max_sld {all_args.max_sld} is smaller than scenario SLD {max_profile_sld}"
        )

    all_args.num_mld = all_args.max_mld
    all_args.num_sld = all_args.max_sld
    all_args.episode_length = all_args.round_length * all_args.rounds_per_update

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError(f"Unknown algorithm: {all_args.algorithm_name}")

    if all_args.cuda and torch.cuda.is_available():
        print("GPU enabled")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("CPU enabled")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    repo_root = Path(__file__).resolve().parents[3]
    base_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
        / "results"
        / all_args.env_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    if all_args.use_wandb:
        import wandb

        wandb_entity = all_args.wandb_entity or all_args.user_name
        wandb_run_name = (
            all_args.wandb_run_name
            or f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}"
        )
        run = wandb.init(
            config=all_args,
            project=all_args.wandb_project,
            entity=wandb_entity,
            group=all_args.wandb_group,
            notes=socket.gethostname(),
            name=wandb_run_name,
            dir=str(base_dir),
            job_type="training",
            reinit=True,
        )
        run_dir = Path(wandb.run.dir)
    else:
        existing = [
            int(str(f.name).split("run")[1])
            for f in base_dir.iterdir()
            if str(f.name).startswith("run")
        ] if base_dir.exists() else []
        curr_run = f"run{max(existing) + 1}" if existing else "run1"
        run_dir = base_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)

    model_save_dir = (
        repo_root
        / "model"
        / all_args.env_name
        / all_args.algorithm_name
        / f"{all_args.experiment_name}_seed{all_args.seed}"
    )
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model checkpoints will be saved to: {model_save_dir}")

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-wifi-v7.1-{all_args.experiment_name}"
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": len(envs.observation_space),
        "device": device,
        "run_dir": run_dir,
        "save_dir": model_save_dir,
    }

    from onpolicy.runner.shared.wifi_v3_runner import WiFiV2Runner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir) + "/summary.json")
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
