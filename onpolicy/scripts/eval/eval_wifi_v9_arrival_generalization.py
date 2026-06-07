#!/usr/bin/env python
"""Run WiFi v9 packet-arrival-rate generalization sweeps.

This wrapper evaluates the same topology under heterogeneous packet-arrival
profiles. Each profile assigns an MLD-specific mu value through --mu_profile,
then compares a trained RL checkpoint against BEB and saves compact CSV/JSON
summaries plus trend plots.
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL_NAME = (
    "wifi_v9_train_airtime50ms_m15m25_s3s5_parallel_vec4_d2lt_mldsucc1_"
    "sld07_10_ntop1_cidle03_1600k_lr1e4_ent5e3_seed1"
)
DEFAULT_RATES = [0.005, 0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.20]
DEFAULT_PROFILE_MODES = [
    "random_id",
    "random_low_ood",
    "random_high_ood",
    "random_wide_ood",
    "random_mixed_ood",
]
PROFILE_MODE_CHOICES = [
    *DEFAULT_PROFILE_MODES,
    "uniform_sweep",
]
METRIC_KEYS = [
    "mbps/system",
    "mbps/mld_total",
    "mbps/sld_total",
    "collision_rate/system_per_event",
    "success_rate/system_per_event",
    "idle_rate/system_per_event",
    "action/transmit_ratio",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    return repo_root().parent


def default_model_dir() -> Path:
    return repo_root() / "model" / "WiFi_v9" / "mappo" / DEFAULT_MODEL_NAME


def parse_scenario(text: str):
    normalized = text.lower().replace("m", "").replace("s", "").replace("_", ":")
    parts = [part for part in normalized.split(":") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid scenario '{text}'. Expected m10_s5 or 10:5."
        )
    mld, sld = int(parts[0]), int(parts[1])
    if mld < 1 or sld < 0:
        raise argparse.ArgumentTypeError("Scenario counts must be positive.")
    return mld, sld


def rate_slug(rate: float) -> str:
    return f"mu{rate:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def scenario_slug(mld: int, sld: int) -> str:
    return f"m{mld}_s{sld}"


def result_dir(env_name: str, eval_name: str, experiment_name: str) -> Path:
    return workspace_root() / "scripts" / "eval_results" / env_name / eval_name / experiment_name


def summary_path(policy: str, experiment_name: str) -> Path:
    eval_name = "wifi_v9_rl_mbps" if policy == "rl" else "wifi_v9_beb_mbps"
    filename = "rl_mbps_summary.json" if policy == "rl" else "beb_mbps_summary.json"
    return result_dir("WiFi_v9", eval_name, experiment_name) / filename


def profile_slug(profile_case):
    name = profile_case["name"]
    seed = profile_case.get("seed")
    if seed is None:
        return name
    return f"{name}_pseed{seed}"


def summarize_profile(values):
    active_values = [float(value) for value in values]
    count = len(active_values)
    mean_value = sum(active_values) / count if count else 0.0
    variance = (
        sum((value - mean_value) ** 2 for value in active_values) / count
        if count
        else 0.0
    )
    low_ood_count = sum(value < 0.01 for value in active_values)
    high_ood_count = sum(value > 0.12 for value in active_values)
    return {
        "mean_mu": mean_value,
        "min_mu": min(active_values) if active_values else 0.0,
        "max_mu": max(active_values) if active_values else 0.0,
        "std_mu": math.sqrt(variance),
        "low_ood_fraction": low_ood_count / count if count else 0.0,
        "high_ood_fraction": high_ood_count / count if count else 0.0,
        "ood_fraction": (low_ood_count + high_ood_count) / count if count else 0.0,
    }


def make_mu_profile(args, active_mld: int, mode: str, profile_seed: int = None, rate=None):
    values = [0.0] * int(args.max_mld)
    if mode == "uniform_sweep":
        active_values = [float(rate)] * active_mld
    else:
        seed = int(args.seed) * 1000003 + int(profile_seed) * 9176 + active_mld
        rng = random.Random(seed)
        if mode == "random_id":
            active_values = [rng.uniform(0.01, 0.12) for _ in range(active_mld)]
        elif mode == "random_low_ood":
            active_values = [rng.uniform(0.001, 0.01) for _ in range(active_mld)]
        elif mode == "random_high_ood":
            active_values = [rng.uniform(0.12, 0.20) for _ in range(active_mld)]
        elif mode == "random_wide_ood":
            active_values = [rng.uniform(0.001, 0.20) for _ in range(active_mld)]
        elif mode == "random_mixed_ood":
            low_count = active_mld * 3 // 10
            id_count = active_mld * 4 // 10
            high_count = active_mld - low_count - id_count
            active_values = (
                [rng.uniform(0.001, 0.01) for _ in range(low_count)]
                + [rng.uniform(0.01, 0.12) for _ in range(id_count)]
                + [rng.uniform(0.12, 0.20) for _ in range(high_count)]
            )
            rng.shuffle(active_values)
        else:
            raise ValueError(f"Unsupported profile mode: {mode}")

    values[:active_mld] = active_values
    return values, summarize_profile(active_values)


def build_profile_cases(args, active_mld: int):
    cases = []
    for mode in args.profile_mode:
        if mode == "uniform_sweep":
            for rate in args.rates:
                values, stats = make_mu_profile(args, active_mld, mode, rate=rate)
                cases.append(
                    {
                        "name": rate_slug(rate),
                        "mode": mode,
                        "seed": None,
                        "rate": float(rate),
                        "values": values,
                        "stats": stats,
                    }
                )
            continue

        for profile_seed in args.profile_seed:
            values, stats = make_mu_profile(args, active_mld, mode, profile_seed=profile_seed)
            cases.append(
                {
                    "name": mode,
                    "mode": mode,
                    "seed": int(profile_seed),
                    "rate": stats["mean_mu"],
                    "values": values,
                    "stats": stats,
                }
            )
    return cases


def mu_profile_text(values):
    return ",".join(f"{float(value):.8f}" for value in values)


def build_common_args(args, mld: int, sld: int, profile_case, experiment_name: str):
    command_args = [
        "--env_name",
        "WiFi_v9",
        "--algorithm_name",
        "mappo",
        "--experiment_name",
        experiment_name,
        "--max_mld",
        str(args.max_mld),
        "--max_sld",
        str(args.max_sld),
        "--num_mld",
        str(mld),
        "--num_sld",
        str(sld),
        "--round_length",
        str(args.round_length),
        "--eta",
        str(args.eta),
        "--zeta",
        str(args.zeta),
        "--c_idle",
        str(args.c_idle),
        "--collision_penalty",
        str(args.collision_penalty),
        "--non_top_tx_penalty",
        str(args.non_top_tx_penalty),
        "--theta_scale",
        str(args.theta_scale),
        "--sld_target_low_scale",
        str(args.sld_target_low_scale),
        "--sld_target_high_scale",
        str(args.sld_target_high_scale),
        "--sld_target_bonus",
        str(args.sld_target_bonus),
        "--mld_success_reward",
        str(args.mld_success_reward),
        "--eval_episodes",
        str(args.eval_episodes),
        "--eval_duration_sec",
        str(args.eval_duration_sec),
        "--live_log_interval_sec",
        str(args.live_log_interval_sec),
        "--slot_time_sec",
        str(args.slot_time_sec),
        "--seed",
        str(args.seed),
        "--use_wandb",
    ]
    if profile_case["mode"] == "uniform_sweep":
        command_args.extend(["--mu_min", str(profile_case["rate"]), "--mu_max", str(profile_case["rate"])])
    else:
        command_args.extend(["--mu_profile", mu_profile_text(profile_case["values"])])
    return command_args


def run_eval(args, policy: str, mld: int, sld: int, profile_case):
    label = f"{scenario_slug(mld, sld)}_{profile_slug(profile_case)}_{args.tag}"
    experiment_name = f"arrival_gen_{policy}_{label}"
    output_summary = summary_path(policy, experiment_name)
    if output_summary.exists() and not args.force:
        print(
            f"[Skip] {policy.upper()} {scenario_slug(mld, sld)} "
            f"{profile_slug(profile_case)}: {output_summary}"
        )
        return experiment_name, output_summary

    module = (
        "onpolicy.scripts.eval.eval_wifi_v9_rl_mbps"
        if policy == "rl"
        else "onpolicy.scripts.eval.eval_wifi_v9_beb_mbps"
    )
    command = [
        sys.executable,
        "-m",
        module,
        *build_common_args(args, mld, sld, profile_case, experiment_name),
    ]
    if policy == "rl":
        command.extend(
            [
                "--model_dir",
                str(args.model_dir),
                "--debug_prob_steps",
                "0",
            ]
        )
        command.append("--deterministic" if args.deterministic else "--stochastic")

    env = os.environ.copy()
    shim_path = repo_root().parent / "cap_sim" / "shims"
    python_paths = [str(repo_root())]
    if shim_path.exists():
        python_paths.append(str(shim_path))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    print(
        f"[Run] {policy.upper()} {scenario_slug(mld, sld)} "
        f"{profile_slug(profile_case)} mean_mu={profile_case['stats']['mean_mu']:.4f}"
    )
    try:
        subprocess.run(command, cwd=str(repo_root()), env=env, check=True)
    except subprocess.CalledProcessError:
        if output_summary.exists():
            print(
                f"[Warn] {policy.upper()} process returned a non-zero exit code, "
                f"but the summary was saved: {output_summary}"
            )
        else:
            raise
    return experiment_name, output_summary


def load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows(run_records):
    rows = []
    for record in run_records:
        mld, sld = record["mld"], record["sld"]
        profile_case = record["profile_case"]
        profile_stats = profile_case["stats"]
        beb = load_summary(record["beb_summary"])
        rl = load_summary(record["rl_summary"])
        row = {
            "scenario": scenario_slug(mld, sld),
            "num_mld": mld,
            "num_sld": sld,
            "profile": profile_slug(profile_case),
            "profile_mode": profile_case["mode"],
            "profile_seed": "" if profile_case["seed"] is None else profile_case["seed"],
            "mean_mu": profile_stats["mean_mu"],
            "min_mu": profile_stats["min_mu"],
            "max_mu": profile_stats["max_mu"],
            "std_mu": profile_stats["std_mu"],
            "low_ood_fraction": profile_stats["low_ood_fraction"],
            "high_ood_fraction": profile_stats["high_ood_fraction"],
            "ood_fraction": profile_stats["ood_fraction"],
        }
        for key in METRIC_KEYS:
            beb_value = float(beb.get(key, 0.0))
            rl_value = float(rl.get(key, 0.0))
            row[f"beb/{key}"] = beb_value
            row[f"rl/{key}"] = rl_value
            if key == "mbps/system":
                row["rl_vs_beb_system_mbps_gain_percent"] = (
                    (rl_value / beb_value - 1.0) * 100.0 if beb_value > 0.0 else 0.0
                )
        rows.append(row)
    return rows


def save_outputs(args, rows):
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{args.tag}_arrival_generalization.csv"
    json_path = output_dir / f"{args.tag}_arrival_generalization.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    plot_paths = save_plots(output_dir, args.tag, rows)
    print("[Saved]")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    for path in plot_paths:
        print(f"  {path}")
    return csv_path, json_path, plot_paths


def upload_wandb(args, rows, csv_path: Path, json_path: Path, plot_paths):
    if not args.wandb_project:
        return

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is required when --wandb_project is provided."
        ) from exc

    run_name = args.wandb_run_name or f"{args.tag}_arrival_generalization"
    group_name = args.wandb_group or "wifi_v9_arrival_generalization"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group_name,
        name=run_name,
        job_type="arrival_generalization",
        config={
            "tag": args.tag,
            "scenarios": [scenario_slug(mld, sld) for mld, sld in args.scenario],
            "profile_mode": args.profile_mode,
            "profile_seed": args.profile_seed,
            "rates": args.rates,
            "model_dir": str(args.model_dir),
            "eval_duration_sec": args.eval_duration_sec,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "max_mld": args.max_mld,
            "max_sld": args.max_sld,
            "round_length": args.round_length,
            "eta": args.eta,
            "zeta": args.zeta,
            "c_idle": args.c_idle,
            "collision_penalty": args.collision_penalty,
            "non_top_tx_penalty": args.non_top_tx_penalty,
            "theta_scale": args.theta_scale,
            "sld_target_low_scale": args.sld_target_low_scale,
            "sld_target_high_scale": args.sld_target_high_scale,
            "sld_target_bonus": args.sld_target_bonus,
            "mld_success_reward": args.mld_success_reward,
            "slot_time_sec": args.slot_time_sec,
        },
        reinit=True,
    )

    if rows:
        columns = list(rows[0].keys())
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[key] for key in columns])
    else:
        table = wandb.Table(columns=[])

    image_payload = {
        f"figures/{path.stem}": wandb.Image(str(path))
        for path in plot_paths
    }
    artifact = wandb.Artifact(
        name=f"{args.tag}_arrival_generalization_outputs",
        type="eval_results",
    )
    artifact.add_file(str(csv_path))
    artifact.add_file(str(json_path))
    for path in plot_paths:
        artifact.add_file(str(path))

    wandb.log(
        {
            "arrival_generalization/table": table,
            "arrival_generalization/row_count": len(rows),
            **image_payload,
        }
    )
    run.log_artifact(artifact)
    run.finish()
    print(
        f"[W&B] Uploaded arrival generalization results to "
        f"{args.wandb_project}/{run_name}"
    )


def save_plots(output_dir: Path, tag: str, rows):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[Plot] matplotlib is not installed; skipping plots.")
        return []

    scenarios = sorted({row["scenario"] for row in rows})
    output_paths = []
    for metric_key, ylabel, suffix in [
        ("mbps/system", "System throughput (Mbps)", "system_mbps"),
        ("collision_rate/system_per_event", "Collision rate / event", "collision_rate"),
        ("success_rate/system_per_event", "Success rate / event", "success_rate"),
    ]:
        fig, axes = plt.subplots(
            1,
            len(scenarios),
            figsize=(max(6.5, 4.3 * len(scenarios)), 4.4),
            sharey=True,
        )
        if len(scenarios) == 1:
            axes = [axes]

        for ax, scenario in zip(axes, scenarios):
            scenario_rows = [row for row in rows if row["scenario"] == scenario]
            grouped = {}
            for row in scenario_rows:
                grouped.setdefault(row["profile"], []).append(row)
            profile_names = sorted(
                grouped,
                key=lambda name: (
                    sum(row["mean_mu"] for row in grouped[name]) / len(grouped[name]),
                    name,
                ),
            )
            x_values = list(range(len(profile_names)))
            beb_values = [
                sum(row[f"beb/{metric_key}"] for row in grouped[name]) / len(grouped[name])
                for name in profile_names
            ]
            rl_values = [
                sum(row[f"rl/{metric_key}"] for row in grouped[name]) / len(grouped[name])
                for name in profile_names
            ]
            ax.plot(x_values, beb_values, marker="o", linewidth=2.0, label="BEB")
            ax.plot(x_values, rl_values, marker="s", linewidth=2.0, label="RL")
            ax.set_title(scenario)
            ax.set_xlabel("traffic profile")
            ax.set_xticks(x_values)
            ax.set_xticklabels(profile_names, rotation=25, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False, fontsize=8)

        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"WiFi v9 Arrival-Rate Generalization - {ylabel}")
        fig.tight_layout()
        output_path = output_dir / f"{tag}_arrival_generalization_{suffix}.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WiFi v9 RL vs BEB over packet arrival rates."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        type=parse_scenario,
        default=None,
        help="Topology to evaluate, e.g. m10_s5 or 10:5. Can be repeated.",
    )
    parser.add_argument(
        "--profile_mode",
        nargs="+",
        choices=PROFILE_MODE_CHOICES,
        default=DEFAULT_PROFILE_MODES,
        help=(
            "Traffic profile modes. Use uniform_sweep to reproduce the old "
            "mu_min=mu_max sweep over --rates."
        ),
    )
    parser.add_argument(
        "--profile_seed",
        nargs="+",
        type=int,
        default=[1],
        help="Seeds used to generate random heterogeneous traffic profiles.",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=DEFAULT_RATES,
        help="Uniform rates used only when --profile_mode includes uniform_sweep.",
    )
    parser.add_argument("--model_dir", type=Path, default=default_model_dir())
    parser.add_argument("--output_dir", type=str, default="scripts/eval_results/WiFi_v9/arrival_generalization")
    parser.add_argument("--tag", type=str, default="m10_s5")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--max_mld", type=int, default=30)
    parser.add_argument("--max_sld", type=int, default=10)
    parser.add_argument("--round_length", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--eval_duration_sec", type=float, default=10.0)
    parser.add_argument("--live_log_interval_sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--zeta", type=float, default=0.2)
    parser.add_argument("--c_idle", type=float, default=0.3)
    parser.add_argument("--collision_penalty", type=float, default=1.0)
    parser.add_argument("--non_top_tx_penalty", type=float, default=1.0)
    parser.add_argument("--theta_scale", type=float, default=1.0)
    parser.add_argument("--sld_target_low_scale", type=float, default=0.7)
    parser.add_argument("--sld_target_high_scale", type=float, default=1.0)
    parser.add_argument("--sld_target_bonus", type=float, default=0.5)
    parser.add_argument("--mld_success_reward", type=float, default=1.0)
    parser.add_argument("--slot_time_sec", type=float, default=9e-6)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()
    if args.scenario is None:
        args.scenario = [(10, 5)]
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    return args


def main():
    args = parse_args()
    run_records = []
    for mld, sld in args.scenario:
        for profile_case in build_profile_cases(args, mld):
            beb_exp, beb_summary = run_eval(args, "beb", mld, sld, profile_case)
            rl_exp, rl_summary = run_eval(args, "rl", mld, sld, profile_case)
            run_records.append(
                {
                    "mld": mld,
                    "sld": sld,
                    "profile_case": profile_case,
                    "beb_experiment": beb_exp,
                    "rl_experiment": rl_exp,
                    "beb_summary": beb_summary,
                    "rl_summary": rl_summary,
                }
            )

    rows = collect_rows(run_records)
    csv_path, json_path, plot_paths = save_outputs(args, rows)
    upload_wandb(args, rows, csv_path, json_path, plot_paths)


if __name__ == "__main__":
    main()
