#!/usr/bin/env python
"""Sweep pure BEB throughput on 2.4 GHz and 5 GHz links.

This is a standalone reference experiment. It does not evaluate RL agents or
the mixed MLD/SLD environment. Instead, it asks a simpler question:

    If every STA on a single link uses BEB, how much throughput does that link
    produce as the number of contending STAs increases?

The BEB update mirrors the WiFi v3 SLD CSMA/CA behavior:
- saturated STAs always have traffic,
- STAs transmit when backoff reaches zero,
- backoff counts down only on idle TXOPs,
- success resets CW to CW_MIN,
- collision doubles CW up to CW_MAX.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CW_MIN = 16
CW_MAX = 1024
RETRY_LIMIT = 6


@dataclass
class StaState:
    cw: int = CW_MIN
    backoff: int = 0
    retry: int = 0


def draw_backoff(rng, cw):
    return int(rng.integers(0, cw))


def reset_states(num_sta, rng):
    return [
        StaState(cw=CW_MIN, backoff=draw_backoff(rng, CW_MIN), retry=0)
        for _ in range(num_sta)
    ]


def simulate_round(num_sta, round_length, rng):
    states = reset_states(num_sta, rng)
    successes = 0
    collisions = 0
    idle = 0
    attempts = 0

    for _ in range(round_length):
        txers = [idx for idx, sta in enumerate(states) if sta.backoff == 0]
        attempts += len(txers)

        if len(txers) == 0:
            idle += 1
            for sta in states:
                if sta.backoff > 0:
                    sta.backoff -= 1
            continue

        if len(txers) == 1:
            successes += 1
            sta = states[txers[0]]
            sta.cw = CW_MIN
            sta.retry = 0
            sta.backoff = draw_backoff(rng, sta.cw)
            continue

        collisions += 1
        for idx in txers:
            sta = states[idx]
            sta.retry += 1
            if sta.retry > RETRY_LIMIT:
                sta.cw = CW_MIN
                sta.retry = 0
            else:
                sta.cw = min(sta.cw * 2, CW_MAX)
            sta.backoff = draw_backoff(rng, sta.cw)

    return {
        "successes": successes,
        "collisions": collisions,
        "idle": idle,
        "attempts": attempts,
        "throughput_per_round": successes,
        "throughput": successes / max(round_length, 1),
        "collision_rate": collisions / max(round_length, 1),
        "idle_rate": idle / max(round_length, 1),
        "attempt_rate": attempts / max(round_length, 1),
    }


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def sweep_link(link_name, min_sta, max_sta, round_length, episodes, seed):
    rows = []
    for num_sta in range(min_sta, max_sta + 1):
        per_metric = {}
        for episode in range(episodes):
            rng = np.random.default_rng(seed + episode + num_sta * 100_003)
            metrics = simulate_round(num_sta, round_length, rng)
            for key, value in metrics.items():
                per_metric.setdefault(key, []).append(value)

        row = {
            "link": link_name,
            "num_sta": num_sta,
            "round_length": round_length,
            "episodes": episodes,
            "cw_min": CW_MIN,
            "cw_max": CW_MAX,
            "retry_limit": RETRY_LIMIT,
        }
        for key, values in sorted(per_metric.items()):
            stats = summarize(values)
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_min"] = stats["min"]
            row[f"{key}_max"] = stats["max"]
        rows.append(row)
    return rows


def _plot_curve(ax, x, y, yerr, hide_error_bars):
    if hide_error_bars:
        ax.plot(x, y, marker="o", linewidth=1.3)
    else:
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.3, capsize=2)


def save_plot(rows, output_dir, hide_error_bars=False, separate_link_plots=False):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[BEB sweep] matplotlib is not installed; skipping plot.")
        return []

    plot_specs = [
        (
            "throughput",
            "Throughput (successes / TXOP)",
            "Pure BEB Link Normalized Throughput Sweep",
            "beb_link_throughput_sweep.png",
        ),
        (
            "throughput_per_round",
            "Throughput (successes / round)",
            "Pure BEB Link Round Throughput Sweep",
            "beb_link_round_throughput_sweep.png",
        ),
    ]
    output_paths = []

    for metric, ylabel, title, filename in plot_specs:
        link_names = ["2.4GHz", "5GHz"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, link_name in zip(axes, link_names):
            link_rows = [row for row in rows if row["link"] == link_name]
            x = [row["num_sta"] for row in link_rows]
            y = [row[f"{metric}_mean"] for row in link_rows]
            yerr = [row[f"{metric}_std"] for row in link_rows]

            _plot_curve(ax, x, y, yerr, hide_error_bars)
            ax.set_title(link_name)
            ax.set_xlabel("Number of STAs")
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0.0)
            ax.grid(alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

        if separate_link_plots:
            for link_name in link_names:
                link_rows = [row for row in rows if row["link"] == link_name]
                x = [row["num_sta"] for row in link_rows]
                y = [row[f"{metric}_mean"] for row in link_rows]
                yerr = [row[f"{metric}_std"] for row in link_rows]

                fig, ax = plt.subplots(figsize=(6, 4.5))
                _plot_curve(ax, x, y, yerr, hide_error_bars)
                ax.set_title(f"{title} - {link_name}")
                ax.set_xlabel("Number of STAs")
                ax.set_ylabel(ylabel)
                ax.set_ylim(bottom=0.0)
                ax.grid(alpha=0.3)
                fig.tight_layout()

                link_slug = link_name.lower().replace(".", "_").replace("ghz", "ghz")
                separate_path = output_dir / f"{Path(filename).stem}_{link_slug}{Path(filename).suffix}"
                fig.savefig(separate_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                output_paths.append(separate_path)

        system_rows = {}
        for row in rows:
            num_sta = row["num_sta"]
            system_rows.setdefault(num_sta, {})[row["link"]] = row

        x = []
        y = []
        yerr = []
        for num_sta in sorted(system_rows):
            link_pair = system_rows[num_sta]
            if "2.4GHz" not in link_pair or "5GHz" not in link_pair:
                continue
            row_24 = link_pair["2.4GHz"]
            row_5 = link_pair["5GHz"]
            x.append(num_sta)
            y.append(row_24[f"{metric}_mean"] + row_5[f"{metric}_mean"])
            # Independent Monte Carlo seeds are used per link, so variances add.
            yerr.append(
                float(
                    np.sqrt(row_24[f"{metric}_std"] ** 2 + row_5[f"{metric}_std"] ** 2)
                )
            )

        fig, ax = plt.subplots(figsize=(6, 4.5))
        _plot_curve(ax, x, y, yerr, hide_error_bars)
        ax.set_title(f"{title} - System")
        ax.set_xlabel("Number of STAs per link")
        ax.set_ylabel(ylabel.replace("Throughput", "System throughput"))
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        system_path = output_dir / f"{Path(filename).stem}_system{Path(filename).suffix}"
        fig.savefig(system_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(system_path)

    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep pure BEB throughput for 2.4 GHz and 5 GHz links."
    )
    parser.add_argument("--env_name", type=str, default="WiFi_beb_eval")
    parser.add_argument("--algorithm_name", type=str, default="beb")
    parser.add_argument("--experiment_name", type=str, default="beb_link_sweep")
    parser.add_argument("--min_sta", type=int, default=1)
    parser.add_argument("--max_sta", type=int, default=30)
    parser.add_argument("--round_length", type=int, default=50)
    parser.add_argument(
        "--episodes",
        "--eval_episodes",
        dest="episodes",
        type=int,
        default=1000,
        help="Number of Monte Carlo rounds per STA count.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--num_mld",
        type=int,
        default=None,
        help="Compatibility with WiFi eval commands; not used by this STA sweep.",
    )
    parser.add_argument(
        "--num_sld",
        type=int,
        default=None,
        help="Compatibility with WiFi eval commands; not used by this STA sweep.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Compatibility with RL eval commands; not used.",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_true",
        help="Compatibility with RL eval commands; not used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mappo/onpolicy/scripts/results/WiFi/wifi_v3/beb_link_sweep",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--hide_error_bars",
        action="store_true",
        help="Plot mean curves without standard-deviation error bars.",
    )
    parser.add_argument(
        "--separate_link_plots",
        action="store_true",
        help="Also save separate image files for 2.4 GHz and 5 GHz plots.",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="beb_link_throughput_sweep")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--user_name",
        type=str,
        default=None,
        help="W&B entity/user name. Used when --wandb_entity is not provided.",
    )
    return parser.parse_args()


def log_to_wandb(args, rows, csv_path, json_path, plot_paths):
    if not args.wandb_project:
        return

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is required when --wandb_project is provided."
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or args.user_name,
        group=args.wandb_group,
        name=args.wandb_run_name,
        job_type="beb_link_sweep",
        config={
            "env_name": args.env_name,
            "algorithm_name": args.algorithm_name,
            "experiment_name": args.experiment_name,
            "min_sta": args.min_sta,
            "max_sta": args.max_sta,
            "round_length": args.round_length,
            "episodes": args.episodes,
            "seed": args.seed,
            "cw_min": CW_MIN,
            "cw_max": CW_MAX,
            "retry_limit": RETRY_LIMIT,
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=[
            "link",
            "num_sta",
            "throughput_mean",
            "throughput_std",
            "throughput_per_round_mean",
            "throughput_per_round_std",
            "collision_rate_mean",
            "idle_rate_mean",
            "attempt_rate_mean",
        ]
    )
    for row in rows:
        table.add_data(
            row["link"],
            row["num_sta"],
            row["throughput_mean"],
            row["throughput_std"],
            row["throughput_per_round_mean"],
            row["throughput_per_round_std"],
            row["collision_rate_mean"],
            row["idle_rate_mean"],
            row["attempt_rate_mean"],
        )

    log_payload = {"beb_link_sweep/table": table}
    for plot_path in plot_paths:
        log_payload[f"beb_link_sweep/{plot_path.stem}"] = wandb.Image(str(plot_path))
    wandb.log(log_payload)

    artifact = wandb.Artifact("beb_link_throughput_sweep", type="evaluation")
    artifact.add_file(str(csv_path))
    artifact.add_file(str(json_path))
    for plot_path in plot_paths:
        artifact.add_file(str(plot_path))
    run.log_artifact(artifact)
    run.finish()


def main():
    args = parse_args()
    if args.min_sta < 1:
        raise ValueError("--min_sta must be >= 1")
    if args.max_sta < args.min_sta:
        raise ValueError("--max_sta must be >= --min_sta")
    if args.round_length < 1:
        raise ValueError("--round_length must be >= 1")
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")

    rows = []
    for link_name in ["2.4GHz", "5GHz"]:
        rows.extend(
            sweep_link(
                link_name=link_name,
                min_sta=args.min_sta,
                max_sta=args.max_sta,
                round_length=args.round_length,
                episodes=args.episodes,
                seed=args.seed + (0 if link_name == "2.4GHz" else 10_000_000),
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "beb_link_throughput_sweep.csv"
    json_path = output_dir / "beb_link_throughput_sweep.json"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    for row in rows:
        print(
            f"{row['link']:>6} STA={row['num_sta']:02d} | "
            f"throughput={row['throughput_mean']:.4f} "
            f"+/- {row['throughput_std']:.4f} | "
            f"successes_round={row['throughput_per_round_mean']:.2f} | "
            f"collision={row['collision_rate_mean']:.4f} | "
            f"idle={row['idle_rate_mean']:.4f}"
        )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")

    plot_paths = []
    if args.plot or args.wandb_project:
        plot_paths = save_plot(
            rows,
            output_dir,
            hide_error_bars=args.hide_error_bars,
            separate_link_plots=args.separate_link_plots,
        )
        for plot_path in plot_paths:
            print(f"Saved plot: {plot_path}")

    log_to_wandb(args, rows, csv_path, json_path, plot_paths)


if __name__ == "__main__":
    main()
