#!/usr/bin/env python
"""Compare WiFi v7.1 BEB and RL Mbps grid summaries."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create BEB-vs-RL Mbps bar charts from WiFi v7.1 grid summaries."
    )
    parser.add_argument("--beb_base_dir", type=str, required=True)
    parser.add_argument("--rl_base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="wifi_v7_1_beb_vs_rl_mbps.png")
    parser.add_argument("--title", type=str, default="WiFi v7.1 BEB vs RL Mbps")
    parser.add_argument("--m_values", nargs="+", type=int, default=[10, 15, 20, 25, 30])
    parser.add_argument("--s_values", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--beb_suffix", type=str, default="_beb_mbps_ep3")
    parser.add_argument(
        "--rl_suffix",
        type=str,
        required=True,
        help="RL experiment suffix after m{m}_s{s}, for example _v7_1_..._mbps.",
    )
    parser.add_argument("--hide_bar_labels", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_image_key", type=str, default="beb_vs_rl_mbps_grid")
    return parser.parse_args()


def load_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_metric(args, cases, metric_key, metric_title, metric_slug, y_label):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [case["label"] for case in cases]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    beb_values = [float(case["beb"].get(metric_key, 0.0)) for case in cases]
    rl_values = [float(case["rl"].get(metric_key, 0.0)) for case in cases]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(labels)), 5.0))
    bars_beb = ax.bar(x - width / 2.0, beb_values, width=width, label="BEB", color="#4c78a8")
    bars_rl = ax.bar(x + width / 2.0, rl_values, width=width, label="RL", color="#f58518")

    if not args.hide_bar_labels:
        for bars in (bars_beb, bars_rl):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(f"{args.title} - {metric_title}")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ymax = max(beb_values + rl_values) if (beb_values or rl_values) else 0.0
    ax.set_ylim(0.0, max(1.0, ymax * 1.18))
    fig.tight_layout()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.output_name).stem
    suffix = Path(args.output_name).suffix or ".png"
    output_path = output_dir / f"{stem}_{metric_slug}{suffix}"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
        import numpy  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and numpy are required for plot_wifi_v7_1_mbps_grid_compare.py"
        ) from exc

    beb_base_dir = Path(args.beb_base_dir).expanduser()
    rl_base_dir = Path(args.rl_base_dir).expanduser()

    cases = []
    for m in args.m_values:
        for s in args.s_values:
            label = f"m{m}_s{s}"
            beb_path = beb_base_dir / f"{label}{args.beb_suffix}" / "beb_mbps_summary.json"
            rl_path = rl_base_dir / f"{label}{args.rl_suffix}" / "rl_mbps_summary.json"
            cases.append(
                {
                    "label": label,
                    "beb_path": str(beb_path),
                    "rl_path": str(rl_path),
                    "beb": load_summary(beb_path),
                    "rl": load_summary(rl_path),
                }
            )

    metric_specs = [
        ("mbps/2_4GHz/total", "2.4GHz", "2_4ghz", "Throughput (Mbps)"),
        ("mbps/5GHz/total", "5GHz", "5ghz", "Throughput (Mbps)"),
        ("mbps/mld_total", "MLD Total", "mld_total", "Throughput (Mbps)"),
        ("mbps/sld_total", "SLD Total", "sld_total", "Throughput (Mbps)"),
        ("mbps/system", "System", "system", "Throughput (Mbps)"),
        ("collision_rate/2_4GHz/per_event", "2.4GHz Collision Rate", "2_4ghz_collision_rate", "Rate / event"),
        ("collision_rate/5GHz/per_event", "5GHz Collision Rate", "5ghz_collision_rate", "Rate / event"),
        ("collision_rate/system_per_event", "System Collision Rate", "system_collision_rate", "Rate / event"),
        ("success_rate/2_4GHz/per_event", "2.4GHz Success Rate", "2_4ghz_success_rate", "Rate / event"),
        ("success_rate/5GHz/per_event", "5GHz Success Rate", "5ghz_success_rate", "Rate / event"),
        ("success_rate/system_per_event", "System Success Rate", "system_success_rate", "Rate / event"),
        ("idle_rate/2_4GHz/per_event", "2.4GHz Idle Rate", "2_4ghz_idle_rate", "Rate / event"),
        ("idle_rate/5GHz/per_event", "5GHz Idle Rate", "5ghz_idle_rate", "Rate / event"),
        ("idle_rate/system_per_event", "System Idle Rate", "system_idle_rate", "Rate / event"),
        ("events/2_4GHz/success", "2.4GHz Success Events", "2_4ghz_success_events", "Events"),
        ("events/5GHz/success", "5GHz Success Events", "5ghz_success_events", "Events"),
        ("events/system/success", "System Success Events", "system_success_events", "Events"),
        ("events/2_4GHz/collision", "2.4GHz Collision Events", "2_4ghz_collision_events", "Events"),
        ("events/5GHz/collision", "5GHz Collision Events", "5ghz_collision_events", "Events"),
        ("events/system/collision", "System Collision Events", "system_collision_events", "Events"),
        ("events/2_4GHz/idle", "2.4GHz Idle Events", "2_4ghz_idle_events", "Events"),
        ("events/5GHz/idle", "5GHz Idle Events", "5ghz_idle_events", "Events"),
        ("events/system/idle", "System Idle Events", "system_idle_events", "Events"),
    ]
    output_paths = [
        plot_metric(args, cases, metric_key, metric_title, metric_slug, y_label)
        for metric_key, metric_title, metric_slug, y_label in metric_specs
    ]

    print("Saved BEB vs RL Mbps comparison charts:")
    for output_path in output_paths:
        print(f"  {output_path}")

    if args.wandb_project:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("wandb is required when --wandb_project is set.") from exc

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            job_type="comparison",
            config={
                "beb_base_dir": str(beb_base_dir),
                "rl_base_dir": str(rl_base_dir),
                "output_name": args.output_name,
                "m_values": args.m_values,
                "s_values": args.s_values,
                "beb_suffix": args.beb_suffix,
                "rl_suffix": args.rl_suffix,
                "cases": cases,
            },
            reinit=True,
        )
        wandb.log(
            {
                f"{args.wandb_image_key}/{path.stem}": wandb.Image(str(path))
                for path in output_paths
            }
        )
        artifact = wandb.Artifact(args.wandb_image_key, type="figure")
        for output_path in output_paths:
            artifact.add_file(str(output_path))
        run.log_artifact(artifact)
        run.finish()
        print(f"Uploaded charts to wandb under key: {args.wandb_image_key}")


if __name__ == "__main__":
    main()
