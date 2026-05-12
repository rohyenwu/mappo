#!/usr/bin/env python
"""Plot WiFi v7.1 BEB Mbps grid summaries and optionally upload to wandb."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create BEB Mbps bar charts from WiFi v7.1 grid summaries."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing m*_s*_beb_mbps*/beb_mbps_summary.json folders.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="wifi_v7_1_beb_mbps_grid.png")
    parser.add_argument("--title", type=str, default="WiFi v7.1 BEB Mbps Grid")
    parser.add_argument("--m_values", nargs="+", type=int, default=[10, 15, 20, 25, 30])
    parser.add_argument("--s_values", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument(
        "--experiment_suffix",
        type=str,
        default="_beb_mbps_ep3",
        help="Folder suffix after m{m}_s{s}; default matches the ep3 run names.",
    )
    parser.add_argument("--hide_bar_labels", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_image_key", type=str, default="beb_mbps_grid")
    return parser.parse_args()


def load_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_single_metric_chart(args, output_dir, cases, metric_key, metric_title, metric_slug, y_label):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [case["label"] for case in cases]
    x = np.arange(len(labels), dtype=float)
    values = [float(case["summary"].get(metric_key, 0.0)) for case in cases]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(labels)), 5.0))
    bars = ax.bar(x, values, width=0.55, color="#4c78a8")
    if not args.hide_bar_labels:
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
    ymax = max(values) if values else 0.0
    ax.set_ylim(0.0, max(1.0, ymax * 1.18))
    fig.tight_layout()

    stem = Path(args.output_name).stem
    suffix = Path(args.output_name).suffix or ".png"
    output_path = output_dir / f"{stem}_{metric_slug}{suffix}"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and numpy are required for plot_wifi_v7_1_beb_mbps_grid.py"
        ) from exc

    base_dir = Path(args.base_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for m in args.m_values:
        for s in args.s_values:
            label = f"m{m}_s{s}"
            summary_path = base_dir / f"{label}{args.experiment_suffix}" / "beb_mbps_summary.json"
            summary = load_summary(summary_path)
            cases.append(
                {
                    "label": label,
                    "summary_path": str(summary_path),
                    "summary": summary,
                }
            )

    labels = [case["label"] for case in cases]
    x = np.arange(len(labels), dtype=float)
    width = 0.26
    metric_specs = [
        ("mbps/system", "System", "#4c78a8", -width),
        ("mbps/mld_total", "MLD", "#f58518", 0.0),
        ("mbps/sld_total", "SLD", "#54a24b", width),
    ]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(labels)), 5.0))
    all_values = []
    for metric_key, metric_label, color, offset in metric_specs:
        values = [float(case["summary"].get(metric_key, 0.0)) for case in cases]
        all_values.extend(values)
        bars = ax.bar(x + offset, values, width=width, label=metric_label, color=color)
        if not args.hide_bar_labels:
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

    ax.set_title(args.title)
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ymax = max(all_values) if all_values else 0.0
    ax.set_ylim(0.0, max(1.0, ymax * 1.18))
    fig.tight_layout()

    output_path = output_dir / args.output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BEB Mbps grid chart: {output_path}")
    output_paths = [output_path]

    extra_metric_specs = [
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
    for metric_key, metric_title, metric_slug, y_label in extra_metric_specs:
        if any(metric_key in case["summary"] for case in cases):
            extra_path = save_single_metric_chart(
                args,
                output_dir,
                cases,
                metric_key,
                metric_title,
                metric_slug,
                y_label,
            )
            output_paths.append(extra_path)
            print(f"Saved BEB metric chart: {extra_path}")

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
                "base_dir": str(base_dir),
                "output_name": args.output_name,
                "m_values": args.m_values,
                "s_values": args.s_values,
                "experiment_suffix": args.experiment_suffix,
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
        for path in output_paths:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
        print(f"Uploaded chart to wandb key: {args.wandb_image_key}")


if __name__ == "__main__":
    main()
