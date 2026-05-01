#!/usr/bin/env python
"""Create a grouped throughput bar chart from evaluation summary JSON files."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MLD/SLD/System throughput across evaluation summaries."
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Summary spec in the form Label=/path/to/summary.json. Repeat this flag.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the comparison chart will be saved.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="throughput_comparison.png",
        help="Output image filename.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="WiFi Throughput Comparison",
        help="Chart title.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Optional wandb project name for uploading the comparison chart.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional wandb group name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional wandb run name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional wandb entity/user name.",
    )
    parser.add_argument(
        "--wandb_image_key",
        type=str,
        default="all_barchart",
        help="wandb key used when uploading the chart image.",
    )
    return parser.parse_args()


def parse_summary_spec(spec: str):
    if "=" not in spec:
        raise ValueError(
            f"Invalid --summary value: {spec}. Expected Label=/path/to/summary.json"
        )

    label, path_str = spec.split("=", 1)
    label = label.strip()
    path = Path(path_str.strip()).expanduser()

    if not label:
        raise ValueError(f"Invalid --summary value: {spec}. Label is empty.")
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return label, data


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and numpy are required for compare_wifi_throughput.py"
        ) from exc

    metric_keys = [
        "throughput/2_4GHz/total",
        "throughput/5GHz/total",
        "throughput/mld_total",
        "throughput/sld_total",
        "throughput/system",
    ]
    metric_labels = ["2.4GHz", "5GHz", "MLD", "SLD", "System"]
    metric_colors = ["#4c78a8", "#72b7b2", "#1f77b4", "#ff7f0e", "#2ca02c"]

    labels = []
    values = []
    for spec in args.summary:
        label, summary = parse_summary_spec(spec)
        labels.append(label)
        values.append([float(summary.get(key, 0.0)) for key in metric_keys])

    values = np.asarray(values, dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = min(0.16, 0.8 / max(len(metric_labels), 1))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    fig, ax = plt.subplots(figsize=(8, 5))
    center_offset = (len(metric_labels) - 1) / 2.0
    for idx, (metric_label, color) in enumerate(zip(metric_labels, metric_colors)):
        offset = (idx - center_offset) * width
        bars = ax.bar(
            x + offset,
            values[:, idx],
            width=width,
            label=metric_label,
            color=color,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput (successes / TXOP)")
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ymax = float(values.max()) if values.size > 0 else 0.0
    ax.set_ylim(0.0, max(0.1, ymax * 1.2))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved throughput comparison chart to: {output_path}")

    if args.wandb_project:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "wandb is required to upload the comparison chart."
            ) from exc

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            job_type="comparison",
            config={
                "title": args.title,
                "summaries": args.summary,
            },
            reinit=True,
        )
        wandb.log({args.wandb_image_key: wandb.Image(str(output_path))})
        run.finish()
        print(
            f"Uploaded throughput comparison chart to wandb "
            f"as '{args.wandb_image_key}'."
        )


if __name__ == "__main__":
    main()
