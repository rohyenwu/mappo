#!/usr/bin/env python
"""Compare WiFi v4 evaluation summaries across scenarios and methods."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped bar charts for WiFi v4 scenarios. "
            "Each --case expects Label|/path/to/beb_summary.json|/path/to/rl_summary.json."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Scenario spec in the form Label|/path/to/beb_summary.json|/path/to/rl_summary.json",
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
        default="wifi_v4_case_comparison.png",
        help="Output image filename.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="WiFi v4 BEB vs RL Comparison",
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
        default="case_comparison",
        help="wandb key used when uploading the chart image.",
    )
    return parser.parse_args()


def load_summary(path_text: str):
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return path, json.load(handle)


def parse_case_spec(spec: str):
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --case value: {spec}. "
            "Expected Label|/path/to/beb_summary.json|/path/to/rl_summary.json"
        )

    label, beb_path_text, rl_path_text = parts
    if not label:
        raise ValueError(f"Invalid --case value: {spec}. Label is empty.")

    beb_path, beb_summary = load_summary(beb_path_text)
    rl_path, rl_summary = load_summary(rl_path_text)
    return {
        "label": label,
        "beb_path": str(beb_path),
        "rl_path": str(rl_path),
        "beb": beb_summary,
        "rl": rl_summary,
    }


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and numpy are required for compare_wifi_v4_cases.py"
        ) from exc

    cases = [parse_case_spec(spec) for spec in args.case]
    metric_keys = [
        ("throughput/mld_total", "MLD Throughput per Round"),
        ("throughput/sld_total", "SLD Throughput per Round"),
        ("throughput/system", "System Throughput per Round"),
    ]
    method_names = ["BEB", "RL"]
    method_colors = ["#4c78a8", "#f58518"]

    labels = [case["label"] for case in cases]
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    fig.suptitle(args.title, fontsize=18)

    for ax, (metric_key, metric_title) in zip(axes, metric_keys):
        beb_values = [float(case["beb"].get(metric_key, 0.0)) for case in cases]
        rl_values = [float(case["rl"].get(metric_key, 0.0)) for case in cases]

        bars_beb = ax.bar(
            x - width / 2.0,
            beb_values,
            width=width,
            label=method_names[0],
            color=method_colors[0],
        )
        bars_rl = ax.bar(
            x + width / 2.0,
            rl_values,
            width=width,
            label=method_names[1],
            color=method_colors[1],
        )

        for bars in [bars_beb, bars_rl]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(metric_title)
        ax.set_ylabel("Throughput")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ymax = max(beb_values + rl_values) if (beb_values or rl_values) else 0.0
        ax.set_ylim(0.0, max(0.1, ymax * 1.25))

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved WiFi v4 case comparison chart to: {output_path}")

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
                "cases": args.case,
            },
            reinit=True,
        )
        wandb.log({args.wandb_image_key: wandb.Image(str(output_path))})
        run.finish()
        print(
            f"Uploaded comparison chart to wandb as '{args.wandb_image_key}'."
        )


if __name__ == "__main__":
    main()
