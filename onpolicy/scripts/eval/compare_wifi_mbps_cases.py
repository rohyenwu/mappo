#!/usr/bin/env python
"""Compare BEB vs RL Mbps summaries across WiFi scenarios."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped bar charts for WiFi Mbps summaries. "
            "Each --case expects Label|/path/to/beb_mbps_summary.json|/path/to/rl_mbps_summary.json."
        )
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="wifi_mbps_comparison.png")
    parser.add_argument("--title", type=str, default="WiFi BEB vs RL Mbps Comparison")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--hide_bar_labels", action="store_true")
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
            "Expected Label|/path/to/beb_mbps_summary.json|/path/to/rl_mbps_summary.json"
        )
    label, beb_path_text, rl_path_text = parts
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
            "matplotlib and numpy are required for compare_wifi_mbps_cases.py"
        ) from exc

    cases = [parse_case_spec(spec) for spec in args.case]
    metric_keys = [
        ("mbps/2_4GHz/total", "2.4GHz Mbps", "2_4ghz_mbps"),
        ("mbps/5GHz/total", "5GHz Mbps", "5ghz_mbps"),
        ("mbps/mld_total", "MLD Mbps", "mld_mbps"),
        ("mbps/sld_total", "SLD Mbps", "sld_mbps"),
        ("mbps/system", "System Mbps", "system_mbps"),
    ]

    labels = [case["label"] for case in cases]
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(args.output_name).stem
    output_suffix = Path(args.output_name).suffix or ".png"
    output_paths = []

    for metric_key, metric_title, metric_slug in metric_keys:
        fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 5.0))
        beb_values = [float(case["beb"].get(metric_key, 0.0)) for case in cases]
        rl_values = [float(case["rl"].get(metric_key, 0.0)) for case in cases]

        bars_beb = ax.bar(x - width / 2.0, beb_values, width=width, label="BEB", color="#4c78a8")
        bars_rl = ax.bar(x + width / 2.0, rl_values, width=width, label="RL", color="#f58518")

        if not args.hide_bar_labels:
            for bars in (bars_beb, bars_rl):
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

        ax.set_title(f"{args.title} - {metric_title}")
        ax.set_ylabel("Throughput (Mbps)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        ymax = max(beb_values + rl_values) if (beb_values or rl_values) else 0.0
        ax.set_ylim(0.0, max(1.0, ymax * 1.25))
        fig.tight_layout()

        output_path = output_dir / f"{output_stem}_{metric_slug}{output_suffix}"
        output_paths.append(output_path)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Saved WiFi Mbps comparison charts to:")
    for output_path in output_paths:
        print(f"  {output_path}")

    if args.wandb_project:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("wandb is required to upload comparison charts.") from exc

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            job_type="comparison",
            config={"title": args.title, "cases": args.case},
            reinit=True,
        )
        for output_path in output_paths:
            wandb.log({output_path.stem: wandb.Image(str(output_path))})
        run.finish()


if __name__ == "__main__":
    main()
