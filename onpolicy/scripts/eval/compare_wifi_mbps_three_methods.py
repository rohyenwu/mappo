#!/usr/bin/env python
"""Compare three WiFi Mbps summary sets across scenarios."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped bar charts for three WiFi Mbps methods. "
            "Each --case expects Label|/path/to/method1.json|/path/to/method2.json|/path/to/method3.json."
        )
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--method_names", type=str, default="BEB,RL,SETL-DQN")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="wifi_mbps_three_methods.png")
    parser.add_argument("--title", type=str, default="WiFi Mbps Comparison")
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
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --case value: {spec}. "
            "Expected Label|/path/to/method1.json|/path/to/method2.json|/path/to/method3.json"
        )
    label = parts[0]
    paths_and_summaries = [load_summary(part) for part in parts[1:]]
    return {
        "label": label,
        "paths": [str(path) for path, _ in paths_and_summaries],
        "summaries": [summary for _, summary in paths_and_summaries],
    }


def fallback_metric(summary, metric_key: str, label: str):
    value = summary.get(metric_key)
    if value is not None:
        return float(value)

    # Older summaries may not have mu_min/mid/max aliases. Use scenario label
    # to recover representative MLD IDs for evenly spaced mu profiles.
    if not metric_key.startswith("mbps/mu_") or not metric_key.endswith("_mld/total"):
        return 0.0
    if not label.startswith("m") or "_s" not in label:
        return 0.0
    try:
        active_mld = int(label.split("_s", 1)[0][1:])
    except ValueError:
        return 0.0
    if active_mld <= 0:
        return 0.0

    if "mu_min_mld" in metric_key:
        mld_id = 0
    elif "mu_max_mld" in metric_key:
        mld_id = active_mld - 1
    else:
        mld_id = int(round((active_mld - 1) / 2.0))
    return float(summary.get(f"mbps/mld_{mld_id}/total", 0.0))


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and numpy are required for compare_wifi_mbps_three_methods.py"
        ) from exc

    method_names = [part.strip() for part in args.method_names.split(",") if part.strip()]
    if len(method_names) != 3:
        raise ValueError("--method_names must contain exactly three comma-separated names")

    cases = [parse_case_spec(spec) for spec in args.case]
    metric_keys = [
        ("mbps/2_4GHz/total", "2.4GHz Mbps", "2_4ghz_mbps"),
        ("mbps/5GHz/total", "5GHz Mbps", "5ghz_mbps"),
        ("mbps/mld_total", "MLD Mbps", "mld_mbps"),
        ("mbps/mu_min_mld/total", "Min Arrival MLD Mbps", "mu_min_mld_mbps"),
        ("mbps/mu_mid_mld/total", "Mid Arrival MLD Mbps", "mu_mid_mld_mbps"),
        ("mbps/mu_max_mld/total", "Max Arrival MLD Mbps", "mu_max_mld_mbps"),
        ("mbps/sld_total", "SLD Mbps", "sld_mbps"),
        ("mbps/system", "System Mbps", "system_mbps"),
    ]

    labels = [case["label"] for case in cases]
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(args.output_name).stem
    output_suffix = Path(args.output_name).suffix or ".png"
    output_paths = []

    for metric_key, metric_title, metric_slug in metric_keys:
        fig, ax = plt.subplots(figsize=(max(8.5, 0.65 * len(labels)), 5.2))
        method_values = []
        for method_idx in range(3):
            values = [
                fallback_metric(case["summaries"][method_idx], metric_key, case["label"])
                for case in cases
            ]
            method_values.append(values)
            bars = ax.bar(
                x + offsets[method_idx],
                values,
                width=width,
                label=method_names[method_idx],
                color=colors[method_idx],
            )
            if not args.hide_bar_labels:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_title(f"{args.title} - {metric_title}")
        ax.set_ylabel("Throughput (Mbps)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=3)
        flat_values = [value for values in method_values for value in values]
        ymax = max(flat_values) if flat_values else 0.0
        ax.set_ylim(0.0, max(1.0, ymax * 1.25))
        fig.tight_layout()

        output_path = output_dir / f"{output_stem}_{metric_slug}{output_suffix}"
        output_paths.append(output_path)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Saved WiFi Mbps three-method comparison charts to:")
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
            config={
                "title": args.title,
                "cases": args.case,
                "method_names": method_names,
            },
            reinit=True,
        )
        for output_path in output_paths:
            wandb.log({output_path.stem: wandb.Image(str(output_path))})
        run.finish()


if __name__ == "__main__":
    main()
