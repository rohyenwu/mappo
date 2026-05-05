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
        "--no_title",
        action="store_true",
        help="Do not draw a chart title. Useful when the title is handled by a paper caption.",
    )
    parser.add_argument(
        "--paper_style",
        action="store_true",
        help="Use compact publication styling: no title, tighter layout, print-friendly fonts, and higher DPI.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=None,
        help="Figure width in inches. Defaults to an automatic width based on the number of cases.",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=None,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--x_label_rotation",
        type=float,
        default=None,
        help="X-axis label rotation in degrees.",
    )
    parser.add_argument(
        "--output_formats",
        nargs="+",
        default=None,
        help="Optional output formats such as png pdf. Defaults to the --output_name suffix.",
    )
    parser.add_argument(
        "--round_length",
        type=float,
        default=500.0,
        help="Round length used to convert throughput from successes/TXOP to successes/round.",
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
    parser.add_argument(
        "--hide_bar_labels",
        action="store_true",
        help="Do not draw numeric value labels above bars.",
    )
    parser.add_argument(
        "--include_packet_throughput",
        action="store_true",
        help="Also plot throughput metrics based on processed packet counts.",
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

    if args.paper_style:
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.labelsize": 9,
                "axes.titlesize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    cases = [parse_case_spec(spec) for spec in args.case]
    metric_keys = [
        ("throughput/2_4GHz/total", "2.4GHz Throughput", "2_4ghz_throughput", None),
        ("throughput/5GHz/total", "5GHz Throughput", "5ghz_throughput", None),
        ("throughput/mld_total", "MLD Throughput", "mld_throughput", None),
        ("throughput/sld_total", "SLD Throughput", "sld_throughput", (0.0, 0.8)),
        ("throughput/system", "System Throughput", "system_throughput", None),
    ]
    if args.include_packet_throughput:
        metric_keys.extend(
            [
                (
                    "packet_throughput/2_4GHz/total",
                    "2.4GHz Packet Throughput",
                    "2_4ghz_packet_throughput",
                    None,
                ),
                (
                    "packet_throughput/5GHz/total",
                    "5GHz Packet Throughput",
                    "5ghz_packet_throughput",
                    None,
                ),
                (
                    "packet_throughput/mld_total",
                    "MLD Packet Throughput",
                    "mld_packet_throughput",
                    None,
                ),
                (
                    "packet_throughput/sld_total",
                    "SLD Packet Throughput",
                    "sld_packet_throughput",
                    (0.0, 0.8),
                ),
                (
                    "packet_throughput/system",
                    "System Packet Throughput",
                    "system_packet_throughput",
                    None,
                ),
            ]
        )
    method_names = ["BEB", "RL"]
    method_colors = ["#4c78a8", "#f58518"]

    labels = [case["label"] for case in cases]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    if args.fig_width is None:
        fig_width = max(4.2, 0.58 * len(labels)) if args.paper_style else 7.0
    else:
        fig_width = args.fig_width
    fig_height = args.fig_height or (2.8 if args.paper_style else 5.0)
    x_label_rotation = (
        args.x_label_rotation
        if args.x_label_rotation is not None
        else (25 if args.paper_style else 20)
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(args.output_name).stem
    output_suffix = Path(args.output_name).suffix or ".png"
    output_formats = args.output_formats or [output_suffix.lstrip(".")]
    output_suffixes = [
        fmt if fmt.startswith(".") else f".{fmt}"
        for fmt in output_formats
    ]
    output_paths = []

    unit_specs = [
        ("txop", "Throughput (successes / TXOP)", 1.0, "normalized"),
        ("round", "Throughput (successes / round)", args.round_length, "round"),
    ]

    for metric_key, metric_title, metric_slug, y_limits in metric_keys:
        base_beb_values = [float(case["beb"].get(metric_key, 0.0)) for case in cases]
        base_rl_values = [float(case["rl"].get(metric_key, 0.0)) for case in cases]

        for unit_suffix, y_label, scale, unit_title in unit_specs:
            fig_size = (fig_width, fig_height)
            fig, ax = plt.subplots(figsize=fig_size)
            beb_values = [value * scale for value in base_beb_values]
            rl_values = [value * scale for value in base_rl_values]

            bars_beb = ax.bar(
                x - width / 2.0,
                beb_values,
                width=width,
                label=method_names[0],
                color=method_colors[0],
                edgecolor="black" if args.paper_style else None,
                linewidth=0.4 if args.paper_style else 0.0,
            )
            bars_rl = ax.bar(
                x + width / 2.0,
                rl_values,
                width=width,
                label=method_names[1],
                color=method_colors[1],
                edgecolor="black" if args.paper_style else None,
                linewidth=0.4 if args.paper_style else 0.0,
            )

            if not args.hide_bar_labels:
                for bars in [bars_beb, bars_rl]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{height:.3f}" if scale == 1.0 else f"{height:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7 if args.paper_style else 8,
                        )

            if not (args.no_title or args.paper_style):
                ax.set_title(f"{args.title} - {metric_title} ({unit_title})")
            ax.set_ylabel(y_label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=x_label_rotation, ha="right")
            ax.grid(axis="y", alpha=0.25, linewidth=0.6 if args.paper_style else 0.8)
            ax.legend(frameon=False)

            if y_limits is not None:
                ax.set_ylim(y_limits[0] * scale, y_limits[1] * scale)
            else:
                ymax = max(beb_values + rl_values) if (beb_values or rl_values) else 0.0
                ax.set_ylim(0.0, max(0.1 * scale, ymax * 1.25))

            fig.tight_layout()
            for suffix in output_suffixes:
                output_path = output_dir / f"{output_stem}_{metric_slug}_{unit_suffix}{suffix}"
                output_paths.append(output_path)
                fig.savefig(output_path, dpi=max(args.dpi, 600) if args.paper_style else args.dpi, bbox_inches="tight")
            plt.close(fig)

    print("Saved WiFi v4 case comparison charts to:")
    for output_path in output_paths:
        print(f"  {output_path}")

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
                "round_length": args.round_length,
            },
            reinit=True,
        )
        image_suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = [
            path for path in output_paths
            if path.suffix.lower() in image_suffixes
        ]
        file_paths = [
            path for path in output_paths
            if path.suffix.lower() not in image_suffixes
        ]

        if image_paths:
            wandb.log(
                {
                    f"{args.wandb_image_key}/{path.stem}": wandb.Image(str(path))
                    for path in image_paths
                }
            )

        if file_paths:
            artifact = wandb.Artifact(
                name=f"{args.wandb_image_key}_files",
                type="figure",
            )
            for path in file_paths:
                artifact.add_file(str(path))
            run.log_artifact(artifact)
        run.finish()
        print(
            f"Uploaded comparison chart to wandb as '{args.wandb_image_key}'."
        )


if __name__ == "__main__":
    main()
