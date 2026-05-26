#!/usr/bin/env python
"""Compare BEB vs RL Mbps summaries across WiFi scenarios."""

import argparse
import json
import re
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
    parser.add_argument(
        "--include_event_doughnuts",
        action="store_true",
        help=(
            "Also create a BEB/RL doughnut chart from aggregated "
            "events/system/{success,collision,idle} counts across all cases."
        ),
    )
    parser.add_argument(
        "--include_improvement_by_sld",
        action="store_true",
        help=(
            "Also create a line chart of system Mbps improvement over BEB, "
            "grouped by SLD count. Case labels must include m<num>_s<num>."
        ),
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


def parse_m_s_label(label: str):
    match = re.search(r"m(\d+)_s(\d+)", label)
    if not match:
        raise ValueError(
            f"Cannot parse MLD/SLD counts from case label '{label}'. "
            "Expected a label containing m<num>_s<num>, for example m10_s2."
        )
    return int(match.group(1)), int(match.group(2))


def save_system_improvement_by_sld_chart(plt, output_path: Path, title: str, cases):
    improvement_by_sld = {}
    for case in cases:
        mld_count, sld_count = parse_m_s_label(case["label"])
        beb_value = float(case["beb"].get("mbps/system", 0.0))
        rl_value = float(case["rl"].get("mbps/system", 0.0))
        improvement = 0.0
        if beb_value > 0.0:
            improvement = (rl_value / beb_value - 1.0) * 100.0
        improvement_by_sld.setdefault(sld_count, []).append(
            {
                "mld_count": mld_count,
                "improvement_percent": improvement,
                "beb_mbps": beb_value,
                "rl_mbps": rl_value,
                "label": case["label"],
            }
        )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = {
        2: "#4c78a8",
        4: "#f58518",
        6: "#54a24b",
    }
    markers = {
        2: "o",
        4: "s",
        6: "^",
    }

    all_improvements = []
    for sld_count in sorted(improvement_by_sld):
        rows = sorted(improvement_by_sld[sld_count], key=lambda row: row["mld_count"])
        x_values = [row["mld_count"] for row in rows]
        y_values = [row["improvement_percent"] for row in rows]
        all_improvements.extend(y_values)
        ax.plot(
            x_values,
            y_values,
            marker=markers.get(sld_count, "o"),
            linewidth=2.2,
            markersize=6.0,
            label=f"SLD={sld_count}",
            color=colors.get(sld_count),
        )
        for x_value, y_value in zip(x_values, y_values):
            ax.text(
                x_value,
                y_value,
                f"{y_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0.0, color="#334155", linewidth=1.0, alpha=0.65)
    ax.set_title(f"{title} - RL Improvement over BEB")
    ax.set_xlabel("Number of MLDs")
    ax.set_ylabel("Throughput Improvement over BEB (%)")
    ax.set_xticks(sorted({parse_m_s_label(case["label"])[0] for case in cases}))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    if all_improvements:
        ymin = min(all_improvements)
        ymax = max(all_improvements)
        padding = max(2.0, (ymax - ymin) * 0.18)
        ax.set_ylim(ymin - padding, ymax + padding)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    data_path = output_path.with_suffix(".json")
    with open(data_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric": "system_mbps_improvement_over_beb_percent",
                "formula": "(RL / BEB - 1) * 100",
                "series": improvement_by_sld,
            },
            handle,
            indent=2,
        )
    return data_path


def case_event_rates(case, policy):
    summary = case[policy]
    values = {
        event_name: float(summary.get(f"events/system/{event_name}", 0.0))
        for event_name in ("success", "collision", "idle")
    }
    total = sum(values.values())
    if total <= 0.0:
        return {event_name: 0.0 for event_name in values}
    return {event_name: value / total for event_name, value in values.items()}


def average_event_rates(cases, policy):
    totals = {"success": 0.0, "collision": 0.0, "idle": 0.0}
    if not cases:
        return totals
    for case in cases:
        rates = case_event_rates(case, policy)
        for event_name in totals:
            totals[event_name] += rates[event_name]
    return {event_name: value / len(cases) for event_name, value in totals.items()}


def draw_event_doughnut(ax, values, title, colors):
    labels = ["Success", "Collision", "Idle"]
    event_names = ["success", "collision", "idle"]
    parts = [values[event_name] for event_name in event_names]
    total = sum(parts)
    if total <= 0.0:
        ax.text(0.5, 0.5, "No event data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.pie(
        parts,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 1.5},
    )
    success_ratio = values["success"] / total
    ax.text(
        0,
        0.06,
        f"{success_ratio * 100:.1f}%",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0,
        -0.14,
        "success",
        ha="center",
        va="center",
        fontsize=8,
        color="#64748b",
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def save_average_event_doughnut_chart(plt, output_path: Path, title: str, cases):
    labels = ["Success", "Collision", "Idle"]
    event_names = ["success", "collision", "idle"]
    colors = ["#16a34a", "#dc2626", "#94a3b8"]
    policy_specs = [
        ("BEB", average_event_rates(cases, "beb")),
        ("RL", average_event_rates(cases, "rl")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, (policy_label, rates) in zip(axes, policy_specs):
        values = [rates[event_name] for event_name in event_names]
        draw_event_doughnut(ax, rates, policy_label, colors)
        legend_labels = [
            f"{label}: {value * 100:.1f}% avg"
            for label, value in zip(labels, values)
        ]
        ax.legend(
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.24),
            frameon=False,
            fontsize=9,
        )

    fig.suptitle(f"{title} - Mean Per-Case Event Result Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_case_event_doughnut_grid(plt, output_path: Path, title: str, cases):
    colors = ["#16a34a", "#dc2626", "#94a3b8"]
    fig_height = max(5.0, 1.45 * len(cases))
    fig, axes = plt.subplots(len(cases), 2, figsize=(8.5, fig_height))
    if len(cases) == 1:
        axes = [axes]

    for row_idx, case in enumerate(cases):
        for col_idx, policy in enumerate(("beb", "rl")):
            ax = axes[row_idx][col_idx]
            rates = case_event_rates(case, policy)
            policy_label = "BEB" if policy == "beb" else "RL"
            draw_event_doughnut(ax, rates, f"{case['label']} {policy_label}", colors)

    fig.suptitle(f"{title} - Per-Case Event Success Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_case_event_doughnut_pair(plt, output_path: Path, title: str, case):
    colors = ["#16a34a", "#dc2626", "#94a3b8"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2))

    for ax, policy in zip(axes, ("beb", "rl")):
        rates = case_event_rates(case, policy)
        policy_label = "BEB" if policy == "beb" else "RL"
        draw_event_doughnut(ax, rates, policy_label, colors)

    fig.suptitle(f"{title} - {case['label']} Event Success Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
        ("mbps/mu_min_mld/total", "Min Arrival MLD Mbps", "mu_min_mld_mbps"),
        ("mbps/mu_mid_mld/total", "Mid Arrival MLD Mbps", "mu_mid_mld_mbps"),
        ("mbps/mu_max_mld/total", "Max Arrival MLD Mbps", "mu_max_mld_mbps"),
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
    data_output_paths = []

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

    if args.include_event_doughnuts:
        avg_output_path = output_dir / f"{output_stem}_event_success_doughnut_mean{output_suffix}"
        save_average_event_doughnut_chart(plt, avg_output_path, args.title, cases)
        output_paths.append(avg_output_path)

        for case in cases:
            case_output_path = (
                output_dir
                / f"{output_stem}_{case['label']}_event_success_doughnut{output_suffix}"
            )
            save_case_event_doughnut_pair(plt, case_output_path, args.title, case)
            output_paths.append(case_output_path)

    if args.include_improvement_by_sld:
        improvement_output_path = (
            output_dir / f"{output_stem}_system_mbps_improvement_by_sld{output_suffix}"
        )
        improvement_data_path = save_system_improvement_by_sld_chart(
            plt, improvement_output_path, args.title, cases
        )
        output_paths.append(improvement_output_path)
        data_output_paths.append(improvement_data_path)

    print("Saved WiFi Mbps comparison charts to:")
    for output_path in output_paths:
        print(f"  {output_path}")
    if data_output_paths:
        print("Saved WiFi Mbps comparison data to:")
        for output_path in data_output_paths:
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
        image_rows = [
            [output_path.stem, wandb.Image(str(output_path))]
            for output_path in output_paths
        ]
        image_table = wandb.Table(columns=["figure", "image"], data=image_rows)
        image_payload = {
            "figures/all_images": image_table,
            "figures/count": len(output_paths),
        }
        image_payload.update(
            {
                f"figures/{output_path.stem}": wandb.Image(str(output_path))
                for output_path in output_paths
            }
        )
        wandb.log(image_payload)
        run.finish()


if __name__ == "__main__":
    main()
