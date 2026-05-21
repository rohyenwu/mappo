#!/usr/bin/env python
"""Create event success doughnut charts from existing WiFi v9 Mbps summaries."""

import argparse
import json
from pathlib import Path


DEFAULT_BEB_SUMMARY = (
    "scripts/eval_results/WiFi_v9/wifi_v9_beb_mbps/"
    "cap_sim_smoke_beb/beb_mbps_summary.json"
)
DEFAULT_RL_SUMMARY = (
    "scripts/eval_results/WiFi_v9/wifi_v9_rl_mbps/"
    "cap_sim_smoke_rl/rl_mbps_summary.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create BEB/RL event success doughnut charts from existing summary JSON files."
    )
    parser.add_argument("--beb_summary", type=str, default=DEFAULT_BEB_SUMMARY)
    parser.add_argument("--rl_summary", type=str, default=DEFAULT_RL_SUMMARY)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scripts/eval_results/WiFi_v9/event_success_doughnuts",
    )
    parser.add_argument("--title", type=str, default="WiFi v9 Total Event Result Ratio")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="wifi_v9_existing_summary_figures")
    parser.add_argument("--wandb_run_name", type=str, default="wifi_v9_event_success_doughnuts")
    parser.add_argument("--wandb_image_key", type=str, default="summary/event_success_doughnut")
    return parser.parse_args()


def load_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def event_values(summary):
    values = [
        max(0.0, float(summary.get("events/system/success", 0.0))),
        max(0.0, float(summary.get("events/system/collision", 0.0))),
        max(0.0, float(summary.get("events/system/idle", 0.0))),
    ]
    total = float(summary.get("events/system/total", sum(values)))
    if total <= 0.0:
        total = sum(values)
    if total <= 0.0:
        raise ValueError("Summary does not contain positive system event counts.")
    return values, total


def save_doughnut(output_dir: Path, policy: str, summary: dict, title: str):
    import matplotlib.pyplot as plt

    labels = ["Success", "Collision", "Idle"]
    colors = ["#16a34a", "#dc2626", "#94a3b8"]
    values, total = event_values(summary)
    success_ratio = values[0] / total

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
    )
    ax.text(
        0,
        0.06,
        f"{success_ratio * 100:.1f}%",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0,
        -0.16,
        "success ratio",
        ha="center",
        va="center",
        fontsize=10,
        color="#64748b",
    )
    legend_labels = [
        f"{label}: {value:.2f} ({value / total * 100:.1f}%)"
        for label, value in zip(labels, values)
    ]
    ax.legend(
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
    )
    ax.set_title(f"{title} - {policy}")
    ax.set_aspect("equal")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"wifi_v9_{policy.lower()}_event_success_doughnut.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plot_wifi_v9_existing_event_doughnuts.py"
        ) from exc

    beb_summary_path = Path(args.beb_summary).expanduser()
    rl_summary_path = Path(args.rl_summary).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    cases = [
        ("BEB", beb_summary_path, load_summary(beb_summary_path)),
        ("RL", rl_summary_path, load_summary(rl_summary_path)),
    ]

    output_paths = []
    table = []
    for policy, summary_path, summary in cases:
        values, total = event_values(summary)
        success_ratio = values[0] / total
        output_path = save_doughnut(output_dir, policy, summary, args.title)
        output_paths.append((policy, output_path))
        table.append(
            {
                "policy": policy,
                "summary_path": str(summary_path),
                "success_events": values[0],
                "collision_events": values[1],
                "idle_events": values[2],
                "total_events": total,
                "success_ratio": success_ratio,
            }
        )

    table_path = output_dir / "wifi_v9_event_success_summary.json"
    with table_path.open("w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, ensure_ascii=False)

    print("Saved event success doughnut charts:")
    for policy, output_path in output_paths:
        print(f"  {policy}: {output_path}")
    print(f"Saved summary table: {table_path}")

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
            job_type="existing-summary-figure",
            config={
                "beb_summary": str(beb_summary_path),
                "rl_summary": str(rl_summary_path),
                "output_dir": str(output_dir),
                "event_success_summary": table,
            },
            reinit=True,
        )
        payload = {
            f"{args.wandb_image_key}/{policy.lower()}": wandb.Image(str(output_path))
            for policy, output_path in output_paths
        }
        payload[f"{args.wandb_image_key}/table"] = wandb.Table(
            columns=[
                "policy",
                "success_events",
                "collision_events",
                "idle_events",
                "total_events",
                "success_ratio",
            ],
            data=[
                [
                    row["policy"],
                    row["success_events"],
                    row["collision_events"],
                    row["idle_events"],
                    row["total_events"],
                    row["success_ratio"],
                ]
                for row in table
            ],
        )
        wandb.log(payload)

        artifact = wandb.Artifact("wifi_v9_event_success_doughnuts", type="figure")
        for _, output_path in output_paths:
            artifact.add_file(str(output_path))
        artifact.add_file(str(table_path))
        run.log_artifact(artifact)
        run.finish()
        print(f"Uploaded doughnut charts to wandb key: {args.wandb_image_key}")


if __name__ == "__main__":
    main()
