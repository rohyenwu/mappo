"""Common fixed-duration Mbps evaluation helpers for WiFi evaluators."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MbpsTimeModel:
    """Time and payload model used for fixed-duration Mbps evaluation."""

    eval_duration_sec: float = 30.0
    slot_time_sec: float = 9e-6
    phy_preamble_sec: float = 20e-6
    sifs_sec: float = 16e-6
    difs_sec: float = 34e-6
    ack_bits: float = 112.0
    payload_bits: float = 131072.0
    mac_header_bits: float = 288.0
    basic_rate_bps: float = 24e6
    data_rate_24_bps: float = 24e6
    data_rate_5_bps: float = 48e6

    def success_time_sec(self, link_id: int) -> float:
        data_rate = self.data_rate_24_bps if link_id == 0 else self.data_rate_5_bps
        data_tx_sec = (self.mac_header_bits + self.payload_bits) / max(data_rate, 1.0)
        ack_tx_sec = self.ack_bits / max(self.basic_rate_bps, 1.0)
        return (
            self.phy_preamble_sec
            + data_tx_sec
            + self.sifs_sec
            + self.phy_preamble_sec
            + ack_tx_sec
            + self.difs_sec
        )

    def collision_time_sec(self, link_id: int) -> float:
        data_rate = self.data_rate_24_bps if link_id == 0 else self.data_rate_5_bps
        data_tx_sec = (self.mac_header_bits + self.payload_bits) / max(data_rate, 1.0)
        return self.phy_preamble_sec + data_tx_sec + self.difs_sec

    def idle_time_sec(self) -> float:
        return self.slot_time_sec


class MbpsAccumulator:
    """Accumulate delivered bits over a fixed wall-clock duration."""

    def __init__(self, time_model: MbpsTimeModel):
        self.time_model = time_model
        self.elapsed_sec = 0.0
        self.bits_mld_24 = 0.0
        self.bits_mld_5 = 0.0
        self.bits_sld = 0.0
        self.step_count = 0
        self.total_step_slots = 0.0
        self.event_counts = {
            link_id: {"success": 0.0, "collision": 0.0, "idle": 0.0}
            for link_id in (0, 1)
        }
        self.success_type_counts = {
            link_id: {"mld": 0.0, "sld": 0.0}
            for link_id in (0, 1)
        }

    def done(self) -> bool:
        return self.elapsed_sec >= self.time_model.eval_duration_sec - 1e-12

    def _step_duration_sec(self, link_events: dict) -> float:
        durations = []
        for link_id in (0, 1):
            result = link_events[link_id]["result"]
            if result == "not_ready":
                continue
            if result == "success":
                durations.append(self.time_model.success_time_sec(link_id))
            elif result == "collision":
                durations.append(self.time_model.collision_time_sec(link_id))
            else:
                durations.append(self.time_model.idle_time_sec())
        return max(durations) if durations else 0.0

    def _step_bits(self, link_events: dict) -> tuple[float, float, float]:
        bits_mld_24 = 0.0
        bits_mld_5 = 0.0
        bits_sld = 0.0

        for link_id in (0, 1):
            if link_events[link_id]["result"] != "success":
                continue
            success_type = link_events[link_id]["success_type"]
            packet_count = float(link_events[link_id].get("packet_count", 1.0))
            if success_type == "mld":
                if link_id == 0:
                    bits_mld_24 += packet_count * self.time_model.payload_bits
                else:
                    bits_mld_5 += packet_count * self.time_model.payload_bits
            elif success_type == "sld":
                bits_sld += packet_count * self.time_model.payload_bits

        return bits_mld_24, bits_mld_5, bits_sld

    def add_step(self, link_events: dict, step_slots: float = 0.0) -> float:
        if self.done():
            return 0.0

        step_slots = max(float(step_slots), 0.0)
        wait_sec = step_slots * self.time_model.slot_time_sec
        duration_sec = wait_sec + self._step_duration_sec(link_events)
        bits_mld_24, bits_mld_5, bits_sld = self._step_bits(link_events)

        remaining_sec = max(self.time_model.eval_duration_sec - self.elapsed_sec, 0.0)
        fraction = 1.0 if duration_sec <= remaining_sec else remaining_sec / max(duration_sec, 1e-12)

        self.bits_mld_24 += bits_mld_24 * fraction
        self.bits_mld_5 += bits_mld_5 * fraction
        self.bits_sld += bits_sld * fraction
        self.elapsed_sec += duration_sec * fraction
        self.step_count += 1
        self.total_step_slots += step_slots * fraction
        for link_id in (0, 1):
            result = link_events[link_id]["result"]
            if result == "not_ready":
                continue
            if result in self.event_counts[link_id]:
                self.event_counts[link_id][result] += fraction
            success_type = link_events[link_id].get("success_type")
            if result == "success" and success_type in self.success_type_counts[link_id]:
                self.success_type_counts[link_id][success_type] += fraction
        return fraction

    def as_metrics(self) -> dict:
        duration_sec = max(self.time_model.eval_duration_sec, 1e-12)
        mbps_24 = self.bits_mld_24 / duration_sec / 1e6
        mbps_5 = self.bits_mld_5 / duration_sec / 1e6
        mbps_sld = self.bits_sld / duration_sec / 1e6
        mbps_mld = mbps_24 + mbps_5
        mbps_system = mbps_mld + mbps_sld

        metrics = {
            "mbps/2_4GHz/mld": float(mbps_24),
            "mbps/2_4GHz/sld": float(mbps_sld),
            "mbps/2_4GHz/total": float(mbps_24 + mbps_sld),
            "mbps/5GHz/mld": float(mbps_5),
            "mbps/5GHz/sld": 0.0,
            "mbps/5GHz/total": float(mbps_5),
            "mbps/mld_total": float(mbps_mld),
            "mbps/sld_total": float(mbps_sld),
            "mbps/system": float(mbps_system),
            "timing/eval_duration_sec": float(duration_sec),
            "timing/elapsed_sec": float(self.elapsed_sec),
            "timing/step_count": float(self.step_count),
            "timing/total_step_slots": float(self.total_step_slots),
            "timing/avg_step_slots": float(
                self.total_step_slots / max(self.step_count, 1)
            ),
        }
        link_names = {0: "2_4GHz", 1: "5GHz"}
        system_events = 0.0
        system_collisions = 0.0
        system_successes = 0.0
        system_idle = 0.0
        for link_id, link_name in link_names.items():
            counts = self.event_counts[link_id]
            success_types = self.success_type_counts[link_id]
            events = counts["success"] + counts["collision"] + counts["idle"]
            system_events += events
            system_collisions += counts["collision"]
            system_successes += counts["success"]
            system_idle += counts["idle"]
            metrics[f"events/{link_name}/success"] = float(counts["success"])
            metrics[f"events/{link_name}/collision"] = float(counts["collision"])
            metrics[f"events/{link_name}/idle"] = float(counts["idle"])
            metrics[f"events/{link_name}/total"] = float(events)
            metrics[f"events/{link_name}/success_mld"] = float(success_types["mld"])
            metrics[f"events/{link_name}/success_sld"] = float(success_types["sld"])
            metrics[f"collision_rate/{link_name}/per_event"] = float(
                counts["collision"] / max(events, 1.0)
            )
            metrics[f"success_rate/{link_name}/per_event"] = float(
                counts["success"] / max(events, 1.0)
            )
            metrics[f"idle_rate/{link_name}/per_event"] = float(
                counts["idle"] / max(events, 1.0)
            )

        metrics["events/system/success"] = float(system_successes)
        metrics["events/system/collision"] = float(system_collisions)
        metrics["events/system/idle"] = float(system_idle)
        metrics["events/system/total"] = float(system_events)
        metrics["collision_rate/system_per_event"] = float(
            system_collisions / max(system_events, 1.0)
        )
        metrics["success_rate/system_per_event"] = float(
            system_successes / max(system_events, 1.0)
        )
        metrics["idle_rate/system_per_event"] = float(
            system_idle / max(system_events, 1.0)
        )
        return metrics


def infer_link_events(env, infos, prev_link_successes, prev_sld_success, prev_link_packet_successes):
    """Infer per-link idle/success/collision and success owner from env deltas."""

    curr_link_successes = env.link_successes.copy()
    curr_link_packet_successes = getattr(env, "link_packet_successes", curr_link_successes).copy()
    curr_sld_success = int(env.round_sld_success)
    delta_link_successes = curr_link_successes - prev_link_successes
    delta_link_packet_successes = curr_link_packet_successes - prev_link_packet_successes
    delta_sld_success = curr_sld_success - prev_sld_success

    link_events = {}
    for link_id in (0, 1):
        active_aids = env._active_link_agents(link_id)
        result = "idle"
        if active_aids:
            result = str(infos[active_aids[0]].get("txop_result", "idle"))

        success_type = None
        packet_count = 0.0
        if result == "success":
            if link_id == 0 and delta_sld_success > 0 and int(delta_link_successes[:, 0].sum()) == 0:
                success_type = "sld"
                packet_count = float(delta_sld_success)
            elif int(delta_link_successes[:, link_id].sum()) > 0:
                success_type = "mld"
                if hasattr(env, "link_packet_successes"):
                    successful_mlds = np.flatnonzero(delta_link_successes[:, link_id] > 0)
                    packet_count = float(delta_link_packet_successes[successful_mlds, link_id].sum())
                else:
                    packet_count = float(delta_link_successes[:, link_id].sum())

        link_events[link_id] = {
            "result": result,
            "success_type": success_type,
            "packet_count": packet_count,
        }

    return link_events, curr_link_successes, curr_sld_success, curr_link_packet_successes


def save_mbps_bar_chart(run_dir: Path, filename: str, summary: dict):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[Eval] matplotlib is not installed; skipping Mbps bar chart generation.")
        return None

    metric_keys = [
        "mbps/2_4GHz/total",
        "mbps/5GHz/total",
        "mbps/mld_total",
        "mbps/sld_total",
        "mbps/system",
    ]
    labels = ["2.4GHz", "5GHz", "MLD", "SLD", "System"]
    values = [float(summary.get(key, 0.0)) for key in metric_keys]
    colors = ["#4c78a8", "#72b7b2", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("WiFi Mbps Evaluation")
    ymax = max(values) if values else 0.0
    ax.set_ylim(0.0, max(1.0, ymax * 1.2))
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    out_path = run_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
