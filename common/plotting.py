"""Reliability curve + latency boxplot, shared by every experiment."""
from __future__ import annotations

import sys
from pathlib import Path

from common.metrics import ece, reliability_bins


def reliability_and_latency(series: list[dict], out_path: Path, title: str, x_label: str,
                            y_label: str, latency_label: str) -> None:
    """series: one dict per model with keys label, preds, labels, latencies, accuracy."""
    import matplotlib.pyplot as plt

    fig, (ax_cal, ax_lat) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_cal.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    for s in series:
        points = reliability_bins(s["preds"], s["labels"])
        xs = [c for c, _, _ in points]
        ys = [a for _, a, _ in points]
        sizes = [20 + n * 4 for _, _, n in points]
        line, = ax_cal.plot(xs, ys, marker="o",
                            label=f"{s['label']} (acc={s['accuracy']:.2f}, ECE={ece(s['preds'], s['labels']):.3f})")
        ax_cal.scatter(xs, ys, s=sizes, color=line.get_color(), alpha=0.5, zorder=3)
    ax_cal.set_xlabel(x_label)
    ax_cal.set_ylabel(y_label)
    ax_cal.set_title(title)
    ax_cal.legend(fontsize=8, loc="upper left")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)

    ax_lat.boxplot([s["latencies"] for s in series], tick_labels=[s["label"] for s in series],
                   orientation="horizontal", showfliers=False)
    ax_lat.set_xlabel(latency_label)
    ax_lat.set_title("Latency")
    ax_lat.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}", file=sys.stderr)
