# python scripts/figures/plot_width_costs.py --out doc/paper/pics/width_costs.pdf
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from types import MappingProxyType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STEP_TIMES = Path("artifacts/outputs/qa/size_sweep_step_times.txt")
STATE_BYTES = Path("artifacts/outputs/qa/width_state_bytes.csv")
STEP_LINE = re.compile(
    r"width (\d+)\s+(\w+)\s+batch \d+\s+frames \d+\s+([\d.]+) ms/step.*peak\s+([\d.]+) GB"
)
COLORS = MappingProxyType({"transformer": "#2a78d6", "mamba": "#eb6834"})
LABELS = MappingProxyType({"transformer": "Transformer", "mamba": "Mamba"})
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
CACHE_HORIZON = 95


def read_step_costs() -> dict[tuple[str, int], tuple[float, float]]:
    costs = {}
    for line in STEP_TIMES.read_text(encoding="utf-8").splitlines():
        match = STEP_LINE.search(line)
        if match:
            width, backbone, step_ms, peak_gb = match.groups()
            costs[(backbone, int(width))] = (float(step_ms), float(peak_gb) * 1000.0)
    return costs


def read_state_bytes() -> dict[tuple[str, int], tuple[float, float]]:
    states = {}
    with STATE_BYTES.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) == CACHE_HORIZON:
                key = (row["backbone"], int(row["width"]))
                states[key] = (float(row["state_bytes"]) / 1e6, float(row["parameters"]) / 1e6)
    return states


def style_axis(axis, title: str, unit: str) -> None:
    axis.set_title(title, color=INK, fontsize=9, loc="left")
    axis.set_ylabel(unit, color=MUTED, fontsize=8)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.set_ylim(bottom=0)


def draw_series(axis, widths: list[int], values: dict[str, list[float]]) -> None:
    for backbone, series in values.items():
        axis.plot(widths, series, color=COLORS[backbone], linewidth=1.6, marker="o", markersize=4)
        axis.annotate(
            f"{series[-1]:.1f}" if series[-1] < 10 else f"{series[-1]:.0f}",
            (widths[-1], series[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=INK,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("doc/paper/pics/width_costs.pdf"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    costs = read_step_costs()
    states = read_state_bytes()
    widths = sorted({width for _, width in costs})
    backbones = ("transformer", "mamba")
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True)
    panels = (
        (
            "Training step (batch 8)",
            "ms",
            {b: [costs[(b, w)][0] for w in widths] for b in backbones},
        ),
        (
            "Peak GPU memory (training)",
            "MB",
            {b: [costs[(b, w)][1] for w in widths] for b in backbones},
        ),
        (
            f"Stream memory, {CACHE_HORIZON} tokens",
            "MB",
            {b: [states[(b, w)][0] for w in widths] for b in backbones},
        ),
    )
    for axis, (title, unit, values) in zip(axes, panels, strict=True):
        draw_series(axis, widths, values)
        style_axis(axis, title, unit)
    tick_labels = [f"{w}\n{states[('transformer', w)][1]:.1f}M" for w in widths]
    for axis in axes:
        axis.set_xticks(widths)
        axis.set_xticklabels(tick_labels)
        axis.set_xlabel("width / transformer parameters", color=MUTED, fontsize=7)
    handles = [
        Line2D([], [], color=COLORS[b], linewidth=1.6, marker="o", markersize=4) for b in backbones
    ]
    figure.legend(
        handles,
        [LABELS[b] for b in backbones],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.out)
    figure.savefig(arguments.out.with_suffix(".png"), dpi=200)
    table = arguments.out.with_suffix(".csv")
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "width",
                "backbone",
                "parameters_m",
                "step_ms",
                "peak_gpu_mb",
                f"stream_mb_at_{CACHE_HORIZON}",
            ]
        )
        for backbone in backbones:
            for width in widths:
                key = (backbone, width)
                writer.writerow(
                    [
                        width,
                        backbone,
                        f"{states[key][1]:.2f}",
                        costs[key][0],
                        costs[key][1],
                        f"{states[key][0]:.3f}",
                    ]
                )
    print(f"wrote {arguments.out}, {arguments.out.with_suffix('.png')} and {table}")


if __name__ == "__main__":
    main()
