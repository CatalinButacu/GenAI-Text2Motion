# python scripts/figures/plot_training_curves.py --out docs/paper/pics/training_curves.pdf
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
from matplotlib.typing import LineStyleType

RUNS: MappingProxyType[str, tuple[str, LineStyleType]] = MappingProxyType(
    {
        "sweep w256 (20 ep)": ("size_sweep_{backbone}_w256.log", ":"),
        "sweep w320 (20 ep)": ("size_sweep_{backbone}_w320.log", "--"),
        "pair 1 w320 (CE stop)": ("p1_{backbone}_w320.log", "-"),
    }
)
LOG_DIR = Path("runs/outputs")
BACKBONES = ("transformer", "mamba")
COLORS = MappingProxyType({"transformer": "#2a78d6", "mamba": "#eb6834"})
CE_LINE = re.compile(r"clean-ce\] train ([\d.]+)\s+val ([\d.]+)\s+gap ([+\-\d.]+)")
FID_LINE = re.compile(r"\[gen-eval:val\] clips \d+\s+FID ([\d.]+)\s+R@1 ([\d.]+)")
EPOCH_LINE = re.compile(r"epoch\s+(\d+)/\d+")
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"


def read_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    epoch = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_LINE.search(line)
        if epoch_match and "clean-ce" not in line:
            epoch = int(epoch_match.group(1))
        ce_match = CE_LINE.search(line)
        if ce_match:
            train, val, gap = (float(value) for value in ce_match.groups())
            rows.append({"epoch": epoch, "train_ce_clean": train, "val_ce": val, "ce_gap": gap})
        fid_match = FID_LINE.search(line)
        if fid_match and rows and rows[-1]["epoch"] == epoch:
            rows[-1]["val_fid"] = float(fid_match.group(1))
            rows[-1]["val_r1"] = float(fid_match.group(2))
    return rows


def style_axis(axis, title: str) -> None:
    axis.set_title(title, color=INK, fontsize=9, loc="left")
    axis.set_xlabel("epoch", color=MUTED, fontsize=7)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/paper/pics/training_curves.pdf"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    curves = {
        (run, backbone): read_curve(LOG_DIR / pattern.format(backbone=backbone))
        for run, (pattern, _) in RUNS.items()
        for backbone in BACKBONES
    }
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    panels = (
        ("val_ce", "Validation CE"),
        ("ce_gap", "CE gap (val - clean train)"),
        ("val_fid", "Validation FID"),
    )
    for axis, (key, title) in zip(axes, panels, strict=True):
        for (run, backbone), rows in curves.items():
            points = [(row["epoch"], row[key]) for row in rows if key in row]
            axis.plot(
                [epoch for epoch, _ in points],
                [value for _, value in points],
                color=COLORS[backbone],
                linestyle=RUNS[run][1],
                linewidth=1.4,
            )
        style_axis(axis, title)
    axes[2].set_ylim(0, 1.5)
    handles = [Line2D([], [], color=COLORS[b], linewidth=1.6) for b in BACKBONES]
    handles += [
        Line2D([], [], color=MUTED, linestyle=style, linewidth=1.4) for _, style in RUNS.values()
    ]
    labels = ["Transformer", "Mamba", *RUNS]
    figure.legend(
        handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=7, labelcolor=INK
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.out)
    figure.savefig(arguments.out.with_suffix(".png"), dpi=200)
    table = arguments.out.with_suffix(".csv")
    fields = ["run", "backbone", "epoch", "train_ce_clean", "val_ce", "ce_gap", "val_fid", "val_r1"]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (run, backbone), rows in curves.items():
            for row in rows:
                writer.writerow({"run": run, "backbone": backbone, **row})
    print(f"wrote {arguments.out}, {arguments.out.with_suffix('.png')} and {table}")


if __name__ == "__main__":
    main()
