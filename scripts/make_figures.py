"""Publication figures from the established results (CPU-only, no GPU/checkpoint needed).

Produces, into paper/figures/:
  1. streaming.png      -- recurrent-state size + per-step latency vs horizon (Contribution B signature)
  2. tokenizer_matrix.png -- recon-FID, FSQ vs RVQ at matched bits/step (Contribution A dominance)
  3. generalization.png -- train vs test recon-FID, the no-overfit scatter (equal-N=1000 audit)

Streaming reads outputs/streaming_bench*.json; the matrix / generalization numbers are the seeded
results (Lesson 7.2 / outputs/generalization.md), inlined so the figure regenerates standalone.

    $env:PYTHONPATH="src"; .venv/Scripts/python.exe scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)


def _load_streaming() -> dict[str, dict[int, dict]]:
    """Merge the short- and long-horizon benchmark runs into {backbone: {horizon: row}}."""
    merged: dict[str, dict[int, dict]] = {"transformer": {}, "mamba": {}}
    for name in ("streaming_bench.json", "streaming_bench_long.json"):
        path = Path("outputs") / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())["results"]
        for bb in merged:
            for row in data.get(bb, []):
                merged[bb][row["horizon"]] = row
    return merged


def fig_streaming() -> None:
    data = _load_streaming()
    fig, (ax_mem, ax_lat) = plt.subplots(1, 2, figsize=(11, 4.2))
    style = {"transformer": ("tab:red", "o", "Transformer (KV-cache)"),
             "mamba": ("tab:blue", "s", "Mamba S6 (SSM state)")}
    for bb, (color, mark, label) in style.items():
        rows = sorted(data[bb].values(), key=lambda r: r["horizon"])
        if not rows:
            continue
        hs = [r["horizon"] for r in rows]
        mem = [r["state_bytes"] / 1e6 for r in rows]
        lat = [r["ms_per_step"] for r in rows]
        ax_mem.plot(hs, mem, color=color, marker=mark, label=label)
        ax_lat.plot(hs, lat, color=color, marker=mark, label=label)

    ax_mem.set(xscale="log", yscale="log", xlabel="horizon (token steps)",
               ylabel="recurrent state (MB)", title="Streaming state size vs horizon")
    ax_mem.grid(True, which="both", alpha=0.3)
    ax_mem.legend()

    ax_lat.axhline(200, color="gray", ls="--", lw=1)
    ax_lat.text(70, 210, "200 ms real-time budget (4 frames @ 20 fps)", color="gray", fontsize=8)
    ax_lat.set(xscale="log", xlabel="horizon (token steps)", ylabel="latency (ms / step)",
               title="Per-step latency vs horizon")
    ax_lat.grid(True, which="both", alpha=0.3)
    ax_lat.legend()

    fig.suptitle("Contribution B: bounded streaming state (100M twins, measured)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "streaming.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "streaming.png")


# Seeded full-test recon-FID (Lesson 7.2). codes/step -> value.
MATRIX = {
    "RVQ-512":  {4: 0.0626, 6: 0.0342, 8: 0.0277},
    "FSQ-512":  {4: 0.0612, 6: 0.0305, 8: 0.0196},
    "RVQ-1024": {4: 0.0558, 6: 0.0310, 8: 0.0221},
    "FSQ-1024": {4: 0.0527, 6: 0.0283, 8: 0.0170},
}
MOMASK = 0.019


def fig_matrix() -> None:
    codes = [4, 6, 8]
    series = ["RVQ-512", "FSQ-512", "RVQ-1024", "FSQ-1024"]
    colors = {"RVQ-512": "#f4a6a6", "FSQ-512": "#5b8def",
              "RVQ-1024": "#d9534f", "FSQ-1024": "#1f3fb0"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    for i, s in enumerate(series):
        xs = [c + (i - 1.5) * width for c in range(len(codes))]
        ys = [MATRIX[s][c] for c in codes]
        ax.bar(xs, ys, width, label=s, color=colors[s])
    ax.axhline(MOMASK, color="black", ls="--", lw=1)
    ax.text(-0.45, MOMASK + 0.0008, "MoMask RVQ 0.019 (cited, more compute)", fontsize=8)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([f"{c} codes/step" for c in codes])
    ax.set(ylabel="recon-FID (test, seed 2026)  -- lower is better",
           title="Contribution A: FSQ dominates strong-RVQ at matched bits/step")
    ax.legend(ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "tokenizer_matrix.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "tokenizer_matrix.png")


# Equal-N=1000 audit (outputs/generalization.md): stem -> (train, test).
GEN = {
    "rvq_l4_512": (0.0631, 0.0661), "rvq_l6_512": (0.0451, 0.0356), "rvq_l8_512": (0.0291, 0.0309),
    "rvq_l4_1024": (0.0573, 0.0547), "rvq_l6_1024": (0.0405, 0.0315), "rvq_l8_1024": (0.0297, 0.0236),
    "fsq_g4_v512": (0.0688, 0.0587), "fsq_g6_v512": (0.0393, 0.0310), "fsq_g8_v512": (0.0287, 0.0208),
    "fsq_g4_v1024": (0.0622, 0.0525), "fsq_g6_v1024": (0.0367, 0.0313), "fsq_g8_v1024": (0.0234, 0.0207),
    "fsq_g4_v1000": (0.0550, 0.0504), "fsq_g6_v1000": (0.0314, 0.0291), "fsq_g8_v1000": (0.0281, 0.0208),
}


def fig_generalization() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for stem, (tr, te) in GEN.items():
        is_fsq = stem.startswith("fsq")
        ax.scatter(tr, te, c="#1f3fb0" if is_fsq else "#d9534f",
                   marker="s" if is_fsq else "o", s=45, zorder=3)
    lo, hi = 0.018, 0.072
    ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="test = train (y = x)")
    ax.fill_between([lo, hi], [lo, hi], hi, color="tab:red", alpha=0.06)
    ax.fill_between([lo, hi], lo, [lo, hi], color="tab:green", alpha=0.06)
    ax.text(0.022, 0.063, "above line: test worse than train\n(overfit hint)", fontsize=8, color="tab:red")
    ax.text(0.045, 0.026, "below line: test <= train\n(generalizes)", fontsize=8, color="tab:green")
    ax.scatter([], [], c="#1f3fb0", marker="s", label="FSQ (9 cells, all below line)")
    ax.scatter([], [], c="#d9534f", marker="o", label="RVQ")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="train recon-FID (N=1000)",
           ylabel="test recon-FID (N=1000)",
           title="No overfitting: held-out test <= train (13/15 cells)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "generalization.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "generalization.png")


if __name__ == "__main__":
    fig_streaming()
    fig_matrix()
    fig_generalization()
    print("done ->", FIG)
