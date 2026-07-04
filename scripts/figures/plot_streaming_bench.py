\
\
\
\
\
\
\
\
   

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"transformer": "tab:red", "mamba": "tab:blue"}


def main() -> None:
    p = argparse.ArgumentParser(description="Plot the streaming benchmark figure.")
    p.add_argument("--bench", default="outputs/streaming_bench.json")
    p.add_argument("--out", default="outputs/streaming_bench.png")
    a = p.parse_args()

    data = json.loads(Path(a.bench).read_text())["results"]
    fig, (ax_mem, ax_lat) = plt.subplots(1, 2, figsize=(12, 5))

    for backbone, rows in data.items():
        hs = [r["horizon"] for r in rows]
        state_mb = [r["state_bytes"] / 1e6 for r in rows]
        ms = [r["ms_per_step"] for r in rows]
        color = COLORS.get(backbone, "tab:green")
        ax_mem.plot(hs, state_mb, "o-", color=color, label=backbone, lw=2)
        ax_lat.plot(hs, ms, "o-", color=color, label=backbone, lw=2)

    tf = data["transformer"][-1]["state_bytes"]
    mb = data["mamba"][-1]["state_bytes"]
    h_max = data["transformer"][-1]["horizon"]
    ax_mem.annotate(
        f"{tf / mb:.0f}x smaller\nat horizon {h_max}",
        xy=(h_max, mb / 1e6), xytext=(h_max * 0.45, tf / 1e6 * 0.6),
        arrowprops={"arrowstyle": "->", "color": "0.3"}, fontsize=11,
    )
    ax_mem.set_title("Recurrent state vs streaming horizon\n(transformer KV-cache grows; Mamba is bounded)")
    ax_mem.set_xlabel("streaming horizon (tokens)")
    ax_mem.set_ylabel("recurrent state (MB)")
    ax_mem.legend()
    ax_mem.grid(alpha=0.3)

    ax_lat.set_title("Per-step latency vs horizon\n(Mamba eager: no CUDA scan kernel on Windows)")
    ax_lat.set_xlabel("streaming horizon (tokens)")
    ax_lat.set_ylabel("median ms / step")
    ax_lat.set_ylim(bottom=0)
    ax_lat.legend()
    ax_lat.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}  (KV {tf / 1e6:.1f}MB vs state {mb / 1e6:.2f}MB at horizon {h_max})")


if __name__ == "__main__":
    main()
