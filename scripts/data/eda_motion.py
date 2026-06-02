"""Exploratory data analysis on the cached motion dataset.

Six dimensions:
  1. Clip length distribution (frames + seconds)
  2. AMASS subset breakdown (or unified-source breakdown)
  3. Per-channel std segmented by SMPL-X block (root, trans, body, hands, jaw/eyes)
  4. PCA cumulative explained variance on z-normalized frames
  5. Symmetric L/R channel correlations (validates mirror augmentation)
  6. Per-channel within-clip vs cross-clip variance decomposition

Read-only: never modifies caches, stats, or training data.

Usage:
    python scripts/data/eda_motion.py --source amass
    python scripts/data/eda_motion.py --source humanml3d
    python scripts/data/eda_motion.py --source all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.data.augmentation import detect_tpose, quality_filter, resample_to_fps
from src.data.dataset_cache import load_or_build_cache
from src.data.motion_normalize import MotionStats, normalize
from src.data.unified import build_sources_buffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig
from src.shared.constants import INGEST_MAX_LENGTH

log = logging.getLogger(__name__)

CHANNEL_BLOCKS = [
    ("root_orient", 0, 3, "tab:red"),
    ("translation", 3, 6, "tab:orange"),
    ("body", 6, 69, "tab:blue"),
    ("left_hand", 69, 114, "tab:green"),
    ("right_hand", 114, 159, "tab:purple"),
    ("jaw_eyes", 159, 168, "tab:gray"),
]

BODY_BLOCK_START = 6
BODY_LR_PAIRS = ((0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20))
LR_NAMES = ["hip", "knee", "ankle", "foot", "collar", "shoulder", "elbow", "wrist"]
FPS = 30.0

def load_samples(source: str, data_dir: str, humanml3d_dir: str) -> tuple[list[dict], str]:
    """Return (samples, descriptor) for the chosen --source."""
    if source == "amass":
        samples, _ = load_or_build_cache(data_dir, INGEST_MAX_LENGTH, None)

        return samples, "AMASS-full"

    if source == "humanml3d":
        cfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, data_dir=humanml3d_dir, amass_dir=data_dir),
        )
        descriptor = "HumanML3D"
    else:
        cfg = UnifiedConfig(
            amass=SourceConfig(enabled=True, data_dir=data_dir),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, data_dir=humanml3d_dir, amass_dir=data_dir),
        )
        descriptor = "AMASS + HumanML3D unified"
    samples = build_sources_buffer(cfg, 30, resample_to_fps, quality_filter, detect_tpose,
                                 max_length=INGEST_MAX_LENGTH)

    return samples, descriptor

def length_distribution(samples: list[dict], out_dir: str) -> dict:
    lengths = np.array([s["motion"].shape[0] for s in samples])
    pcts = np.percentile(lengths, [10, 25, 50, 75, 90, 99])
    summary = {
        "n": int(len(lengths)), "min": int(lengths.min()), "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "p10": float(pcts[0]), "p25": float(pcts[1]), "p50": float(pcts[2]),
        "p75": float(pcts[3]), "p90": float(pcts[4]), "p99": float(pcts[5]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(lengths, bins=60, color="steelblue")
    axes[0].set_xlabel("frames")
    axes[0].set_ylabel("count")
    axes[0].set_title("Clip length (frames @ 30 fps)")
    axes[0].axvline(200, color="red", linestyle="--", label="train window 200")
    axes[0].axvline(INGEST_MAX_LENGTH, color="orange", linestyle="--",
                    label=f"ingest cap {INGEST_MAX_LENGTH}")
    axes[0].legend()
    axes[1].hist(lengths / FPS, bins=60, color="seagreen")
    axes[1].set_xlabel("seconds")
    axes[1].set_ylabel("count")
    axes[1].set_title("Clip length (seconds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_length_dist.png"), dpi=110)
    plt.close(fig)

    return summary

def source_breakdown(samples: list[dict], out_dir: str, source: str) -> dict:
    """For AMASS, parse the subset prefix from sample_id. For unified, count by 'source'."""
    counts: dict[str, int] = {}

    if source == "amass":
        for s in samples:
            sid = s.get("sample_id", "")
            sub = sid.split("/")[0] if "/" in sid else "unknown"
            counts[sub] = counts.get(sub, 0) + 1
        title = "AMASS subset breakdown"
    else:

        for s in samples:
            src = s.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        title = "Source breakdown"

    items = sorted(counts.items(), key=lambda x: -x[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, max(3, len(items) * 0.3)))
    ax.barh(labels, values, color="steelblue")
    ax.set_xlabel("clip count")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_source_breakdown.png"), dpi=110)
    plt.close(fig)

    return {"counts": counts, "top5": items[:5]}

def channel_std_analysis(all_frames: np.ndarray, out_dir: str) -> dict:
    ch_std = all_frames.std(axis=0)
    dead_idx = np.where(ch_std < 1e-3)[0].tolist()

    fig, ax = plt.subplots(figsize=(13, 4))

    for name, start, end, color in CHANNEL_BLOCKS:
        ax.bar(range(start, end), ch_std[start:end], color=color, label=name)
    ax.axhline(1e-3, color="red", linestyle="--", alpha=0.6, label="dead threshold")
    ax.set_xlabel("channel index")
    ax.set_ylabel("std")
    ax.set_title(f"Per-channel std (n_dead={len(dead_idx)} of 168)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_channel_std.png"), dpi=110)
    plt.close(fig)

    block_stats = {}

    for name, start, end, _ in CHANNEL_BLOCKS:
        sub = ch_std[start:end]
        block_stats[name] = {
            "mean_std": float(sub.mean()), "max_std": float(sub.max()),
            "n_dead": int((sub < 1e-3).sum()), "size": int(end - start),
        }

    return {"dead_channels": dead_idx, "block_stats": block_stats}

def pca_analysis(frames_norm: np.ndarray, out_dir: str, n_components: int = 50) -> dict:
    sub = frames_norm[::10]
    pca = PCA(n_components=n_components)
    pca.fit(sub)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    knees = {}

    for thresh in (0.50, 0.80, 0.90, 0.95, 0.99):
        idx = int(np.searchsorted(cum_var, thresh)) + 1
        knees[f"k_for_{int(thresh * 100)}pct"] = min(idx, n_components)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, n_components + 1), cum_var * 100, marker="o", color="steelblue")
    ax.axhline(95, color="red", linestyle="--", label="95%")
    ax.axhline(99, color="orange", linestyle="--", label="99%")
    ax.set_xlabel("# principal components")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title(f"PCA on z-normalized frames (subsampled to {len(sub)} rows)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_pca_variance.png"), dpi=110)
    plt.close(fig)

    return knees

def symmetric_correlations(all_frames: np.ndarray, out_dir: str) -> dict:
    correlations = {}

    for (l_joint, r_joint), name in zip(BODY_LR_PAIRS, LR_NAMES):
        l_start = BODY_BLOCK_START + l_joint * 3
        r_start = BODY_BLOCK_START + r_joint * 3
        rs = []

        for k in range(3):
            lc = all_frames[:, l_start + k]
            rc = all_frames[:, r_start + k]

            if lc.std() < 1e-6 or rc.std() < 1e-6:
                continue
            rs.append(float(np.corrcoef(lc, rc)[0, 1]))
        correlations[name] = float(np.mean(rs)) if rs else 0.0

    fig, ax = plt.subplots(figsize=(9, 4))
    keys = list(correlations.keys())
    vals = list(correlations.values())
    colors = ["seagreen" if v > 0 else "coral" for v in vals]
    ax.bar(keys, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("mean Pearson r over the 3 axes")
    ax.set_title("L/R channel correlation by body joint")
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_symmetric_corr.png"), dpi=110)
    plt.close(fig)

    return correlations

def variance_decomposition(samples: list[dict], out_dir: str) -> dict:
    per_clip_means = []
    per_clip_stds = []

    for s in samples:
        m = s["motion"]
        per_clip_means.append(m.mean(axis=0))
        per_clip_stds.append(m.std(axis=0))
    per_clip_means = np.stack(per_clip_means)
    per_clip_stds = np.stack(per_clip_stds)

    within_std = per_clip_stds.mean(axis=0)
    cross_std = per_clip_means.std(axis=0)
    safe_cross = np.where(cross_std < 1e-6, 1e-6, cross_std)
    ratio = within_std / safe_cross

    ymax = float(max(within_std.max(), cross_std.max())) * 1.1

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(within_std, label="within-clip std (temporal motion)", color="steelblue")
    ax.plot(cross_std, label="cross-clip std (clip identity)", color="orange", alpha=0.8)

    for name, start, end, _ in CHANNEL_BLOCKS:
        ax.axvspan(start, end, alpha=0.04)
        ax.text((start + end) / 2, ymax * 0.95, name, ha="center", fontsize=8, color="gray")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("channel")
    ax.set_ylabel("std")
    ax.set_title("Per-channel variance decomposition")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_var_decomposition.png"), dpi=110)
    plt.close(fig)

    finite_ratio = ratio[np.isfinite(ratio)]

    return {
        "mean_within_std": float(within_std.mean()),
        "mean_cross_std": float(cross_std.mean()),
        "mean_ratio": float(finite_ratio.mean()) if len(finite_ratio) else 0.0,
    }

def write_report(out_dir: str, source: str, n: int, length: dict, breakdown: dict,
                channel_std: dict, pca: dict, corr: dict, var_decomp: dict) -> None:
    lines: list[str] = []
    lines.append(f"# EDA report — {source}")
    lines.append("")
    lines.append(f"Total clips: **{n}**")
    lines.append("")

    lines.append("## 1. Clip length distribution")
    lines.append("")
    lines.append(f"- min/median/max: {length['min']}/{int(length['p50'])}/{length['max']} frames "
                 f"({length['min'] / FPS:.1f}/{length['p50'] / FPS:.1f}/"
                 f"{length['max'] / FPS:.1f} s)")
    lines.append(f"- p10/p90/p99: {int(length['p10'])}/{int(length['p90'])}/"
                 f"{int(length['p99'])} frames")
    lines.append(f"- mean: {length['mean']:.1f} frames ({length['mean'] / FPS:.1f} s)")
    lines.append("")

    lines.append("## 2. Source breakdown (top 5)")
    lines.append("")

    for k, v in breakdown["top5"]:
        lines.append(f"- **{k}**: {v} clips ({100 * v / n:.1f}%)")
    lines.append("")

    lines.append("## 3. Channel statistics")
    lines.append("")
    lines.append(f"- Dead channels (std < 1e-3): **{len(channel_std['dead_channels'])} of 168**")

    if channel_std["dead_channels"]:
        lines.append(f"  indices: {channel_std['dead_channels']}")
    lines.append("")
    lines.append("| block | size | mean std | max std | n_dead |")
    lines.append("|---|---|---|---|---|")

    for name, st in channel_std["block_stats"].items():
        lines.append(f"| {name} | {st['size']} | {st['mean_std']:.4f} | "
                     f"{st['max_std']:.4f} | {st['n_dead']} |")
    lines.append("")

    lines.append("## 4. PCA explained variance")
    lines.append("")

    for k, v in pca.items():
        lines.append(f"- {k}: **{v} components**")
    lines.append("")

    lines.append("## 5. L/R symmetric channel correlations")
    lines.append("")

    for joint, r in corr.items():
        lines.append(f"- {joint}: r = {r:.3f}")
    avg_corr = float(np.mean(list(corr.values()))) if corr else 0.0
    lines.append("")
    lines.append(f"Mean across all L/R pairs: **{avg_corr:.3f}**")
    lines.append("")

    lines.append("## 6. Variance decomposition")
    lines.append("")
    lines.append(f"- Mean within-clip std (temporal motion): {var_decomp['mean_within_std']:.4f}")
    lines.append(f"- Mean cross-clip std (clip identity):   {var_decomp['mean_cross_std']:.4f}")
    lines.append(f"- Within / cross ratio: **{var_decomp['mean_ratio']:.3f}**")
    lines.append("")
    lines.append("- > 1: temporal motion dominates (codec must learn dynamics)")
    lines.append("- < 1: clip-to-clip variation dominates")
    lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="amass", choices=["amass", "humanml3d", "all"],
                        dest="source")
    parser.add_argument("--data-dir", default="data/AMASS", dest="data_dir")
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3d_dir")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="stats_path")
    parser.add_argument("--output-dir", default="data/.cache/eda", dest="output_dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    out_dir = os.path.join(args.output_dir, args.source)
    os.makedirs(out_dir, exist_ok=True)

    log.info("[eda] loading samples (source=%s)", args.source)
    samples, descriptor = load_samples(args.source, args.data_dir, args.humanml3d_dir)
    log.info("[eda] %d clips loaded", len(samples))

    log.info("[eda] loading shared stats from %s", args.stats_path)
    loaded = np.load(args.stats_path)
    stats = MotionStats(mean=loaded["mean"], std=loaded["std"])

    log.info("[eda] step 1/6 length distribution")
    length_summary = length_distribution(samples, out_dir)

    log.info("[eda] step 2/6 source breakdown")
    breakdown = source_breakdown(samples, out_dir, args.source)

    log.info("[eda] step 3/6 stacking frames (this can take a minute)")
    all_frames = np.concatenate([s["motion"] for s in samples], axis=0).astype(np.float32)
    log.info("[eda]   total frames: %d", len(all_frames))

    log.info("[eda] step 4/6 per-channel std")
    channel_std = channel_std_analysis(all_frames, out_dir)

    log.info("[eda] step 5/6 PCA")
    frames_norm = normalize(all_frames, stats, clip_value=None)
    pca_result = pca_analysis(frames_norm, out_dir)

    log.info("[eda] step 6a/6 symmetric correlations")
    corr_result = symmetric_correlations(all_frames, out_dir)

    log.info("[eda] step 6b/6 variance decomposition")
    var_decomp = variance_decomposition(samples, out_dir)

    log.info("[eda] writing report")
    write_report(out_dir, descriptor, len(samples), length_summary, breakdown,
                channel_std, pca_result, corr_result, var_decomp)

    log.info("[eda] DONE — outputs in %s", out_dir)

    return 0

if __name__ == "__main__":
    sys.exit(main())
