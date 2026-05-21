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

from src.data.augmentation import detectTpose, qualityFilter, resampleToFps
from src.data.dataset_cache import INGEST_MAX_LENGTH, loadOrBuildCache
from src.data.motion_normalize import MotionStats, normalize
from src.data.unified import buildSourcesBuffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig

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

def loadSamples(source: str, dataDir: str, humanml3dDir: str) -> tuple[list[dict], str]:
    """Return (samples, descriptor) for the chosen --source."""
    if source == "amass":
        samples, _ = loadOrBuildCache(dataDir, INGEST_MAX_LENGTH, None)

        return samples, "AMASS-full"

    if source == "humanml3d":
        cfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, dataDir=humanml3dDir, amassDir=dataDir),
        )
        descriptor = "HumanML3D"
    else:
        cfg = UnifiedConfig(
            amass=SourceConfig(enabled=True, dataDir=dataDir),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, dataDir=humanml3dDir, amassDir=dataDir),
        )
        descriptor = "AMASS + HumanML3D unified"
    samples = buildSourcesBuffer(cfg, 30, resampleToFps, qualityFilter, detectTpose,
                                 maxLength=INGEST_MAX_LENGTH)

    return samples, descriptor

def lengthDistribution(samples: list[dict], outDir: str) -> dict:
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
    fig.savefig(os.path.join(outDir, "01_length_dist.png"), dpi=110)
    plt.close(fig)

    return summary

def sourceBreakdown(samples: list[dict], outDir: str, source: str) -> dict:
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
    fig.savefig(os.path.join(outDir, "02_source_breakdown.png"), dpi=110)
    plt.close(fig)

    return {"counts": counts, "top5": items[:5]}

def channelStdAnalysis(allFrames: np.ndarray, outDir: str) -> dict:
    chStd = allFrames.std(axis=0)
    deadIdx = np.where(chStd < 1e-3)[0].tolist()

    fig, ax = plt.subplots(figsize=(13, 4))

    for name, start, end, color in CHANNEL_BLOCKS:
        ax.bar(range(start, end), chStd[start:end], color=color, label=name)
    ax.axhline(1e-3, color="red", linestyle="--", alpha=0.6, label="dead threshold")
    ax.set_xlabel("channel index")
    ax.set_ylabel("std")
    ax.set_title(f"Per-channel std (n_dead={len(deadIdx)} of 168)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "03_channel_std.png"), dpi=110)
    plt.close(fig)

    blockStats = {}

    for name, start, end, _ in CHANNEL_BLOCKS:
        sub = chStd[start:end]
        blockStats[name] = {
            "mean_std": float(sub.mean()), "max_std": float(sub.max()),
            "n_dead": int((sub < 1e-3).sum()), "size": int(end - start),
        }

    return {"dead_channels": deadIdx, "block_stats": blockStats}

def pcaAnalysis(framesNorm: np.ndarray, outDir: str, nComponents: int = 50) -> dict:
    sub = framesNorm[::10]
    pca = PCA(n_components=nComponents)
    pca.fit(sub)
    cumVar = np.cumsum(pca.explained_variance_ratio_)

    knees = {}

    for thresh in (0.50, 0.80, 0.90, 0.95, 0.99):
        idx = int(np.searchsorted(cumVar, thresh)) + 1
        knees[f"k_for_{int(thresh * 100)}pct"] = min(idx, nComponents)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, nComponents + 1), cumVar * 100, marker="o", color="steelblue")
    ax.axhline(95, color="red", linestyle="--", label="95%")
    ax.axhline(99, color="orange", linestyle="--", label="99%")
    ax.set_xlabel("# principal components")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title(f"PCA on z-normalized frames (subsampled to {len(sub)} rows)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "04_pca_variance.png"), dpi=110)
    plt.close(fig)

    return knees

def symmetricCorrelations(allFrames: np.ndarray, outDir: str) -> dict:
    correlations = {}

    for (lJoint, rJoint), name in zip(BODY_LR_PAIRS, LR_NAMES):
        lStart = BODY_BLOCK_START + lJoint * 3
        rStart = BODY_BLOCK_START + rJoint * 3
        rs = []

        for k in range(3):
            lc = allFrames[:, lStart + k]
            rc = allFrames[:, rStart + k]

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
    fig.savefig(os.path.join(outDir, "05_symmetric_corr.png"), dpi=110)
    plt.close(fig)

    return correlations

def varianceDecomposition(samples: list[dict], outDir: str) -> dict:
    perClipMeans = []
    perClipStds = []

    for s in samples:
        m = s["motion"]
        perClipMeans.append(m.mean(axis=0))
        perClipStds.append(m.std(axis=0))
    perClipMeans = np.stack(perClipMeans)
    perClipStds = np.stack(perClipStds)

    withinStd = perClipStds.mean(axis=0)
    crossStd = perClipMeans.std(axis=0)
    safeCross = np.where(crossStd < 1e-6, 1e-6, crossStd)
    ratio = withinStd / safeCross

    ymax = float(max(withinStd.max(), crossStd.max())) * 1.1

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(withinStd, label="within-clip std (temporal motion)", color="steelblue")
    ax.plot(crossStd, label="cross-clip std (clip identity)", color="orange", alpha=0.8)

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
    fig.savefig(os.path.join(outDir, "06_var_decomposition.png"), dpi=110)
    plt.close(fig)

    finiteRatio = ratio[np.isfinite(ratio)]

    return {
        "mean_within_std": float(withinStd.mean()),
        "mean_cross_std": float(crossStd.mean()),
        "mean_ratio": float(finiteRatio.mean()) if len(finiteRatio) else 0.0,
    }

def writeReport(outDir: str, source: str, n: int, length: dict, breakdown: dict,
                channelStd: dict, pca: dict, corr: dict, varDecomp: dict) -> None:
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
    lines.append(f"- Dead channels (std < 1e-3): **{len(channelStd['dead_channels'])} of 168**")

    if channelStd["dead_channels"]:
        lines.append(f"  indices: {channelStd['dead_channels']}")
    lines.append("")
    lines.append("| block | size | mean std | max std | n_dead |")
    lines.append("|---|---|---|---|---|")

    for name, st in channelStd["block_stats"].items():
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
    avgCorr = float(np.mean(list(corr.values()))) if corr else 0.0
    lines.append("")
    lines.append(f"Mean across all L/R pairs: **{avgCorr:.3f}**")
    lines.append("")

    lines.append("## 6. Variance decomposition")
    lines.append("")
    lines.append(f"- Mean within-clip std (temporal motion): {varDecomp['mean_within_std']:.4f}")
    lines.append(f"- Mean cross-clip std (clip identity):   {varDecomp['mean_cross_std']:.4f}")
    lines.append(f"- Within / cross ratio: **{varDecomp['mean_ratio']:.3f}**")
    lines.append("")
    lines.append("- > 1: temporal motion dominates (codec must learn dynamics)")
    lines.append("- < 1: clip-to-clip variation dominates")
    lines.append("")

    with open(os.path.join(outDir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="amass", choices=["amass", "humanml3d", "all"],
                        dest="source")
    parser.add_argument("--data-dir", default="data/AMASS", dest="dataDir")
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3dDir")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="statsPath")
    parser.add_argument("--output-dir", default="data/.cache/eda", dest="outputDir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    outDir = os.path.join(args.outputDir, args.source)
    os.makedirs(outDir, exist_ok=True)

    log.info("[eda] loading samples (source=%s)", args.source)
    samples, descriptor = loadSamples(args.source, args.dataDir, args.humanml3dDir)
    log.info("[eda] %d clips loaded", len(samples))

    log.info("[eda] loading shared stats from %s", args.statsPath)
    loaded = np.load(args.statsPath)
    stats = MotionStats(mean=loaded["mean"], std=loaded["std"])

    log.info("[eda] step 1/6 length distribution")
    lengthSummary = lengthDistribution(samples, outDir)

    log.info("[eda] step 2/6 source breakdown")
    breakdown = sourceBreakdown(samples, outDir, args.source)

    log.info("[eda] step 3/6 stacking frames (this can take a minute)")
    allFrames = np.concatenate([s["motion"] for s in samples], axis=0).astype(np.float32)
    log.info("[eda]   total frames: %d", len(allFrames))

    log.info("[eda] step 4/6 per-channel std")
    channelStd = channelStdAnalysis(allFrames, outDir)

    log.info("[eda] step 5/6 PCA")
    framesNorm = normalize(allFrames, stats, clipValue=None)
    pcaResult = pcaAnalysis(framesNorm, outDir)

    log.info("[eda] step 6a/6 symmetric correlations")
    corrResult = symmetricCorrelations(allFrames, outDir)

    log.info("[eda] step 6b/6 variance decomposition")
    varDecomp = varianceDecomposition(samples, outDir)

    log.info("[eda] writing report")
    writeReport(outDir, descriptor, len(samples), lengthSummary, breakdown,
                channelStd, pcaResult, corrResult, varDecomp)

    log.info("[eda] DONE — outputs in %s", outDir)

    return 0

if __name__ == "__main__":
    sys.exit(main())
