"""Evaluate a trained RVQ tokenizer.

Two halves:
  A) Reconstruction quality
       - per-clip MSE distribution (z-norm space)
       - per-channel-block MSE (which body parts reconstruct best/worst)
       - codebook usage (active fraction, entropy, top-N entries per layer)

  B) Latent clustering
       - encoder mean-pooled latent per clip -> t-SNE 2D
       - colored by action keyword (HumanML3D) or AMASS subset
       - reveals whether semantically similar motions cluster together

Defaults to CPU so this can run while another training job is using the GPU.

Usage:
    python scripts/evaluation/eval_rvq.py \
        --checkpoint checkpoints/rvq_tokenizer/<run_id>/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.augmentation import detectTpose, qualityFilter, resampleToFps
from src.data.dataset_cache import INGEST_MAX_LENGTH, loadOrBuildCache
from src.data.motion_dataset import MotionDataset
from src.data.motion_normalize import MotionStats
from src.data.unified import buildOrLoadUnifiedBuffer, buildSourcesBuffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig, UnifiedMotionDataset
from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer
from src.shared.constants import MOTION_DIM

log = logging.getLogger(__name__)

CHANNEL_BLOCKS = [
    ("root_orient", 0, 3),
    ("translation", 3, 6),
    ("body", 6, 69),
    ("left_hand", 69, 114),
    ("right_hand", 114, 159),
    ("jaw_eyes", 159, 168),
]

ACTION_PATTERNS = [
    ("walk", re.compile(r"\b(walk|walking|walks|stroll|strolls)\b", re.I)),
    ("run", re.compile(r"\b(run|running|runs|jog|jogging|jogs)\b", re.I)),
    ("jump", re.compile(r"\b(jump|jumping|jumps|leap|leaps|hop|hops)\b", re.I)),
    ("kick", re.compile(r"\b(kick|kicks|kicking)\b", re.I)),
    ("punch", re.compile(r"\b(punch|punches|punching|hit|hits)\b", re.I)),
    ("wave", re.compile(r"\b(wave|waves|waving)\b", re.I)),
    ("sit", re.compile(r"\b(sit|sits|sitting|seated)\b", re.I)),
    ("stand", re.compile(r"\b(stand|stands|standing)\b", re.I)),
    ("turn", re.compile(r"\b(turn|turns|turning|rotate|rotates)\b", re.I)),
    ("dance", re.compile(r"\b(dance|dances|dancing)\b", re.I)),
    ("throw", re.compile(r"\b(throw|throws|throwing|toss)\b", re.I)),
    ("bend", re.compile(r"\b(bend|bends|bending|crouch|crouches|squat|squats)\b", re.I)),
    ("kneel", re.compile(r"\b(kneel|kneels|kneeling)\b", re.I)),
    ("climb", re.compile(r"\b(climb|climbs|climbing)\b", re.I)),
]


def actionLabel(text: str) -> str:
    for label, pat in ACTION_PATTERNS:
        if pat.search(text):
            return label

    return "other"


def amassSubset(sampleId: str) -> str:
    return sampleId.split("/")[0] if "/" in sampleId else "unknown"


def loadCheckpoint(ckPath: str, device: torch.device) -> tuple[MotionRVQTokenizer, dict]:
    log.info("[eval] loading %s", ckPath)
    ck = torch.load(ckPath, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = MotionRVQTokenizer(
        motionDim=MOTION_DIM,
        latentDim=cfg["latentDim"],
        nCodebooks=cfg["nCodebooks"],
        codebookSize=cfg["codebookSize"],
        downT=cfg["downT"],
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    log.info("[eval] checkpoint epoch=%d  val_loss=%.4f", ck["epoch"], ck.get("val_loss", -1))

    return model, cfg


def buildTestDataset(cfg: dict, statsPath: str | None) -> tuple[Dataset, str]:
    """Mirror the dataset construction used at train time, test split only."""
    src = cfg.get("dataSource", "amass")
    sharedStats = None

    if statsPath and os.path.exists(statsPath):
        s = np.load(statsPath)
        sharedStats = MotionStats(mean=s["mean"], std=s["std"])

    if src == "amass":
        ds = MotionDataset(cfg["dataDir"], "test", cfg["maxMotionLength"],
                           augment=False, stats=sharedStats)

        return ds, "amass"

    if src == "humanml3d":
        ucfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, dataDir=cfg["humanml3dDir"],
                                   amassDir=cfg["dataDir"]),
        )
    else:
        ucfg = UnifiedConfig(
            amass=SourceConfig(enabled=True, dataDir=cfg["dataDir"]),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, dataDir=cfg["humanml3dDir"],
                                   amassDir=cfg["dataDir"]),
        )
    buf = buildOrLoadUnifiedBuffer(ucfg, src)
    ds = UnifiedMotionDataset("test", cfg["maxMotionLength"], augment=False,
                              preloadedBuf=buf, stats=sharedStats)

    return ds, src


# buildOrLoadUnifiedBuffer is now imported from src.data.unified at the top of this file.


@torch.no_grad()
def evaluateBatch(model: MotionRVQTokenizer, motion: torch.Tensor, mask: torch.Tensor):
    """Returns (recon, indices, latentMean) for a batch.

    latentMean is the encoder pre-quantization output mean-pooled over time -> (B, latent_dim).
    Used for downstream clustering.
    """
    x = motion.transpose(1, 2)  # (B, D, T)
    z = model.encoder(x).transpose(1, 2)  # (B, T', latent_dim)
    quantized, indices, _ = model.rvq(z)
    recon = model.decoder(quantized.transpose(1, 2)).transpose(1, 2)
    # Mean-pool z over time using the (downsampled) mask.
    latentMean = z.mean(dim=1)  # (B, latent_dim)

    return recon, indices, latentMean


def reconstructionQuality(model: MotionRVQTokenizer, dataset, device: torch.device,
                           batchSize: int, maxBatches: int | None) -> dict:
    """Per-clip MSE + per-block MSE + worst/best clips."""
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)
    perClipMse: list[float] = []
    perBlockSse: dict[str, float] = {name: 0.0 for name, _, _ in CHANNEL_BLOCKS}
    perBlockCount: dict[str, float] = {name: 0.0 for name, _, _ in CHANNEL_BLOCKS}
    nBatches = 0

    for batch in loader:
        if maxBatches is not None and nBatches >= maxBatches:
            break
        motion = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)
        recon, _, _ = evaluateBatch(model, motion, mask)
        # Mask out padded frames (mask is (B, T) with 1.0 = real, 0.0 = pad)
        diffSq = (recon - motion) ** 2  # (B, T, D)
        maskExp = mask.unsqueeze(-1)  # (B, T, 1)
        # Per-clip MSE
        clipNumer = (diffSq * maskExp).sum(dim=(1, 2))
        clipDenom = (maskExp.sum(dim=1).squeeze(-1) * motion.shape[2]).clamp(min=1.0)
        perClipMse.extend((clipNumer / clipDenom).cpu().tolist())

        for name, start, end in CHANNEL_BLOCKS:
            blkSse = (diffSq[:, :, start:end] * maskExp).sum().item()
            blkN = (maskExp.sum() * (end - start)).item()
            perBlockSse[name] += blkSse
            perBlockCount[name] += blkN
        nBatches += 1

    mseArr = np.array(perClipMse)
    blockMse = {n: perBlockSse[n] / max(perBlockCount[n], 1.0) for n in perBlockSse}

    return {
        "n_clips": int(len(mseArr)),
        "mse_mean": float(mseArr.mean()),
        "mse_p10": float(np.percentile(mseArr, 10)),
        "mse_p50": float(np.percentile(mseArr, 50)),
        "mse_p90": float(np.percentile(mseArr, 90)),
        "mse_p99": float(np.percentile(mseArr, 99)),
        "block_mse": blockMse,
        "per_clip_mse": mseArr.tolist(),
    }


def codebookStats(model: MotionRVQTokenizer) -> dict:
    util = model.codebookUtilization()

    return {
        "active_pct": [u["active_fraction"] * 100 for u in util],
        "entropy_pct": [u["entropy"] / max(u["max_entropy"], 1e-8) * 100 for u in util],
    }


def collectLatents(model: MotionRVQTokenizer, dataset, device: torch.device,
                    batchSize: int, maxClips: int | None) -> tuple[np.ndarray, list, list]:
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)
    latents: list[np.ndarray] = []
    texts: list[str] = []
    sources: list[str] = []
    n = 0

    for batch in loader:
        motion = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)
        _, _, latentMean = evaluateBatch(model, motion, mask)
        latents.append(latentMean.cpu().numpy())

        for t in batch["texts"]:
            texts.append(t)
        # source field present on UnifiedMotionDataset items; AMASS dataset has none
        srcs = batch.get("source", ["amass"] * motion.shape[0])

        for s in srcs:
            sources.append(s if isinstance(s, str) else "unknown")
        n += motion.shape[0]

        if maxClips is not None and n >= maxClips:
            break

    return np.concatenate(latents, axis=0)[:maxClips or n], texts[:maxClips or n], sources[:maxClips or n]


def plotMseHist(perClipMse: list[float], outDir: str, title: str) -> None:
    arr = np.array(perClipMse)
    med = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr, bins=50, color="steelblue")
    ax.axvline(med, color="red", linestyle="--", label=f"median={med:.4f}")
    ax.axvline(p90, color="orange", linestyle="--", label=f"p90={p90:.4f}")
    ax.set_xlabel("per-clip MSE (z-norm space)")
    ax.set_ylabel("clip count")
    ax.set_title(f"{title} — reconstruction MSE distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "01_recon_mse_hist.png"), dpi=110)
    plt.close(fig)


def plotBlockMse(blockMse: dict, outDir: str, title: str) -> None:
    names = list(blockMse.keys())
    vals = [blockMse[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color=["tab:red", "tab:orange", "tab:blue", "tab:green",
                                "tab:purple", "tab:gray"])
    ax.set_ylabel("MSE (z-norm space)")
    ax.set_title(f"{title} — per-block reconstruction MSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "02_block_mse.png"), dpi=110)
    plt.close(fig)


def plotCodebookUsage(cbStats: dict, outDir: str, title: str) -> None:
    nLayers = len(cbStats["active_pct"])
    x = np.arange(nLayers)
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, cbStats["active_pct"], width, label="active %", color="seagreen")
    ax.bar(x + width / 2, cbStats["entropy_pct"], width, label="entropy %", color="steelblue")
    ax.set_xlabel("codebook layer (0=coarsest)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title(f"{title} — codebook utilization per RVQ layer")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "03_codebook_usage.png"), dpi=110)
    plt.close(fig)


def plotTsneClusters(latents: np.ndarray, labels: list, outDir: str, title: str,
                      labelKind: str) -> dict:
    """Run t-SNE and color points by category. Returns count by category."""
    nClips = latents.shape[0]
    log.info("[eval] running t-SNE on %d clips x %d dims", nClips, latents.shape[1])
    perplexity = min(30, max(5, nClips // 4 - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=42, max_iter=500)
    coords = tsne.fit_transform(latents)

    counts: dict[str, int] = {}

    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.colormaps["tab20"]
    sortedCats = sorted(counts.keys(), key=lambda c: -counts[c])

    for i, cat in enumerate(sortedCats):
        idx = [j for j, lab in enumerate(labels) if lab == cat]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.55,
                   color=cmap(i % 20), label=f"{cat} (n={counts[cat]})")
    ax.set_title(f"{title} — t-SNE of clip latents (colored by {labelKind})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "04_tsne_clusters.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    return counts


def writeReport(outDir: str, ckPath: str, cfg: dict, recon: dict, cbStats: dict,
                clusterCounts: dict, labelKind: str) -> None:
    lines: list[str] = []
    lines.append(f"# RVQ tokenizer evaluation\n")
    lines.append(f"**Checkpoint:** `{ckPath}`")
    lines.append(f"**Source:** {cfg.get('dataSource', 'amass')}")
    lines.append(f"**Latent dim:** {cfg['latentDim']}, **codebooks:** {cfg['nCodebooks']}, "
                 f"**codebook size:** {cfg['codebookSize']}, **downT:** {cfg['downT']}\n")

    lines.append("## A) Reconstruction quality\n")
    lines.append(f"- Test clips evaluated: **{recon['n_clips']}**")
    lines.append(f"- Mean MSE: **{recon['mse_mean']:.4f}**  (z-norm space)")
    lines.append(f"- Percentiles p10 / p50 / p90 / p99: "
                 f"{recon['mse_p10']:.4f} / {recon['mse_p50']:.4f} / "
                 f"{recon['mse_p90']:.4f} / {recon['mse_p99']:.4f}\n")

    lines.append("### Per-block MSE (lower = better)\n")
    lines.append("| block | MSE |")
    lines.append("|---|---|")

    for name, m in recon["block_mse"].items():
        lines.append(f"| {name} | {m:.4f} |")
    lines.append("")

    lines.append("## B) Codebook utilization (per RVQ layer)\n")
    lines.append("| layer | active % | entropy % |")
    lines.append("|---|---|---|")

    for i, (a, e) in enumerate(zip(cbStats["active_pct"], cbStats["entropy_pct"])):
        lines.append(f"| {i} | {a:.1f} | {e:.1f} |")
    lines.append("")

    lines.append(f"## C) Latent cluster counts (label kind: {labelKind})\n")
    sortedCats = sorted(clusterCounts.items(), key=lambda x: -x[1])

    for cat, n in sortedCats:
        lines.append(f"- {cat}: {n}")
    lines.append("")

    with open(os.path.join(outDir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(os.path.join(outDir, "metrics.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "checkpoint": ckPath, "config": cfg,
            "recon": {k: v for k, v in recon.items() if k != "per_clip_mse"},
            "codebook": cbStats, "cluster_counts": clusterCounts,
        }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32, dest="batchSize")
    parser.add_argument("--max-clips-cluster", type=int, default=2000, dest="maxClipsCluster",
                        help="Cap clips fed to t-SNE (slow on >2k)")
    parser.add_argument("--max-batches-mse", type=int, default=None, dest="maxBatchesMse",
                        help="Cap batches for MSE; default = whole test set")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="statsPath",
                        help="Override stats path; defaults to AMASS-full shared stats")
    parser.add_argument("--output-dir", default=None, dest="outputDir",
                        help="Default: alongside checkpoint as eval_<run_id>/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.outputDir is None:
        ckDir = os.path.dirname(args.checkpoint)
        args.outputDir = os.path.join(ckDir, "eval")
    os.makedirs(args.outputDir, exist_ok=True)
    log.info("[eval] outputs -> %s", args.outputDir)

    device = torch.device(args.device)
    model, cfg = loadCheckpoint(args.checkpoint, device)

    # Use stats from --stats-path (shared) so cross-checkpoint MSE is comparable.
    statsPath = args.statsPath if os.path.exists(args.statsPath) else None
    log.info("[eval] building test dataset (source=%s)", cfg.get("dataSource", "amass"))
    dataset, src = buildTestDataset(cfg, statsPath)
    log.info("[eval] test set: %d samples", len(dataset))

    title = f"{src} (run_id {os.path.basename(os.path.dirname(args.checkpoint))})"

    log.info("[eval] step A: reconstruction quality")
    recon = reconstructionQuality(model, dataset, device, args.batchSize, args.maxBatchesMse)
    plotMseHist(recon["per_clip_mse"], args.outputDir, title)
    plotBlockMse(recon["block_mse"], args.outputDir, title)
    log.info("[eval]   mean MSE=%.4f  p50=%.4f  p90=%.4f",
             recon["mse_mean"], recon["mse_p50"], recon["mse_p90"])

    log.info("[eval] step B: codebook stats")
    cbStats = codebookStats(model)
    plotCodebookUsage(cbStats, args.outputDir, title)

    log.info("[eval] step C: collecting latents for clustering")
    latents, texts, sources = collectLatents(model, dataset, device, args.batchSize,
                                              args.maxClipsCluster)
    log.info("[eval]   collected %d clip latents", len(latents))
    # Pick label kind: HumanML3D / unified -> action keyword. AMASS -> AMASS subset.
    if src == "amass":
        labels = []

        for i in range(len(latents)):
            sid = dataset.samples[i].get("sample_id", "")
            labels.append(amassSubset(sid))
        labelKind = "AMASS subset"
    else:
        labels = [actionLabel(t) for t in texts]
        labelKind = "action keyword"

    clusterCounts = plotTsneClusters(latents, labels, args.outputDir, title, labelKind)

    writeReport(args.outputDir, args.checkpoint, cfg, recon, cbStats, clusterCounts, labelKind)
    log.info("[eval] DONE — see %s/report.md", args.outputDir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
