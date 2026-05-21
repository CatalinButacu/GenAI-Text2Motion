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

from src.data.motion_dataset import MotionDataset
from src.data.motion_normalize import MotionStats
from src.data.unified import build_or_load_unified_buffer
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

def action_label(text: str) -> str:
    for label, pat in ACTION_PATTERNS:
        if pat.search(text):
            return label

    return "other"

def amass_subset(sample_id: str) -> str:
    return sample_id.split("/")[0] if "/" in sample_id else "unknown"

def load_checkpoint(ck_path: str, device: torch.device) -> tuple[MotionRVQTokenizer, dict]:
    log.info("[eval] loading %s", ck_path)
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = MotionRVQTokenizer(
        motion_dim=MOTION_DIM,
        latent_dim=cfg["latent_dim"],
        n_codebooks=cfg["n_codebooks"],
        codebook_size=cfg["codebook_size"],
        down_t=cfg["down_t"],
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    log.info("[eval] checkpoint epoch=%d  val_loss=%.4f", ck["epoch"], ck.get("val_loss", -1))

    return model, cfg

def build_test_dataset(cfg: dict, stats_path: str | None) -> tuple[Dataset, str]:
    """Mirror the dataset construction used at train time, test split only."""
    src = cfg.get("data_source", "amass")
    shared_stats = None

    if stats_path and os.path.exists(stats_path):
        s = np.load(stats_path)
        shared_stats = MotionStats(mean=s["mean"], std=s["std"])

    if src == "amass":
        ds = MotionDataset(cfg["data_dir"], "test", cfg["max_motion_length"],
                           augment=False, stats=shared_stats)

        return ds, "amass"

    if src == "humanml3d":
        ucfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, data_dir=cfg["humanml3d_dir"],
                                   amass_dir=cfg["data_dir"]),
        )
    else:
        ucfg = UnifiedConfig(
            amass=SourceConfig(enabled=True, data_dir=cfg["data_dir"]),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=True, data_dir=cfg["humanml3d_dir"],
                                   amass_dir=cfg["data_dir"]),
        )
    buf = build_or_load_unified_buffer(ucfg, src)
    ds = UnifiedMotionDataset("test", cfg["max_motion_length"], augment=False,
                              preloaded_buf=buf, stats=shared_stats)

    return ds, src

# build_or_load_unified_buffer is now imported from src.data.unified at the top of this file.

@torch.no_grad()
def evaluate_batch(model: MotionRVQTokenizer, motion: torch.Tensor, mask: torch.Tensor):
    """Returns (recon, indices, latent_mean) for a batch.

    latent_mean is the encoder pre-quantization output mean-pooled over time -> (B, latent_dim).
    Used for downstream clustering.
    """
    x = motion.transpose(1, 2)  # (B, D, T)
    z = model.encoder(x).transpose(1, 2)  # (B, T', latent_dim)
    quantized, indices, _ = model.rvq(z)
    recon = model.decoder(quantized.transpose(1, 2)).transpose(1, 2)
    # Mean-pool z over time using the (downsampled) mask.
    latent_mean = z.mean(dim=1)  # (B, latent_dim)

    return recon, indices, latent_mean

def reconstruction_quality(model: MotionRVQTokenizer, dataset, device: torch.device,
                           batch_size: int, max_batches: int | None) -> dict:
    """Per-clip MSE + per-block MSE + worst/best clips."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    per_clip_mse: list[float] = []
    per_block_sse: dict[str, float] = {name: 0.0 for name, _, _ in CHANNEL_BLOCKS}
    per_block_count: dict[str, float] = {name: 0.0 for name, _, _ in CHANNEL_BLOCKS}
    n_batches = 0

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        motion = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)
        recon, _, _ = evaluate_batch(model, motion, mask)
        # Mask out padded frames (mask is (B, T) with 1.0 = real, 0.0 = pad)
        diff_sq = (recon - motion) ** 2  # (B, T, D)
        mask_exp = mask.unsqueeze(-1)  # (B, T, 1)
        # Per-clip MSE
        clip_numer = (diff_sq * mask_exp).sum(dim=(1, 2))
        clip_denom = (mask_exp.sum(dim=1).squeeze(-1) * motion.shape[2]).clamp(min=1.0)
        per_clip_mse.extend((clip_numer / clip_denom).cpu().tolist())

        for name, start, end in CHANNEL_BLOCKS:
            blk_sse = (diff_sq[:, :, start:end] * mask_exp).sum().item()
            blk_n = (mask_exp.sum() * (end - start)).item()
            per_block_sse[name] += blk_sse
            per_block_count[name] += blk_n
        n_batches += 1

    mse_arr = np.array(per_clip_mse)
    block_mse = {n: per_block_sse[n] / max(per_block_count[n], 1.0) for n in per_block_sse}

    return {
        "n_clips": int(len(mse_arr)),
        "mse_mean": float(mse_arr.mean()),
        "mse_p10": float(np.percentile(mse_arr, 10)),
        "mse_p50": float(np.percentile(mse_arr, 50)),
        "mse_p90": float(np.percentile(mse_arr, 90)),
        "mse_p99": float(np.percentile(mse_arr, 99)),
        "block_mse": block_mse,
        "per_clip_mse": mse_arr.tolist(),
    }

def codebook_stats(model: MotionRVQTokenizer) -> dict:
    util = model.codebook_utilization()

    return {
        "active_pct": [u["active_fraction"] * 100 for u in util],
        "entropy_pct": [u["entropy"] / max(u["max_entropy"], 1e-8) * 100 for u in util],
    }

def collect_latents(model: MotionRVQTokenizer, dataset, device: torch.device,
                    batch_size: int, max_clips: int | None) -> tuple[np.ndarray, list, list]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    latents: list[np.ndarray] = []
    texts: list[str] = []
    sources: list[str] = []
    n = 0

    for batch in loader:
        motion = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)
        _, _, latent_mean = evaluate_batch(model, motion, mask)
        latents.append(latent_mean.cpu().numpy())

        for t in batch["texts"]:
            texts.append(t)
        # source field present on UnifiedMotionDataset items; AMASS dataset has none
        srcs = batch.get("source", ["amass"] * motion.shape[0])

        for s in srcs:
            sources.append(s if isinstance(s, str) else "unknown")
        n += motion.shape[0]

        if max_clips is not None and n >= max_clips:
            break

    limit = max_clips or n
    return np.concatenate(latents, axis=0)[:limit], texts[:limit], sources[:limit]

def plot_mse_hist(per_clip_mse: list[float], out_dir: str, title: str) -> None:
    arr = np.array(per_clip_mse)
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
    fig.savefig(os.path.join(out_dir, "01_recon_mse_hist.png"), dpi=110)
    plt.close(fig)

def plot_block_mse(block_mse: dict, out_dir: str, title: str) -> None:
    names = list(block_mse.keys())
    vals = [block_mse[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color=["tab:red", "tab:orange", "tab:blue", "tab:green",
                                "tab:purple", "tab:gray"])
    ax.set_ylabel("MSE (z-norm space)")
    ax.set_title(f"{title} — per-block reconstruction MSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_block_mse.png"), dpi=110)
    plt.close(fig)

def plot_codebook_usage(cb_stats: dict, out_dir: str, title: str) -> None:
    n_layers = len(cb_stats["active_pct"])
    x = np.arange(n_layers)
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, cb_stats["active_pct"], width, label="active %", color="seagreen")
    ax.bar(x + width / 2, cb_stats["entropy_pct"], width, label="entropy %", color="steelblue")
    ax.set_xlabel("codebook layer (0=coarsest)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title(f"{title} — codebook utilization per RVQ layer")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_codebook_usage.png"), dpi=110)
    plt.close(fig)

def plot_tsne_clusters(latents: np.ndarray, labels: list, out_dir: str, title: str,
                      label_kind: str) -> dict:
    """Run t-SNE and color points by category. Returns count by category."""
    n_clips = latents.shape[0]
    log.info("[eval] running t-SNE on %d clips x %d dims", n_clips, latents.shape[1])
    perplexity = min(30, max(5, n_clips // 4 - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=42, max_iter=500)
    coords = tsne.fit_transform(latents)

    counts: dict[str, int] = {}

    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.colormaps["tab20"]
    sorted_cats = sorted(counts.keys(), key=lambda c: -counts[c])

    for i, cat in enumerate(sorted_cats):
        idx = [j for j, lab in enumerate(labels) if lab == cat]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.55,
                   color=cmap(i % 20), label=f"{cat} (n={counts[cat]})")
    ax.set_title(f"{title} — t-SNE of clip latents (colored by {label_kind})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_tsne_clusters.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    return counts

def write_report(out_dir: str, ck_path: str, cfg: dict, recon: dict, cb_stats: dict,
                cluster_counts: dict, label_kind: str) -> None:
    lines: list[str] = []
    lines.append("# RVQ tokenizer evaluation\n")
    lines.append(f"**Checkpoint:** `{ck_path}`")
    lines.append(f"**Source:** {cfg.get('data_source', 'amass')}")
    lines.append(f"**Latent dim:** {cfg['latent_dim']}, **codebooks:** {cfg['n_codebooks']}, "
                 f"**codebook size:** {cfg['codebook_size']}, **down_t:** {cfg['down_t']}\n")

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

    for i, (a, e) in enumerate(zip(cb_stats["active_pct"], cb_stats["entropy_pct"])):
        lines.append(f"| {i} | {a:.1f} | {e:.1f} |")
    lines.append("")

    lines.append(f"## C) Latent cluster counts (label kind: {label_kind})\n")
    sorted_cats = sorted(cluster_counts.items(), key=lambda x: -x[1])

    for cat, n in sorted_cats:
        lines.append(f"- {cat}: {n}")
    lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "checkpoint": ck_path, "config": cfg,
            "recon": {k: v for k, v in recon.items() if k != "per_clip_mse"},
            "codebook": cb_stats, "cluster_counts": cluster_counts,
        }, indent=2))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--max-clips-cluster", type=int, default=2000, dest="max_clips_cluster",
                        help="Cap clips fed to t-SNE (slow on >2k)")
    parser.add_argument("--max-batches-mse", type=int, default=None, dest="max_batches_mse",
                        help="Cap batches for MSE; default = whole test set")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="stats_path",
                        help="Override stats path; defaults to AMASS-full shared stats")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Default: alongside checkpoint as eval_<run_id>/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.output_dir is None:
        ck_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(ck_dir, "eval")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info("[eval] outputs -> %s", args.output_dir)

    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)

    # Use stats from --stats-path (shared) so cross-checkpoint MSE is comparable.
    stats_path = args.stats_path if os.path.exists(args.stats_path) else None
    log.info("[eval] building test dataset (source=%s)", cfg.get("data_source", "amass"))
    dataset, src = build_test_dataset(cfg, stats_path)
    log.info("[eval] test set: %d samples", len(dataset))

    title = f"{src} (run_id {os.path.basename(os.path.dirname(args.checkpoint))})"

    log.info("[eval] step A: reconstruction quality")
    recon = reconstruction_quality(model, dataset, device, args.batch_size, args.max_batches_mse)
    plot_mse_hist(recon["per_clip_mse"], args.output_dir, title)
    plot_block_mse(recon["block_mse"], args.output_dir, title)
    log.info("[eval]   mean MSE=%.4f  p50=%.4f  p90=%.4f",
             recon["mse_mean"], recon["mse_p50"], recon["mse_p90"])

    log.info("[eval] step B: codebook stats")
    cb_stats = codebook_stats(model)
    plot_codebook_usage(cb_stats, args.output_dir, title)

    log.info("[eval] step C: collecting latents for clustering")
    latents, texts, sources = collect_latents(model, dataset, device, args.batch_size,
                                              args.max_clips_cluster)
    log.info("[eval]   collected %d clip latents", len(latents))
    # Pick label kind: HumanML3D / unified -> action keyword. AMASS -> AMASS subset.
    if src == "amass":
        labels = []

        for i in range(len(latents)):
            sid = dataset.samples[i].get("sample_id", "")
            labels.append(amass_subset(sid))
        label_kind = "AMASS subset"
    else:
        labels = [action_label(t) for t in texts]
        label_kind = "action keyword"

    cluster_counts = plot_tsne_clusters(latents, labels, args.output_dir, title, label_kind)

    write_report(args.output_dir, args.checkpoint, cfg, recon, cb_stats, cluster_counts, label_kind)
    log.info("[eval] DONE — see %s/report.md", args.output_dir)

    return 0

if __name__ == "__main__":
    sys.exit(main())
