"""Hierarchical clustering of trained RVQ tokenizer's clip latents.

Reveals how the latent space decomposes from a single root cluster down to
fine-grained sub-clusters. For HumanML3D / unified checkpoints, auto-labels
each cluster via TF-IDF on the per-clip texts.

CPU-only — does not compete with a running training job.

Usage:
    python scripts/evaluation/cluster_rvq_hierarchical.py \
        --checkpoint checkpoints/rvq_tokenizer/<run_id>/best_model.pt
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
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from scripts.evaluation.eval_rvq import (
    action_label,
    amass_subset,
    build_test_dataset,
    collect_latents,
    load_checkpoint,
)

log = logging.getLogger(__name__)

K_LEVELS = (2, 4, 8, 16, 32)
ENGLISH_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "with", "to", "of", "in", "on", "at", "by", "for", "from", "as", "and",
    "or", "but", "his", "her", "him", "she", "he", "they", "their", "them",
    "this", "that", "these", "those", "it", "its", "person", "someone",
    "human", "subject", "actor", "user", "people",
}

def compute_linkage(latents: np.ndarray) -> np.ndarray:
    log.info("[hcluster] running ward linkage on %d x %d latents", *latents.shape)

    return linkage(latents, method="ward")

def plot_dendrogram(Z: np.ndarray, out_dir: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True, ax=ax,
               color_threshold=0)
    ax.set_xlabel("cluster (truncated to top 30 leaves)")
    ax.set_ylabel("ward distance")
    ax.set_title(f"{title} — hierarchical dendrogram")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_dendrogram.png"), dpi=110)
    plt.close(fig)

def run_tsne(latents: np.ndarray) -> np.ndarray:
    n_clips = latents.shape[0]
    perplexity = min(30, max(5, n_clips // 4 - 1))
    log.info("[hcluster] t-SNE 2D on %d clips (perplexity=%d)", n_clips, perplexity)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=42, max_iter=500)

    return tsne.fit_transform(latents)

def top_words_per_cluster(texts: list[str], cluster_ids: np.ndarray,
                       k: int, top_n: int = 5) -> dict[int, list[str]]:
    if not any(t.strip() for t in texts):
        return {c: [] for c in range(1, k + 1)}
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP), max_features=2000,
                                  token_pattern=r"\b[a-zA-Z]{3,}\b")
    tfidf = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    out: dict[int, list[str]] = {}

    for c in range(1, k + 1):
        idx = np.where(cluster_ids == c)[0]

        if len(idx) == 0:
            out[c] = []
            continue
        mean_vec = np.asarray(tfidf[idx].mean(axis=0)).flatten()
        top_idx = np.argsort(-mean_vec)[:top_n]
        out[c] = [vocab[i] for i in top_idx if mean_vec[i] > 0]

    return out

def action_distribution(texts: list[str], cluster_ids: np.ndarray, k: int) -> dict:
    """For each cluster, report how its clips break down across the action keywords."""
    out: dict[int, dict[str, int]] = {}

    for c in range(1, k + 1):
        idx = np.where(cluster_ids == c)[0]
        counts: dict[str, int] = {}

        for i in idx:
            label = action_label(texts[i]) if i < len(texts) else "other"
            counts[label] = counts.get(label, 0) + 1
        out[c] = dict(sorted(counts.items(), key=lambda x: -x[1]))

    return out

def amass_distribution(samples: list, cluster_ids: np.ndarray, k: int) -> dict:
    out: dict[int, dict[str, int]] = {}

    for c in range(1, k + 1):
        idx = np.where(cluster_ids == c)[0]
        counts: dict[str, int] = {}

        for i in idx:
            sid = samples[i].get("sample_id", "") if i < len(samples) else ""
            sub = amass_subset(sid)
            counts[sub] = counts.get(sub, 0) + 1
        out[c] = dict(sorted(counts.items(), key=lambda x: -x[1]))

    return out

def plot_k_grid(coords: np.ndarray, cluster_map: dict[int, np.ndarray], out_dir: str,
              title: str) -> None:
    n_k = len(cluster_map)
    fig, axes = plt.subplots(1, n_k, figsize=(4 * n_k, 4.5), squeeze=False)
    cmap = plt.colormaps["tab20"]

    for i, (k, ids) in enumerate(cluster_map.items()):
        ax = axes[0, i]

        for c in range(1, k + 1):
            idx = np.where(ids == c)[0]

            if len(idx) == 0:
                continue
            ax.scatter(coords[idx, 0], coords[idx, 1], s=6, alpha=0.55,
                       color=cmap((c - 1) % 20))
        ax.set_title(f"k={k}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{title} — t-SNE colored by hierarchical cluster id")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_hierarchical_kgrid.png"), dpi=110)
    plt.close(fig)

def write_report(out_dir: str, title: str, n: int, cluster_map: dict[int, np.ndarray],
                words_by_k: dict, action_dist_by_k: dict, amass_dist_by_k: dict | None) -> None:
    lines: list[str] = []
    lines.append(f"# Hierarchical cluster report — {title}\n")
    lines.append(f"Total clips clustered: **{n}**\n")
    lines.append("Each section reports cuts of the dendrogram at successive k values, with cluster")
    lines.append("size, top characteristic words (TF-IDF on per-clip texts), and the distribution")
    lines.append("of regex-matched action keywords across each cluster's members.\n")

    for k, ids in cluster_map.items():
        lines.append(f"## k = {k}\n")
        lines.append("| cluster | size | top words | action keyword distribution |")
        lines.append("|---|---|---|---|")
        sizes = [(c, int(np.sum(ids == c))) for c in range(1, k + 1)]
        sizes.sort(key=lambda x: -x[1])

        for c, sz in sizes:
            words = ", ".join(words_by_k[k].get(c, [])) or "—"
            act_dist = action_dist_by_k[k].get(c, {})
            dist_str = ", ".join(f"{lab}:{n}" for lab, n in list(act_dist.items())[:5]) or "—"

            if amass_dist_by_k is not None:
                amass_str = ", ".join(f"{lab}:{n}" for lab, n in
                                     list(amass_dist_by_k[k].get(c, {}).items())[:3])
                lines.append(f"| {c} | {sz} | {words or amass_str or '—'} | {dist_str} |")
            else:
                lines.append(f"| {c} | {sz} | {words} | {dist_str} |")
        lines.append("")

    with open(os.path.join(out_dir, "report_clusters.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--max-clips", type=int, default=1500, dest="max_clips")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="stats_path")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    stats_path = args.stats_path if os.path.exists(args.stats_path) else None
    dataset, src = build_test_dataset(cfg, stats_path)
    log.info("[hcluster] test set: %d samples (source=%s)", len(dataset), src)

    run_title = f"{src} ({os.path.basename(os.path.dirname(args.checkpoint))})"
    latents, texts, _ = collect_latents(model, dataset, device, args.batch_size, args.max_clips)
    log.info("[hcluster] collected %d latents", len(latents))

    Z = compute_linkage(latents)
    plot_dendrogram(Z, args.output_dir, run_title)

    coords = run_tsne(latents)
    cluster_map: dict[int, np.ndarray] = {}

    for k in K_LEVELS:
        ids = fcluster(Z, t=k, criterion="maxclust")
        cluster_map[k] = ids
    plot_k_grid(coords, cluster_map, args.output_dir, run_title)

    words_by_k: dict[int, dict[int, list[str]]] = {}
    action_dist_by_k: dict[int, dict[int, dict[str, int]]] = {}

    for k, ids in cluster_map.items():
        words_by_k[k] = top_words_per_cluster(texts, ids, k)
        action_dist_by_k[k] = action_distribution(texts, ids, k)

    amass_dist_by_k: dict[int, dict[int, dict[str, int]]] | None = None

    if src == "amass":
        amass_dist_by_k = {}

        for k, ids in cluster_map.items():
            amass_dist_by_k[k] = amass_distribution(dataset.samples, ids, k)  # type: ignore[attr-defined]
    n_clips = int(latents.shape[0])
    write_report(args.output_dir, run_title, n_clips, cluster_map, words_by_k, action_dist_by_k,
                amass_dist_by_k)
    log.info("[hcluster] DONE — see %s/report_clusters.md", args.output_dir)

    return 0

if __name__ == "__main__":
    sys.exit(main())
