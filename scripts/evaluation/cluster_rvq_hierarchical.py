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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.eval_rvq import (
    ACTION_PATTERNS,
    actionLabel,
    amassSubset,
    buildTestDataset,
    collectLatents,
    loadCheckpoint,
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


def computeLinkage(latents: np.ndarray) -> np.ndarray:
    log.info("[hcluster] running ward linkage on %d x %d latents", *latents.shape)

    return linkage(latents, method="ward")


def plotDendrogram(Z: np.ndarray, outDir: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True, ax=ax,
               color_threshold=0)
    ax.set_xlabel("cluster (truncated to top 30 leaves)")
    ax.set_ylabel("ward distance")
    ax.set_title(f"{title} — hierarchical dendrogram")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "05_dendrogram.png"), dpi=110)
    plt.close(fig)


def runTsne(latents: np.ndarray) -> np.ndarray:
    nClips = latents.shape[0]
    perplexity = min(30, max(5, nClips // 4 - 1))
    log.info("[hcluster] t-SNE 2D on %d clips (perplexity=%d)", nClips, perplexity)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=42, max_iter=500)

    return tsne.fit_transform(latents)


def topWordsPerCluster(texts: list[str], clusterIds: np.ndarray,
                       k: int, topN: int = 5) -> dict[int, list[str]]:
    if not any(t.strip() for t in texts):
        return {c: [] for c in range(1, k + 1)}
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP), max_features=2000,
                                  token_pattern=r"\b[a-zA-Z]{3,}\b")
    tfidf = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    out: dict[int, list[str]] = {}

    for c in range(1, k + 1):
        idx = np.where(clusterIds == c)[0]

        if len(idx) == 0:
            out[c] = []
            continue
        meanVec = np.asarray(tfidf[idx].mean(axis=0)).flatten()
        topIdx = np.argsort(-meanVec)[:topN]
        out[c] = [vocab[i] for i in topIdx if meanVec[i] > 0]

    return out


def actionDistribution(texts: list[str], clusterIds: np.ndarray, k: int) -> dict:
    """For each cluster, report how its clips break down across the action keywords."""
    out: dict[int, dict[str, int]] = {}

    for c in range(1, k + 1):
        idx = np.where(clusterIds == c)[0]
        counts: dict[str, int] = {}

        for i in idx:
            label = actionLabel(texts[i]) if i < len(texts) else "other"
            counts[label] = counts.get(label, 0) + 1
        out[c] = dict(sorted(counts.items(), key=lambda x: -x[1]))

    return out


def amassDistribution(samples: list, clusterIds: np.ndarray, k: int) -> dict:
    out: dict[int, dict[str, int]] = {}

    for c in range(1, k + 1):
        idx = np.where(clusterIds == c)[0]
        counts: dict[str, int] = {}

        for i in idx:
            sid = samples[i].get("sample_id", "") if i < len(samples) else ""
            sub = amassSubset(sid)
            counts[sub] = counts.get(sub, 0) + 1
        out[c] = dict(sorted(counts.items(), key=lambda x: -x[1]))

    return out


def plotKGrid(coords: np.ndarray, clusterMap: dict[int, np.ndarray], outDir: str,
              title: str) -> None:
    nK = len(clusterMap)
    fig, axes = plt.subplots(1, nK, figsize=(4 * nK, 4.5), squeeze=False)
    cmap = plt.colormaps["tab20"]

    for i, (k, ids) in enumerate(clusterMap.items()):
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
    fig.savefig(os.path.join(outDir, "06_hierarchical_kgrid.png"), dpi=110)
    plt.close(fig)


def writeReport(outDir: str, title: str, n: int, clusterMap: dict[int, np.ndarray],
                wordsByK: dict, actionDistByK: dict, amassDistByK: dict | None) -> None:
    lines: list[str] = []
    lines.append(f"# Hierarchical cluster report — {title}\n")
    lines.append(f"Total clips clustered: **{n}**\n")
    lines.append("Each section reports cuts of the dendrogram at successive k values, with cluster")
    lines.append("size, top characteristic words (TF-IDF on per-clip texts), and the distribution")
    lines.append("of regex-matched action keywords across each cluster's members.\n")

    for k, ids in clusterMap.items():
        lines.append(f"## k = {k}\n")
        lines.append("| cluster | size | top words | action keyword distribution |")
        lines.append("|---|---|---|---|")
        sizes = [(c, int(np.sum(ids == c))) for c in range(1, k + 1)]
        sizes.sort(key=lambda x: -x[1])

        for c, sz in sizes:
            words = ", ".join(wordsByK[k].get(c, [])) or "—"
            actDist = actionDistByK[k].get(c, {})
            distStr = ", ".join(f"{lab}:{n}" for lab, n in list(actDist.items())[:5]) or "—"

            if amassDistByK is not None:
                amassStr = ", ".join(f"{lab}:{n}" for lab, n in
                                     list(amassDistByK[k].get(c, {}).items())[:3])
                lines.append(f"| {c} | {sz} | {words or amassStr or '—'} | {distStr} |")
            else:
                lines.append(f"| {c} | {sz} | {words} | {distStr} |")
        lines.append("")

    with open(os.path.join(outDir, "report_clusters.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32, dest="batchSize")
    parser.add_argument("--max-clips", type=int, default=1500, dest="maxClips")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="statsPath")
    parser.add_argument("--output-dir", default=None, dest="outputDir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.outputDir is None:
        args.outputDir = os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(args.outputDir, exist_ok=True)

    device = torch.device(args.device)
    model, cfg = loadCheckpoint(args.checkpoint, device)
    statsPath = args.statsPath if os.path.exists(args.statsPath) else None
    dataset, src = buildTestDataset(cfg, statsPath)
    log.info("[hcluster] test set: %d samples (source=%s)", len(dataset), src)

    runTitle = f"{src} ({os.path.basename(os.path.dirname(args.checkpoint))})"
    latents, texts, _ = collectLatents(model, dataset, device, args.batchSize, args.maxClips)
    log.info("[hcluster] collected %d latents", len(latents))

    Z = computeLinkage(latents)
    plotDendrogram(Z, args.outputDir, runTitle)

    coords = runTsne(latents)
    clusterMap: dict[int, np.ndarray] = {}

    for k in K_LEVELS:
        ids = fcluster(Z, t=k, criterion="maxclust")
        clusterMap[k] = ids
    plotKGrid(coords, clusterMap, args.outputDir, runTitle)

    wordsByK: dict[int, dict[int, list[str]]] = {}
    actionDistByK: dict[int, dict[int, dict[str, int]]] = {}

    for k, ids in clusterMap.items():
        wordsByK[k] = topWordsPerCluster(texts, ids, k)
        actionDistByK[k] = actionDistribution(texts, ids, k)

    amassDistByK: dict[int, dict[int, dict[str, int]]] | None = None

    if src == "amass":
        amassDistByK = {}

        for k, ids in clusterMap.items():
            amassDistByK[k] = amassDistribution(dataset.samples, ids, k)  # type: ignore[attr-defined]
    nClips = int(latents.shape[0])
    writeReport(args.outputDir, runTitle, nClips, clusterMap, wordsByK, actionDistByK,
                amassDistByK)
    log.info("[hcluster] DONE — see %s/report_clusters.md", args.outputDir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
