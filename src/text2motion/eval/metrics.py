"""Text-to-motion metrics in the matcher's embedding space (Guo et al. protocol).

FID, R-precision (top-1/2/3), Diversity, MM-Dist. MultiModality is added once a generator can
produce multiple samples per prompt. All operate on (N, 512) L2-normalised features from `matcher`.
Reference: Guo et al., CVPR 2022 (github.com/EricGuo5513/text-to-motion); same definitions used by
T2M-GPT (arXiv:2301.06052) and MoMask (arXiv:2312.00063).
"""

import numpy as np
from scipy import linalg


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Fréchet distance between two Gaussians — the FID core."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fid(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    """FID between real and generated motion embeddings, each (N, 512)."""
    mu_r, cov_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, cov_g = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)

    return frechet_distance(mu_r, cov_r, mu_g, cov_g)


def r_precision(
    text_feats: np.ndarray,
    motion_feats: np.ndarray,
    pool_size: int = 32,
    top_k: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Top-1..top_k retrieval accuracy. For each text, rank its true motion against pool_size-1
    distractors by Euclidean distance; count when the true match lands in the top-k.

    Returns array of length top_k (cumulative top-1, top-2, ... accuracies)."""
    n = len(text_feats)
    rng = np.random.default_rng(seed)
    hits = np.zeros(top_k)
    groups = 0

    for start in range(0, n - pool_size + 1, pool_size):
        idx = np.arange(start, start + pool_size)

        for i in range(pool_size):
            distractors = rng.choice([j for j in idx if j != idx[i]], pool_size - 1, replace=False)
            pool = np.concatenate([[idx[i]], distractors])  # position 0 = the true match
            dists = np.linalg.norm(text_feats[idx[i]] - motion_feats[pool], axis=1)
            rank = int(np.argsort(dists).tolist().index(0))

            for k in range(top_k):
                if rank <= k:
                    hits[k] += 1

            groups += 1

    return hits / max(groups, 1)


def mm_dist(text_feats: np.ndarray, motion_feats: np.ndarray) -> float:
    """Mean Euclidean distance between paired text and motion embeddings."""
    return float(np.linalg.norm(text_feats - motion_feats, axis=1).mean())


def diversity(motion_feats: np.ndarray, num_pairs: int = 300, seed: int = 0) -> float:
    """Average distance over random pairs of motion embeddings."""
    n = len(motion_feats)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, num_pairs)
    b = rng.integers(0, n, num_pairs)

    return float(np.linalg.norm(motion_feats[a] - motion_feats[b], axis=1).mean())
