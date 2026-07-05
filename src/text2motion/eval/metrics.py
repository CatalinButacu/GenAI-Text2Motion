import numpy as np
from scipy import linalg


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fid(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
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
    return float(np.linalg.norm(text_feats - motion_feats, axis=1).mean())


def diversity(motion_feats: np.ndarray, num_pairs: int = 300, seed: int = 0) -> float:
    n = len(motion_feats)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, num_pairs)
    b = rng.integers(0, n, num_pairs)

    return float(np.linalg.norm(motion_feats[a] - motion_feats[b], axis=1).mean())
