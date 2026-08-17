import numpy as np
import torch
from scipy import linalg


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fid(real_feats: np.ndarray, gen_feats: np.ndarray, warn_rank_deficient: bool = True) -> float:
    n_real, n_gen = len(real_feats), len(gen_feats)
    if n_real < 2 or n_gen < 2:
        raise ValueError(f"FID needs >= 2 samples per side, got {n_real} real / {n_gen} generated")

    dim = real_feats.shape[-1]
    if warn_rank_deficient and min(n_real, n_gen) <= dim:
        print(
            f"WARNING: FID over {min(n_real, n_gen)} samples of {dim}-dim features -- the "
            f"covariance is rank-deficient, so this value carries a large sample-size-dependent "
            f"bias. It is comparable ONLY to other FIDs at the same sample count."
        )

    mu_r, cov_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, cov_g = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)

    return frechet_distance(mu_r, cov_r, mu_g, cov_g)


def bootstrap_fid(
    real_feats: np.ndarray, gen_feats: np.ndarray, resamples: int = 200, seed: int = 0
) -> dict[str, float]:
    n = len(gen_feats)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        pick = rng.integers(0, n, n)
        values.append(fid(real_feats[pick], gen_feats[pick], warn_rank_deficient=False))

    arr = np.array(values)
    return {
        "fid_boot_mean": float(arr.mean()),
        "fid_std": float(arr.std(ddof=1)),
        "fid_ci_lo": float(np.percentile(arr, 2.5)),
        "fid_ci_hi": float(np.percentile(arr, 97.5)),
        "fid_resamples": int(resamples),
    }


def paired_fid_delta(
    real_feats: np.ndarray,
    gen_a: np.ndarray,
    gen_b: np.ndarray,
    resamples: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    n = len(gen_a)
    if len(gen_b) != n or len(real_feats) != n:
        raise ValueError("paired FID needs the same clips, in the same order, for both models")

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        pick = rng.integers(0, n, n)
        deltas.append(
            fid(real_feats[pick], gen_a[pick], warn_rank_deficient=False)
            - fid(real_feats[pick], gen_b[pick], warn_rank_deficient=False)
        )

    arr = np.array(deltas)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "delta_mean": float(arr.mean()),
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "separates": bool(lo > 0.0 or hi < 0.0),
    }


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
            pool = np.concatenate([[idx[i]], distractors])
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


@torch.no_grad()
def embed_motions(matcher, feats, eval_mean, eval_std, device, batch=32):
    out = []
    for start in range(0, len(feats), batch):
        group = feats[start : start + batch]
        max_t = max(f.shape[0] for f in group)
        padded = np.zeros((len(group), max_t, group[0].shape[-1]), np.float32)
        lengths = [f.shape[0] for f in group]
        for row, feat in enumerate(group):
            padded[row, : feat.shape[0]] = (feat - eval_mean) / eval_std
        emb = matcher(torch.from_numpy(padded).to(device), torch.tensor(lengths, device=device))
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(matcher, pairs, device, batch=32):
    out = []
    for start in range(0, len(pairs), batch):
        group = pairs[start : start + batch]
        max_l = max(we.shape[0] for we, _ in group)
        we_pad = np.zeros((len(group), max_l, group[0][0].shape[-1]), np.float32)
        pe_pad = np.zeros((len(group), max_l, group[0][1].shape[-1]), np.float32)
        lengths = [we.shape[0] for we, _ in group]
        for row, (we, pe) in enumerate(group):
            we_pad[row, : we.shape[0]] = we
            pe_pad[row, : pe.shape[0]] = pe
        emb = matcher(
            torch.from_numpy(we_pad).to(device),
            torch.from_numpy(pe_pad).to(device),
            lengths=torch.tensor(lengths, device=device),
        )
        out.append(emb.cpu().numpy())
    return np.concatenate(out)
