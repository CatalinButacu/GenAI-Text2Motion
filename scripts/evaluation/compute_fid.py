#!/usr/bin/env python
"""Compute FID and R-Precision for a trained TextToMotionSSM checkpoint.

Evaluation protocol follows HumanML3D / T2M standard (Guo et al., CVPR 2022):
  - FID: Frechet Inception Distance in T2M motion-encoder feature space
  - R-Precision: fraction of prompts where the generated motion ranks within
    the Top-K (K=1,2,3) against a pool of 31 random text distractor candidates

Usage
-----
  # Minimal --generate from the test split and evaluate:
  python scripts/evaluation/compute_fid.py \\
      --checkpoint checkpoints/motion_ssm_hml3d/best_model.pt \\
      --data-dir data/humanml3d \\
      --split test

  # Also compare against a reference model:
  python scripts/evaluation/compute_fid.py \\
      --checkpoint checkpoints/motion_ssm_hml3d/best_model.pt \\
      --data-dir data/humanml3d \\
      --split test \\
      --ref-checkpoint checkpoints/motion_ssm_hml3d/baseline.pt \\
      --output results/fid_report.json

  # Use SBERT-trained checkpoint:
  python scripts/evaluation/compute_fid.py \\
      --checkpoint checkpoints/motion_ssm_hml3d/best_model.pt \\
      --use-sbert \\
      --data-dir data/humanml3d

Design decisions
----------------
- Pool size for R-Precision = 32 (31 distractors + 1 ground truth). This is
  the T2M standard --see Guo et al. 5 "Evaluation".
- Generated motions are produced at the ground-truth frame count (+/-0 frames).
- FID is computed with the full Frechet formula (not diagonal approximation).
  Requires N >= motion_dim for the covariance matrix to be non-singular.
  With <512 samples a diagonal approximation is used as fallback.

References
----------
  Guo et al. "Generating Diverse and Natural 3D Human Motions from Text."
  CVPR 2022. https://github.com/EricGuo5513/text-to-motion
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm

from scripts.evaluation.motion_encoder import (
    encode_texts_t2m,
    extract_features,
    load_encoder,
    load_text_encoder,
)
from src.architecture.nn_models import TextToMotionSSM
from src.architecture.rvq_tokenizer import MotionRVQTokenizer
from src.architecture.training.trainer_utils import load_strict
from src.data.humanml3d_loader import HumanML3DMotionDataset
from src.data.motion_normalize import MotionStats, normalize
from src.modules.motion.ssm_model import sample_indices
from src.shared.config import TrainingConfig
from src.shared.constants import CONSTS, SMPLX
from src.shared.tokenizer import tokenize as tok

try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SBERT_AVAILABLE = False

log = logging.getLogger(__name__)

#  Constants
RPREC_POOL_SIZE: int = 32  # 31 distractors + 1 GT (T2M standard)
FEAT_DIM: int = 512  # T2MMotionEncoder output dim
MIN_FULL_FID: int = 512  # minimum samples for full (non-diagonal) FID

#  Data structures


@dataclass
class EvalSample:
    text: str  # text prompt
    gt_motion: np.ndarray  # (T, 168) ground-truth motion
    gen_motion: np.ndarray  # (T, 168) generated motion
    clip_id: str


@dataclass
class EvalResults:
    n_samples: int
    fid: float  # Frechet Inception Distance (lower = better)
    r_prec_top1: float  # R-Precision Top-1 (higher = better)
    r_prec_top2: float
    r_prec_top3: float
    diversity: float  # average pairwise L2 distance in feature space
    multimodality: float  # average feature distance across 10 gens of same prompt
    using_pretrained_encoder: bool

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "fid": self.fid,
            "r_precision": {
                "top1": self.r_prec_top1,
                "top2": self.r_prec_top2,
                "top3": self.r_prec_top3,
            },
            "diversity": self.diversity,
            "multimodality": self.multimodality,
            "using_pretrained_encoder": self.using_pretrained_encoder,
        }

    def print_table(self) -> None:
        enc_note = (
            "(pretrained)"
            if self.using_pretrained_encoder
            else "(random --NOT comparable to literature)"
        )
        print()
        print(f"{'=' * 60}")
        print(f"  T2M EVALUATION RESULTS  {enc_note}")
        print(f"{'=' * 60}")
        print(f"  Samples evaluated : {self.n_samples}")
        print(f"  FID               : {self.fid:.4f}   [target: <=1.0]")
        print(f"  R-Precision Top-1 : {self.r_prec_top1:.4f}   [target: >=0.50]")
        print(f"  R-Precision Top-2 : {self.r_prec_top2:.4f}")
        print(f"  R-Precision Top-3 : {self.r_prec_top3:.4f}   [target: >=0.70]")
        print(f"  Diversity         : {self.diversity:.4f}   [higher = more varied]")
        print(f"  Multimodality     : {self.multimodality:.4f}   [higher = richer variety]")
        print(f"{'=' * 60}")
        print()


#  FID computation


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Full matrix Frechet distance between two Gaussians.

    FID = ||mu1-mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

    Uses scipy.linalg.sqrtm for the matrix square root.
    Falls back to the diagonal approximation if sqrtm fails (e.g. non-PSD).
    """

    diff = mu1 - mu2
    mean_term = float(diff @ diff)

    result = sqrtm(sigma1 @ sigma2)
    # scipy < 1.16 with disp=False returns (matrix, errest); newer just returns matrix
    covmean = result[0] if isinstance(result, tuple) else result
    if np.iscomplexobj(covmean):
        # Numerical error: imaginary parts should be negligible
        imag_norm = np.abs(covmean.imag).max()
        if imag_norm > 1e-3:
            log.warning(
                "[FID] sqrtm imaginary part large (%.2e) - using diagonal fallback", imag_norm
            )
            return diagonal_fid(mu1, np.diag(sigma1), mu2, np.diag(sigma2))
        covmean = covmean.real

    trace_term = float(np.trace(sigma1 + sigma2 - 2 * covmean))
    return mean_term + trace_term


def diagonal_fid(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> float:
    """Diagonal-covariance FID (used when N < _MIN_FULL_FID)."""
    diff = mu1 - mu2
    covmean = np.sqrt(np.maximum(var1 * var2, 0.0))
    return float(diff @ diff + (var1 + var2 - 2 * covmean).sum())


def compute_fid(gen_feats: np.ndarray, real_feats: np.ndarray) -> float:
    """Compute FID between generated and real motion feature distributions.

    Selects full vs diagonal covariance based on sample count.

    Args:
        gen_feats:  (N_gen, 512) generated motion embeddings
        real_feats: (N_real, 512) real motion embeddings

    Returns:
        FID scalar (lower is better).
    """
    mu_g = gen_feats.mean(axis=0)
    mu_r = real_feats.mean(axis=0)

    # Full covariance requires N > D (512 here)
    if min(len(gen_feats), len(real_feats)) >= MIN_FULL_FID:
        sigma_g = np.cov(gen_feats, rowvar=False)
        sigma_r = np.cov(real_feats, rowvar=False)
        try:
            return frechet_distance(mu_g, sigma_g, mu_r, sigma_r)
        except Exception as exc:
            log.warning("[FID] full covariance failed (%s), using diagonal", exc)

    log.info(
        "[FID] using diagonal approximation (N=%d/%d < %d)",
        len(gen_feats),
        len(real_feats),
        MIN_FULL_FID,
    )
    return diagonal_fid(mu_g, gen_feats.var(axis=0), mu_r, real_feats.var(axis=0))


#  R-Precision


def compute_text_features(
    texts: list[str],
    device: str = "cpu",
) -> np.ndarray:
    """Encode a list of texts using a sentence-level encoder.

    Uses SBERT (all-MiniLM-L6-v2) when available, falls back to a simple
    bag-of-characters hash when not installed.

    Returns: (N, D) float32 array.
    """
    if SBERT_AVAILABLE and SentenceTransformer is not None:
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    else:
        log.warning(
            "[R-Precision] sentence-transformers not installed. "
            "Using hash-based text features (results will be meaningless)."
        )
        # Deterministic per-text random vectors (very crude)
        feats = []
        for t in texts:
            h = abs(hash(t)) % (2**31)
            rng2 = np.random.default_rng(seed=h)
            feats.append(rng2.standard_normal(384).astype(np.float32))
        return np.stack(feats)


def compute_precision_r(
    samples: list[EvalSample],
    encoder: object,  # T2MMotionEncoder
    device: str = "cpu",
    pool_size: int = RPREC_POOL_SIZE,
    rng_seed: int = 42,
    text_encoder=None,  # T2MTextEncoder from motion_encoder --None falls back to SBERT
) -> tuple[float, float, float]:
    """Compute R-Precision@(1,2,3) following T2M evaluation protocol.

    For valid R-Precision, ``text_encoder`` must be the T2M text encoder from
    finest.tar --the same checkpoint that produced ``encoder`` (motion encoder).
    Both were trained jointly to produce features in a shared 512-dim space.
    Comparing features from the SAME joint space gives meaningful cosine
    similarities and therefore valid R-Precision values.

    When ``text_encoder`` is None, SBERT text features are used as a fallback.
    SBERT and T2M motion features occupy different, unaligned embedding spaces,
    so cosine similarities are meaningless and R-Precision values must NOT be
    reported as valid evaluation results.

    For each test sample:
      1. Take the generated motion and the correct text.
      2. Sample (pool_size - 1) distractor texts from the rest of the test set.
      3. Rank all pool_size texts by cosine similarity to the generated motion embedding.
      4. Score: 1 if correct text appears in Top-K.

    Returns:
        (top1, top2, top3) accuracy fractions.
    """

    n_unique = len({s.text for s in samples})

    if n_unique < pool_size:
        log.warning(
            "[R-Precision] eval set has only %d unique texts but pool_size=%d. "
            "Distractor sampling will draw duplicates of the GT text and "
            "the resulting R-Precision values are NOT comparable to T2M-protocol "
            "numbers. Use a larger eval set (>=%d unique prompts) or lower pool_size.",
            n_unique,
            pool_size,
            pool_size,
        )
    rng = random.Random(rng_seed)
    all_texts = [s.text for s in samples]

    #  Text features  must come from the T2M joint space for valid R-Precision
    if text_encoder is not None:
        text_feats = encode_texts_t2m(
            text_encoder,
            all_texts,
            device=device,  # type: ignore[arg-type]
        )
        # encode_texts_t2m already returns L2-normalised (512,) vectors --same dim as motion
    else:
        log.warning(
            "[R-Precision] T2M text encoder unavailable --falling back to SBERT.\n"
            "  SBERT and T2M motion features occupy DIFFERENT embedding spaces.\n"
            "  Cosine similarity across spaces is undefined; R-Precision values\n"
            "  are NOT valid and must NOT be reported as evaluation results.\n"
            "  Ensure finest.tar is at data/t2m/text_mot_match/model/finest.tar"
        )
        text_feats = compute_text_features(all_texts, device=device)
        text_feats = text_feats / (np.linalg.norm(text_feats, axis=1, keepdims=True) + 1e-8)

    #  Motion features
    gen_motions = [s.gen_motion for s in samples]
    motion_feats = extract_features(encoder, gen_motions, device=device)  # type: ignore[arg-type]
    # motion_feats already L2-normalised by the encoder

    #  Align feature dimensions (text 512 = motion 512 for T2M; SBERT=384->pad)
    d_txt = text_feats.shape[1]
    if d_txt >= FEAT_DIM:
        text_feats_aligned = text_feats[:, :FEAT_DIM]
    else:
        text_feats_aligned = np.pad(text_feats, ((0, 0), (0, FEAT_DIM - d_txt)))

    top1_hits = top2_hits = top3_hits = 0
    N = len(samples)

    for i in range(N):
        # Distractor indices (anything except ourselves)
        other_ids = list(range(N))
        other_ids.remove(i)
        distractor_ids = rng.sample(other_ids, min(pool_size - 1, len(other_ids)))
        pool_ids = [i] + distractor_ids  # GT text always at index 0 in pool

        pool_txt = text_feats_aligned[pool_ids]  # (pool_size, 512)
        mot_vec = motion_feats[i : i + 1]  # (1, 512)

        sims = (pool_txt @ mot_vec.T).squeeze(1).astype(np.float64)  # (pool_size,)
        ranked = np.argsort(-sims)  # descending similarity

        # GT index within pool is 0
        gt_rank = int(np.nonzero(ranked == 0)[0][0]) + 1  # 1-indexed rank

        if gt_rank <= 1:
            top1_hits += 1
        if gt_rank <= 2:
            top2_hits += 1
        if gt_rank <= 3:
            top3_hits += 1

    return top1_hits / N, top2_hits / N, top3_hits / N


#  Diversity and Multimodality


def compute_diversity(feats: np.ndarray, n_pairs: int = 300, seed: int = 42) -> float:
    """Average pairwise L2 distance between motion features (random sample of pairs).

    T2M paper 5: diversity = average over 300 random pairs.
    Higher = more diverse generated motions.
    """
    rng = np.random.default_rng(seed=seed)
    N = len(feats)
    if N < 2:
        return 0.0
    pairs = rng.choice(N, size=(min(n_pairs, N * (N - 1) // 2), 2), replace=False)
    dists = [float(np.linalg.norm(feats[a] - feats[b])) for a, b in pairs]
    return float(np.mean(dists))


def compute_multimodality(
    prompt_to_feats: dict[str, list[np.ndarray]],
    n_pairs: int = 10,
    seed: int = 42,
) -> float:
    """Average L2 distance between feature pairs generated from the same prompt.

    Requires multiple generations per prompt. If not available (only one
    generation per prompt), returns 0.0 with a warning.

    T2M paper 5: "sample 10 pairs of motions for 30 unique prompts".
    """
    rng = np.random.default_rng(seed=seed)
    per_prompt_dists = []
    for text, feat_list in prompt_to_feats.items():
        if len(feat_list) < 2:
            continue
        for _ in range(n_pairs):
            a, b = rng.choice(len(feat_list), size=2, replace=False)
            per_prompt_dists.append(float(np.linalg.norm(feat_list[a] - feat_list[b])))
    if not per_prompt_dists:
        log.warning(
            "[Multimodality] No prompt has multiple generations - returning 0.0. "
            "Use --multi-gen N>1 to generate N motions per prompt."
        )
        return 0.0
    return float(np.mean(per_prompt_dists))


#  Data loading


def load_test_samples(
    data_dir: str,
    split: str = "test",
    max_samples: int | None = None,
) -> list[dict]:
    """Load (text, motion) pairs from HumanML3D test split.

    Returns list of dicts with keys: 'text', 'motion' (T, 168), 'clip_id'.
    """

    log.info("[Eval] loading HumanML3D %s split from %s ...", split, data_dir)
    try:
        ds = HumanML3DMotionDataset(
            data_dir=data_dir,
            split=split,
            max_motion_length=200,
            preload=True,
            augment=False,
        )
    except FileNotFoundError as exc:
        log.exception("[Eval] %s", exc)
        sys.exit(1)

    samples = []
    indices = list(range(len(ds)))
    if max_samples:
        rng = random.Random(42)
        rng.shuffle(indices)
        indices = indices[:max_samples]

    for i in indices:
        item = ds.samples[i]
        motion_key = item["clip_id"]
        if ds.cache is not None:
            motion = ds.cache[motion_key].copy()
        else:
            ds_item = ds[i]
            motion = ds_item["motion"][: ds_item["length"]].numpy()
        samples.append(
            {
                "text": item["text"],
                "motion": motion,
                "clip_id": item["clip_id"],
            }
        )

    log.info("[Eval] %d samples loaded from %s", len(samples), split)
    return samples


#  Generator


def load_frozen_tokenizer(cfg: TrainingConfig, device: str) -> MotionRVQTokenizer:
    """Load the same RVQ tokenizer the SSM was trained against, in eval mode."""

    path = cfg.rvq_checkpoint_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[Eval] RVQ tokenizer not found at {path!r}. "
            "Train it first with scripts/training/train_rvq_tokenizer.py."
        )
    tokenizer = MotionRVQTokenizer(
        motion_dim=cfg.motion_dim,
        latent_dim=cfg.rvq_latent_dim,
        n_codebooks=cfg.rvq_n_codebooks,
        codebook_size=cfg.rvq_codebook_size,
        down_t=cfg.rvq_down_t,
    ).to(device)
    rvq_ck = torch.load(path, map_location=device, weights_only=False)  # NOSONAR
    tokenizer.load_state_dict(rvq_ck["model_state_dict"])
    tokenizer.eval()
    log.info("[Eval] frozen RVQ tokenizer loaded from %s", path)
    return tokenizer


def encodeTextInputs(text: str, use_sbert: bool, vocab: dict | None,
                     cfg: TrainingConfig, device: str):
    if use_sbert:
        return [text]

    v = vocab or {}
    ids = tok(text, v, max_len=cfg.max_text_length)
    return torch.tensor([ids], dtype=torch.long, device=device)


def generate_motions(
    checkpoint: str,
    samples: list[dict],
    device: str,
    use_sbert: bool = False,
    multi_gen: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[list[EvalSample], MotionStats | None]:
    """Generate motions for all test samples using a trained checkpoint.

    Args:
        checkpoint: Path to .pt checkpoint (TextToMotionSSM format).
        samples:    list of {'text', 'motion', 'clip_id'} dicts.
        device:     Torch device string.
        use_sbert:   Use SBERT encoder (must match training config).
        multi_gen:   Number of generations per prompt (for multimodality).

    Returns:
        (eval_samples, motion_stats) where eval_samples is one EvalSample per
        (input * multi_gen) and motion_stats is the train-time normalisation
        the model was trained against (None if missing). Both gt_motion and
        gen_motion are returned in NORMALISED space so FID is consistent.
    """

    log.info("[Eval] loading generator checkpoint: %s", checkpoint)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)  # NOSONAR
    cfg: TrainingConfig = ck.get("config", TrainingConfig())
    # Always respect the CLI override for encoder type
    cfg.use_sbert = use_sbert
    cfg.device = device

    # Restore vocab from checkpoint (if SBERT not used)
    if not use_sbert and "vocab" in ck:
        cfg.vocab_size = len(ck["vocab"])

    model = TextToMotionSSM(cfg).to(device)

    # Load full model weights. Default = lenient; SSM_STRICT_LOAD=1 enforces strict.
    state_key = "model_state_dict" if "model_state_dict" in ck else "motion_ssm_state_dict"
    strict = os.environ.get("SSM_STRICT_LOAD", "").lower() in ("1", "true", "yes")

    if strict:
        load_strict(model, ck[state_key], "motion_ssm")
    else:
        model_sd = model.state_dict()
        compat = {
            k: v for k, v in ck[state_key].items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        missed = len(ck[state_key]) - len(compat)

        if missed:
            log.error(
                "[Eval] LENIENT LOAD: %d/%d checkpoint keys did NOT load (set "
                "SSM_STRICT_LOAD=1 to fail loudly). FID numbers will reflect a "
                "partially-trained model.",
                missed, len(ck[state_key]),
            )
        model.load_state_dict(compat, strict=False)
        log.info("[Eval] loaded %d/%d model weights", len(compat), len(ck[state_key]))
    model.eval()

    tokenizer = load_frozen_tokenizer(cfg, device)
    motion_stats: MotionStats | None = ck.get("motion_stats")
    if motion_stats is None:
        log.warning(
            "[Eval] checkpoint has no 'motion_stats' - cannot align GT and generated "
            "scales. FID may be dominated by normalisation offset, not motion quality."
        )

    # Restore vocab for tokenization
    vocab = ck.get("vocab", None)

    eval_samples: list[EvalSample] = []
    t0 = time.time()

    with torch.no_grad():
        for idx, s in enumerate(samples):
            text = s["text"]
            gt_motion = s["motion"]  # un-normalised (T, 168)
            num_frames = min(gt_motion.shape[0], cfg.max_motion_length)
            gt_norm = (
                normalize(gt_motion[:num_frames], motion_stats)
                if motion_stats is not None
                else gt_motion[:num_frames]
            )

            for _ in range(multi_gen):
                inputs = encodeTextInputs(text, use_sbert, vocab, cfg, device)
                logits, _ = model(inputs, num_frames)  # (1, T', K, V)
                # multi_gen > 1 with greedy sampling (temp=1.0+top_p=1.0) collapses
                # to identical samples -> multimodality trivially 0. Auto-bump to
                # mild stochastic sampling unless the caller explicitly overrode.
                eff_temp, eff_top_p = temperature, top_p

                if multi_gen > 1 and abs(temperature - 1.0) < 1e-9 and top_p >= 1.0 - 1e-9:
                    eff_temp, eff_top_p = 1.1, 0.95
                indices = sample_indices(logits, temperature=eff_temp, top_p=eff_top_p)
                motion = tokenizer.decode(indices)  # (1, T, 168) in normalised space
                gen_motion = motion[0, :num_frames].cpu().numpy()

                eval_samples.append(
                    EvalSample(
                        text=text,
                        gt_motion=gt_norm,
                        gen_motion=gen_motion,
                        clip_id=s["clip_id"],
                    )
                )

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                log.info("[Eval] generated %d/%d (%.1fs)", idx + 1, len(samples), elapsed)

    log.info("[Eval] generation complete: %d samples in %.1fs", len(eval_samples), time.time() - t0)
    return eval_samples, motion_stats


#  Main computation


def run_evaluation(
    checkpoint: str,
    data_dir: str,
    split: str = "test",
    max_samples: int | None = None,
    device: str = "cpu",
    use_sbert: bool = False,
    multi_gen: int = 1,
    encoder_weights: str | None = None,
    output: str = "results/fid_report.json",
    ref_checkpoint: str | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> EvalResults:
    """Full evaluation pipeline: generate -> encode -> FID, R-Prec, Diversity."""

    # 1. Load test data
    samples = load_test_samples(data_dir, split=split, max_samples=max_samples)

    # 2. Generate motions
    eval_samples, _motionStats = generate_motions(
        checkpoint,
        samples,
        device,
        use_sbert,
        multi_gen,
        temperature=temperature,
        top_p=top_p,
    )

    # 3. Load motion encoder
    enc = load_encoder(input_dim=SMPLX.pose_dim, weights_path=encoder_weights, device=device)
    pretrained = getattr(enc, "_loaded_pretrained", False)

    # 3b. Load T2M text encoder from the same finest.tar for valid R-Precision

    txt_enc = load_text_encoder(weights_path=encoder_weights, device=device)

    # 4. Extract features for generated and real motions
    log.info("[Eval] extracting features for %d generated motions ...", len(eval_samples))
    gen_feats = extract_features(enc, [s.gen_motion for s in eval_samples], device=device)
    real_feats = extract_features(enc, [s.gt_motion for s in eval_samples], device=device)

    # 5. FID
    fid = compute_fid(gen_feats, real_feats)
    log.info("[Eval] FID = %.4f", fid)

    # 6. R-Precision (one per prompt, first gen only)
    # Use unique samples (first generation per clip)
    seen: set[str] = set()
    unique_samples: list[EvalSample] = []
    for s in eval_samples:
        if s.clip_id not in seen:
            seen.add(s.clip_id)
            unique_samples.append(s)

    top1, top2, top3 = compute_precision_r(
        unique_samples,
        enc,
        device=device,
        text_encoder=txt_enc,
    )
    log.info("[Eval] R-Precision Top-1=%.4f Top-2=%.4f Top-3=%.4f", top1, top2, top3)

    # 7. Diversity (all generated motions)
    diversity = compute_diversity(gen_feats)
    log.info("[Eval] Diversity = %.4f", diversity)

    # 8. Multimodality (multiple gens per prompt)
    prompt_to_feats: dict[str, list[np.ndarray]] = {}
    for s, f in zip(eval_samples, gen_feats):
        prompt_to_feats.setdefault(s.text, []).append(f)
    multimodality = compute_multimodality(prompt_to_feats)
    log.info("[Eval] Multimodality = %.4f", multimodality)

    results = EvalResults(
        n_samples=len(unique_samples),
        fid=fid,
        r_prec_top1=top1,
        r_prec_top2=top2,
        r_prec_top3=top3,
        diversity=diversity,
        multimodality=multimodality,
        using_pretrained_encoder=pretrained,
    )

    # 9. Optional reference comparison
    report = {"generated": results.to_dict()}
    if ref_checkpoint and Path(ref_checkpoint).exists():
        log.info("[Eval] evaluating reference checkpoint: %s", ref_checkpoint)
        ref_eval, _ = generate_motions(ref_checkpoint, samples, device, use_sbert, multi_gen=1)
        ref_feats = extract_features(enc, [s.gen_motion for s in ref_eval], device=device)
        ref_fid = compute_fid(ref_feats, real_feats)
        log.info("[Eval] Reference FID = %.4f", ref_fid)
        report["reference"] = {"fid": ref_fid, "checkpoint": ref_checkpoint}

    # 10. Save report
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    log.info("[Eval] report saved to %s", output)

    return results


#  CLI


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Compute FID + R-Precision for TextToMotionSSM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint", required=True, help="Path to trained TextToMotionSSM .pt checkpoint"
    )
    p.add_argument(
        "--data-dir", default="data/humanml3d", help="HumanML3D data directory", dest="data_dir"
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N random samples (default: all)",
        dest="max_samples",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--use-sbert",
        action="store_true",
        help="Use SBERT encoder (must match training config)",
        dest="use_sbert",
    )
    p.add_argument(
        "--multi-gen",
        type=int,
        default=1,
        help="Number of generations per prompt (used for multimodality)",
        dest="multi_gen",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for the RVQ head; >1.0 diversifies. With --multi-gen>1, "
             "defaults of 1.0/1.0 auto-bump to 1.1/0.95 so multimodality is not trivially 0.",
    )
    p.add_argument(
        "--top-p", type=float, default=1.0, dest="top_p",
        help="Nucleus filter for the RVQ head; <1.0 trims low-probability tails.",
    )
    p.add_argument(
        "--encoder-weights",
        default=None,
        help="Path to T2M motion encoder .pt weights (optional)",
        dest="encoder_weights",
    )
    p.add_argument(
        "--ref-checkpoint",
        default=None,
        help="Baseline checkpoint to compare against",
        dest="ref_checkpoint",
    )
    p.add_argument("--output", default="results/fid_report.json", help="Output JSON path")
    args = p.parse_args()

    results = run_evaluation(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
        device=args.device,
        use_sbert=args.use_sbert,
        multi_gen=args.multi_gen,
        encoder_weights=args.encoder_weights,
        output=args.output,
        ref_checkpoint=args.ref_checkpoint,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    results.print_table()


if __name__ == "__main__":
    main()
