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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.motion_encoder import (
    encodeTextsT2M,
    extractFeatures,
    loadEncoder,
    loadTextEncoder,
)
from src.data.humanml3d_loader import HumanML3DMotionDataset
from src.data.motion_normalize import MotionStats, normalize
from src.modules.motion.config import TrainingConfig
from src.modules.motion.nn_models import TextToMotionSSM
from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer
from src.modules.motion.ssm_model import sampleIndices
from src.shared.constants import MOTION_DIM
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
    gtMotion: np.ndarray  # (T, 168) ground-truth motion
    genMotion: np.ndarray  # (T, 168) generated motion
    clipId: str


@dataclass
class EvalResults:
    nSamples: int
    fid: float  # Frechet Inception Distance (lower = better)
    rPrecTop1: float  # R-Precision Top-1 (higher = better)
    rPrecTop2: float
    rPrecTop3: float
    diversity: float  # average pairwise L2 distance in feature space
    multimodality: float  # average feature distance across 10 gens of same prompt
    usingPretrainedEncoder: bool

    def toDict(self) -> dict:
        return {
            "n_samples": self.nSamples,
            "fid": self.fid,
            "r_precision": {
                "top1": self.rPrecTop1,
                "top2": self.rPrecTop2,
                "top3": self.rPrecTop3,
            },
            "diversity": self.diversity,
            "multimodality": self.multimodality,
            "using_pretrained_encoder": self.usingPretrainedEncoder,
        }

    def printTable(self) -> None:
        encNote = (
            "(pretrained)"
            if self.usingPretrainedEncoder
            else "(random --NOT comparable to literature)"
        )
        print()
        print(f"{'=' * 60}")
        print(f"  T2M EVALUATION RESULTS  {encNote}")
        print(f"{'=' * 60}")
        print(f"  Samples evaluated : {self.nSamples}")
        print(f"  FID               : {self.fid:.4f}   [target: <=1.0]")
        print(f"  R-Precision Top-1 : {self.rPrecTop1:.4f}   [target: >=0.50]")
        print(f"  R-Precision Top-2 : {self.rPrecTop2:.4f}")
        print(f"  R-Precision Top-3 : {self.rPrecTop3:.4f}   [target: >=0.70]")
        print(f"  Diversity         : {self.diversity:.4f}   [higher = more varied]")
        print(f"  Multimodality     : {self.multimodality:.4f}   [higher = richer variety]")
        print(f"{'=' * 60}")
        print()


#  FID computation


def frechetDistance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Full matrix Frechet distance between two Gaussians.

    FID = ||mu1-mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

    Uses scipy.linalg.sqrtm for the matrix square root.
    Falls back to the diagonal approximation if sqrtm fails (e.g. non-PSD).
    """

    diff = mu1 - mu2
    meanTerm = float(diff @ diff)

    result = sqrtm(sigma1 @ sigma2)
    # scipy < 1.16 with disp=False returns (matrix, errest); newer just returns matrix
    covmean = result[0] if isinstance(result, tuple) else result
    if np.iscomplexobj(covmean):
        # Numerical error: imaginary parts should be negligible
        imagNorm = np.abs(covmean.imag).max()
        if imagNorm > 1e-3:
            log.warning(
                "[FID] sqrtm imaginary part large (%.2e) - using diagonal fallback", imagNorm
            )
            return diagonalFID(mu1, np.diag(sigma1), mu2, np.diag(sigma2))
        covmean = covmean.real

    traceTerm = float(np.trace(sigma1 + sigma2 - 2 * covmean))
    return meanTerm + traceTerm


def diagonalFID(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> float:
    """Diagonal-covariance FID (used when N < _MIN_FULL_FID)."""
    diff = mu1 - mu2
    covmean = np.sqrt(np.maximum(var1 * var2, 0.0))
    return float(diff @ diff + (var1 + var2 - 2 * covmean).sum())


def computeFID(genFeats: np.ndarray, realFeats: np.ndarray) -> float:
    """Compute FID between generated and real motion feature distributions.

    Selects full vs diagonal covariance based on sample count.

    Args:
        genFeats:  (N_gen, 512) generated motion embeddings
        realFeats: (N_real, 512) real motion embeddings

    Returns:
        FID scalar (lower is better).
    """
    muG = genFeats.mean(axis=0)
    muR = realFeats.mean(axis=0)

    # Full covariance requires N > D (512 here)
    if min(len(genFeats), len(realFeats)) >= MIN_FULL_FID:
        sigmaG = np.cov(genFeats, rowvar=False)
        sigmaR = np.cov(realFeats, rowvar=False)
        try:
            return frechetDistance(muG, sigmaG, muR, sigmaR)
        except Exception as exc:
            log.warning("[FID] full covariance failed (%s), using diagonal", exc)

    log.info(
        "[FID] using diagonal approximation (N=%d/%d < %d)",
        len(genFeats),
        len(realFeats),
        MIN_FULL_FID,
    )
    return diagonalFID(muG, genFeats.var(axis=0), muR, realFeats.var(axis=0))


#  R-Precision


def computeTextFeatures(
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


def computePrecisionR(
    samples: list[EvalSample],
    encoder: object,  # T2MMotionEncoder
    device: str = "cpu",
    poolSize: int = RPREC_POOL_SIZE,
    rngSeed: int = 42,
    textEncoder=None,  # T2MTextEncoder from motion_encoder --None falls back to SBERT
) -> tuple[float, float, float]:
    """Compute R-Precision@(1,2,3) following T2M evaluation protocol.

    For valid R-Precision, ``textEncoder`` must be the T2M text encoder from
    finest.tar --the same checkpoint that produced ``encoder`` (motion encoder).
    Both were trained jointly to produce features in a shared 512-dim space.
    Comparing features from the SAME joint space gives meaningful cosine
    similarities and therefore valid R-Precision values.

    When ``textEncoder`` is None, SBERT text features are used as a fallback.
    SBERT and T2M motion features occupy different, unaligned embedding spaces,
    so cosine similarities are meaningless and R-Precision values must NOT be
    reported as valid evaluation results.

    For each test sample:
      1. Take the generated motion and the correct text.
      2. Sample (poolSize - 1) distractor texts from the rest of the test set.
      3. Rank all poolSize texts by cosine similarity to the generated motion embedding.
      4. Score: 1 if correct text appears in Top-K.

    Returns:
        (top1, top2, top3) accuracy fractions.
    """

    rng = random.Random(rngSeed)
    allTexts = [s.text for s in samples]

    #  Text features  must come from the T2M joint space for valid R-Precision
    if textEncoder is not None:
        textFeats = encodeTextsT2M(
            textEncoder,
            allTexts,
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
        textFeats = computeTextFeatures(allTexts, device=device)
        textFeats = textFeats / (np.linalg.norm(textFeats, axis=1, keepdims=True) + 1e-8)

    #  Motion features
    genMotions = [s.genMotion for s in samples]
    motionFeats = extractFeatures(encoder, genMotions, device=device)  # type: ignore[arg-type]
    # motion_feats already L2-normalised by the encoder

    #  Align feature dimensions (text 512 = motion 512 for T2M; SBERT=384->pad)
    dTxt = textFeats.shape[1]
    if dTxt >= FEAT_DIM:
        textFeatsAligned = textFeats[:, :FEAT_DIM]
    else:
        textFeatsAligned = np.pad(textFeats, ((0, 0), (0, FEAT_DIM - dTxt)))

    top1Hits = top2Hits = top3Hits = 0
    N = len(samples)

    for i in range(N):
        # Distractor indices (anything except ourselves)
        otherIds = list(range(N))
        otherIds.remove(i)
        distractorIds = rng.sample(otherIds, min(poolSize - 1, len(otherIds)))
        poolIds = [i] + distractorIds  # GT text always at index 0 in pool

        poolTxt = textFeatsAligned[poolIds]  # (poolSize, 512)
        motVec = motionFeats[i : i + 1]  # (1, 512)

        sims = (poolTxt @ motVec.T).squeeze(1).astype(np.float64)  # (poolSize,)
        ranked = np.argsort(-sims)  # descending similarity

        # GT index within pool is 0
        gtRank = int(np.nonzero(ranked == 0)[0][0]) + 1  # 1-indexed rank

        if gtRank <= 1:
            top1Hits += 1
        if gtRank <= 2:
            top2Hits += 1
        if gtRank <= 3:
            top3Hits += 1

    return top1Hits / N, top2Hits / N, top3Hits / N


#  Diversity and Multimodality


def computeDiversity(feats: np.ndarray, nPairs: int = 300, seed: int = 42) -> float:
    """Average pairwise L2 distance between motion features (random sample of pairs).

    T2M paper 5: diversity = average over 300 random pairs.
    Higher = more diverse generated motions.
    """
    rng = np.random.default_rng(seed=seed)
    N = len(feats)
    if N < 2:
        return 0.0
    pairs = rng.choice(N, size=(min(nPairs, N * (N - 1) // 2), 2), replace=False)
    dists = [float(np.linalg.norm(feats[a] - feats[b])) for a, b in pairs]
    return float(np.mean(dists))


def computeMultimodality(
    promptToFeats: dict[str, list[np.ndarray]],
    nPairs: int = 10,
    seed: int = 42,
) -> float:
    """Average L2 distance between feature pairs generated from the same prompt.

    Requires multiple generations per prompt. If not available (only one
    generation per prompt), returns 0.0 with a warning.

    T2M paper 5: "sample 10 pairs of motions for 30 unique prompts".
    """
    rng = np.random.default_rng(seed=seed)
    perPromptDists = []
    for text, feat_list in promptToFeats.items():
        if len(feat_list) < 2:
            continue
        for _ in range(nPairs):
            a, b = rng.choice(len(feat_list), size=2, replace=False)
            perPromptDists.append(float(np.linalg.norm(feat_list[a] - feat_list[b])))
    if not perPromptDists:
        log.warning(
            "[Multimodality] No prompt has multiple generations - returning 0.0. "
            "Use --multi-gen N>1 to generate N motions per prompt."
        )
        return 0.0
    return float(np.mean(perPromptDists))


#  Data loading


def loadTestSamples(
    dataDir: str,
    split: str = "test",
    maxSamples: int | None = None,
) -> list[dict]:
    """Load (text, motion) pairs from HumanML3D test split.

    Returns list of dicts with keys: 'text', 'motion' (T, 168), 'clip_id'.
    """

    log.info("[Eval] loading HumanML3D %s split from %s ...", split, dataDir)
    try:
        ds = HumanML3DMotionDataset(
            dataDir=dataDir,
            split=split,
            maxMotionLength=200,
            preload=True,
            augment=False,
        )
    except FileNotFoundError as exc:
        log.error("[Eval] %s", exc)
        sys.exit(1)

    samples = []
    indices = list(range(len(ds)))
    if maxSamples:
        rng = random.Random(42)
        rng.shuffle(indices)
        indices = indices[:maxSamples]

    for i in indices:
        item = ds.samples[i]
        motionKey = item["clip_id"]
        if ds.cache is not None:
            motion = ds.cache[motionKey].copy()
        else:
            dsItem = ds[i]
            motion = dsItem["motion"][: dsItem["length"]].numpy()
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


def loadFrozenTokenizer(
    cfg: TrainingConfig, device: str
) -> MotionRVQTokenizer:
    """Load the same RVQ tokenizer the SSM was trained against, in eval mode."""

    path = cfg.rvqCheckpointPath
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[Eval] RVQ tokenizer not found at {path!r}. "
            "Train it first with scripts/training/train_rvq_tokenizer.py."
        )
    tokenizer = MotionRVQTokenizer(
        motionDim=cfg.motionDim,
        latentDim=cfg.rvqLatentDim,
        nCodebooks=cfg.rvqNCodebooks,
        codebookSize=cfg.rvqCodebookSize,
        downT=cfg.rvqDownT,
    ).to(device)
    rvqCk = torch.load(path, map_location=device, weights_only=False)
    tokenizer.load_state_dict(rvqCk["model_state_dict"])
    tokenizer.eval()
    log.info("[Eval] frozen RVQ tokenizer loaded from %s", path)
    return tokenizer


def generateMotions(
    checkpoint: str,
    samples: list[dict],
    device: str,
    useSBERT: bool = False,
    multiGen: int = 1,
) -> tuple[list[EvalSample], MotionStats | None]:
    """Generate motions for all test samples using a trained checkpoint.

    Args:
        checkpoint: Path to .pt checkpoint (TextToMotionSSM format).
        samples:    list of {'text', 'motion', 'clip_id'} dicts.
        device:     Torch device string.
        useSBERT:   Use SBERT encoder (must match training config).
        multiGen:   Number of generations per prompt (for multimodality).

    Returns:
        (evalSamples, motionStats) where evalSamples is one EvalSample per
        (input * multiGen) and motionStats is the train-time normalisation
        the model was trained against (None if missing). Both gtMotion and
        genMotion are returned in NORMALISED space so FID is consistent.
    """

    log.info("[Eval] loading generator checkpoint: %s", checkpoint)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg: TrainingConfig = ck.get("config", TrainingConfig())
    # Always respect the CLI override for encoder type
    cfg.useSBERT = useSBERT
    cfg.device = device

    # Restore vocab from checkpoint (if SBERT not used)
    if not useSBERT and "vocab" in ck:
        cfg.vocabSize = len(ck["vocab"])

    model = TextToMotionSSM(cfg).to(device)

    # Load full model weights
    stateKey = "model_state_dict" if "model_state_dict" in ck else "motion_ssm_state_dict"
    modelSd = model.state_dict()
    compat = {
        k: v for k, v in ck[stateKey].items() if k in modelSd and v.shape == modelSd[k].shape
    }
    model.load_state_dict(compat, strict=False)
    log.info("[Eval] loaded %d/%d model weights", len(compat), len(ck[stateKey]))
    model.eval()

    tokenizer = loadFrozenTokenizer(cfg, device)
    motionStats: MotionStats | None = ck.get("motion_stats")
    if motionStats is None:
        log.warning(
            "[Eval] checkpoint has no 'motion_stats' - cannot align GT and generated "
            "scales. FID may be dominated by normalisation offset, not motion quality."
        )

    # Restore vocab for tokenization
    vocab = ck.get("vocab", None)

    evalSamples: list[EvalSample] = []
    t0 = time.time()

    with torch.no_grad():
        for idx, s in enumerate(samples):
            text = s["text"]
            gtMotion = s["motion"]  # un-normalised (T, 168)
            numFrames = min(gtMotion.shape[0], cfg.maxMotionLength)
            gtNorm = (
                normalize(gtMotion[:numFrames], motionStats)
                if motionStats is not None
                else gtMotion[:numFrames]
            )

            for gen in range(multiGen):
                # Encode text
                if useSBERT:
                    inputs = [text]
                else:

                    vocab = vocab or {}
                    ids = tok(text, vocab, maxLen=cfg.maxTextLength)
                    inputs = torch.tensor([ids], dtype=torch.long, device=device)

                logits, _ = model(inputs, numFrames)  # (1, T', K, V)
                indices = sampleIndices(logits)  # (1, T', K) — argmax for deterministic eval
                motion = tokenizer.decode(indices)  # (1, T, 168) in normalised space
                genMotion = motion[0, :numFrames].cpu().numpy()

                evalSamples.append(
                    EvalSample(
                        text=text,
                        gtMotion=gtNorm,
                        genMotion=genMotion,
                        clipId=s["clip_id"],
                    )
                )

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                log.info("[Eval] generated %d/%d (%.1fs)", idx + 1, len(samples), elapsed)

    log.info("[Eval] generation complete: %d samples in %.1fs", len(evalSamples), time.time() - t0)
    return evalSamples, motionStats


#  Main computation


def runEvaluation(
    checkpoint: str,
    dataDir: str,
    split: str = "test",
    maxSamples: int | None = None,
    device: str = "cpu",
    useSBERT: bool = False,
    multiGen: int = 1,
    encoderWeights: str | None = None,
    output: str = "results/fid_report.json",
    refCheckpoint: str | None = None,
) -> EvalResults:
    """Full evaluation pipeline: generate -> encode -> FID, R-Prec, Diversity."""

    # 1. Load test data
    samples = loadTestSamples(dataDir, split=split, maxSamples=maxSamples)

    # 2. Generate motions
    evalSamples, _motionStats = generateMotions(
        checkpoint, samples, device, useSBERT, multiGen,
    )

    # 3. Load motion encoder
    enc = loadEncoder(inputDim=MOTION_DIM, weightsPath=encoderWeights, device=device)
    pretrained = getattr(enc, "_loaded_pretrained", False)

    # 3b. Load T2M text encoder from the same finest.tar for valid R-Precision

    txtEnc = loadTextEncoder(weightsPath=encoderWeights, device=device)

    # 4. Extract features for generated and real motions
    log.info("[Eval] extracting features for %d generated motions ...", len(evalSamples))
    genFeats = extractFeatures(enc, [s.genMotion for s in evalSamples], device=device)
    realFeats = extractFeatures(enc, [s.gtMotion for s in evalSamples], device=device)

    # 5. FID
    fid = computeFID(genFeats, realFeats)
    log.info("[Eval] FID = %.4f", fid)

    # 6. R-Precision (one per prompt, first gen only)
    # Use unique samples (first generation per clip)
    seen: set[str] = set()
    uniqueSamples: list[EvalSample] = []
    for s in evalSamples:
        if s.clipId not in seen:
            seen.add(s.clipId)
            uniqueSamples.append(s)

    top1, top2, top3 = computePrecisionR(
        uniqueSamples,
        enc,
        device=device,
        textEncoder=txtEnc,
    )
    log.info("[Eval] R-Precision Top-1=%.4f Top-2=%.4f Top-3=%.4f", top1, top2, top3)

    # 7. Diversity (all generated motions)
    diversity = computeDiversity(genFeats)
    log.info("[Eval] Diversity = %.4f", diversity)

    # 8. Multimodality (multiple gens per prompt)
    promptToFeats: dict[str, list[np.ndarray]] = {}
    for s, f in zip(evalSamples, genFeats):
        promptToFeats.setdefault(s.text, []).append(f)
    multimodality = computeMultimodality(promptToFeats)
    log.info("[Eval] Multimodality = %.4f", multimodality)

    results = EvalResults(
        nSamples=len(uniqueSamples),
        fid=fid,
        rPrecTop1=top1,
        rPrecTop2=top2,
        rPrecTop3=top3,
        diversity=diversity,
        multimodality=multimodality,
        usingPretrainedEncoder=pretrained,
    )

    # 9. Optional reference comparison
    report = {"generated": results.toDict()}
    if refCheckpoint and Path(refCheckpoint).exists():
        log.info("[Eval] evaluating reference checkpoint: %s", refCheckpoint)
        refEval, _ = generateMotions(refCheckpoint, samples, device, useSBERT, multiGen=1)
        refFeats = extractFeatures(enc, [s.genMotion for s in refEval], device=device)
        refFid = computeFID(refFeats, realFeats)
        log.info("[Eval] Reference FID = %.4f", refFid)
        report["reference"] = {"fid": refFid, "checkpoint": refCheckpoint}

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
    p.add_argument("--data-dir", default="data/humanml3d",
                   help="HumanML3D data directory", dest="dataDir")
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
    dest="maxSamples")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--use-sbert", action="store_true", help="Use SBERT encoder (must match training config)"
    , dest="useSBERT")
    p.add_argument(
        "--multi-gen",
        type=int,
        default=1,
        help="Number of generations per prompt (used for multimodality)",
    dest="multiGen")
    p.add_argument(
        "--encoder-weights", default=None, help="Path to T2M motion encoder .pt weights (optional)"
    , dest="encoderWeights")
    p.add_argument("--ref-checkpoint", default=None,
                   help="Baseline checkpoint to compare against",
                   dest="refCheckpoint")
    p.add_argument("--output", default="results/fid_report.json", help="Output JSON path")
    args = p.parse_args()

    results = runEvaluation(
        checkpoint=args.checkpoint,
        dataDir=args.dataDir,
        split=args.split,
        maxSamples=args.maxSamples,
        device=args.device,
        useSBERT=args.useSBERT,
        multiGen=args.multiGen,
        encoderWeights=args.encoderWeights,
        output=args.output,
        refCheckpoint=args.refCheckpoint,
    )
    results.printTable()


if __name__ == "__main__":
    main()
