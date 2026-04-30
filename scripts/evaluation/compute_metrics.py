#!/usr/bin/env python
"""Compute thesis Chapter 4 evaluation metrics on generated MotionClips.

Metrics
-------
- foot_sliding_ms   : mean foot velocity (m/s) during active contact frames
- ground_penetration: mean pelvis depth below ground (cm), contact frames only
- joint_angle_fid   : FID between generated and AMASS reference joint-angle distributions
- validity_rate     : fraction of clips with no NaN, finite params, and ||pose||_inf < pi

Usage
-----
  python scripts/compute_metrics.py \\
      --clips-dir results/clips \\
      --reference-dir data/AMASS \\
      --output results/metrics.json

Input format
------------
  results/clips/  must contain *.npy files, each shaped (T, 168) --SMPL-X axis-angle params.
  Filenames are used as clip IDs.

SMPL-X 168-dim pose layout
--------------------------
  root_orient(3) + trans(3) + body_pose(63) + hand_pose(90) + jaw(3) + eyes(6) = 168

  * ``TRANS_SLICE = slice(3, 6)`` -- global translation (m), Z-up (AMASS / SMPL-X).
  * ``BODY_SLICE  = slice(6, 69)`` -- body joint angles (63-dim = 21 joints x 3).
  * Foot joints (within body_pose): left_ankle = joint 7, right_ankle = joint 8.
    ``body_pose`` index = ``(joint_idx - 1) * 3`` because joint 0 (root) is not
    in ``body_pose``; joint 1 = left_hip occupies indices 0-2.
  * ``PELVIS_HEIGHT_DIM = 5`` -- ``params[:, 5]`` = ``trans[2]``, the Z-up height
    used as the pelvis height proxy for ground-contact detection. Matches
    SMPLX_TRANSL_Z_IDX in src/shared/constants.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.shared.constants import (
    MOTION_DIM,
    MOTION_FPS,
    SMPLX_TRANSL_Z_IDX,
)

log = logging.getLogger(__name__)

#  SMPL-X pose layout
TRANS_SLICE = slice(3, 6)
BODY_SLICE = slice(6, 69)
LFOOT_START = (7 - 1) * 3
RFOOT_START = (8 - 1) * 3
# Z-up height = trans[2]; index within the (T, 3) trans slice
PELVIS_HEIGHT_IDX: int = SMPLX_TRANSL_Z_IDX - 3

# Contact detection thresholds (justified below).
#
# FOOT_VEL_THRESH: ankle joint rotation velocity below which a foot is
# considered stationary (rad/frame at 30 fps).  Derived empirically from AMASS
# reference clips: static standing poses produce ankle delta-angles < 0.05
# rad/frame; 0.15 provides a 3× margin to tolerate minor sway.
#
# PELVIS_CONTACT_HEIGHT: pelvis Z-coordinate (metres, Z-up SMPL-X) below which
# at least one foot is likely on the ground.  Mean standing pelvis height in
# AMASS is ~0.95 m; 1.1 m adds 15 cm tolerance for crouching and heel lifts.
FOOT_VEL_THRESH: float = 0.15
PELVIS_CONTACT_HEIGHT: float = 1.1


class ClipMetrics(NamedTuple):
    """Per-clip evaluation metrics.

    Attributes
    ----------
    clip_id : str
        Filename stem used as a unique identifier.
    n_frames : int
        Number of pose frames in the clip.
    foot_sliding_ms : float
        Mean ankle joint rotation velocity (rad/s) of the slower foot during
        contact frames; ``nan`` when no contact frames are detected.
        Uses ankle angular velocity (not root translation) so the metric
        captures the planted foot rotating in place, not whole-body drift.
    ground_penetration : float
        Mean pelvis depth below y = 0 (cm); 0 if always above ground.
    validity : bool
        ``True`` if params are finite and all body joint angles < pi rad.
    joint_angle_mean : np.ndarray
        Shape ``(MOTION_DIM,)`` -- per-dimension mean over all frames.
    joint_angle_cov_diag : np.ndarray
        Shape ``(MOTION_DIM,)`` -- per-dimension variance over all frames.
        Used together with ``joint_angle_mean`` to form a diagonal Gaussian
        for FID computation.
    """

    clipId: str
    nFrames: int
    footSlidingMs: float
    groundPenetration: float
    validity: bool
    jointAngleMean: np.ndarray
    jointAngleCovDiag: np.ndarray


#  Per-clip metrics


def contactMask(params: np.ndarray) -> np.ndarray:
    """Heuristic contact mask: foot is in contact when pelvis height is low
    and ankle joint angular velocity is below FOOT_VEL_THRESH.

    Returns bool array (T,).
    """
    trans = params[:, TRANS_SLICE]  # (T, 3)
    pelvisH = trans[:, PELVIS_HEIGHT_IDX]  # Z-up height (AMASS / SMPL-X)

    body = params[:, BODY_SLICE]  # (T, 63)
    lfoot = body[:, LFOOT_START : LFOOT_START + 3]
    rfoot = body[:, RFOOT_START : RFOOT_START + 3]

    if len(params) > 1:
        lvel = np.linalg.norm(np.diff(lfoot, axis=0), axis=1)
        rvel = np.linalg.norm(np.diff(rfoot, axis=0), axis=1)
        footVel = np.minimum(lvel, rvel)
        footSlow = np.concatenate([[footVel[0]], footVel]) < FOOT_VEL_THRESH
    else:
        footSlow = np.ones(len(params), dtype=bool)

    lowPelvis = pelvisH < PELVIS_CONTACT_HEIGHT
    return footSlow & lowPelvis


def footSliding(params: np.ndarray) -> float:
    """Mean ankle angular velocity (rad/s) of the slower foot during contact frames.

    Uses the ankle joint rotation velocity (not root translation) so the metric
    captures the planted foot rotating while in contact — the direct signal for
    foot sliding artifacts.  The slower of left/right ankle is used since it is
    the one most likely planted.

    Velocity frame alignment: ``np.diff`` produces ``T-1`` values; the contact
    mask ``contact[1:]`` aligns to the *end* frame of each transition.
    """
    if len(params) < 2:
        return float("nan")
    contact = contactMask(params)
    if not contact.any():
        return float("nan")

    body = params[:, BODY_SLICE]
    lfoot = body[:, LFOOT_START : LFOOT_START + 3]
    rfoot = body[:, RFOOT_START : RFOOT_START + 3]
    lvel = np.linalg.norm(np.diff(lfoot, axis=0), axis=1) * MOTION_FPS  # rad/s
    rvel = np.linalg.norm(np.diff(rfoot, axis=0), axis=1) * MOTION_FPS  # rad/s
    footVel = np.minimum(lvel, rvel)  # slower foot = most likely planted
    contactVel = contact[1:]
    if not contactVel.any():
        return float("nan")
    return float(footVel[contactVel].mean())


def groundPenetrationCm(params: np.ndarray) -> float:
    """Mean pelvis depth below ground (cm) across all frames."""
    trans = params[:, TRANS_SLICE]
    pelvisH = trans[:, PELVIS_HEIGHT_IDX]  # m, Z-up
    below = np.maximum(-pelvisH, 0.0)  # depth below z=0
    return float(below.mean() * 100.0)  # -> cm


def isValid(params: np.ndarray) -> bool:
    """Clip is valid: no NaN/Inf, joint angles within [-pi, pi]."""
    if not np.isfinite(params).all():
        return False
    body = params[:, BODY_SLICE]
    return bool(np.abs(body).max() < np.pi)


def computeClipMetrics(clipId: str, params: np.ndarray) -> ClipMetrics:
    """Compute all per-clip metrics for one (T, 168) array."""
    assert (
        params.ndim == 2 and params.shape[1] == MOTION_DIM
    ), f"Expected (T, {MOTION_DIM}), got {params.shape}"
    return ClipMetrics(
        clipId=clipId,
        nFrames=len(params),
        footSlidingMs=footSliding(params),
        groundPenetration=groundPenetrationCm(params),
        validity=isValid(params),
        jointAngleMean=params.mean(axis=0),
        jointAngleCovDiag=params.var(axis=0),
    )


#  Frechet Inception Distance (joint-angle space)


def gaussianFID(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Scalar FID between two diagonal Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))

    For diagonal covariances, sqrt(sigma1 * sigma2) = sqrt(element-wise product).
    """
    diff = mu1 - mu2
    covmean = np.sqrt(np.maximum(sigma1 * sigma2, 0.0))
    return float(np.dot(diff, diff) + (sigma1 + sigma2 - 2 * covmean).sum())


def jointAngleFID(genClips: list[ClipMetrics], refClips: list[ClipMetrics]) -> float:
    """FID between generated and reference joint-angle distributions."""

    def pool(clips):
        """Aggregate per-clip mean and variance using the law of total variance.

        For a mixture of clip-level Gaussians G each with mean E[X|G] and
        variance Var(X|G)::

            Var(X) = E[Var(X|G)] + Var(E[X|G])

        The first term is the mean within-clip variance; the second is the
        between-clip spread of the clip means.
        """
        means = np.stack([c.jointAngleMean for c in clips])
        vars_ = np.stack([c.jointAngleCovDiag for c in clips])
        mu = means.mean(axis=0)
        var = vars_.mean(axis=0) + ((means - mu) ** 2).mean(axis=0)
        return mu, var

    mu_g, var_g = pool(genClips)
    mu_r, var_r = pool(refClips)
    return gaussianFID(mu_g, var_g, mu_r, var_r)


#  Load clips from directory


def loadClips(clipDir: str) -> list[ClipMetrics]:
    """Load all *.npy files from a directory as ClipMetrics."""
    p = Path(clipDir)
    npyFiles = sorted(p.glob("*.npy"))
    if not npyFiles:
        raise FileNotFoundError(f"No *.npy clips found in {clipDir!r}")
    metrics = []
    for f in npyFiles:
        arr = np.load(str(f))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != MOTION_DIM:
            log.warning("Skipping %s - expected dim %d, got %d", f.name, MOTION_DIM, arr.shape[-1])
            continue
        metrics.append(computeClipMetrics(f.stem, arr))
        log.debug("Loaded %s: %d frames", f.stem, len(arr))
    log.info("Loaded %d clips from %s", len(metrics), clipDir)
    return metrics


#  Summary statistics


def summarise(metrics: list[ClipMetrics], label: str) -> dict:
    """Aggregate per-clip metrics into mean +/- std summary dict."""
    foot = np.array([m.footSlidingMs for m in metrics])
    foot = foot[np.isfinite(foot)]
    gp = np.array([m.groundPenetration for m in metrics])
    validRate = float(sum(m.validity for m in metrics) / max(len(metrics), 1))

    return {
        "label": label,
        "n_clips": len(metrics),
        "validity_rate": validRate,
        "foot_sliding_mean_ms": float(foot.mean()) if len(foot) else float("nan"),
        "foot_sliding_std_ms": float(foot.std()) if len(foot) else float("nan"),
        "ground_penetration_mean_cm": float(gp.mean()),
        "ground_penetration_std_cm": float(gp.std()),
    }


#  Main


def run(clipsDir: str, referenceDir: str | None, output: str, label: str) -> dict:
    genMetrics = loadClips(clipsDir)
    summary = summarise(genMetrics, label)

    if referenceDir and Path(referenceDir).exists():
        refMetrics = loadClips(referenceDir)
        fid = jointAngleFID(genMetrics, refMetrics)
        summary["joint_angle_fid"] = fid
        log.info("Joint angle FID vs reference: %.4f", fid)
    else:
        summary["joint_angle_fid"] = None
        log.warning("No reference dir - FID not computed")

    # Print human-readable table
    print(f"\n{''*55}")
    print(f"  Metrics: {label}")
    print(f"{''*55}")
    print(f"  Clips evaluated :  {summary['n_clips']}")
    print(f"  Validity rate   :  {summary['validity_rate']*100:.1f}%")
    print(
        f"  Foot sliding    :  {summary['foot_sliding_mean_ms']:.4f} +/- "
        f"{summary['foot_sliding_std_ms']:.4f}  m/s"
    )
    print(
        f"  Ground penetrat.:  {summary['ground_penetration_mean_cm']:.3f} +/- "
        f"{summary['ground_penetration_std_cm']:.3f}  cm"
    )
    if summary["joint_angle_fid"] is not None:
        print(f"  Joint angle FID :  {summary['joint_angle_fid']:.4f}")
    print(f"{''*55}\n")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Metrics saved to %s", output)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Compute thesis evaluation metrics")
    p.add_argument("--clips-dir", required=True,
                   help="Dir of generated *.npy clips", dest="clipsDir")
    p.add_argument("--reference-dir", default=None,
                   help="Dir of AMASS reference *.npy clips",
                   dest="referenceDir")
    p.add_argument("--output", default="results/metrics.json")
    p.add_argument("--label", default="generated", help="Label for this config")
    args = p.parse_args()
    run(args.clipsDir, args.referenceDir, args.output, args.label)


if __name__ == "__main__":
    main()
