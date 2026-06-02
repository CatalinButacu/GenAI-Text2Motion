"""Motion-X smplx-322 -> repo 168-dim SMPL-X converter.

Motion-X (NeurIPS 2023) distributes ~96 K text-paired motions in SMPL-X 322-dim
parameter form. The repo trains on 168-dim SMPL-X (root + transl + body + both
hands + jaw/eyes). This script remaps 322 -> 168 and dumps .npz files compatible
with ``data/AMASS/`` / ``src/data/motion_dataset.MotionDataset``.

Canonical Motion-X smplx-322 layout (per their official preprocessing):
    [0:3]     root_orient
    [3:66]    body_pose         (21 joints * 3)
    [66:111]  left_hand_pose    (15 joints * 3)
    [111:156] right_hand_pose   (15 joints * 3)
    [156:159] jaw_pose
    [159:162] leye_pose
    [162:165] reye_pose
    [165:215] expression        (50)
    [215:218] transl
    [218:228] betas             (10)
    [228:322] face_shape        (94)

Repo 168-dim layout (see src/shared/constants.build_smplx_spec):
    [0:3]     root_orient
    [3:6]     transl
    [6:69]    body_pose
    [69:114]  left_hand_pose
    [114:159] right_hand_pose
    [159:168] jaw + leye + reye  (9)

Face expression / shape / betas are discarded — the SMPL-X spec the renderer
uses ignores them.

NOTE: Motion-X is published with the body in **Z-up** (matches raw AMASS).
The repo expects **Y-up**. This script writes Z-up output and points to the
AMASS preprocessing pipeline (scripts/data/preprocess_amass.py-equivalent) for
the axis swap. If you need a one-shot pipeline, run the AMASS preprocessor on
the produced .npz files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MOTIONX_DIM = 322
REPO_DIM = 168


def map_motionx_322_to_168(arr: np.ndarray) -> np.ndarray:
    """Map a (T, 322) Motion-X smplx parameter array to the repo (T, 168) layout.

    Raises ``ValueError`` if the second dimension is not 322.
    """

    if arr.ndim != 2 or arr.shape[1] != MOTIONX_DIM:
        raise ValueError(
            f"expected (T, {MOTIONX_DIM}) Motion-X smplx array, got shape={arr.shape}"
        )
    out = np.zeros((arr.shape[0], REPO_DIM), dtype=arr.dtype)
    out[:, 0:3]     = arr[:, 0:3]        # root_orient
    out[:, 3:6]     = arr[:, 215:218]    # transl
    out[:, 6:69]    = arr[:, 3:66]       # body_pose
    out[:, 69:114]  = arr[:, 66:111]     # left_hand_pose
    out[:, 114:159] = arr[:, 111:156]    # right_hand_pose
    out[:, 159:168] = arr[:, 156:165]    # jaw + leye + reye

    return out


def convert_file(src: Path, dst: Path) -> dict:
    payload = np.load(src, allow_pickle=True)
    # Motion-X .npy / .npz commonly stores under key "smplx_322" or "motion" or as raw array.
    key = None

    if isinstance(payload, np.ndarray):
        arr = payload
    else:
        for k in ("smplx_322", "motion", "poses", "smplx"):
            if k in payload:
                arr = payload[k]
                key = k
                break
        else:
            raise KeyError(
                f"{src}: no recognised key (tried smplx_322, motion, poses, smplx); "
                f"available keys: {list(payload.keys())}"
            )

    if arr.ndim == 1:
        # Possibly stored flattened
        arr = arr.reshape(-1, MOTIONX_DIM)

    out = map_motionx_322_to_168(arr.astype(np.float32))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, poses=out)
    log.info("[motionx] %s -> %s  shape=%s  src_key=%s", src.name, dst.name, out.shape, key)

    return {"src": str(src), "dst": str(dst), "frames": int(out.shape[0])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, type=Path, dest="input_dir",
                   help="Root directory containing Motion-X smplx 322-dim .npy/.npz files")
    p.add_argument("--output-dir", required=True, type=Path, dest="output_dir",
                   help="Destination directory; mirrors input subtree, .npz with key 'poses'")
    p.add_argument("--pattern", default="**/*.np[yz]",
                   help="Glob pattern relative to --input-dir (default: **/*.np[yz])")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N files (smoke test)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    files = sorted(args.input_dir.glob(args.pattern))

    if not files:
        raise SystemExit(f"no files matched {args.pattern!r} under {args.input_dir}")

    if args.limit is not None:
        files = files[: args.limit]
    log.info("[motionx] converting %d files: %s -> %s",
             len(files), args.input_dir, args.output_dir)

    ok = 0
    failed: list[tuple[Path, str]] = []

    for src in files:
        rel = src.relative_to(args.input_dir).with_suffix(".npz")
        dst = args.output_dir / rel

        try:
            convert_file(src, dst)
            ok += 1
        except (ValueError, KeyError, OSError) as e:
            failed.append((src, str(e)))

    log.info("[motionx] done: %d ok, %d failed", ok, len(failed))

    for src, err in failed[:10]:
        log.error("  FAIL %s: %s", src, err)


if __name__ == "__main__":
    main()
