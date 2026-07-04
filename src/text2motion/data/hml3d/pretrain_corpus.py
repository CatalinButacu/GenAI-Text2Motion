"""Build the ALL-AMASS pretraining corpus (E7/E7b) -- 263 features for every AMASS sequence.

`regenerate.py` stage 1 already forwarded the FULL donor AMASS through SMPL-X (pose_data covers
~16.4k sequences); only its index stage filters down to HumanML3D's captioned clips. This module
walks pose_data wholesale instead: per sequence it applies the SAME canonicalization the official
index loop applies to AMASS sources (dataset head-trim, x-flip), runs `process_file`, and writes
full-length 263 features to ``<out_dir>/new_joint_vecs``.

**Eval-leakage guard:** sequences that underlie HumanML3D *val/test* clips (resolved through
index.csv) are EXCLUDED, so pretraining never sees held-out motion. Train-clip sources stay in.

**No new Mean/Std:** the corpus is consumed in the OFFICIAL HumanML3D normalization (the frozen
tokenizer's space) -- see ``tokenize_corpus.py``. Computing corpus-specific stats would silently
shift the token lattice. Failures are loud, per file.

    python -m text2motion.data.hml3d.pretrain_corpus --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from text2motion.shared.config import Config, load_config
from text2motion.shared.run_log import log_metrics, start_run

from . import param_util
from .feature import build_tgt_offsets, default_params, process_file
from .regenerate import _DATASET_HEAD_TRIM_S, _resolve_pose_path


def heldout_sources(cfg: Config, pose_root: Path) -> set[Path]:
    """Resolve every val/test HumanML3D clip back to its AMASS pose file (the exclusion set)."""
    if cfg.paths.hml3d_index_csv is None:
        raise ValueError("paths.hml3d_index_csv must be set (leakage guard needs the index)")
    out_dir = Path(cfg.paths.hml3d_out_dir)
    held_ids: set[str] = set()
    for split in ("val", "test"):
        names = (out_dir / f"{split}.txt").read_text().splitlines()
        held_ids |= {
            n.strip()[1:] if n.strip().startswith("M") else n.strip() for n in names if n.strip()
        }

    index = pd.read_csv(cfg.paths.hml3d_index_csv)
    excluded: set[Path] = set()
    for i in range(index.shape[0]):
        if Path(str(index.loc[i]["new_name"])).stem in held_ids:
            resolved = _resolve_pose_path(index.loc[i]["source_path"], pose_root)
            if resolved is not None:
                excluded.add(resolved.resolve())
    return excluded


def head_trim_frames(rel_path: Path, fps: int) -> int:
    """Per-dataset leading trim, matching the official index loop (donor dataset names included)."""
    renames = {"HDM05": "MPI_HDM05", "PosePrior": "MPI_Limits", "Transitions": "Transitions_mocap"}
    dataset = rel_path.parts[0]
    official = renames.get(dataset, dataset)
    return int(round(_DATASET_HEAD_TRIM_S.get(official, 0.0) * fps))


def build_corpus(
    cfg: Config, regen_dir: Path, out_dir: Path, min_frames: int, limit: int = 0
) -> None:
    pose_root = regen_dir / "pose_data"
    joints_dir = regen_dir / "joints"
    if not pose_root.is_dir():
        raise FileNotFoundError(f"{pose_root} missing -- run regenerate.py --stage amass first")

    # Same reference skeleton as the HumanML3D regen (all clips share SMPL-X bone lengths).
    ref_path = joints_dir / f"{param_util.t2m_tgt_skel_id}.npy"
    if not ref_path.is_file():
        candidates = sorted(joints_dir.glob("[0-9]*.npy"))
        if not candidates:
            raise FileNotFoundError(f"no reference joints in {joints_dir}")
        ref_path = candidates[0]
    reference = np.load(ref_path)[:, : cfg.hml3d.num_joints].reshape(-1, cfg.hml3d.num_joints, 3)
    params = default_params(build_tgt_offsets(reference))

    excluded = heldout_sources(cfg, pose_root)
    vec_dir = out_dir / "new_joint_vecs"
    vec_dir.mkdir(parents=True, exist_ok=True)
    run_dir = start_run(
        "pretrain_corpus",
        cfg,
        cfg.paths.outputs_dir,
        {"regen_dir": str(regen_dir), "out_dir": str(out_dir), "excluded": len(excluded)},
    )

    sources = sorted(pose_root.rglob("*.npy"))
    if limit:
        sources = sources[:limit]
    written = skipped_heldout = skipped_short = failed = total_frames = 0
    for src in tqdm(sources, desc="pose->263 (all AMASS)"):
        if src.resolve() in excluded:
            skipped_heldout += 1
            continue
        rel = src.relative_to(pose_root)
        flat_name = "__".join(rel.with_suffix("").parts) + ".npy"
        save_path = vec_dir / flat_name
        if save_path.is_file():
            written += 1
            continue
        joints = np.load(src)[:, : cfg.hml3d.num_joints]
        trim = head_trim_frames(rel, cfg.hml3d.fps)
        if trim:
            joints = joints[trim:]
        if joints.shape[0] < min_frames:
            skipped_short += 1
            continue
        joints = joints.copy()
        joints[..., 0] *= -1  # the official index loop's x-flip for AMASS sources
        try:
            features, _, _, _ = process_file(joints, params)
            if not np.isfinite(features).all():
                raise ValueError("non-finite features")
            np.save(save_path, features.astype(np.float32))
            written += 1
            total_frames += features.shape[0]
        except Exception as exc:  # loud per-file, never a silent fallback
            failed += 1
            print(f"FAILED {rel}: {exc}")

    summary = {
        "written": written,
        "skipped_heldout": skipped_heldout,
        "skipped_short": skipped_short,
        "failed": failed,
        "frames": total_frames,
        "hours": round(total_frames / cfg.hml3d.fps / 3600, 2),
    }
    log_metrics(run_dir, summary)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the all-AMASS 263 pretraining corpus.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--regen_dir", default="data/HumanML3D_263", help="regen root with pose_data/"
    )
    parser.add_argument("--out_dir", default="data/AMASS_263")
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="process only the first N (smoke)")
    args = parser.parse_args()
    build_corpus(
        load_config(args.config),
        Path(args.regen_dir),
        Path(args.out_dir),
        args.min_frames,
        args.limit,
    )


if __name__ == "__main__":
    main()
