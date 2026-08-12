from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_HEAD_TRIM_S = {
    "Eyes_Japan_Dataset": 3.0,
    "MPI_HDM05": 3.0,
    "TotalCapture": 1.0,
    "MPI_Limits": 1.0,
    "Transitions_mocap": 0.5,
}

AMASS_DATASET_RENAME = {
    "MPI_HDM05": "HDM05",
    "BioMotionLab_NTroje": "BMLrub",
    "MPI_Limits": "PosePrior",
    "MPI_mosh": "MoSh",
    "DFaust_67": "DFaust",
    "SSM_synced": "SSM",
    "TCD_handMocap": "TCDHands",
    "Transitions_mocap": "Transitions",
    "Eyes_Japan_Dataset": "EyesJapanDataset",
}

SHORT_TO_OFFICIAL = {short: official for official, short in AMASS_DATASET_RENAME.items()}


def dataset_of(source_path: str | Path) -> str:
    parts = [p for p in Path(source_path).parts if p not in (".", "pose_data")]
    if not parts:
        return ""
    return SHORT_TO_OFFICIAL.get(parts[0], parts[0])


def head_trim_frames(source_path: str | Path, fps: int) -> int:
    return int(round(DATASET_HEAD_TRIM_S.get(dataset_of(source_path), 0.0) * fps))


def pose_path_candidates(source_path: str, pose_root: Path) -> tuple[Path | None, list[Path]]:
    parts = [p for p in Path(source_path).parts if p not in (".", "pose_data")]
    if not parts:
        return None, []
    dataset = AMASS_DATASET_RENAME.get(parts[0], parts[0])
    rel = Path(dataset, *parts[1:])
    stem = rel.stem
    base = stem[:-6] if stem.endswith("_poses") else stem
    folder = pose_root / rel.parent

    # the SMPL-X release underscores what the SMPL+H index spells with spaces
    stems = list(dict.fromkeys([base, base.replace(" ", "_"), stem, stem.replace(" ", "_")]))
    names = [f"{s}{suffix}.npy" for s in stems for suffix in ("_stageii", "_poses", "")]
    return folder, [folder / name for name in dict.fromkeys(names)]


def resolve_pose_path(source_path: str, pose_root: Path) -> Path | None:
    folder, candidates = pose_path_candidates(source_path, pose_root)
    if folder is None:
        return None
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def read_heldout_ids(out_dir: Path) -> set[str]:
    held: set[str] = set()
    for split in ("val", "test"):
        names = (out_dir / f"{split}.txt").read_text().splitlines()
        held |= {
            n.strip()[1:] if n.strip().startswith("M") else n.strip() for n in names if n.strip()
        }
    return held


def heldout_sources(index_csv: Path, out_dir: Path, pose_root: Path) -> set[Path]:
    held_ids = read_heldout_ids(out_dir)
    index = pd.read_csv(index_csv)

    excluded: set[Path] = set()
    unexcludable: list[str] = []
    for i in range(index.shape[0]):
        if Path(str(index.loc[i]["new_name"])).stem not in held_ids:
            continue
        source_path = str(index.loc[i]["source_path"])
        resolved = resolve_pose_path(source_path, pose_root)
        if resolved is not None:
            excluded.add(resolved.resolve())
            continue
        folder, _ = pose_path_candidates(source_path, pose_root)
        if folder is not None and folder.is_dir():
            unexcludable.append(source_path)

    if unexcludable:
        raise RuntimeError(
            f"leakage guard failed: {len(unexcludable)} held-out clips have their AMASS capture "
            f"folder present under {pose_root} but no filename matched the known spellings, so "
            f"they would NOT be excluded from the unlabeled pretraining corpus. Fix the spelling "
            f"candidates before pretraining. First offenders: {unexcludable[:5]}"
        )
    return excluded
