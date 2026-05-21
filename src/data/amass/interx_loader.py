from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import numpy as np

from .smplx_pack import SMPLXSample, pack_smplx_pose

log = logging.getLogger(__name__)

INTERX_FPS = 30.0
ACTION_CODE_RE = re.compile(r"A(\d{3})")


def load_action_map(repo_datasets_dir: Path) -> dict[str, str]:
    # action_setting.txt: line N (0-indexed) is the human-readable name for "A%03d" % N
    f = repo_datasets_dir / "action_setting.txt"

    if not f.exists():
        return {}
    actions = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]

    return {f"A{i:03d}": name for i, name in enumerate(actions)}


def parse_seq_action(seq_id: str, action_map: dict[str, str]) -> str:
    # seq_id pattern: G???T???A???R??? — extract the A-prefixed 3-digit action code
    m = ACTION_CODE_RE.search(seq_id)

    if m is None:
        return "interaction"
    code = f"A{m.group(1)}"

    return action_map.get(code, code)


def load_texts(texts_dir: Path, seq_id: str) -> list[str]:
    f = texts_dir / f"{seq_id}.txt"

    if not f.exists():
        return []

    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


class InterXLoader:
    """Inter-X dataset loader.

    Inter-X has 11,387 two-person interactions; we treat each person as an independent
    motion clip, so the loader yields up to ~22,774 SMPLXSamples. Each NPZ holds
    SMPL-X body+hand parameters but no jaw/eye poses — those channels are zero-padded.
    """

    def __init__(self, data_dir: str = "data/inter-x"):
        self.data_dir = Path(data_dir)
        self.motions_dir = self.data_dir / "motions"
        self.texts_dir = self.data_dir / "texts"
        self.repo_datasets_dir = self.data_dir / "Inter-X-main" / "datasets"
        self.action_map = load_action_map(self.repo_datasets_dir)

    def discover_sequences(self, split_name: str | None = None) -> list[str]:
        if split_name:
            split_file = self.repo_datasets_dir / f"{split_name}.txt"

            if split_file.exists():
                ids = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines()
                       if ln.strip()]
                log.info("InterXLoader: split=%s -> %d ids", split_name, len(ids))

                return ids

        if not self.motions_dir.exists():
            log.warning("InterXLoader: motions dir not found: %s", self.motions_dir)

            return []
        ids = sorted([d.name for d in self.motions_dir.iterdir() if d.is_dir()])
        log.info("InterXLoader: found %d sequences on disk", len(ids))

        return ids

    def load_person(self, seq_id: str, person_idx: int) -> SMPLXSample | None:
        npz_path = self.motions_dir / seq_id / f"P{person_idx}.npz"

        if not npz_path.exists():
            return None

        try:
            d = np.load(npz_path, allow_pickle=True)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.warning("InterXLoader: failed to load %s: %s", npz_path, e)

            return None
        root_orient = d["root_orient"].astype(np.float32)
        trans = d["trans"].astype(np.float32)
        T = root_orient.shape[0]
        pose_body = d["pose_body"].astype(np.float32).reshape(T, -1)
        pose_lhand = d["pose_lhand"].astype(np.float32).reshape(T, -1)
        pose_rhand = d["pose_rhand"].astype(np.float32).reshape(T, -1)
        pose_hand = np.concatenate([pose_lhand, pose_rhand], axis=1)
        pose_jaw = np.zeros((T, 3), dtype=np.float32)
        pose_eye = np.zeros((T, 6), dtype=np.float32)
        motion = pack_smplx_pose(root_orient, trans, pose_body, pose_hand, pose_jaw, pose_eye)

        action = parse_seq_action(seq_id, self.action_map)
        texts = load_texts(self.texts_dir, seq_id)
        text = texts[0] if texts else f"two people {action.lower()}"
        gender = str(d["gender"]) if "gender" in d.files else "neutral"

        return SMPLXSample(
            sample_id=f"interx/{seq_id}/P{person_idx}",
            motion=motion,
            betas=np.zeros(16, dtype=np.float32),
            fps=INTERX_FPS,
            duration=T / INTERX_FPS,
            gender=gender,
            text=text,
            source="interx",
        )

    def load_dataset(self, max_samples: int | None = None, min_frames: int = 30,
                    split_name: str | None = None) -> list[SMPLXSample]:
        seqs = self.discover_sequences(split_name=split_name)
        out: list[SMPLXSample] = []

        for seq_id in seqs:
            for person_idx in (1, 2):
                s = self.load_person(seq_id, person_idx)

                if s is not None and s.motion.shape[0] >= min_frames:
                    out.append(s)

                    if max_samples and len(out) >= max_samples:
                        log.info("InterXLoader: loaded %d samples (capped)", len(out))

                        return out
        log.info("InterXLoader: loaded %d / %d samples", len(out), 2 * len(seqs))

        return out
