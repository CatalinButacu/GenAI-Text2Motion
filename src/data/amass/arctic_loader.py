from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import numpy as np

from src.shared.constants import CONSTS

from .smplx_pack import SMPLXSample, pack_smplx_pose

log = logging.getLogger(__name__)


class ARCTICLoader:
    def __init__(self, data_dir: str = "data/ARCTIC/unpack"):
        self.data_dir = Path(data_dir)
        self.raw_seqs = self.data_dir / "raw_seqs"

    def discover_sequences(self) -> list[tuple[str, str, str]]:
        seqs = []

        if not self.raw_seqs.exists():
            return seqs

        for subj_dir in sorted(self.raw_seqs.iterdir()):
            if not subj_dir.is_dir():
                continue

            seen: set[str] = set()

            for f in sorted(subj_dir.glob("*.smplx.npy")):
                base = f.name.replace(".smplx.npy", "")

                if base not in seen:
                    seen.add(base)
                    seqs.append((subj_dir.name, base, f"{subj_dir.name}/{base}"))
        log.info("ARCTICLoader: found %d sequences", len(seqs))

        return seqs

    def load_sequence(self, subject: str, seq_name: str) -> SMPLXSample | None:
        base = self.raw_seqs / subject / seq_name
        smplx_path = Path(f"{base}.smplx.npy")

        if not smplx_path.exists():
            return None

        try:
            body = np.load(smplx_path, allow_pickle=True).item()
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.warning("Failed to load %s: %s", smplx_path, e)

            return None

        T = body["body_pose"].shape[0]
        lhand = body["left_hand_pose"].astype(np.float32)
        rhand = body["right_hand_pose"].astype(np.float32)
        motion = pack_smplx_pose(
            body["global_orient"].astype(np.float32),
            body["transl"].astype(np.float32),
            body["body_pose"].astype(np.float32),
            np.concatenate([lhand, rhand], axis=1),
            body["jaw_pose"].astype(np.float32),
            np.concatenate(
                [body["leye_pose"].astype(np.float32), body["reye_pose"].astype(np.float32)],
                axis=1,
            ),
        )
        obj_path = Path(f"{base}.object.npy")
        obj_motion = None

        if obj_path.exists():
            obj_data = np.load(obj_path, allow_pickle=True)

            if isinstance(obj_data, np.ndarray):
                obj_array = obj_data.item() if obj_data.dtype == object else obj_data
                obj_motion = obj_array.astype(np.float32)

        obj_name = seq_name.split("_")[0] if "_" in seq_name else seq_name

        return SMPLXSample(
            sample_id=f"arctic/{subject}/{seq_name}",
            motion=motion, betas=np.zeros(16, dtype=np.float32),
            fps=float(CONSTS.runtime.motion_fps),
            duration=T / float(CONSTS.runtime.motion_fps),
            text=f"person interacts with {obj_name}: {seq_name.replace('_', ' ')}",
            source="arctic", object_motion=obj_motion,
        )

    def load_dataset(
        self, max_samples: int | None = None, min_frames: int = 30,
    ) -> list[SMPLXSample]:
        seqs = self.discover_sequences()

        if max_samples:
            seqs = seqs[:max_samples]
        samples = [
            s for subj, sn, _ in seqs
            if (s := self.load_sequence(subj, sn)) and s.motion.shape[0] >= min_frames
        ]
        log.info("ARCTICLoader: loaded %d / %d sequences", len(samples), len(seqs))

        return samples
