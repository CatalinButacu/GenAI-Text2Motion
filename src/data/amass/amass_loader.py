"""AMASS .npz loader.

Coordinate-system contract: this loader does NOT apply a Z-up -> Y-up
rotation. It expects the input .npz files to already be in the Y-up
convention used throughout the pipeline (matches HumanML3D and aitviewer).

If you point this loader at raw AMASS-native .npz (Z-up), the entire
pipeline silently produces sideways motion. Either:
  - preprocess your AMASS dump to Y-up upstream (preferred), or
  - set RenderConfig.input_coord_system='zup' so the renderer rotates
    on the way out (works for visualization, not for training -- the
    SSM learns the wrong manifold).
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import numpy as np

from src.shared.constants import MOTION_FPS

from .amass_text import text_from_filename
from .smplx_pack import SMPLXSample, pack_smplx_pose

log = logging.getLogger(__name__)


class AMASSLoader:
    def __init__(self, data_dir: str = "data/AMASS"):
        self.data_dir = Path(data_dir)

    text_from_filename = staticmethod(text_from_filename)

    def discover_files(self) -> list[Path]:
        files = sorted(self.data_dir.rglob("*.npz"))
        log.info("AMASSLoader: found %d files in %s", len(files), self.data_dir)

        return files

    def load_file(self, path: Path) -> SMPLXSample | None:
        try:
            data = np.load(path, allow_pickle=True)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.warning("Failed to load %s: %s", path, e)

            return None

        if not all(k in data for k in ("root_orient", "trans", "pose_body")):
            return None

        T = data["root_orient"].shape[0]
        fps = float(data.get("mocap_frame_rate", MOTION_FPS))
        root_orient = data["root_orient"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
        pose_body = data["pose_body"].astype(np.float32)

        if "pose_hand" in data:
            pose_hand = data["pose_hand"].astype(np.float32)

            if pose_hand.shape[1] < 90:
                pose_hand = np.pad(pose_hand, ((0, 0), (0, 90 - pose_hand.shape[1])))
            elif pose_hand.shape[1] > 90:
                pose_hand = pose_hand[:, :90]
        else:
            pose_hand = np.zeros((T, 90), dtype=np.float32)
        pose_jaw = (data["pose_jaw"].astype(np.float32)
                   if "pose_jaw" in data else np.zeros((T, 3), dtype=np.float32))
        pose_eye = (data["pose_eye"].astype(np.float32)
                   if "pose_eye" in data else np.zeros((T, 6), dtype=np.float32))
        motion = pack_smplx_pose(root_orient, trans, pose_body, pose_hand, pose_jaw, pose_eye)

        if np.isnan(motion).any() or np.isinf(motion).any():
            log.warning("NaN/Inf detected in %s -- skipping", path)

            return None

        betas = (data["betas"].astype(np.float32)
                 if "betas" in data else np.zeros(16, dtype=np.float32))

        if len(betas) < 16:
            betas = np.pad(betas, (0, 16 - len(betas)))
        gender_val = data.get("gender", "neutral")
        gender = (
            str(gender_val)
            if not isinstance(gender_val, np.ndarray)
            else str(gender_val.item())
        )
        sample_id = str(path.relative_to(self.data_dir).with_suffix("")).replace(os.sep, "/")

        return SMPLXSample(
            sample_id=sample_id, motion=motion, betas=betas, fps=fps,
            duration=T / fps, gender=gender, text=text_from_filename(sample_id), source="amass",
        )

    def load_dataset(
        self, max_samples: int | None = None, min_frames: int = 30,
    ) -> list[SMPLXSample]:
        files = self.discover_files()

        if max_samples:
            files = files[:max_samples]
        samples = [s for f in files if (s := self.load_file(f)) and s.motion.shape[0] >= min_frames]
        log.info("AMASSLoader: loaded %d / %d sequences", len(samples), len(files))

        return samples
