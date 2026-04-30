from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import numpy as np

from src.shared.constants import MOTION_FPS

from .amass_text import textFromFilename
from .smplx_pack import SMPLXSample, packSmplxPose

log = logging.getLogger(__name__)


class AMASSLoader:
    def __init__(self, dataDir: str = "data/AMASS"):
        self.dataDir = Path(dataDir)

    textFromFilename = staticmethod(textFromFilename)

    def discoverFiles(self) -> list[Path]:
        files = sorted(self.dataDir.rglob("*.npz"))
        log.info("AMASSLoader: found %d files in %s", len(files), self.dataDir)

        return files

    def loadFile(self, path: Path) -> SMPLXSample | None:
        try:
            data = np.load(path, allow_pickle=True)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.warning("Failed to load %s: %s", path, e)

            return None

        if not all(k in data for k in ("root_orient", "trans", "pose_body")):
            return None

        T = data["root_orient"].shape[0]
        fps = float(data.get("mocap_frame_rate", MOTION_FPS))
        rootOrient = data["root_orient"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
        poseBody = data["pose_body"].astype(np.float32)

        if "pose_hand" in data:
            poseHand = data["pose_hand"].astype(np.float32)

            if poseHand.shape[1] < 90:
                poseHand = np.pad(poseHand, ((0, 0), (0, 90 - poseHand.shape[1])))
            elif poseHand.shape[1] > 90:
                poseHand = poseHand[:, :90]
        else:
            poseHand = np.zeros((T, 90), dtype=np.float32)
        poseJaw = (data["pose_jaw"].astype(np.float32)
                   if "pose_jaw" in data else np.zeros((T, 3), dtype=np.float32))
        poseEye = (data["pose_eye"].astype(np.float32)
                   if "pose_eye" in data else np.zeros((T, 6), dtype=np.float32))
        motion = packSmplxPose(rootOrient, trans, poseBody, poseHand, poseJaw, poseEye)

        if np.isnan(motion).any() or np.isinf(motion).any():
            log.warning("NaN/Inf detected in %s -- skipping", path)

            return None

        betas = (data["betas"].astype(np.float32)
                 if "betas" in data else np.zeros(16, dtype=np.float32))

        if len(betas) < 16:
            betas = np.pad(betas, (0, 16 - len(betas)))
        genderVal = data.get("gender", "neutral")
        gender = str(genderVal) if not isinstance(genderVal, np.ndarray) else str(genderVal.item())
        sampleId = str(path.relative_to(self.dataDir).with_suffix("")).replace(os.sep, "/")

        return SMPLXSample(
            sampleId=sampleId, motion=motion, betas=betas, fps=fps,
            duration=T / fps, gender=gender, text=textFromFilename(sampleId), source="amass",
        )

    def loadDataset(self, maxSamples: int | None = None, minFrames: int = 30) -> list[SMPLXSample]:
        files = self.discoverFiles()

        if maxSamples:
            files = files[:maxSamples]
        samples = [s for f in files if (s := self.loadFile(f)) and s.motion.shape[0] >= minFrames]
        log.info("AMASSLoader: loaded %d / %d sequences", len(samples), len(files))

        return samples
