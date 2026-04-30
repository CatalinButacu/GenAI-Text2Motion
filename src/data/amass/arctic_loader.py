from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import numpy as np

from src.shared.constants import MOTION_FPS

from .smplx_pack import SMPLXSample, packSmplxPose

log = logging.getLogger(__name__)


class ARCTICLoader:
    def __init__(self, dataDir: str = "data/ARCTIC/unpack"):
        self.dataDir = Path(dataDir)
        self.raw_seqs = self.dataDir / "raw_seqs"

    def discoverSequences(self) -> list[tuple[str, str, str]]:
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

    def loadSequence(self, subject: str, seqName: str) -> SMPLXSample | None:
        base = self.raw_seqs / subject / seqName
        smplxPath = Path(f"{base}.smplx.npy")

        if not smplxPath.exists():
            return None

        try:
            body = np.load(smplxPath, allow_pickle=True).item()
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.warning("Failed to load %s: %s", smplxPath, e)

            return None

        T = body["body_pose"].shape[0]
        lhand = body["left_hand_pose"].astype(np.float32)
        rhand = body["right_hand_pose"].astype(np.float32)
        motion = packSmplxPose(
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
        objPath = Path(f"{base}.object.npy")
        objMotion = None

        if objPath.exists():
            objData = np.load(objPath, allow_pickle=True)
            objMotion = (
                (objData.item() if objData.dtype == object else objData).astype(np.float32)
                if isinstance(objData, np.ndarray)
                else None
            )
        objName = seqName.split("_")[0] if "_" in seqName else seqName

        return SMPLXSample(
            sampleId=f"arctic/{subject}/{seqName}",
            motion=motion, betas=np.zeros(16, dtype=np.float32),
            fps=30.0, duration=T / 30.0,
            text=f"person interacts with {objName}: {seqName.replace('_', ' ')}",
            source="arctic", objectMotion=objMotion,
        )

    def loadDataset(self, maxSamples: int | None = None, minFrames: int = 30) -> list[SMPLXSample]:
        seqs = self.discoverSequences()

        if maxSamples:
            seqs = seqs[:maxSamples]
        samples = [
            s for subj, sn, _ in seqs
            if (s := self.loadSequence(subj, sn)) and s.motion.shape[0] >= minFrames
        ]
        log.info("ARCTICLoader: loaded %d / %d sequences", len(samples), len(seqs))

        return samples
