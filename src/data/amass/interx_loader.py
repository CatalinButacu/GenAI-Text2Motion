from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import numpy as np

from .smplx_pack import SMPLXSample, packSmplxPose

log = logging.getLogger(__name__)

INTERX_FPS = 30.0
ACTION_CODE_RE = re.compile(r"A(\d{3})")


def loadActionMap(repoDatasetsDir: Path) -> dict[str, str]:
    # action_setting.txt: line N (0-indexed) is the human-readable name for "A%03d" % N
    f = repoDatasetsDir / "action_setting.txt"

    if not f.exists():
        return {}
    actions = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]

    return {f"A{i:03d}": name for i, name in enumerate(actions)}


def parseSeqAction(seqId: str, actionMap: dict[str, str]) -> str:
    # seqId pattern: G???T???A???R??? — extract the A-prefixed 3-digit action code
    m = ACTION_CODE_RE.search(seqId)

    if m is None:
        return "interaction"
    code = f"A{m.group(1)}"

    return actionMap.get(code, code)


def loadTexts(textsDir: Path, seqId: str) -> list[str]:
    f = textsDir / f"{seqId}.txt"

    if not f.exists():
        return []

    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


class InterXLoader:
    """Inter-X dataset loader.

    Inter-X has 11,387 two-person interactions; we treat each person as an independent
    motion clip, so the loader yields up to ~22,774 SMPLXSamples. Each NPZ holds
    SMPL-X body+hand parameters but no jaw/eye poses — those channels are zero-padded.
    """

    def __init__(self, dataDir: str = "data/inter-x"):
        self.dataDir = Path(dataDir)
        self.motionsDir = self.dataDir / "motions"
        self.textsDir = self.dataDir / "texts"
        self.repoDatasetsDir = self.dataDir / "Inter-X-main" / "datasets"
        self.actionMap = loadActionMap(self.repoDatasetsDir)

    def discoverSequences(self, splitName: str | None = None) -> list[str]:
        if splitName:
            splitFile = self.repoDatasetsDir / f"{splitName}.txt"

            if splitFile.exists():
                ids = [ln.strip() for ln in splitFile.read_text(encoding="utf-8").splitlines()
                       if ln.strip()]
                log.info("InterXLoader: split=%s -> %d ids", splitName, len(ids))

                return ids

        if not self.motionsDir.exists():
            log.warning("InterXLoader: motions dir not found: %s", self.motionsDir)

            return []
        ids = sorted([d.name for d in self.motionsDir.iterdir() if d.is_dir()])
        log.info("InterXLoader: found %d sequences on disk", len(ids))

        return ids

    def loadPerson(self, seqId: str, personIdx: int) -> SMPLXSample | None:
        npzPath = self.motionsDir / seqId / f"P{personIdx}.npz"

        if not npzPath.exists():
            return None

        try:
            d = np.load(npzPath, allow_pickle=True)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.warning("InterXLoader: failed to load %s: %s", npzPath, e)

            return None
        rootOrient = d["root_orient"].astype(np.float32)
        trans = d["trans"].astype(np.float32)
        T = rootOrient.shape[0]
        poseBody = d["pose_body"].astype(np.float32).reshape(T, -1)
        poseLhand = d["pose_lhand"].astype(np.float32).reshape(T, -1)
        poseRhand = d["pose_rhand"].astype(np.float32).reshape(T, -1)
        poseHand = np.concatenate([poseLhand, poseRhand], axis=1)
        poseJaw = np.zeros((T, 3), dtype=np.float32)
        poseEye = np.zeros((T, 6), dtype=np.float32)
        motion = packSmplxPose(rootOrient, trans, poseBody, poseHand, poseJaw, poseEye)

        action = parseSeqAction(seqId, self.actionMap)
        texts = loadTexts(self.textsDir, seqId)
        text = texts[0] if texts else f"two people {action.lower()}"
        gender = str(d["gender"]) if "gender" in d.files else "neutral"

        return SMPLXSample(
            sampleId=f"interx/{seqId}/P{personIdx}",
            motion=motion,
            betas=np.zeros(16, dtype=np.float32),
            fps=INTERX_FPS,
            duration=T / INTERX_FPS,
            gender=gender,
            text=text,
            source="interx",
        )

    def loadDataset(self, maxSamples: int | None = None, minFrames: int = 30,
                    splitName: str | None = None) -> list[SMPLXSample]:
        seqs = self.discoverSequences(splitName=splitName)
        out: list[SMPLXSample] = []

        for seqId in seqs:
            for personIdx in (1, 2):
                s = self.loadPerson(seqId, personIdx)

                if s is not None and s.motion.shape[0] >= minFrames:
                    out.append(s)

                    if maxSamples and len(out) >= maxSamples:
                        log.info("InterXLoader: loaded %d samples (capped)", len(out))

                        return out
        log.info("InterXLoader: loaded %d / %d samples", len(out), 2 * len(seqs))

        return out
