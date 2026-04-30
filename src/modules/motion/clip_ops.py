"""Clip generation and blending -- M4 module logic."""

import logging
import time
from typing import Any

import numpy as np

from src.shared.constants import MOTION_FPS
from src.shared.vocabulary import ACTIONS

from .blend import slerpBlendFrames
from .config import MotionConfig
from .generator import MotionGenerator
from .models import MotionClip, MotionSource

log = logging.getLogger(__name__)


#  Clip blending


def crossfadeArrays(
    featureParts: list,
    jointParts: list,
    f: np.ndarray,
    j: np.ndarray | None,
    blendFrames: int,
) -> None:
    """Crossfade the new arrays into the running parts lists in-place."""
    n = min(blendFrames, len(featureParts[-1]), len(f))

    if n <= 0:
        featureParts.append(f)

        if j is not None:
            jointParts.append(j)

        return

    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)
    tail, head = featureParts[-1][-n:], f[:n]
    blended = slerpBlendFrames(tail, head, alpha)

    featureParts[-1] = featureParts[-1][:-n]
    featureParts.append(blended)
    featureParts.append(f[n:])

    if j is not None and jointParts and jointParts[-1] is not None:
        jt, jh = jointParts[-1][-n:], j[:n]
        jb = jt * (1 - alpha[:, None, None]) + jh * alpha[:, None, None]
        jointParts[-1] = jointParts[-1][:-n]
        jointParts.append(jb)
        jointParts.append(j[n:])


def blendClips(clips: list[MotionClip], blendFrames: int) -> MotionClip:
    """Merge a sequence of clips for one actor into a single clip with crossfades."""
    featureParts: list[np.ndarray] = []
    jointParts: list[np.ndarray] = []
    labels: list[str] = []

    for i, clip in enumerate(clips):
        labels.append(clip.action)
        f, j = clip.smplxParams, clip.rawJoints

        if i > 0 and blendFrames > 0:
            crossfadeArrays(featureParts, jointParts, f, j, blendFrames)
        else:
            featureParts.append(f)

            if j is not None:
                jointParts.append(j)

    missing = [i for i, clip in enumerate(clips) if clip.rawJoints is None]
    raw = None

    if jointParts and not missing:
        raw = np.concatenate(jointParts, axis=0)
    elif jointParts and missing:
        log.warning(
            "[M4] blend_clips: %d/%d clips have raw_joints=None (indices %s) -- "
            "dropping ALL raw_joints for this actor. Physics retargeting will fail.",
            len(missing),
            len(clips),
            missing,
        )

    return MotionClip(
        action=" then ".join(labels),
        smplxParams=np.concatenate(featureParts, axis=0),
        fps=clips[0].fps,
        source=MotionSource.SEQUENCED,
        rawJoints=raw,
    )


def sequenceClips(actionClips: list, blendFrames: int) -> dict[str, MotionClip]:
    """Group per-action clips by actor and blend each actor's sequence into one clip."""
    actorSeqs: dict[str, list] = {}

    for actor, clip in actionClips:
        actorSeqs.setdefault(actor, []).append(clip)

    return {
        actor: seq[0] if len(seq) == 1 else blendClips(seq, blendFrames)
        for actor, seq in actorSeqs.items()
    }


#  Clip generation


def buildActionQuery(action, actDef) -> str:
    """Build the text query used to retrieve a motion clip for one action."""
    base = (
        action.rawText
        or (actDef.motionClip if actDef and actDef.motionClip else None)
        or action.actionType.replace("_", " ")
    )

    return f"{action.modifier} {base}".strip() if action.modifier else base


def computeActionFrames(action, totalFrames: int, nActions: int, config: MotionConfig) -> int:
    """Compute the frame budget for a single action."""
    if action.duration is not None:
        return max(int(action.duration * MOTION_FPS), config.minActionFrames)

    return max(totalFrames // nActions, config.minActionFrames)


def lastPoseOf(clip: MotionClip | None) -> np.ndarray | None:
    """Extract the final SMPL-X pose from a clip to seed the next generation."""
    if clip is None or clip.smplxParams is None or len(clip.smplxParams) == 0:
        return None

    return clip.smplxParams[-1]


def generateActionClips(
    planned, totalFrames: int, motionGen: MotionGenerator, config: MotionConfig
) -> list[tuple[str, Any]]:
    """Generate one MotionClip per action, ordered and seeded from the previous clip."""
    actions = getattr(planned, "actions", [])

    if not actions:
        return []

    sortedActions = sorted(actions, key=lambda a: a.order)
    nActions = len(sortedActions)
    clips: list[tuple] = []
    lastClip: dict[str, MotionClip] = {}

    for action in sortedActions:
        actDef = ACTIONS.get(action.actionType)
        n = computeActionFrames(action, totalFrames, nActions, config)
        query = buildActionQuery(action, actDef)
        seed = lastPoseOf(lastClip.get(action.actor))
        t0 = time.time()
        clip = motionGen.generate(query, numFrames=n, initPose=seed)
        log.info(
            "[M4] generate %r (order=%d) for '%s' -> %d frames (src=%s, seeded=%s) in %.2fs",
            query,
            action.order,
            action.actor,
            clip.numFrames,
            clip.source,
            seed is not None,
            time.time() - t0,
        )
        lastClip[action.actor] = clip
        clips.append((action.actor, clip))

    return clips
