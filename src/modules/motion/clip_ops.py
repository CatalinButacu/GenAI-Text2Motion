"""Clip generation and blending -- M4 module logic."""

import logging
import time
from typing import Any

import numpy as np

from src.shared.constants import MOTION_FPS
from src.shared.vocabulary import ACTIONS

from .blend import slerp_blend_frames
from .config import MotionConfig
from .generator import MotionGenerator
from .models import MotionClip, MotionSource

log = logging.getLogger(__name__)


#  Clip blending


def crossfade_arrays(
    feature_parts: list,
    joint_parts: list,
    f: np.ndarray,
    j: np.ndarray | None,
    blend_frames: int,
) -> None:
    """Crossfade the new arrays into the running parts lists in-place."""
    n = min(blend_frames, len(feature_parts[-1]), len(f))

    if n <= 0:
        feature_parts.append(f)

        if j is not None:
            joint_parts.append(j)

        return

    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)
    tail, head = feature_parts[-1][-n:], f[:n]
    blended = slerp_blend_frames(tail, head, alpha)

    feature_parts[-1] = feature_parts[-1][:-n]
    feature_parts.append(blended)
    feature_parts.append(f[n:])

    if j is not None and joint_parts and joint_parts[-1] is not None:
        jt, jh = joint_parts[-1][-n:], j[:n]
        jb = jt * (1 - alpha[:, None, None]) + jh * alpha[:, None, None]
        joint_parts[-1] = joint_parts[-1][:-n]
        joint_parts.append(jb)
        joint_parts.append(j[n:])


def blend_clips(clips: list[MotionClip], blend_frames: int) -> MotionClip:
    """Merge a sequence of clips for one actor into a single clip with crossfades."""
    feature_parts: list[np.ndarray] = []
    joint_parts: list[np.ndarray] = []
    labels: list[str] = []

    for i, clip in enumerate(clips):
        labels.append(clip.action)
        f, j = clip.smplx_params, clip.raw_joints

        if i > 0 and blend_frames > 0:
            crossfade_arrays(feature_parts, joint_parts, f, j, blend_frames)
        else:
            feature_parts.append(f)

            if j is not None:
                joint_parts.append(j)

    missing = [i for i, clip in enumerate(clips) if clip.raw_joints is None]
    raw = None

    if joint_parts and not missing:
        raw = np.concatenate(joint_parts, axis=0)
    elif joint_parts and missing:
        log.warning(
            "[M4] blend_clips: %d/%d clips have raw_joints=None (indices %s) -- "
            "dropping ALL raw_joints for this actor. Physics retargeting will fail.",
            len(missing),
            len(clips),
            missing,
        )

    return MotionClip(
        action=" then ".join(labels),
        smplx_params=np.concatenate(feature_parts, axis=0),
        fps=clips[0].fps,
        source=MotionSource.SEQUENCED,
        raw_joints=raw,
    )


def sequence_clips(action_clips: list, blend_frames: int) -> dict[str, MotionClip]:
    """Group per-action clips by actor and blend each actor's sequence into one clip."""
    actor_seqs: dict[str, list] = {}

    for actor, clip in action_clips:
        actor_seqs.setdefault(actor, []).append(clip)

    return {
        actor: seq[0] if len(seq) == 1 else blend_clips(seq, blend_frames)
        for actor, seq in actor_seqs.items()
    }


#  Clip generation


def build_action_query(action, act_def) -> str:
    """Build the text query used to retrieve a motion clip for one action."""
    base = (
        action.raw_text
        or (act_def.motion_clip if act_def and act_def.motion_clip else None)
        or action.action_type.replace("_", " ")
    )

    return f"{action.modifier} {base}".strip() if action.modifier else base


def compute_action_frames(action, total_frames: int, n_actions: int, config: MotionConfig) -> int:
    """Compute the frame budget for a single action."""
    if action.duration is not None:
        return max(int(action.duration * MOTION_FPS), config.min_action_frames)

    return max(total_frames // n_actions, config.min_action_frames)


def last_pose_of(clip: MotionClip | None) -> np.ndarray | None:
    """Extract the final SMPL-X pose from a clip to seed the next generation."""
    if clip is None or clip.smplx_params is None or len(clip.smplx_params) == 0:
        return None

    return clip.smplx_params[-1]


def generate_action_clips(
    planned, total_frames: int, motion_gen: MotionGenerator, config: MotionConfig
) -> list[tuple[str, Any]]:
    """Generate one MotionClip per action, ordered and seeded from the previous clip."""
    actions = getattr(planned, "actions", [])

    if not actions:
        return []

    sorted_actions = sorted(actions, key=lambda a: a.order)
    n_actions = len(sorted_actions)
    clips: list[tuple] = []
    last_clip: dict[str, MotionClip] = {}

    for action in sorted_actions:
        act_def = ACTIONS.get(action.action_type)
        n = compute_action_frames(action, total_frames, n_actions, config)
        query = build_action_query(action, act_def)
        seed = last_pose_of(last_clip.get(action.actor))
        t0 = time.time()
        clip = motion_gen.generate(query, num_frames=n, init_pose=seed)
        log.info(
            "[M4] generate %r (order=%d) for '%s' -> %d frames (src=%s, seeded=%s) in %.2fs",
            query,
            action.order,
            action.actor,
            clip.num_frames,
            clip.source,
            seed is not None,
            time.time() - t0,
        )
        last_clip[action.actor] = clip
        clips.append((action.actor, clip))

    return clips
