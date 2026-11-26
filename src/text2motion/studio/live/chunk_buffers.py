from __future__ import annotations

from text2motion.motion.representation import JOINTS
from text2motion.studio.live.actions import upsample_joints


def prepare_joints(state, chunk):
    buffers = state.stream_buffers
    joints = upsample_joints(buffers.interpolation_anchor, chunk.joints, state.upsample)
    buffers.interpolation_anchor = chunk.joints[-1]
    buffers.add_source_joints(chunk.joints)
    assert joints.ndim == 3 and joints.shape[1:] == (JOINTS, 3), f"bad joints {joints.shape}"
    return joints


def publish_chunk(state, joints, emitted: int, total_frames: int) -> None:
    state.stream_buffers.frames_emitted = emitted
    state.stream_buffers.gen_progress = (
        0.0 if total_frames == 0 else min(1.0, emitted / total_frames)
    )
    with state.stream_buffers.lock:
        state.stream_buffers.new_chunks.append(joints)
