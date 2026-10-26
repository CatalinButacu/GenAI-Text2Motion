import threading
import time
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from text2motion.studio import avatar as avatar_module
from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.generation import StudioGenerationController
from text2motion.studio.scene import (
    SmplxFitState,
    StreamSceneUpdater,
    StudioSceneNodes,
    StudioStreamBuffers,
)


class FakeScene:
    def __init__(self) -> None:
        self.nodes = []
        self.selected_object = None
        self.current_frame_id = 0
        self.n_frames = 0

    def add(self, node) -> None:
        self.nodes.append(node)

    def remove(self, node) -> None:
        self.nodes.remove(node)

    def select(self, node) -> None:
        self.selected_object = node


def __updater_with(skeleton_frames: int, body_frames: int, fitting: bool, playhead: int):
    scene = FakeScene()
    host = SimpleNamespace(scene=scene)
    buffers = StudioStreamBuffers()
    buffers.fitting = fitting
    nodes = StudioSceneNodes(scene)
    nodes.motion = SimpleNamespace(joint_positions=np.zeros((skeleton_frames, 22, 3)), enabled=True)
    if body_frames:
        nodes.body = SimpleNamespace(enabled=False)
        buffers.body_verts = np.zeros((body_frames, 4, 3))
    state = SimpleNamespace(stream_buffers=buffers, nodes=nodes, show_skin=True)
    scene.current_frame_id = playhead
    return StreamSceneUpdater(cast(ViewerHost, host), state), scene


def test_playhead_waits_at_the_first_frame_until_the_first_mesh_is_fitted():
    updater, scene = __updater_with(skeleton_frames=160, body_frames=0, fitting=True, playhead=90)
    updater.hold_playhead_at_fitted_body()
    assert scene.current_frame_id == 0


def test_playhead_is_held_at_the_last_fitted_body_frame_while_fitting():
    updater, scene = __updater_with(skeleton_frames=160, body_frames=48, fitting=True, playhead=90)
    updater.hold_playhead_at_fitted_body()
    assert scene.current_frame_id == 47
    updater.sync_display_nodes()
    assert updater.state.nodes.body.enabled
    assert not updater.state.nodes.motion.enabled


def test_playhead_runs_free_once_fitting_has_stopped():
    updater, scene = __updater_with(skeleton_frames=160, body_frames=48, fitting=False, playhead=90)
    updater.hold_playhead_at_fitted_body()
    assert scene.current_frame_id == 90


def test_a_new_take_makes_the_running_fit_stale():
    buffers = StudioStreamBuffers()
    buffers.fit_take = buffers.take
    assert not buffers.fit_is_stale()
    buffers.take += 1
    assert buffers.fit_is_stale()
    buffers.fit_take = buffers.take
    buffers.cancel = True
    assert buffers.fit_is_stale()


@pytest.mark.parametrize("upsample", [1, 3])
def test_every_streamed_chunk_reaches_the_body_fitter_after_a_normal_finish(upsample):
    fitted = []
    counted = []
    fitter_done = threading.Event()

    def fit_chunk(joints, warm_start, n_frames, factor):
        time.sleep(0.05)
        fitted.append((len(joints), factor))
        counted.append(n_frames)
        return warm_start

    chunks = [
        SimpleNamespace(
            joints=np.zeros((16, 22, 3), np.float32),
            features=np.zeros((16, 263), np.float32),
            yaw=0.0,
        )
        for _ in range(6)
    ]
    client = SimpleNamespace(generate=lambda *args, **kwargs: iter(chunks), last_stats=None)
    buffers = StudioStreamBuffers()
    state = SimpleNamespace(
        stream_buffers=buffers,
        smplx_fit=SmplxFitState(),
        client=client,
        fit_lookahead_chunks=2,
        stream_lead_frames=10_000,
        upsample=upsample,
        generating=True,
        status="",
    )
    host = SimpleNamespace(
        playback_fps=20.0,
        run_animations=True,
        toggle_animation=lambda run: None,
        scene=FakeScene(),
    )
    avatar = SimpleNamespace(fit_single_chunk=fit_chunk)
    controller = StudioGenerationController(
        cast(ViewerHost, host), state, cast(SmplxAvatarController, avatar)
    )
    original = controller.fit_motion_chunks

    def fit_then_signal(fit_queue, take):
        original(fit_queue, take)
        fitter_done.set()

    controller.fit_motion_chunks = fit_then_signal
    buffers.take = 1
    params = {
        "steps": 24,
        "downsample": 4,
        "upsample": upsample,
        "fit_body": True,
        "temperature": 1.0,
        "top_p": 0.9,
        "cfg_scale": 6.0,
    }
    controller.stream_generation("walk", params)
    assert fitter_done.wait(timeout=5)
    assert fitted == [(16, upsample)] * 6
    assert counted[-1] == 96 * upsample
    assert not buffers.fitting


def test_a_dead_fit_thread_does_not_block_the_stream():
    def broken_fit(joints, warm_start, n_frames, factor):
        raise RuntimeError("fitter crashed")

    chunks = [SimpleNamespace(joints=np.zeros((16, 22, 3), np.float32), yaw=0.0) for _ in range(6)]
    client = SimpleNamespace(generate=lambda *args, **kwargs: iter(chunks), last_stats=None)
    buffers = StudioStreamBuffers()
    buffers.take = 1
    state = SimpleNamespace(
        stream_buffers=buffers,
        smplx_fit=SmplxFitState(),
        client=client,
        fit_lookahead_chunks=1,
        stream_lead_frames=10_000,
        upsample=1,
        generating=True,
        status="",
    )
    host = SimpleNamespace(
        playback_fps=20.0, run_animations=True, toggle_animation=lambda run: None, scene=FakeScene()
    )
    avatar = SimpleNamespace(fit_single_chunk=broken_fit)
    controller = StudioGenerationController(
        cast(ViewerHost, host), state, cast(SmplxAvatarController, avatar)
    )
    params = {
        "steps": 24,
        "downsample": 4,
        "upsample": 1,
        "fit_body": True,
        "temperature": 1.0,
        "top_p": 0.9,
        "cfg_scale": 6.0,
    }

    stream = threading.Thread(
        target=controller.stream_generation, args=("walk", params), daemon=True
    )
    stream.start()
    stream.join(timeout=5)

    assert not stream.is_alive()
    assert len(buffers.new_chunks) == 6
    assert "body fit failed" in state.status


def test_a_new_take_replaces_the_previous_skeleton_instead_of_appending():
    scene = FakeScene()
    host = SimpleNamespace(
        scene=scene,
        toggle_animation=lambda run: None,
    )
    scene.n_frames = 0
    buffers = StudioStreamBuffers()
    nodes = StudioSceneNodes(scene)
    state = SimpleNamespace(
        stream_buffers=buffers, nodes=nodes, window_frames=600, show_skin=True, upsample=1
    )
    updater = StreamSceneUpdater(cast(ViewerHost, host), state)
    updater.append_chunks([np.zeros((160, 22, 3), np.float32)])
    assert nodes.motion is not None
    assert len(nodes.motion.joint_positions) == 160
    buffers.started = False
    buffers.accum_joints = None
    updater.append_chunks([np.ones((16, 22, 3), np.float32)])
    assert nodes.motion is not None
    assert len(nodes.motion.joint_positions) == 16
    assert buffers.accum_joints is not None
    assert len(buffers.accum_joints) == 16
    assert sum(node.name == "Motion (streaming)" for node in scene.nodes) == 1


def test_the_window_cut_keeps_display_and_source_frames_aligned():
    scene = FakeScene()
    host = SimpleNamespace(scene=scene, toggle_animation=lambda run: None)
    buffers = StudioStreamBuffers()
    nodes = StudioSceneNodes(scene)
    state = SimpleNamespace(
        stream_buffers=buffers, nodes=nodes, window_frames=32, show_skin=True, upsample=6
    )
    updater = StreamSceneUpdater(cast(ViewerHost, host), state)
    for index in range(3):
        source = np.full((4, 22, 3), index, np.float32)
        buffers.add_source_joints(source)
        updater.append_chunks([np.repeat(source, 6, axis=0)])

    assert buffers.accum_joints is not None
    assert buffers.source_joints is not None
    assert len(buffers.accum_joints) == 36
    assert nodes.motion is not None
    assert len(nodes.motion.joint_positions) == 36
    assert len(buffers.source_joints) == 36 // 6 + 1


class FakeBody:
    def __init__(self, frames: int) -> None:
        self.vertices = np.zeros((frames, 4, 3), np.float32)
        self.enabled = True

    def remove_frames(self, dropped: list[int]) -> None:
        self.vertices = self.vertices[len(dropped) :]


def test_the_window_cut_trims_the_fitted_track_with_the_body():
    scene = FakeScene()
    host = SimpleNamespace(scene=scene, toggle_animation=lambda run: None)
    buffers = StudioStreamBuffers()
    nodes = StudioSceneNodes(scene)
    fit = SmplxFitState()
    state = SimpleNamespace(
        stream_buffers=buffers,
        nodes=nodes,
        window_frames=32,
        show_skin=True,
        upsample=1,
        smplx_fit=fit,
    )
    updater = StreamSceneUpdater(cast(ViewerHost, host), state)
    updater.append_chunks([np.zeros((30, 22, 3), np.float32)])
    nodes.body = FakeBody(60)
    for chunk in range(6):
        fit.orient.append(np.full((10, 3), chunk, np.float32))
        fit.pose.append(np.zeros((10, 63), np.float32))
        fit.transl.append(np.zeros((10, 3), np.float32))

    updater.append_chunks([np.zeros((30, 22, 3), np.float32)])

    track = np.concatenate(fit.orient)
    assert len(nodes.body.vertices) == 32
    assert len(track) == 32
    assert len(np.concatenate(fit.pose)) == len(np.concatenate(fit.transl)) == 32
    assert track[0, 0] == 2.0


def test_the_retrofit_fits_source_frames_like_the_live_stream(monkeypatch):
    calls = []
    buffers = StudioStreamBuffers()
    buffers.accum_joints = np.zeros((36, 22, 3), np.float32)
    buffers.source_joints = np.arange(7, dtype=np.float32)[:, None, None] * np.ones((7, 22, 3))
    fit_state = SmplxFitState()
    fit_state.model_cache[(1, "neutral")] = "anchor model"
    state = SimpleNamespace(
        stream_buffers=buffers,
        smplx_fit=fit_state,
        generating=False,
        fit_body=True,
        model_dir="models",
        upsample=6,
        gender="neutral",
        device="cpu",
        status="",
        config=SimpleNamespace(schedule=SimpleNamespace(retrofit_window=16)),
    )
    host = SimpleNamespace(run_animations=True, toggle_animation=lambda run: None)
    avatar = SmplxAvatarController(cast(ViewerHost, host), state)

    def fake_solve(model, joints, device):
        return SimpleNamespace(
            global_orient=joints[:, 0], body_pose=joints[:, 1], transl=joints[:, 2]
        )

    def record_fit(joints, warm_start, n_frames, factor):
        calls.append((joints[:, 0, 0].tolist(), factor, buffers.fit_anchor))
        return warm_start

    monkeypatch.setattr(avatar, "fit_single_chunk", record_fit)
    monkeypatch.setattr(avatar_module, "solve_smplx_pose_from_joints", fake_solve)
    avatar.on_fit_body_toggled()
    deadline = time.time() + 5
    while state.generating and time.time() < deadline:
        time.sleep(0.01)

    fitted = [frame for frames, _, _ in calls for frame in frames]
    assert fitted == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert {factor for _, factor, _ in calls} == {6}
    assert calls[0][2] is not None
    assert float(calls[0][2][0][0]) == 0.0
    assert len(fitted) * 6 == len(buffers.accum_joints)


def test_the_stream_waits_until_the_playhead_is_within_the_lead():
    buffers = StudioStreamBuffers()
    scene = FakeScene()
    host = SimpleNamespace(scene=scene)
    state = SimpleNamespace(stream_buffers=buffers, stream_lead_frames=32, upsample=1)
    controller = StudioGenerationController(
        cast(ViewerHost, host), state, cast(SmplxAvatarController, SimpleNamespace())
    )
    assert controller.frames_ahead_of_playhead(48) == 48
    released = threading.Event()

    def accept():
        controller.wait_for_playback(48)
        released.set()

    threading.Thread(target=accept, daemon=True).start()
    assert not released.wait(0.2)
    scene.current_frame_id = 20
    assert released.wait(2.0)
    buffers.trimmed_frames = 100
    scene.current_frame_id = 0
    assert controller.frames_ahead_of_playhead(148) == 48


def test_stop_releases_a_stream_that_is_waiting_for_playback():
    buffers = StudioStreamBuffers()
    host = SimpleNamespace(scene=FakeScene())
    state = SimpleNamespace(stream_buffers=buffers, stream_lead_frames=8, upsample=1)
    controller = StudioGenerationController(
        cast(ViewerHost, host), state, cast(SmplxAvatarController, SimpleNamespace())
    )
    released = threading.Event()

    def accept():
        controller.wait_for_playback(500)
        released.set()

    threading.Thread(target=accept, daemon=True).start()
    assert not released.wait(0.2)
    buffers.cancel = True
    assert released.wait(2.0)


def test_upsampling_interpolates_between_frames_and_ends_on_each_source_frame():
    from text2motion.studio.generation import upsample_joints

    joints = np.stack([np.full((22, 3), float(i), np.float32) for i in range(4)])
    upsampled = upsample_joints(None, joints, 3)
    assert upsampled.shape == (12, 22, 3)
    np.testing.assert_allclose(upsampled[2::3, 0, 0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(upsampled[3:6, 0, 0], [1 / 3, 2 / 3, 1.0], rtol=1e-6)
    continued = upsample_joints(joints[-1], joints[:1] + 4.0, 3)
    np.testing.assert_allclose(continued[:, 0, 0], [3 + 1 / 3, 3 + 2 / 3, 4.0], rtol=1e-6)
    assert upsample_joints(None, joints, 1) is joints


def __z_rotation(degrees: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.radians(degrees)], np.float32)


def test_pose_upsampling_ends_on_each_source_frame_and_moves_translation_linearly():
    from text2motion.studio.avatar import upsample_pose_parameters

    orient = np.stack([__z_rotation(0), __z_rotation(30), __z_rotation(60)])
    pose = np.zeros((3, 63), np.float32)
    transl = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]], np.float32)

    up_orient, up_pose, up_transl = upsample_pose_parameters(None, orient, pose, transl, 3)

    assert up_orient.shape == (9, 3)
    assert up_pose.shape == (9, 63)
    np.testing.assert_allclose(up_orient[2::3], orient, atol=1e-5)
    np.testing.assert_allclose(up_orient[3:6, 2], np.radians([10, 20, 30]), atol=1e-5)
    np.testing.assert_allclose(up_transl[3:6, 0], [1, 2, 3], atol=1e-5)
    assert upsample_pose_parameters(None, orient, pose, transl, 1)[0] is orient


def test_pose_upsampling_takes_the_short_way_across_180_degrees():
    from text2motion.studio.avatar import upsample_pose_parameters

    anchor = (__z_rotation(170), np.zeros(63, np.float32), np.zeros(3, np.float32))
    orient = __z_rotation(-170)[None]

    up_orient, _, _ = upsample_pose_parameters(
        anchor, orient, np.zeros((1, 63), np.float32), np.zeros((1, 3), np.float32), 2
    )

    angle = np.linalg.norm(up_orient[0])
    axis_z = abs(up_orient[0, 2]) / angle
    assert angle == pytest.approx(np.pi, abs=1e-4)
    assert axis_z == pytest.approx(1.0, abs=1e-4)
