from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from scripts.render import render_clip


@pytest.mark.parametrize("ignore_pose", [False, True])
def test_review_calibration_requires_horizontal_arms_and_y_up(ignore_pose):
    def model(**kwargs):
        joints = torch.zeros((1, 22, 3))
        joints[0, 15, 1] = 1.7
        for shoulder, wrist, sign in ((16, 20, 1), (17, 21, -1)):
            joints[0, shoulder] = torch.tensor([sign * 0.15, 1.4, 0])
            delta = torch.tensor([sign * 0.5, -0.05, 0])
            angle = 0 if ignore_pose else float(kwargs["body_pose"][0, (shoulder - 1) * 3 + 2])
            rotation = torch.tensor(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
            )
            joints[0, wrist] = joints[0, shoulder] + rotation @ delta
        return SimpleNamespace(joints=joints)

    if ignore_pose:
        with pytest.raises(ValueError, match="horizontal"):
            render_clip.review_tpose(model)
    else:
        joints = render_clip.review_tpose(model)
        np.testing.assert_allclose(joints[0, (20, 21), 1], joints[0, (16, 17), 1])
        assert joints[0, 15, 1] > joints[0, (7, 8, 10, 11), 1].max()
    assert len(render_clip.REVIEW_FRAMES) == 113
    assert render_clip.REVIEW_FRAMES[0] == 176 and render_clip.REVIEW_FRAMES[-1] == 288


def test_skeleton_factory_preserves_default_radius(monkeypatch):
    from text2motion.studio.scene import create_skeleton_sequence_node

    calls = []
    monkeypatch.setattr(
        "aitviewer.renderables.skeletons.Skeletons", lambda **kwargs: calls.append(kwargs)
    )
    joints = np.zeros((1, 22, 3))
    create_skeleton_sequence_node(joints)
    create_skeleton_sequence_node(joints, radius=0.025)
    assert calls[0]["radius"] == 0.01 and calls[1]["radius"] == 0.025
    assert calls[0]["joint_positions"] is joints
    np.testing.assert_array_equal(calls[0]["joint_connections"], calls[1]["joint_connections"])


def test_review_frame_sets_the_requested_frame_without_policy_labels():
    renderer = SimpleNamespace(
        scene=SimpleNamespace(current_frame_id=-1),
        get_frame=lambda: Image.new("RGB", (800, 480), "white"),
    )
    case = {
        "case": "case_001",
        "backbone": "mamba",
        "old_caption": "a person walks",
        "new_caption": "a person kneels",
    }
    image = render_clip.review_frame(renderer, case, 192)
    assert renderer.scene.current_frame_id == 192
    assert image.size == (800, 630)


def test_render_cli_keeps_prompt_mode(monkeypatch):
    captured = []
    monkeypatch.setattr("sys.argv", ["render_clip", "--prompt", "a person walks"])
    monkeypatch.setattr(render_clip, "run", captured.append)
    render_clip.main()
    assert captured[0].prompt == "a person walks" and captured[0].review is None


def test_render_cli_accepts_archived_review_and_rejects_both_inputs(monkeypatch):
    captured = []
    monkeypatch.setattr("sys.argv", ["render_clip", "--review", "audit.json"])
    monkeypatch.setattr(render_clip, "run", captured.append)
    render_clip.main()
    assert captured[0].review.name == "audit.json" and captured[0].prompt is None
    monkeypatch.setattr("sys.argv", ["render_clip", "--review", "audit.json", "--prompt", "walk"])
    with pytest.raises(SystemExit):
        render_clip.main()
