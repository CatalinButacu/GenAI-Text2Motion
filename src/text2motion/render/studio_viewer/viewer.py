from __future__ import annotations

import argparse

from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer

from text2motion.render.joints2smpl import FitConfig, rest_pose_body
from text2motion.render.studio_viewer.avatar import Avatar
from text2motion.render.studio_viewer.constants import SKY_COLOR
from text2motion.render.studio_viewer.display import Display
from text2motion.render.studio_viewer.generation import GenerationControl
from text2motion.render.studio_viewer.gui_panels import GuiPanels
from text2motion.render.studio_viewer.studio_state import StudioState


class StreamingStudioViewer(Viewer):
    def __init__(self, client, model_dir: str, device: str, args: argparse.Namespace) -> None:
        super().__init__()
        self.state = StudioState(client, model_dir, device, args, self.scene)
        self.display = Display(self, self.state)
        self.avatar = Avatar(self, self.state)
        self.generation = GenerationControl(self, self.state, self.avatar)
        self.panels = GuiPanels(self, self.state, self.avatar, self.generation)

        self.playback_fps = float(args.fps)
        self.run_animations = False
        self.scene.background_color = SKY_COLOR

        self.panels.install()
        self._add_rest_pose()

    def _add_rest_pose(self) -> None:
        state = self.state
        if not state.model_dir:
            return
        try:
            verts, faces = rest_pose_body(
                FitConfig(model_dir=state.model_dir, gender=state.gender), "cpu"
            )
            state.nodes.show_rest(
                Meshes(verts, faces, color=state.skin_color, name="Avatar (rest)")
            )
            state.buf.need_reset_view = True
        except Exception as exc:
            print(f"[warn] rest-pose SMPL-X unavailable ({exc}); skeleton-only until first prompt")
            state.model_dir = ""
            state.fit_body = False

    def gui_scene(self) -> None:
        self.panels.gui_scene()

    def gui_playback(self) -> None:
        self.panels.gui_playback()

    def on_render(self, time, frame_time, **kwargs) -> None:
        self.generation._tick_live()
        self.display._consume()
        self.display._sync_display_nodes()
        self.display._update_follow_cam()
        super().on_render(time, frame_time, **kwargs)
