from __future__ import annotations

import logging
import traceback
from dataclasses import replace
from typing import Any

import imgui
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer

from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.config import StudioConfig
from text2motion.studio.contracts import (
    MotionInferenceClientPort,
    StudioLaunchArguments,
)
from text2motion.studio.fitting import create_smplx_rest_pose_mesh
from text2motion.studio.generation import (
    StudioGenerationController,
    log_action,
    log_result,
    log_toggle,
)
from text2motion.studio.panels import GuiPanels
from text2motion.studio.scene import StreamSceneUpdater
from text2motion.studio.session import StudioSession

logger = logging.getLogger(__name__)


############################################################
#       StreamingStudioViewer
############################################################
class StreamingStudioViewer(Viewer):
    scene: Any

    def __init__(
        self,
        client: MotionInferenceClientPort,
        model_dir: str,
        device: str,
        args: StudioLaunchArguments,
        config: StudioConfig,
        reload_config=None,
    ) -> None:
        super().__init__()
        self._reload_config = reload_config
        self.state = StudioSession(client, model_dir, device, args, self.scene, config)
        self.scene_updater = StreamSceneUpdater(self, self.state)
        self.avatar = SmplxAvatarController(self, self.state)
        self.generation = StudioGenerationController(self, self.state, self.avatar)
        self.panels = GuiPanels(self, self.state, self.avatar, self.generation)

        self.playback_fps = float(args.fps) * self.state.upsample
        self.run_animations = False
        self.scene.background_color = config.appearance.sky_color

        self.panels.install()
        self.__add_rest_pose()

    def __add_rest_pose(self) -> None:
        state = self.state
        if not state.model_dir:
            return
        try:
            verts, faces = create_smplx_rest_pose_mesh(
                replace(state.config.fit, model_dir=state.model_dir, gender=state.gender), "cpu"
            )
            state.nodes.show_rest(
                Meshes(verts, faces, color=state.skin_color, name="Avatar (rest)")
            )
            state.stream_buffers.need_reset_view = True
        except Exception as exc:
            logger.info(
                f"[warn] rest-pose SMPL-X unavailable ({exc}); skeleton-only until first prompt"
            )
            state.model_dir = ""
            state.fit_body = False

    def gui_scene(self) -> None:
        self.panels.gui_scene()

    def gui_playback(self) -> None:
        self.panels.gui_playback()

    def reload_settings(self) -> None:
        if self._reload_config is None:
            self.state.load_status = "no config file to reload"
            return
        try:
            self.state.load_status = self.state.apply_config(self._reload_config())
            log_result("reload settings", True, self.state.load_status)
        except Exception as exc:
            traceback.print_exc()
            log_result("reload settings", False, f"{type(exc).__name__}: {exc}")
            self.state.load_status = f"reload failed: {type(exc).__name__}: {exc}"

    def key_event(self, key, action, modifiers) -> None:
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS or imgui.get_io().want_capture_keyboard:
            return
        if key == self.wnd.keys.H:
            self.state.show_skin = not self.state.show_skin
            log_toggle("skin [H]", self.state.show_skin)
        elif key == self.wnd.keys.R:
            log_action("key R")
            self.reload_settings()

    def __advance_playhead(self, time) -> None:
        if not self.run_animations or self.scene.n_frames == 0:
            return
        frames = int((time - self._last_frame_rendered_at) * self.playback_fps)
        if frames <= 0:
            return
        if self.playback_without_skipping:
            frames = 1
        self.scene.current_frame_id = (self.scene.current_frame_id + frames) % self.scene.n_frames
        self._last_frame_rendered_at += frames * (1.0 / self.playback_fps)

    def on_render(self, time, frame_time, export=False, transparent_background=False) -> None:
        self.generation.tick_live()
        self.scene_updater.consume()
        self.__advance_playhead(time)
        self.scene_updater.hold_playhead_at_fitted_body()
        self.scene_updater.sync_display_nodes()
        self.scene_updater.update_follow_cam()
        super().on_render(
            time, frame_time, export=export, transparent_background=transparent_background
        )
