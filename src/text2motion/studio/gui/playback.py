from __future__ import annotations

from array import array

import imgui
import numpy as np

from text2motion.studio.contracts import ViewerHost
from text2motion.studio.generation import log_toggle
from text2motion.studio.gui.layout import WINDOW_FLAGS, PanelLayout


############################################################
#       PlaybackPanel
############################################################
class PlaybackPanel:
    def __init__(
        self,
        host: ViewerHost,
        state,
        layout: PanelLayout,
    ) -> None:
        self.host = host
        self.state = state
        self.layout = layout

    def gui_playback(self) -> None:
        *_, playback, _ = self.layout.rects()
        self.layout.dock(playback)
        imgui.begin("Playback", False, WINDOW_FLAGS)

        self.frame_controls()
        self.camera_controls()
        self.render_rate_plot()
        imgui.end()

    def frame_controls(self) -> None:
        changed, run = imgui.checkbox("play [space]", self.host.run_animations)
        if changed:
            log_toggle("play", run)
            self.host.toggle_animation(run)
        imgui.same_line()
        n_frames = max(1, self.host.scene.n_frames)
        self.layout.dim_text(f"frame {self.host.scene.current_frame_id + 1}/{n_frames}")
        imgui.push_item_width(-1)
        moved, frame = imgui.slider_int(
            "##frame", self.host.scene.current_frame_id, 0, n_frames - 1
        )
        if moved:
            self.host.scene.current_frame_id = frame
        imgui.pop_item_width()
        _, self.host.playback_fps = imgui.drag_float(
            "speed (fps)",
            self.host.playback_fps,
            0.1,
            min_value=1.0,
            max_value=120.0,
            format="%.1f",
        )

    def camera_controls(self) -> None:
        changed, self.state.follow_cam = imgui.checkbox(
            "camera follows avatar", self.state.follow_cam
        )
        if changed:
            log_toggle("camera follows avatar", self.state.follow_cam)
        imgui.same_line()
        if imgui.small_button("center view"):
            self.state.stream_buffers.need_reset_view = True

    def render_rate_plot(self) -> None:
        valid = self.host._past_frametimes[self.host._past_frametimes > 0.0]
        if valid.size > 0:
            fps_avg = 1.0 / float(np.mean(valid))
            imgui.plot_lines(
                f"render {fps_avg:.0f} fps",
                array("f", (1.0 / valid).tolist()),
                scale_min=0.0,
                scale_max=120.0,
                graph_size=(imgui.get_content_region_available()[0] - 110, 24),
            )
