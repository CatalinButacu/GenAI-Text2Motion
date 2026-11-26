from __future__ import annotations

import imgui

from text2motion.studio.contracts import ViewerHost
from text2motion.studio.gui.layout import WINDOW_FLAGS, PanelLayout
from text2motion.studio.live.actions import log_toggle


############################################################
#       ScenePanel
############################################################
class ScenePanel:
    def __init__(
        self,
        host: ViewerHost,
        layout: PanelLayout,
    ) -> None:
        self.host = host
        self.layout = layout

    def gui_scene(self) -> None:
        self.layout.apply_style()
        editor, *_ = self.layout.rects()
        self.layout.dock(editor)
        expanded, _ = imgui.begin("Scene", False, WINDOW_FLAGS)
        if expanded:
            self.host.scene.gui_editor(imgui, self.host.viewports, self.host.viewport_mode)
        imgui.end()


############################################################
#       EnvironmentPanel
############################################################
class EnvironmentPanel:
    def __init__(
        self,
        host: ViewerHost,
        layout: PanelLayout,
    ) -> None:
        self.host = host
        self.layout = layout

    def gui_environment(self) -> None:
        _, env, *_ = self.layout.rects()
        self.layout.dock(env)
        imgui.begin("Environment", False, WINDOW_FLAGS)

        self.scene_controls()
        self.layout.section("lights")
        for i, light in enumerate(self.host.scene.lights):
            self.light_controls(i, light)
        self.floor_controls()
        imgui.end()

    def scene_controls(self) -> None:
        self.layout.section("scene")
        changed, color = imgui.color_edit4("background", *self.host.scene.background_color)
        if changed:
            self.host.scene.background_color = color
        _, self.host.scene.ambient_strength = imgui.slider_float(
            "ambient", self.host.scene.ambient_strength, 0.0, 4.0
        )
        imgui.text("mood:")
        for mood in ("default", "dark", "diffuse"):
            imgui.same_line()
            if imgui.small_button(mood):
                self.host.scene.light_mode = mood
        _, self.host.shadows_enabled = imgui.checkbox("shadows [S]", self.host.shadows_enabled)

    def light_controls(self, i: int, light) -> None:
        flags = imgui.TREE_NODE_DEFAULT_OPEN if i == 0 else 0
        if not imgui.tree_node(f"light {i + 1}##light{i}", flags):
            return
        _, light.strength = imgui.slider_float(f"strength##l{i}", light.strength, 0.0, 10.0)
        changed, elevation = imgui.slider_float(f"elevation##l{i}", light.elevation, -90.0, 90.0)
        if changed:
            light.elevation = elevation
        changed, azimuth = imgui.slider_float(f"azimuth##l{i}", light.azimuth, 0.0, 360.0)
        if changed:
            light.azimuth = azimuth
        changed, col = imgui.color_edit3(f"color##l{i}", *light.light_color[:3])
        if changed:
            light.light_color = col
        _, light.shadow_enabled = imgui.checkbox(f"cast shadows##l{i}", light.shadow_enabled)
        imgui.tree_pop()

    def floor_controls(self) -> None:
        self.layout.section("floor & origin")
        floor = self.host.scene.floor
        changed, floor.enabled = imgui.checkbox("show floor", floor.enabled)
        if changed:
            log_toggle("show floor", floor.enabled)
        changed, col = imgui.color_edit3("tile a", *[float(v) for v in floor.c1[:3]])
        if changed:
            floor.c1[:3] = col
        changed, col = imgui.color_edit3("tile b", *[float(v) for v in floor.c2[:3]])
        if changed:
            floor.c2[:3] = col
        _, self.host.scene.origin.enabled = imgui.checkbox(
            "origin axes", self.host.scene.origin.enabled
        )
