from __future__ import annotations

import imgui

from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.generation import log_action, log_toggle
from text2motion.studio.gui.layout import PanelLayout


############################################################
#       AvatarSection
############################################################
class AvatarSection:
    def __init__(
        self,
        state,
        avatar: SmplxAvatarController,
        layout: PanelLayout,
    ) -> None:
        self.state = state
        self.avatar = avatar
        self.layout = layout

    def avatar_controls(self) -> None:
        self.layout.section("avatar")
        self.skin_toggle()
        self.gender_selector()
        self.skin_color_editor()
        self.shape_controls()
        if self.state.smplx_fit.note:
            self.layout.dim_text(self.state.smplx_fit.note)

    def skin_toggle(self) -> None:
        clicked, self.state.fit_body = imgui.checkbox(
            "avatar skin (SMPL-X body)", self.state.fit_body
        )
        if not self.state.model_dir:
            self.layout.dim_text("SMPL-X models not found: skeleton only.")
        if not clicked:
            return
        log_toggle("avatar skin (fit body)", self.state.fit_body)
        if self.state.fit_body and not self.state.model_dir:
            self.state.fit_body = False
            return
        self.avatar.on_fit_body_toggled()

    def gender_selector(self) -> None:
        genders = self.state.config.genders
        for i, name in enumerate(genders):
            if imgui.radio_button(name, self.state.gender_idx == i) and self.state.gender_idx != i:
                self.state.gender_idx = i
                self.avatar.on_avatar_changed()
            if i < len(genders) - 1:
                imgui.same_line()

    def skin_color_editor(self) -> None:
        changed, col = imgui.color_edit3("skin", *self.state.skin_color[:3])
        if not changed:
            return
        self.state.skin_color = (col[0], col[1], col[2], 1.0)
        for node in (self.state.nodes.body, self.state.nodes.rest):
            if node is not None:
                node.color = self.state.skin_color

    def shape_controls(self) -> None:
        self.beta_slider(0, "height")
        self.beta_slider(1, "build")
        if imgui.collapsing_header("more shape dimensions")[0]:
            for i in range(2, self.state.config.fit.num_betas):
                self.beta_slider(i, f"shape {i}")
        half = imgui.get_content_region_available()[0] / 2 - 4
        if imgui.button("reset shape", width=half) and self.state.user_betas.any():
            log_action("press reset shape")
            self.state.user_betas[:] = 0.0
            self.avatar.on_avatar_changed()
        imgui.same_line()
        if imgui.button("refresh avatar", width=half):
            log_action("press refresh avatar")
            self.avatar.on_avatar_changed()

    def beta_slider(self, i: int, label: str) -> None:
        changed, value = imgui.slider_float(label, float(self.state.user_betas[i]), -3.0, 3.0)
        if changed:
            self.state.user_betas[i] = value
        if imgui.is_item_deactivated_after_edit():
            self.avatar.on_avatar_changed()
