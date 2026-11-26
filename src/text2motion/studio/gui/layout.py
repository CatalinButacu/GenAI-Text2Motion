from __future__ import annotations

from types import MappingProxyType

import imgui

_COLOR_SLOTS = MappingProxyType(
    {
        "window_background": imgui.COLOR_WINDOW_BACKGROUND,
        "title_background": imgui.COLOR_TITLE_BACKGROUND,
        "title_background_active": imgui.COLOR_TITLE_BACKGROUND_ACTIVE,
        "frame_background": imgui.COLOR_FRAME_BACKGROUND,
        "frame_background_hovered": imgui.COLOR_FRAME_BACKGROUND_HOVERED,
        "frame_background_active": imgui.COLOR_FRAME_BACKGROUND_ACTIVE,
        "button": imgui.COLOR_BUTTON,
        "button_hovered": imgui.COLOR_BUTTON_HOVERED,
        "button_active": imgui.COLOR_BUTTON_ACTIVE,
        "header": imgui.COLOR_HEADER,
        "header_hovered": imgui.COLOR_HEADER_HOVERED,
        "header_active": imgui.COLOR_HEADER_ACTIVE,
        "slider_grab": imgui.COLOR_SLIDER_GRAB,
        "slider_grab_active": imgui.COLOR_SLIDER_GRAB_ACTIVE,
        "check_mark": imgui.COLOR_CHECK_MARK,
        "separator": imgui.COLOR_SEPARATOR,
        "plot_lines": imgui.COLOR_PLOT_LINES,
        "plot_histogram": imgui.COLOR_PLOT_HISTOGRAM,
    }
)

WINDOW_FLAGS = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE

_STYLE_FIELDS = (
    "window_rounding",
    "child_rounding",
    "frame_rounding",
    "grab_rounding",
    "popup_rounding",
    "scrollbar_rounding",
    "window_border_size",
    "window_padding",
    "frame_padding",
    "item_spacing",
)


############################################################
#       PanelLayout
############################################################
class PanelLayout:
    def __init__(
        self,
        state,
    ) -> None:
        self.state = state

    def rects(self):
        layout = self.state.config.layout
        w, h = imgui.get_io().display_size
        top = imgui.get_frame_height()
        left_w = w * layout.left_width
        right_w = w * layout.right_width
        prompt_h = h * layout.prompt_height
        editor_h = (h - top) * layout.editor_height
        play_h = (h - top) * layout.playback_height
        editor = (0.0, top, left_w, editor_h)
        env = (0.0, top + editor_h, left_w, h - top - editor_h)
        params = (w - right_w, top, right_w, h - top - play_h)
        playback = (w - right_w, h - play_h, right_w, play_h)
        prompt = (left_w, h - prompt_h, w - left_w - right_w, prompt_h)
        return editor, env, params, playback, prompt

    def dock(self, rect) -> None:
        x, y, w, h = rect
        imgui.set_next_window_position(x, y, imgui.ALWAYS)
        imgui.set_next_window_size(w, h, imgui.ALWAYS)

    def apply_style(self) -> None:
        if self.state.styled:
            return
        self.state.styled = True
        theme = self.state.config.theme
        style = imgui.get_style()
        for name in _STYLE_FIELDS:
            setattr(style, name, getattr(theme, name))
        for name, rgba in theme.colors.items():
            slot = _COLOR_SLOTS.get(name)
            if slot is None:
                raise KeyError(
                    f"unknown imgui colour slot {name!r}; known slots: {sorted(_COLOR_SLOTS)}"
                )
            style.colors[slot] = rgba

    def section(self, label: str) -> None:
        imgui.spacing()
        imgui.push_style_color(imgui.COLOR_TEXT, *self.state.config.theme.section_label_color)
        imgui.text(label.upper())
        imgui.pop_style_color()
        imgui.separator()

    def dim_text(self, text: str) -> None:
        imgui.push_style_color(imgui.COLOR_TEXT, 0.62, 0.66, 0.74, 1.0)
        imgui.text_wrapped(text)
        imgui.pop_style_color()
