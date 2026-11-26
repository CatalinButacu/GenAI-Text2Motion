from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.gui import sampling
from text2motion.studio.gui.panels import GuiPanels
from text2motion.studio.live.controller import StudioGenerationController
from text2motion.studio.scene import StreamSceneUpdater, StudioSceneNodes
from text2motion.studio.session.buffers import StudioStreamBuffers
from text2motion.studio.session.debounce import LiveGenerationDebouncer
from text2motion.studio.session.fit_state import SmplxFitState


class FakeScene:
    def __init__(self) -> None:
        self.nodes = []
        self.selected_object = None
        self.current_frame_id = 0

    def add(self, node) -> None:
        self.nodes.append(node)

    def remove(self, node) -> None:
        if node in self.nodes:
            self.nodes.remove(node)

    def select(self, node) -> None:
        self.selected_object = node


class FakeHost:
    def __init__(self) -> None:
        self.scene = FakeScene()
        self.gui_controls = {}
        self.animated = None

    def toggle_animation(self, on) -> None:
        self.animated = on

    def center_view_on_node(self, node) -> None:
        self.centered = node


class FakeState:
    def __init__(self) -> None:
        self.stream_buffers = StudioStreamBuffers()
        self.smplx_fit = SmplxFitState()
        self.live = LiveGenerationDebouncer()
        self.nodes = StudioSceneNodes(FakeScene())
        self.follow_cam = True
        self.skin_color = (1.0, 1.0, 1.0)


@pytest.fixture
def wired():
    host, state = FakeHost(), FakeState()
    viewer_host = cast(ViewerHost, host)
    avatar = SmplxAvatarController(viewer_host, state)
    generation = StudioGenerationController(viewer_host, state, avatar)
    panels = GuiPanels(viewer_host, state, avatar, generation)
    return host, state, StreamSceneUpdater(viewer_host, state), avatar, generation, panels


def test_components_construct_without_a_real_viewer(wired):
    host, state, display, avatar, generation, panels = wired
    for component in (display, avatar, generation, panels):
        assert component.host is host
        assert component.state is state


def test_collaborators_are_explicit_not_inherited(wired):
    _, _, display, avatar, generation, panels = wired

    assert generation.avatar is avatar  # generation reaches avatar by reference, not by MRO
    assert panels.avatar is avatar
    assert panels.generation is generation
    assert not hasattr(display, "avatar")  # display never needed avatar; it must not see it


def test_no_component_inherits_from_another(wired):
    _, _, display, avatar, generation, panels = wired
    types_ = [type(display), type(avatar), type(generation), type(panels)]
    for a in types_:
        for b in types_:
            if a is not b:
                assert not issubclass(a, b)


def test_panels_install_registers_the_three_docked_panels(wired):
    host, _, _, _, _, panels = wired
    panels.install()
    assert sorted(host.gui_controls) == ["environment", "params", "prompt"]


def test_each_panel_is_composed_on_the_stub_host_with_one_shared_layout(wired):
    host, state, _, avatar, generation, panels = wired
    params = panels.params
    hosted = (panels.scene, panels.environment, params, params.sampling, panels.playback)
    drawn = (*hosted, params.avatar_section, panels.prompt)
    assert all(panel.host is host for panel in hosted)
    assert all(panel.layout is panels.scene.layout for panel in drawn)
    assert panels.scene.layout.state is state
    assert params.avatar_section.avatar is avatar
    assert params.generation is generation
    assert panels.prompt.generation is generation


def test_install_binds_each_docked_key_to_its_panel(wired):
    host, _, _, _, _, panels = wired
    panels.install()
    assert host.gui_controls == {
        "environment": panels.environment.gui_environment,
        "params": panels.params.gui_params,
        "prompt": panels.prompt.gui_prompt,
    }


def test_run_forever_toggle_draws_against_a_stub_host(wired, monkeypatch):
    imgui = MagicMock()
    imgui.checkbox.return_value = (True, False)
    monkeypatch.setattr(sampling, "imgui", imgui)
    section = wired[5].params.sampling
    section.state = SimpleNamespace(steps=0, default_steps=50)
    section.forever_toggle()
    assert section.state.steps == 50


def test_display_consume_runs_against_a_stub_host(wired):
    host, state, display, _, _, _ = wired
    display.consume()  # empty queues: must be a clean no-op, not an attribute error
    assert state.stream_buffers.drain() == ([], [], None)


def test_viewer_no_longer_uses_mixin_inheritance():
    from aitviewer.viewer import Viewer

    from text2motion.studio.viewer import StreamingStudioViewer

    assert StreamingStudioViewer.__bases__ == (Viewer,)
    for hook in ("gui_scene", "gui_playback", "on_render"):
        assert hook in StreamingStudioViewer.__dict__  # aitviewer hooks delegate to components
