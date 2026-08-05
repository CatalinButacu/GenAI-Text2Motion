import pytest

from text2motion.render.studio_viewer.avatar import Avatar
from text2motion.render.studio_viewer.display import Display
from text2motion.render.studio_viewer.generation import GenerationControl
from text2motion.render.studio_viewer.gui_panels import GuiPanels
from text2motion.render.studio_viewer.scene_nodes import (
    FitState,
    LiveRegen,
    SceneNodes,
    StreamBuffers,
)


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
        self.buf = StreamBuffers()
        self.fit = FitState()
        self.live = LiveRegen()
        self.nodes = SceneNodes(FakeScene())
        self.follow_cam = True
        self.skin_color = (1.0, 1.0, 1.0)


@pytest.fixture
def wired():
    host, state = FakeHost(), FakeState()
    avatar = Avatar(host, state)
    generation = GenerationControl(host, state, avatar)
    panels = GuiPanels(host, state, avatar, generation)
    return host, state, Display(host, state), avatar, generation, panels


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


def test_display_consume_runs_against_a_stub_host(wired):
    host, state, display, _, _, _ = wired
    display._consume()  # empty queues: must be a clean no-op, not an attribute error
    assert state.buf.drain() == ([], [], None)


def test_viewer_no_longer_uses_mixin_inheritance():
    from aitviewer.viewer import Viewer

    from text2motion.render.studio_viewer.viewer import StreamingStudioViewer

    assert StreamingStudioViewer.__bases__ == (Viewer,)
    for hook in ("gui_scene", "gui_playback", "on_render"):
        assert hook in StreamingStudioViewer.__dict__  # aitviewer hooks delegate to components
