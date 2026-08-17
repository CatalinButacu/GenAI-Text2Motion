from text2motion.studio.scene import SceneNodes


class FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeScene:
    def __init__(self) -> None:
        self.nodes: list[FakeNode] = []
        self.selected_object = None

    def add(self, node) -> None:
        self.nodes.append(node)

    def remove(self, node) -> None:
        if node in self.nodes:
            self.nodes.remove(node)

    def select(self, node) -> None:
        self.selected_object = node


def make() -> tuple[SceneNodes, FakeScene]:
    scene = FakeScene()
    return SceneNodes(scene), scene


def test_show_replaces_previous_node_in_scene():
    nodes, scene = make()
    first = FakeNode("motion-1")
    second = FakeNode("motion-2")

    nodes.show_motion(first)
    nodes.show_motion(second)

    assert nodes.motion is second
    assert scene.nodes == [second]  # the first was removed, not leaked


def test_clear_deselects_when_the_node_is_selected():
    nodes, scene = make()
    body = FakeNode("body")
    nodes.show_body(body)
    scene.select(body)

    nodes.clear_body()

    assert nodes.body is None
    assert scene.selected_object is None  # stale selection must not survive removal
    assert scene.nodes == []


def test_clear_all_empties_every_slot():
    nodes, scene = make()
    nodes.show_rest(FakeNode("rest"))
    nodes.show_motion(FakeNode("motion"))
    nodes.show_body(FakeNode("body"))

    nodes.clear_all()

    assert (nodes.rest, nodes.motion, nodes.body) == (None, None, None)
    assert scene.nodes == []


def test_focus_prefers_body_then_motion_then_rest():
    nodes, _ = make()
    assert nodes.focus is None

    rest = FakeNode("rest")
    nodes.show_rest(rest)
    assert nodes.focus is rest

    motion = FakeNode("motion")
    nodes.show_motion(motion)
    assert nodes.focus is motion

    body = FakeNode("body")
    nodes.show_body(body)
    assert nodes.focus is body


def test_has_motion_or_body_tracks_clears():
    nodes, _ = make()
    assert not nodes.has_motion_or_body

    nodes.show_rest(FakeNode("rest"))
    assert not nodes.has_motion_or_body  # rest alone must not block the rest-pose remesh

    nodes.show_motion(FakeNode("motion"))
    assert nodes.has_motion_or_body

    nodes.clear_motion()
    assert not nodes.has_motion_or_body


def test_clearing_an_empty_slot_is_a_noop():
    nodes, scene = make()
    nodes.clear_all()
    nodes.clear_body()
    assert scene.nodes == []


def test_append_fitted_chunk_keeps_body_and_fit_track_aligned():
    from text2motion.studio.scene import FitState, StreamBuffers

    class FitResult:
        global_orient = "o"
        body_pose = "p"
        transl = "t"
        betas = "b"

    buf, fit = StreamBuffers(), FitState()
    for _ in range(3):
        buf.append_fitted_chunk(fit, "verts", "faces", FitResult())

    assert len(buf.pending_body_chunks) == len(fit.orient) == 3
    assert len(fit.pose) == len(fit.transl) == 3
    assert fit.betas == "b"  # captured once, not overwritten per chunk


def test_reset_fit_track_clears_both_sides_together():
    from text2motion.studio.scene import FitState, StreamBuffers

    class FitResult:
        global_orient = "o"
        body_pose = "p"
        transl = "t"
        betas = "b"

    buf, fit = StreamBuffers(), FitState()
    buf.append_fitted_chunk(fit, "v", "f", FitResult())

    buf.reset_fit_track(fit)

    assert buf.pending_body_chunks == []
    assert (fit.orient, fit.pose, fit.transl, fit.betas) == ([], [], [], None)


def test_drain_empties_buffers_and_returns_contents():
    from text2motion.studio.scene import StreamBuffers

    buf = StreamBuffers()
    buf.push_joints("j")
    buf.push_body("v", "f")
    buf.push_remesh("v", "f", True)

    chunks, body, remesh = buf.drain()

    assert (chunks, body, remesh) == (["j"], [("v", "f")], ("v", "f", True))
    assert buf.drain() == ([], [], None)  # second drain sees an empty queue


def test_live_regen_debounce_waits_then_fires():
    from text2motion.studio.scene import LiveRegen

    live = LiveRegen()
    assert not live.debounce_elapsed(now=0.0, debounce_s=0.5)  # nothing typed yet

    live.mark_dirty("a person walks", now=10.0)
    assert not live.debounce_elapsed(now=10.2, debounce_s=0.5)  # still typing
    assert live.debounce_elapsed(now=10.5, debounce_s=0.5)  # settled


def test_live_regen_consumed_edit_stops_being_ready():
    from text2motion.studio.scene import LiveRegen

    live = LiveRegen()
    live.mark_dirty("jump", now=0.0)
    assert live.debounce_elapsed(now=1.0, debounce_s=0.5)

    live.prev_prompt = ""  # how _tick_live consumes a dispatched edit
    assert not live.debounce_elapsed(now=2.0, debounce_s=0.5)


def test_live_regen_starts_disabled_with_no_signature():
    from text2motion.studio.scene import LiveRegen

    live = LiveRegen()
    assert live.enabled is False
    assert live.signature is None
    assert live.regen_pending is False
    assert live.last_dispatched == ""
