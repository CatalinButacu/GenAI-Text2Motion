"""Logic-level unit tests: duration helpers, clip operations, config consistency.

These tests have zero external dependencies (no GPU, no files, no spaCy model).
They target previously untested code paths and expose concrete implementation
bugs such as the render-fps/motion-fps mismatch.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


class TestExtractDuration(unittest.TestCase):
    """src.modules.understanding.actions.extract_duration"""

    @classmethod
    def setUpClass(cls):
        from src.modules.understanding.actions import extractDuration

        cls.fn = staticmethod(extractDuration)

    def test_seconds_with_for(self):
        self.assertEqual(self.fn("walk for 3s"), 3.0)

    def test_seconds_word(self):
        self.assertEqual(self.fn("run for 2.5 seconds"), 2.5)

    def test_seconds_abbreviation(self):
        self.assertEqual(self.fn("jump for 10 sec"), 10.0)

    def test_minutes_converts_to_seconds(self):
        # "2 minutes" must be scaled to 120 seconds
        result = self.fn("walk for 2 minutes")
        assert result is not None
        self.assertAlmostEqual(result, 120.0)

    def test_minutes_abbreviation(self):
        result = self.fn("run for 1.5 min")
        assert result is not None
        self.assertAlmostEqual(result, 90.0)

    def test_no_duration_returns_none(self):
        self.assertIsNone(self.fn("a person walks"))

    def test_without_for_prefix(self):
        # "3 seconds" without "for" should still match
        self.assertEqual(self.fn("holds pose 3 seconds"), 3.0)

    def test_decimal_seconds(self):
        result = self.fn("for 0.5 seconds")
        assert result is not None
        self.assertAlmostEqual(result, 0.5)


class TestExtractModifier(unittest.TestCase):
    """src.modules.understanding.actions.extract_modifier"""

    @classmethod
    def setUpClass(cls):
        from src.modules.understanding.actions import extractModifier

        cls.fn = staticmethod(extractModifier)

    def test_single_modifier(self):
        self.assertEqual(self.fn("the person walks quickly"), "quickly")

    def test_no_modifier(self):
        self.assertEqual(self.fn("a person walks"), "")

    def test_multiple_modifiers_are_alphabetically_sorted(self):
        # extract_modifier sorts matches for determinism
        result = self.fn("runs quickly and violently")
        self.assertEqual(result, "quickly violently")

    def test_modifier_reversed_input_same_output(self):
        # Regardless of input order, output is alphabetical
        self.assertEqual(
            self.fn("runs violently and quickly"),
            self.fn("runs quickly and violently"),
        )

    def test_case_insensitive(self):
        self.assertEqual(self.fn("Quickly runs away"), "quickly")


class TestComputeSceneDuration(unittest.TestCase):
    """src.modules.understanding.actions.compute_scene_duration"""

    @classmethod
    def setUpClass(cls):
        from src.modules.understanding.actions import computeSceneDuration
        from src.modules.understanding.models import ParsedAction

        cls.fn = staticmethod(computeSceneDuration)
        cls.PA = ParsedAction

    def _action(self, order: int, duration):
        return self.PA(actionType="walk", actor="p", order=order, duration=duration)

    def test_all_none_returns_default(self):
        actions = [self._action(0, None), self._action(1, None)]
        total, explicit = self.fn(actions)
        self.assertEqual(total, 5.0)
        self.assertFalse(explicit)

    def test_empty_list_returns_default(self):
        total, explicit = self.fn([])
        self.assertEqual(total, 5.0)
        self.assertFalse(explicit)

    def test_single_explicit_duration(self):
        total, explicit = self.fn([self._action(0, 3.0)])
        self.assertEqual(total, 3.0)
        self.assertTrue(explicit)

    def test_sequential_explicit_durations_summed(self):
        # Two sequential actions (different orders) -> sum
        actions = [self._action(0, 2.0), self._action(1, 4.0)]
        total, explicit = self.fn(actions)
        self.assertAlmostEqual(total, 6.0)
        self.assertTrue(explicit)

    def test_concurrent_actions_take_max(self):
        # Two concurrent actions (same order) -> max, not sum
        actions = [self._action(0, 3.0), self._action(0, 7.0)]
        total, explicit = self.fn(actions)
        self.assertAlmostEqual(total, 7.0)
        self.assertTrue(explicit)

    def test_mixed_order_skips_implicit_slots(self):
        """When order=1 slot has no explicit duration it is excluded from the sum.

        This means the returned total only covers the explicitly-timed portion
        of the timeline, not the full scene.  M2/M4 must handle the implicit
        remainder.
        """
        actions = [self._action(0, 2.0), self._action(1, None)]
        total, explicit = self.fn(actions)
        # Only the explicit slot (order 0 = 2.0 s) is counted.
        self.assertAlmostEqual(total, 2.0)
        self.assertTrue(explicit)
        # NOTE: order=1 needs ~base_duration more seconds -- the planner must
        # account for this rather than relying solely on scene.duration.


# ---------------------------------------------------------------------------
# Clause splitting
# ---------------------------------------------------------------------------


class TestSplitIntoClauses(unittest.TestCase):
    """src.modules.understanding.parsing_utils.split_into_clauses"""

    @classmethod
    def setUpClass(cls):
        from src.modules.understanding.parsing_utils import splitIntoClauses

        cls.fn = staticmethod(splitIntoClauses)

    def test_no_marker_single_clause(self):
        clauses = self.fn("a person walks")
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0][0], "a person walks")
        self.assertFalse(clauses[0][1])

    def test_then_makes_two_sequential(self):
        clauses = self.fn("a person walks then runs")
        self.assertEqual(len(clauses), 2)
        self.assertFalse(clauses[0][1])
        self.assertFalse(clauses[1][1])

    def test_while_marks_concurrent(self):
        clauses = self.fn("person walks while ball rolls")
        self.assertEqual(len(clauses), 2)
        self.assertTrue(clauses[1][1])

    def test_simultaneous_marks_concurrent(self):
        clauses = self.fn("a kicks b simultaneously c dances")
        concurrent_flags = [c[1] for c in clauses]
        self.assertIn(True, concurrent_flags)

    def test_followed_by_sequential(self):
        clauses = self.fn("person waves followed by person bows")
        self.assertEqual(len(clauses), 2)
        self.assertFalse(clauses[1][1])

    def test_empty_string_returns_one_clause(self):
        clauses = self.fn("")
        # Fallback path: returns [(original_text, False)]
        self.assertEqual(len(clauses), 1)


# ---------------------------------------------------------------------------
# Clip operations (no GPU, no checkpoints)
# ---------------------------------------------------------------------------


def _make_clip(
    nFrames: int, action: str = "walk", fps: int = 30, nJoints: int = 22, rawJoints: bool = True
):
    """Create a minimal MotionClip with synthetic numpy data."""
    from src.modules.motion.models import MotionClip, MotionSource

    smplx = np.zeros((nFrames, 168), dtype=np.float32)
    joints = np.zeros((nFrames, nJoints, 3), dtype=np.float32) if rawJoints else None
    return MotionClip(
        action=action,
        smplxParams=smplx,
        fps=fps,
        source=MotionSource.SSM,
        rawJoints=joints,
    )


class TestLastPoseOf(unittest.TestCase):
    """src.modules.motion.clip_ops.last_pose_of"""

    @classmethod
    def setUpClass(cls):
        from src.modules.motion.clip_ops import lastPoseOf

        cls.fn = staticmethod(lastPoseOf)

    def test_none_clip_returns_none(self):
        self.assertIsNone(self.fn(None))

    def test_none_params_returns_none(self):
        from src.modules.motion.models import MotionClip

        clip = MotionClip(action="walk", smplxParams=np.zeros((0, 168)))
        self.assertIsNone(self.fn(clip))

    def test_empty_params_returns_none(self):
        clip = _make_clip(0)
        self.assertIsNone(self.fn(clip))

    def test_valid_clip_returns_last_frame(self):
        clip = _make_clip(5)
        clip.smplxParams[4, 0] = 99.0  # mark last frame
        result = self.fn(clip)
        assert result is not None
        self.assertEqual(result[0], 99.0)

    def test_returns_array_not_copy_issue(self):
        """last_pose_of uses [-1] indexing, ensure correct shape."""
        clip = _make_clip(3)
        result = self.fn(clip)
        assert result is not None
        self.assertEqual(result.shape, (168,))


class TestBlendClips(unittest.TestCase):
    """src.modules.motion.clip_ops.blend_clips"""

    @classmethod
    def setUpClass(cls):
        from src.modules.motion.clip_ops import blendClips

        cls.fn = staticmethod(blendClips)

    def test_two_clips_concatenated_without_blend(self):
        c1 = _make_clip(10)
        c2 = _make_clip(8)
        result = self.fn([c1, c2], blendFrames=0)
        self.assertEqual(result.numFrames, 18)

    def test_blend_shortens_total_by_blend_frames(self):
        # blend_frames=4 -> 4 frames overlap -> total = 10+8-4 = 14
        c1 = _make_clip(10)
        c2 = _make_clip(8)
        result = self.fn([c1, c2], blendFrames=4)
        self.assertEqual(result.numFrames, 14)

    def test_action_label_joined(self):
        c1 = _make_clip(5, action="walk")
        c2 = _make_clip(5, action="run")
        result = self.fn([c1, c2], blendFrames=0)
        self.assertEqual(result.action, "walk then run")

    def test_fps_preserved_from_first_clip(self):
        c1 = _make_clip(10, fps=30)
        c2 = _make_clip(10, fps=30)
        result = self.fn([c1, c2], blendFrames=0)
        self.assertEqual(result.fps, 30)

    def test_raw_joints_concatenated_when_all_present(self):
        c1 = _make_clip(5, rawJoints=True)
        c2 = _make_clip(7, rawJoints=True)
        result = self.fn([c1, c2], blendFrames=0)
        assert result.rawJoints is not None
        self.assertEqual(result.rawJoints.shape[0], 12)

    def test_raw_joints_dropped_when_any_missing(self):
        c1 = _make_clip(5, rawJoints=True)
        c2 = _make_clip(5, rawJoints=False)
        result = self.fn([c1, c2], blendFrames=0)
        self.assertIsNone(result.rawJoints)

    def test_source_is_sequenced(self):
        from src.modules.motion.models import MotionSource

        c1 = _make_clip(5)
        c2 = _make_clip(5)
        result = self.fn([c1, c2], blendFrames=0)
        self.assertEqual(result.source, MotionSource.SEQUENCED)


class TestSequenceClips(unittest.TestCase):
    """src.modules.motion.clip_ops.sequence_clips"""

    @classmethod
    def setUpClass(cls):
        from src.modules.motion.clip_ops import sequenceClips

        cls.fn = staticmethod(sequenceClips)

    def test_single_clip_per_actor_passthrough(self):
        c1 = _make_clip(10)
        result = self.fn([("alice", c1)], blendFrames=0)
        self.assertIs(result["alice"], c1)

    def test_multiple_actors_split_correctly(self):
        c1 = _make_clip(10)
        c2 = _make_clip(8)
        result = self.fn([("alice", c1), ("bob", c2)], blendFrames=0)
        self.assertIn("alice", result)
        self.assertIn("bob", result)
        self.assertIs(result["alice"], c1)
        self.assertIs(result["bob"], c2)

    def test_same_actor_multiple_clips_blended(self):
        c1 = _make_clip(10)
        c2 = _make_clip(8)
        result = self.fn([("alice", c1), ("alice", c2)], blendFrames=0)
        self.assertEqual(result["alice"].numFrames, 18)

    def test_empty_returns_empty_dict(self):
        result = self.fn([], blendFrames=0)
        self.assertEqual(result, {})


class TestCrossfadeArrays(unittest.TestCase):
    """src.modules.motion.clip_ops.crossfade_arrays"""

    @classmethod
    def setUpClass(cls):
        from src.modules.motion.clip_ops import crossfadeArrays

        cls.fn = staticmethod(crossfadeArrays)

    def _run(self, nExisting, nNew, blend):
        f_existing = np.ones((nExisting, 168), dtype=np.float32)
        f_new = np.ones((nNew, 168), dtype=np.float32) * 2.0
        parts = [f_existing]
        jparts = []
        self.fn(parts, jparts, f_new, None, blend)
        total = sum(len(p) for p in parts)
        return total

    def test_zero_blend_just_appends(self):
        total = self._run(10, 8, blend=0)
        self.assertEqual(total, 18)

    def test_blend_reduces_total_length(self):
        # n=min(4, 10, 8)=4 frames of overlap -> total = 10+8-4 = 14
        total = self._run(10, 8, blend=4)
        self.assertEqual(total, 14)

    def test_blend_larger_than_new_uses_new_length(self):
        # blend=20 but new has only 5 frames -> n=min(20,10,5)=5 -> total=10+5-5=10
        total = self._run(10, 5, blend=20)
        self.assertEqual(total, 10)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


class TestBackfillActorTargets(unittest.TestCase):
    """src.modules.understanding.actions.backfill_actor_targets"""

    @classmethod
    def setUpClass(cls):
        from src.modules.understanding.actions import backfillActorTargets
        from src.modules.understanding.models import ParsedAction, ParsedEntity

        cls.fn = staticmethod(backfillActorTargets)
        cls.PA = ParsedAction
        cls.PE = ParsedEntity

    def test_fills_empty_actor(self):
        actor = self.PE(name="alice", objectType="humanoid", isActor=True)
        a = self.PA(actionType="walk", actor="")
        self.fn([a], [actor])
        self.assertEqual(a.actor, "alice")

    def test_preserves_existing_actor(self):
        actor = self.PE(name="alice", objectType="humanoid", isActor=True)
        a = self.PA(actionType="walk", actor="bob")
        self.fn([a], [actor])
        self.assertEqual(a.actor, "bob")

    def test_fills_missing_target_when_required(self):
        from src.shared.vocabulary import ACTIONS

        # Find an action that requires_target
        target_action = next((name for name, d in ACTIONS.items() if d.requiresTarget), None)
        if target_action is None:
            self.skipTest("no requires_target action found in vocabulary")
        actor = self.PE(name="person", objectType="humanoid", isActor=True)
        obj = self.PE(name="cube", objectType="sphere", isActor=False)
        a = self.PA(actionType=target_action, actor="person", target="")
        self.fn([a], [actor, obj])
        self.assertEqual(a.target, "cube")

    def test_no_actors_leaves_actor_empty(self):
        a = self.PA(actionType="walk", actor="")
        self.fn([a], [])
        self.assertEqual(a.actor, "")


class TestPropagateRename(unittest.TestCase):
    """src.modules.understanding.actions.propagate_rename"""

    @classmethod
    def setUpClass(cls):
        from src.modules.understanding.actions import propagateRename
        from src.modules.understanding.models import ParsedAction

        cls.fn = staticmethod(propagateRename)
        cls.PA = ParsedAction

    def test_renames_actor(self):
        a = self.PA(actionType="walk", actor="humanoid")
        self.fn([a], "humanoid", "humanoid_1")
        self.assertEqual(a.actor, "humanoid_1")

    def test_renames_target(self):
        a = self.PA(actionType="kick", actor="person", target="ball")
        self.fn([a], "ball", "ball_1")
        self.assertEqual(a.target, "ball_1")

    def test_no_match_leaves_unchanged(self):
        a = self.PA(actionType="walk", actor="alice", target="")
        self.fn([a], "bob", "bob_1")
        self.assertEqual(a.actor, "alice")

    def test_empty_action_list_no_error(self):
        self.fn([], "humanoid", "humanoid_1")  # must not raise


# ---------------------------------------------------------------------------
# Compute action frames
# ---------------------------------------------------------------------------


class TestComputeActionFrames(unittest.TestCase):
    """src.modules.motion.clip_ops.compute_action_frames"""

    @classmethod
    def setUpClass(cls):
        from src.modules.motion.clip_ops import computeActionFrames
        from src.modules.motion.config import MotionConfig
        from src.modules.understanding.models import ParsedAction

        cls.fn = staticmethod(computeActionFrames)
        cls.cfg = MotionConfig(minActionFrames=20)
        cls.PA = ParsedAction

    def test_explicit_duration_converted_to_frames(self):
        from src.shared.constants import MOTION_FPS

        a = self.PA(actionType="walk", duration=2.0)
        frames = self.fn(a, totalFrames=100, nActions=5, config=self.cfg)
        self.assertEqual(frames, max(int(2.0 * MOTION_FPS), 20))

    def test_implicit_duration_uses_budget(self):
        a = self.PA(actionType="walk", duration=None)
        frames = self.fn(a, totalFrames=100, nActions=5, config=self.cfg)
        self.assertEqual(frames, 20)

    def test_budget_too_small_returns_min_frames(self):
        a = self.PA(actionType="walk", duration=None)
        frames = self.fn(a, totalFrames=10, nActions=5, config=self.cfg)
        self.assertEqual(frames, 20)  # 10//5=2, clamped to min=20

    def test_explicit_duration_too_short_returns_min_frames(self):
        a = self.PA(actionType="walk", duration=0.1)
        frames = self.fn(a, totalFrames=100, nActions=1, config=self.cfg)
        self.assertEqual(frames, 20)  # int(0.1*30)=3, clamped to 20


# ---------------------------------------------------------------------------
# Build action query
# ---------------------------------------------------------------------------


class TestBuildActionQuery(unittest.TestCase):
    """src.modules.motion.clip_ops.build_action_query"""

    @classmethod
    def setUpClass(cls):
        from src.modules.motion.clip_ops import buildActionQuery

        cls.fn = staticmethod(buildActionQuery)

    def _action(self, rawText="", actionType="walk", modifier=""):
        from src.modules.understanding.models import ParsedAction

        return ParsedAction(actionType=actionType, rawText=rawText, modifier=modifier)

    def _act_def(self, motionClip=None):
        return types.SimpleNamespace(motionClip=motionClip)

    def test_raw_text_takes_priority(self):
        a = self._action(rawText="a person walks fast")
        result = self.fn(a, self._act_def(motionClip="walking"))
        self.assertEqual(result, "a person walks fast")

    def test_motion_clip_fallback_when_no_raw_text(self):
        a = self._action(rawText="")
        result = self.fn(a, self._act_def(motionClip="brisk walk"))
        self.assertEqual(result, "brisk walk")

    def test_action_type_final_fallback(self):
        a = self._action(rawText="", actionType="jump_rope")
        result = self.fn(a, None)
        self.assertEqual(result, "jump rope")

    def test_modifier_prepended(self):
        a = self._action(rawText="walk", modifier="quickly")
        result = self.fn(a, None)
        self.assertEqual(result, "quickly walk")

    def test_no_modifier_no_prepend(self):
        a = self._action(rawText="walk", modifier="")
        result = self.fn(a, None)
        self.assertEqual(result, "walk")


# ---------------------------------------------------------------------------
# Config consistency: FPS mismatch is a real bug
# ---------------------------------------------------------------------------


class TestConfigConsistency(unittest.TestCase):
    """Cross-module config invariants."""

    def test_render_fps_matches_motion_fps(self):
        """RenderConfig.fps must equal MOTION_FPS so video plays at the correct speed.

        MotionClip.fps = MOTION_FPS = 30, but RenderConfig.fps defaults to 50.
        When M6 renders with fps=50 a clip recorded at 30fps, the output video
        plays at 30/50 = 60% of normal speed (wrong timing).
        """
        from src.modules.render.config import RenderConfig
        from src.shared.constants import MOTION_FPS

        self.assertEqual(
            RenderConfig().fps,
            MOTION_FPS,
            msg=(
                f"RenderConfig.fps={RenderConfig().fps} != MOTION_FPS={MOTION_FPS}. "
                "The render will produce videos with wrong playback speed."
            ),
        )

    def test_pipeline_fps_propagated_to_render(self):
        """PipelineConfig.__post_init__ must propagate fps to render sub-config."""
        from src.shared.config import PipelineConfig

        cfg = PipelineConfig(fps=24)
        self.assertEqual(cfg.render.fps, 24)

    def test_pipeline_duration_propagated_to_planner(self):
        """PipelineConfig.__post_init__ must propagate duration to planner."""
        from src.shared.config import PipelineConfig

        cfg = PipelineConfig(duration=7.5)
        self.assertEqual(cfg.planner.baseDuration, 7.5)

    def test_pipeline_custom_fps_does_not_affect_motion_fps(self):
        """MOTION_FPS is a constant; PipelineConfig.fps is for output video only."""
        from src.shared.config import PipelineConfig
        from src.shared.constants import MOTION_FPS

        cfg = PipelineConfig(fps=24)
        # Motion clips are always generated at MOTION_FPS regardless of output fps
        self.assertEqual(MOTION_FPS, 30)
        self.assertNotEqual(cfg.fps, cfg.motion.minActionFrames)  # unrelated fields

    def test_planner_config_has_base_duration(self):
        from src.modules.planner.config import PlannerConfig

        cfg = PlannerConfig()
        self.assertIsInstance(cfg.baseDuration, float)

    def test_planner_config_has_duration_jitter(self):
        from src.modules.planner.config import PlannerConfig

        cfg = PlannerConfig()
        self.assertIsInstance(cfg.durationJitter, float)


# ---------------------------------------------------------------------------
# Duration jitter in planner
# ---------------------------------------------------------------------------


class TestPlannerDurationJitter(unittest.TestCase):
    """ScenePlanner.compute_duration jitter behaviour."""

    @classmethod
    def setUpClass(cls):
        from src.modules.planner.config import PlannerConfig
        from src.modules.planner.planner import ScenePlanner

        cls.PlannerConfig = PlannerConfig
        cls.ScenePlanner = ScenePlanner

    def test_no_jitter_returns_raw(self):
        cfg = self.PlannerConfig(baseDuration=5.0, durationJitter=0.0)
        planner = self.ScenePlanner(cfg)
        self.assertEqual(planner.computeDuration(5.0, explicit=False), 5.0)

    def test_explicit_duration_never_jittered(self):
        cfg = self.PlannerConfig(baseDuration=5.0, durationJitter=2.0)
        planner = self.ScenePlanner(cfg)
        for _ in range(20):
            self.assertEqual(planner.computeDuration(3.0, explicit=True), 3.0)

    def test_jitter_stays_within_bounds(self):
        cfg = self.PlannerConfig(baseDuration=5.0, durationJitter=1.0)
        planner = self.ScenePlanner(cfg)
        for _ in range(50):
            d = planner.computeDuration(5.0, explicit=False)
            self.assertGreaterEqual(d, 1.0)  # clamped to 1.0 minimum
            self.assertLessEqual(d, 6.0)

    def test_jitter_duration_always_positive(self):
        # Even with large jitter the result is clamped to 1.0
        cfg = self.PlannerConfig(baseDuration=0.5, durationJitter=10.0)
        planner = self.ScenePlanner(cfg)
        for _ in range(30):
            d = planner.computeDuration(0.5, explicit=False)
            self.assertGreaterEqual(d, 1.0)


# ---------------------------------------------------------------------------
# MotionClip model properties
# ---------------------------------------------------------------------------


class TestMotionClipModel(unittest.TestCase):
    """src.modules.motion.models.MotionClip properties."""

    def test_duration_computed_from_fps(self):
        clip = _make_clip(60, fps=30)
        self.assertAlmostEqual(clip.duration, 2.0)

    def test_num_frames_matches_array_length(self):
        clip = _make_clip(45)
        self.assertEqual(clip.numFrames, 45)

    def test_duration_with_non_standard_fps(self):
        clip = _make_clip(50, fps=25)
        self.assertAlmostEqual(clip.duration, 2.0)


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------


class TestBuildActionLemmaMap(unittest.TestCase):
    """src.modules.understanding.actions.build_action_lemma_map"""

    @classmethod
    def setUpClass(cls):
        from src.modules.understanding.actions import buildActionLemmaMap

        cls.lemma_map = buildActionLemmaMap()

    def test_returns_dict(self):
        self.assertIsInstance(self.lemma_map, dict)

    def test_all_values_are_action_names(self):
        from src.shared.vocabulary import ACTIONS

        for v in self.lemma_map.values():
            self.assertIn(v, ACTIONS, msg=f"{v!r} not in ACTIONS")

    def test_known_action_keyword_present(self):
        from src.shared.vocabulary import ACTIONS

        # Pick the first keyword of the first action
        first_action = next(iter(ACTIONS.values()))
        if first_action.keywords:
            kw = first_action.keywords[0].lower()
            self.assertIn(kw, self.lemma_map)

    def test_no_empty_keys(self):
        for k in self.lemma_map:
            self.assertTrue(k, msg="Empty string is a key in lemma map")


if __name__ == "__main__":
    unittest.main()
