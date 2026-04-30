"""
Unit Tests for Pipeline Components
==================================
Covers: Shared Vocabulary, M1 (SpacyParser), M2 (ScenePlanner),
M4 (MotionGenerator + SSM), M5 (PhysicsScene + HumanoidBody),
and end-to-end Pipeline integration.

Not covered here: M3 (AssetGenerator), M6 (RenderEngine), M7 (AIEnhancer).

Run with: pytest tests/test_modules.py -v
Or: python tests/test_modules.py
"""

import os
import sys
import unittest

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.motion import MotionGenerator, SSMMotionModel
from src.modules.motion.ssm import getSsmInfo
from src.modules.planner import ScenePlanner
from src.modules.understanding import SpacyParser
from src.pipeline import Pipeline, PipelineConfig
from src.shared.vocabulary import (
    ACTIONS,
    OBJECTS,
    ActionCategory,
    getActionByKeyword,
)


class TestSharedVocabulary(unittest.TestCase):
    """Tests for the shared vocabulary system."""

    def test_actions_exist(self):
        """Vocabulary contains expected action categories."""
        self.assertGreater(len(ACTIONS), 0)

        # Check key actions exist
        self.assertIn("walk", ACTIONS)
        self.assertIn("kick", ACTIONS)
        self.assertIn("fall", ACTIONS)

    def test_objects_exist(self):
        """Vocabulary contains expected objects."""
        # Objects use canonical names (sphere, cube) not keywords (ball)
        self.assertIn("sphere", OBJECTS)  # 'ball' is a keyword alias for 'sphere'
        self.assertIn("cube", OBJECTS)
        self.assertIn("humanoid", OBJECTS)

    def test_action_lookup(self):
        """Action lookup by keyword works."""
        action = getActionByKeyword("walks")
        self.assertIsNotNone(action)
        assert action is not None  # narrow for type checker
        self.assertEqual(action.name, "walk")  # pyright: ignore[reportAttributeAccessIssue]

        action = getActionByKeyword("kicks")
        self.assertIsNotNone(action)
        assert action is not None  # narrow for type checker
        self.assertEqual(action.name, "kick")  # pyright: ignore[reportAttributeAccessIssue]

    def test_physics_actions(self):
        """Physics actions (fall, roll, etc.) are defined."""
        physics_actions = [a for a in ACTIONS.values() if a.category == ActionCategory.PHYSICS]
        self.assertGreater(len(physics_actions), 0)

        # fall does not require a target ("a ball falls" is valid without one)
        fall = ACTIONS.get("fall")
        self.assertIsNotNone(fall)
        assert fall is not None
        self.assertFalse(fall.requiresTarget)  # pyright: ignore[reportAttributeAccessIssue]


class TestSpacyParser(unittest.TestCase):
    """Tests for Module 1: SpacyParser."""

    def setUp(self):
        self.parser = SpacyParser()

    def test_simple_parse(self):
        """Parse simple prompt with one entity."""
        result = self.parser.parse("A red ball")

        self.assertEqual(len(result.entities), 1)
        # Parser may normalize 'ball' to 'sphere' based on vocabulary
        self.assertIn(result.entities[0].objectType, ["ball", "sphere"])

    def test_action_parse(self):
        """Parse prompt with action."""
        result = self.parser.parse("A person walks forward")

        self.assertGreater(len(result.actions), 0)
        self.assertEqual(result.actions[0].actionType, "walk")

    def test_fall_action(self):
        """Parse physics action (fall)."""
        result = self.parser.parse("A ball falls on a cube")

        actions = [a.actionType for a in result.actions]
        self.assertIn("fall", actions)

    def test_multiple_entities(self):
        """Parse prompt with multiple entities."""
        result = self.parser.parse("A red ball and a blue cube")

        self.assertGreaterEqual(len(result.entities), 2)


class TestParsedEntity(unittest.TestCase):
    """Unit tests for the ParsedEntity dataclass."""

    def test_defaults(self):
        """All optional fields default to None/False."""
        from src.modules.understanding.models import ParsedEntity

        e = ParsedEntity(name="sphere", objectType="sphere")
        self.assertFalse(e.isActor)
        self.assertIsNone(e.skin)

    def test_actor_flag(self):
        """is_actor flag is preserved."""
        from src.modules.understanding.models import ParsedEntity

        e = ParsedEntity(name="person", objectType="humanoid", isActor=True)
        self.assertTrue(e.isActor)

    def test_skin_assigned(self):
        """skin field accepts descriptive strings for objects and actors."""
        from src.modules.understanding.models import ParsedEntity

        e = ParsedEntity(name="red_sphere", objectType="sphere", skin="rubber")
        self.assertEqual(e.skin, "rubber")

        actor = ParsedEntity(name="person", objectType="humanoid", isActor=True, skin="dark skin")
        self.assertEqual(actor.skin, "dark skin")

    def test_slots_no_arbitrary_attributes(self):
        """slots=True prevents setting undeclared attributes."""
        from src.modules.understanding.models import ParsedEntity

        e = ParsedEntity(name="sphere", objectType="sphere")
        with self.assertRaises(AttributeError):
            e.nonexistent = "x"  # type: ignore[attr-defined]

    def test_parser_sets_skin_from_colour(self):
        """SpacyParser extracts colour word and stores it in skin."""
        parser = SpacyParser()
        scene = parser.parse("a red ball")
        self.assertEqual(len(scene.entities), 1)
        self.assertIsNotNone(scene.entities[0].skin)

    def test_parser_actor_has_body_params(self):
        """SpacyParser sets is_actor=True for humanoid entities."""
        parser = SpacyParser()
        scene = parser.parse("a person walks")
        actors = [e for e in scene.entities if e.isActor]
        self.assertGreater(len(actors), 0)


class TestSpatialRelation(unittest.TestCase):
    """Unit tests for the SpatialRelation dataclass."""

    def test_construction(self):
        """All four fields are required and stored correctly."""
        from src.modules.understanding.models import SpatialRelation

        r = SpatialRelation(subject="ball", predicate="on top of", relation="ON", object="cube")
        self.assertEqual(r.subject, "ball")
        self.assertEqual(r.predicate, "on top of")
        self.assertEqual(r.relation, "ON")
        self.assertEqual(r.object, "cube")

    def test_slots_no_arbitrary_attributes(self):
        """slots=True prevents setting undeclared attributes."""
        from src.modules.understanding.models import SpatialRelation

        r = SpatialRelation(subject="ball", predicate="next to", relation="BESIDE", object="cube")
        with self.assertRaises(AttributeError):
            r.nonexistent = "x"  # type: ignore[attr-defined]

    def test_parser_returns_typed_list(self):
        """SpacyParser produces list[SpatialRelation], not list[dict]."""
        from src.modules.understanding.models import SpatialRelation

        parser = SpacyParser()
        scene = parser.parse("a ball on top of a cube")
        self.assertGreater(len(scene.spatialRelations), 0)
        for sr in scene.spatialRelations:
            self.assertIsInstance(sr, SpatialRelation)

    def test_parser_resolves_entities(self):
        """subject and object are resolved to entity names, not empty strings."""
        parser = SpacyParser()
        scene = parser.parse("a ball on top of a cube")
        resolved = [sr for sr in scene.spatialRelations if sr.subject and sr.object]
        self.assertGreater(len(resolved), 0)

    def test_canonical_relation_set(self):
        """relation field holds a canonical value from SPATIAL_RELATIONS."""
        from src.modules.understanding.parsing_utils import SPATIAL_RELATIONS

        parser = SpacyParser()
        scene = parser.parse("a ball next to a cube")
        canonical_values = set(SPATIAL_RELATIONS.values())
        for sr in scene.spatialRelations:
            self.assertIn(sr.relation, canonical_values)

    def test_constraint_solver_uses_spatial_relation(self):
        """SpatialRelation objects feed directly into the constraint layout solver."""
        from src.modules.planner.constraint_layout import solveLayout
        from src.modules.understanding.models import SpatialRelation

        relations = [
            SpatialRelation(subject="ball", predicate="on top of", relation="ON", object="cube")
        ]
        positions = solveLayout(["ball", "cube"], relations)
        self.assertIn("ball", positions)
        self.assertIn("cube", positions)
        # ON constraint: ball should be above cube
        self.assertGreater(positions["ball"][2], positions["cube"][2])


class TestScenePlanner(unittest.TestCase):
    """Tests for Module 2: Scene Planner."""

    def setUp(self):
        self.parser = SpacyParser()
        self.planner = ScenePlanner()

    def test_basic_positioning(self):
        """Entities are assigned 3D positions."""
        parsed = self.parser.parse("A ball and a cube")
        planned = self.planner.plan(parsed)

        self.assertGreater(len(planned.entities), 0)

        for entity in planned.entities:
            self.assertIsNotNone(entity.position)
            self.assertIsNotNone(entity.position.x)
            self.assertIsNotNone(entity.position.y)
            self.assertIsNotNone(entity.position.z)

    def test_fall_positioning(self):
        """Falling objects are positioned above targets via constraint solver.

        With STACK_GAP=0.25m and ground_z=0.15m, the ball sits at
        cube.z + STACK_GAP ~= 0.40m.  0.30m threshold verifies the solver
        raised the ball above ground, without being sensitive to exact tuning.
        """
        parsed = self.parser.parse("A ball falls on a cube")
        planned = self.planner.plan(parsed)

        # At least one object should be meaningfully above ground (z > 0.30m)
        max_z = max(e.position.z for e in planned.entities)
        self.assertGreater(max_z, 0.30)


class TestMotionGenerator(unittest.TestCase):
    """Tests for Module 4: Motion Generator."""

    def test_generator_raises_when_ssm_checkpoint_missing(self):
        """MotionGenerator fails fast at construction when the SSM checkpoint is missing."""
        from src.modules.motion.config import MotionConfig

        cfg = MotionConfig(checkpointPath="checkpoints/nonexistent.pt")
        with self.assertRaises(FileNotFoundError):
            MotionGenerator(cfg)

    def test_motion_clip_structure(self):
        """MotionClip has correct field names after rename."""
        import numpy as np

        from src.modules.motion.models import MotionClip, MotionSource

        clip = MotionClip(
            action="walk",
            smplxParams=np.zeros((30, 168), dtype=np.float32),
            source=MotionSource.RETRIEVAL,
        )
        self.assertEqual(clip.numFrames, 30)
        self.assertIsNotNone(clip.source)
        self.assertEqual(clip.source, MotionSource.RETRIEVAL)


class TestSSMMotionGenerator(unittest.TestCase):
    """Tests for SSM-enhanced Motion Generator."""

    def test_ssm_generation(self):
        """SSM motion model generates a non-empty clip when a current checkpoint is present."""
        import os

        ckpt = "checkpoints/motion_ssm/best_model.pt"
        rvq_ckpt = "checkpoints/rvq_tokenizer/best_model.pt"

        if not (os.path.exists(ckpt) and os.path.exists(rvq_ckpt)):
            self.skipTest(f"checkpoints missing (need {ckpt} and {rvq_ckpt})")

        model = SSMMotionModel(checkpointPath=ckpt)
        clip = model.generateFromTextTokens("walk", numFrames=30)
        self.assertIsNotNone(clip)
        self.assertGreater(clip.numFrames, 0)

    def test_ssm_missing_checkpoint_raises(self):
        """SSM raises FileNotFoundError when the checkpoint is missing -- no silent fallback."""
        with self.assertRaises(FileNotFoundError):
            SSMMotionModel(checkpointPath="nonexistent/path.pt")


class TestSSMCore(unittest.TestCase):
    """Tests for SSM module core components."""

    def test_ssm_info(self):
        """SSM info returns expected structure."""
        info = getSsmInfo()

        self.assertIn("torch_available", info)
        self.assertIn("layers", info)
        self.assertIn("references", info)
        self.assertIn("novel_contribution", info)


class TestBiMambaLayer(unittest.TestCase):
    """Tests for the BiMambaLayer bidirectional Mamba wrapper."""

    def test_forward_shape(self):
        """BiMambaLayer output shape equals input shape (drop-in for MambaLayer)."""
        import torch

        from src.modules.motion.ssm import BiMambaLayer, SSMConfig

        cfg = SSMConfig(dModel=32, dState=8)
        layer = BiMambaLayer(cfg)
        layer.eval()

        x = torch.randn(2, 10, 32)  # (batch=2, T=10, d_model=32)
        with torch.no_grad():
            y = layer(x)

        self.assertEqual(y.shape, x.shape, f"BiMambaLayer output {y.shape} != input {x.shape}")

    def test_fwd_bwd_differ(self):
        """Forward and backward scans produce different outputs (not identical copies)."""
        import torch

        from src.modules.motion.ssm import BiMambaLayer, SSMConfig

        cfg = SSMConfig(dModel=16, dState=4)
        layer = BiMambaLayer(cfg)
        layer.eval()

        x = torch.randn(1, 8, 16)
        with torch.no_grad():
            fwd = layer.fwd(x)
            bwd = layer.bwd(x.flip(1)).flip(1)

        # Forward and backward outputs should differ (different learned parameters)
        self.assertFalse(
            torch.allclose(fwd, bwd), "Forward and backward scans are identical --likely a bug"
        )


class TestSBERTTextEncoder(unittest.TestCase):
    """Tests for SBERTTextEncoder - validates fallback when SBERT not installed."""

    def test_proj_layer_shape(self):
        """Projection layer has correct input/output dimensions regardless of SBERT."""
        import torch

        from src.modules.motion.nn_models import SBERTTextEncoder

        enc = SBERTTextEncoder(dModel=64)
        # The projection linear layer should be (384, 64)
        proj_linear = enc.proj[0]  # first element in Sequential
        self.assertEqual(proj_linear.in_features, SBERTTextEncoder.SBERT_DIM)
        self.assertEqual(proj_linear.out_features, 64)

    def test_forward_raises_without_sbert(self):
        """forward() raises a RuntimeError with a clear message when SBERT unavailable."""
        from src.modules.motion.nn_models import SBERTTextEncoder

        enc = SBERTTextEncoder(dModel=64)
        if enc.available:
            self.skipTest("sentence-transformers is installed --skipping unavailability test")

        with self.assertRaises(RuntimeError):
            enc(["a person walks"])


class TestSMPLXConstants(unittest.TestCase):
    """Tests the self-consistency of SMPL-X architectural constants."""

    def test_transl_y_idx_in_range(self):
        """SMPLX_TRANSL_Y_IDX must be within the global translation slice [3:6]."""
        from src.shared.constants import SMPLX_TRANSL_SLICE, SMPLX_TRANSL_Y_IDX

        self.assertGreaterEqual(SMPLX_TRANSL_Y_IDX, SMPLX_TRANSL_SLICE.start)
        self.assertLess(SMPLX_TRANSL_Y_IDX, SMPLX_TRANSL_SLICE.stop)

    def test_motion_dim(self):
        """Calculated dim 168 should match root, transl, body, hands, jaw, eyes."""
        from src.shared.constants import MOTION_DIM

        computed = 3 + 3 + 63 + 45 + 45 + 3 + 6
        self.assertEqual(MOTION_DIM, computed)


class TestFIDEvaluator(unittest.TestCase):
    """Tests for the T2M FID + R-Precision evaluation pipeline."""

    def test_encoder_output_shape(self):
        """T2MMotionEncoder produces (B, 512) L2-normalised embeddings."""
        import torch

        from scripts.evaluation.motion_encoder import T2MMotionEncoder

        enc = T2MMotionEncoder(inputDim=168)
        enc.eval()

        x = torch.randn(4, 50, 168)
        with torch.no_grad():
            feats = enc(x)

        self.assertEqual(feats.shape, (4, 512))
        # L2-normalised
        norms = feats.norm(dim=1)
        self.assertTrue(
            torch.allclose(norms, torch.ones(4), atol=1e-5),
            f"Embeddings not L2-normalised: {norms}",
        )

    def test_encoder_with_lengths(self):
        """L2-normalised output even with variable-length masking."""
        import torch

        from scripts.evaluation.motion_encoder import T2MMotionEncoder

        enc = T2MMotionEncoder(inputDim=168).eval()
        x = torch.randn(3, 60, 168)
        lengths = torch.tensor([20, 40, 60])
        with torch.no_grad():
            feats = enc(x, lengths)
        norms = feats.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(3), atol=1e-5))

    def test_extract_features_shape(self):
        """extract_features() handles variable-length clips and returns (N, 512)."""
        from scripts.evaluation.motion_encoder import extractFeatures, loadEncoder

        enc = loadEncoder(inputDim=168, device="cpu")
        rng = np.random.default_rng(42)
        motions = [rng.standard_normal((t, 168)).astype("float32") for t in [30, 60, 90, 45]]
        feats = extractFeatures(enc, motions, device="cpu")
        self.assertEqual(feats.shape, (4, 512))

    def test_fid_identical_distributions(self):
        """FID between identical distributions should be ~0."""
        from scripts.evaluation.compute_fid import computeFID

        # Same feature matrix -> FID should be 0
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((600, 512)).astype("float32")
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        fid = computeFID(feats, feats.copy())
        self.assertAlmostEqual(
            fid, 0.0, places=3, msg=f"FID on identical distributions = {fid}, expected ~0"
        )

    def test_fid_different_distributions(self):
        """FID between very different distributions should be large."""
        from scripts.evaluation.compute_fid import computeFID

        rng = np.random.default_rng(1)
        feats_a = rng.standard_normal((600, 512)).astype("float32")
        feats_b = rng.standard_normal((600, 512)).astype("float32") + 10.0  # large shift

        fid = computeFID(feats_a, feats_b)
        self.assertGreater(fid, 10.0, msg=f"FID on shifted dists = {fid}, expected > 10")

    def test_diversity_non_negative(self):
        """Diversity metric is always >= 0."""
        from scripts.evaluation.compute_fid import computeDiversity

        rng = np.random.default_rng(2)
        feats = rng.standard_normal((50, 512)).astype("float32")
        d = computeDiversity(feats, nPairs=20)
        self.assertGreaterEqual(d, 0.0)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for full pipeline (requires all modules + PyBullet)."""

    pytestmark = [pytest.mark.slow]

    def test_pipeline_setup(self):
        """Pipeline can be set up with all modules."""
        config = PipelineConfig()

        pipeline = Pipeline(config)
        self.assertIsNotNone(pipeline)

    def test_pipeline_run(self):
        """Pipeline can process a prompt end-to-end (rule-based parser, no checkpoints needed)."""
        import shutil

        if not shutil.which("ffmpeg"):
            self.skipTest("FFmpeg not installed -- skipping render step")

        config = PipelineConfig(duration=1.0, fps=12)

        pipeline = Pipeline(config)
        result = pipeline.run("A ball falls", outputName="test_integration")

        self.assertIsInstance(result, dict)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)
