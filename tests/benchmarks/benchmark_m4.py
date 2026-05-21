"""M4 Motion Generator (SSM) — benchmark suite.

Tests the SSM-backed text-to-motion generator. Run:
    py tests/benchmarks/benchmark_m4.py

Skips gracefully when the trained checkpoint is missing.
"""

import os

import numpy as np

from src.modules.motion import MotionGenerator
from src.shared.constants import MOTION_DIM, MOTION_FPS

GEN_NOT_READY = "Generator not ready"
TEST_WALK_RUN = "10. Walk != Run (Semantic)"
TEST_KICK_JUMP = "11. Kick != Jump (Semantic)"

class M4Benchmark:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.generator: MotionGenerator | None = None

    def test(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"

        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {name} {details}")

    def testImports(self) -> bool:
        print("\n[1-4] IMPORTS")
        print("-" * 50)
        self.test("1. MotionGenerator import", True)

        try:
            from src.modules.motion.ssm import MambaLayer  # noqa: F401

            self.test("2. SSM modules import", True)
        except ImportError as e:
            self.test("2. SSM modules import", False, str(e)[:30])

        try:
            from src.modules.motion.training import SSMTrainer  # noqa: F401

            self.test("3. Training module import", True)
        except ImportError as e:
            self.test("3. Training module import", False, str(e)[:30])

        try:
            from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer  # noqa: F401

            self.test("4. RVQ tokenizer import", True)
        except ImportError as e:
            self.test("4. RVQ tokenizer import", False, str(e)[:30])

        return True

    def buildGenerator(self) -> None:
        print("\n[5-7] GENERATOR INSTANTIATION")
        print("-" * 50)

        ckpt = "checkpoints/motion_ssm/best_model.pt"
        rvq = "checkpoints/rvq_tokenizer/best_model.pt"
        ckptOk = os.path.exists(ckpt)
        rvqOk = os.path.exists(rvq)
        self.test("5. Motion SSM checkpoint exists", ckptOk, f"path={ckpt}")
        self.test("6. RVQ tokenizer checkpoint exists", rvqOk, f"path={rvq}")

        if not (ckptOk and rvqOk):
            self.test("7. MotionGenerator construct", False, GEN_NOT_READY)

            return

        try:
            self.generator = MotionGenerator()
            self.test("7. MotionGenerator construct", True)
        except Exception as e:
            self.test("7. MotionGenerator construct", False, str(e)[:50])

    def testGeneration(self) -> None:
        print("\n[8-12] SSM GENERATION")
        print("-" * 50)

        if self.generator is None:
            for i in range(8, 13):
                self.test(f"{i}. Motion test", False, GEN_NOT_READY)

            return

        try:
            clip = self.generator.generate("a person walks forward", numFrames=60)
            self.test("8. Walk motion shape", clip.smplxParams.shape == (60, MOTION_DIM),
                      f"shape={clip.smplxParams.shape}")
        except Exception as e:
            self.test("8. Walk motion shape", False, str(e)[:30])

        try:
            clip = self.generator.generate("a person runs", numFrames=60)
            self.test("9. FPS preserved", clip.fps == MOTION_FPS, f"fps={clip.fps}")
        except Exception as e:
            self.test("9. FPS preserved", False, str(e)[:30])

        try:
            walk = self.generator.generate("walk forward", numFrames=60)
            run = self.generator.generate("run fast", numFrames=60)
            diff = abs(np.std(walk.smplxParams) - np.std(run.smplxParams))
            self.test(TEST_WALK_RUN, bool(diff > 0.001), f"diff={diff:.4f}")
        except Exception as e:
            self.test(TEST_WALK_RUN, False, str(e)[:30])

        try:
            kick = self.generator.generate("kick", numFrames=60)
            jump = self.generator.generate("jump", numFrames=60)
            diff = bool(
                kick.numFrames != jump.numFrames
                or np.std(kick.smplxParams) != np.std(jump.smplxParams)
            )
            self.test(TEST_KICK_JUMP, diff)
        except Exception as e:
            self.test(TEST_KICK_JUMP, False, str(e)[:30])

        try:
            clip = self.generator.generate("idle", numFrames=120)
            ok = clip.smplxParams.shape[0] == 120 and bool(np.isfinite(clip.smplxParams).all())
            self.test("12. Long generation (120f, finite)", ok)
        except Exception as e:
            self.test("12. Long generation", False, str(e)[:30])

    def runAll(self) -> bool:
        print("=" * 70)
        print("M4 MOTION GENERATOR (SSM) - BENCHMARK SUITE")
        print("=" * 70)

        self.testImports()
        self.buildGenerator()
        self.testGeneration()

        print("\n" + "=" * 70)
        total = self.passed + self.failed
        print(f"M4 BENCHMARK: {self.passed}/{total} PASSED")
        print("=" * 70)

        return self.passed >= max(1, total - 5)

if __name__ == "__main__":
    benchmark = M4Benchmark()
    success = benchmark.runAll()
    exit(0 if success else 1)
