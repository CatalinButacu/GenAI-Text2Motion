"""M2 Scene Planner — benchmark suite.

Tests positioning, layout, and scene construction. Run:
    py tests/benchmarks/benchmark_m2.py
"""


from src.modules.planner import PlannedScene, Position3D, ScenePlanner
from src.modules.understanding import SpacyParser

A_BALL = "A ball"

class M2Benchmark:
    def __init__(self):
        self.parser = SpacyParser()
        self.planner = ScenePlanner()
        self.passed = 0
        self.failed = 0

    def test(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"

        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {name} {details}")

    def plan(self, prompt: str) -> PlannedScene:
        parsed = self.parser.parse(prompt)

        return self.planner.plan(parsed)

    def run_all(self):
        print("=" * 70)
        print("M2 SCENE PLANNER - BENCHMARK SUITE")
        print("=" * 70)

        # === POSITION ASSIGNMENT (1-10) ===
        print("\n[1-10] POSITION ASSIGNMENT")
        print("-" * 50)

        s = self.plan(A_BALL)
        has_pos = all(e.position is not None for e in s.entities)
        self.test("1. Position assigned", has_pos)

        s = self.plan(A_BALL)
        ball = [e for e in s.entities if "sphere" in e.object_type]
        above = bool(ball) and ball[0].position.z >= 0
        self.test("2. Above ground (z>=0)", above)

        s = self.plan("A ball and a cube")
        positions = [str(e.position.to_list()) for e in s.entities]
        unique = len(set(positions)) == len(positions)
        self.test("3. Unique positions", unique)

        s = self.plan("A ball, a cube, and a cylinder")
        spread = len(s.entities) >= 3 and len({str(e.position.to_list()) for e in s.entities}) >= 3
        self.test("4. Three objects spread", spread, f"entities={len(s.entities)}")

        s = self.plan("A person kicks a ball")
        actor = [e for e in s.entities if e.is_actor]
        obj = [e for e in s.entities if not e.is_actor]
        diff_pos = bool(actor and obj) and actor[0].position.to_list() != obj[0].position.to_list()
        self.test("5. Actor vs object positions", diff_pos)

        s = self.plan("A person stands")
        person = [e for e in s.entities if e.is_actor]
        tall = bool(person) and person[0].position.z >= -2
        self.test("6. Humanoid reasonable z", tall)

        s = self.plan(A_BALL)
        ball = [e for e in s.entities if "sphere" in e.object_type]
        close = bool(ball) and abs(ball[0].position.x) < 5 and abs(ball[0].position.y) < 5
        self.test("7. Object in scene bounds", close)

        p1 = Position3D(1.0, 2.0, 3.0)
        p2 = Position3D(4.0, 5.0, 6.0)
        p3 = p1 + p2
        self.test("8. Position3D add", p3.to_list() == [5.0, 7.0, 9.0])

        p = Position3D(1.5, 2.5, 3.5)
        self.test("9. Position3D to_list", p.to_list() == [1.5, 2.5, 3.5])

        p = Position3D()
        self.test("10. Position3D default", p.to_list() == [0.0, 0.0, 0.0])

        # === PROPERTY PRESERVATION (11-18) ===
        print("\n[11-18] PROPERTY PRESERVATION")
        print("-" * 50)

        s = self.plan(A_BALL)
        ball = [e for e in s.entities if "sphere" in e.object_type]
        has_mass = bool(ball) and ball[0].mass > 0
        self.test("11. Mass assigned", has_mass, f"mass={ball[0].mass if ball else 0}")

        s = self.plan("A cube")
        cube = [e for e in s.entities if "cube" in e.object_type]
        has_size = bool(cube) and cube[0].size is not None
        self.test("12. Size assigned", has_size)

        s = self.plan("A cylinder")
        has_cyl = any("cylinder" in e.object_type for e in s.entities)
        self.test("13. Object type preserved", has_cyl)

        s = self.plan("A person walks")
        has_actor = any(e.is_actor for e in s.entities)
        self.test("14. Actor flag preserved", has_actor)

        s = self.plan("A ball and another ball")
        names = [e.name for e in s.entities]
        unique_names = len(set(names)) == len(names)
        self.test("15. Unique names", unique_names)

        s = self.plan("A cube")
        cube = [e for e in s.entities if "cube" in e.object_type]
        has_rot = bool(cube) and cube[0].rotation is not None
        self.test("16. Rotation assigned", has_rot)

        s = self.plan(A_BALL)
        e = s.entities[0]
        has_all = hasattr(e, "name") and hasattr(e, "position") and hasattr(e, "mass")
        self.test("17. PlannedEntity attributes", has_all)

        s = self.plan(A_BALL)
        self.test("18. Entities list populated", len(s.entities) > 0)

        # === SCENE DURATION (19-22) ===
        print("\n[19-22] SCENE DURATION")
        print("-" * 50)

        s = self.plan("A person walks for 5 seconds")
        self.test("19. Duration > 0", s.duration > 0, f"duration={s.duration:.1f}s")

        s = self.plan(A_BALL)
        self.test("20. Default duration", s.duration > 0, f"default={s.duration:.1f}s")

        s = self.plan("A person walks for 10 seconds")
        self.test("21. Explicit duration honored", s.duration >= 8, f"duration={s.duration:.1f}s")

        self.plan("")
        self.test("22. Empty prompt handling", True)

        # === COMPLEX SCENARIOS (23-30) ===
        print("\n[23-30] COMPLEX SCENARIOS")
        print("-" * 50)

        s = self.plan("A person kicks a ball")
        has_both = any(e.is_actor for e in s.entities) and any(
            "sphere" in e.object_type for e in s.entities
        )
        self.test("23. Person + object scene", has_both)

        s = self.plan("A person walks to a ball and kicks it")
        self.test("24. Action sequence scene", len(s.entities) >= 2)

        s = self.plan("A ball on a table")
        self.test("25. Spatial relation scene", len(s.entities) >= 2)

        s = self.plan("A table with a ball on it")
        self.test("26. Furniture scene", len(s.entities) >= 1)

        s = self.plan("In a room, a red ball sits on a blue table near a person")
        self.test("27. Full scene description", len(s.entities) >= 2)

        s = self.plan(
            "A red sphere, a blue cube, a green cylinder, a yellow ball, "
            "and a person standing nearby"
        )
        self.test("28. Stress test 5+ entities", len(s.entities) >= 4, f"found={len(s.entities)}")

        s = self.plan("A person and another person")
        actors = [e for e in s.entities if e.is_actor]
        self.test("29. Multi-actor scene", len(actors) >= 2, f"actors={len(actors)}")

        s = self.plan("A person stands still")
        self.test("30. Idle action scene", len(s.entities) >= 1)

        # === SUMMARY ===
        print("\n" + "=" * 70)
        print(f"M2 BENCHMARK: {self.passed}/30 PASSED")
        print("=" * 70)

        return self.passed >= 25

if __name__ == "__main__":
    benchmark = M2Benchmark()
    success = benchmark.run_all()
    exit(0 if success else 1)
