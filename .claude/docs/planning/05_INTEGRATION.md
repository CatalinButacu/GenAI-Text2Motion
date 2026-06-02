# Pipeline Integration

Data flow contracts, error handling, and coordinate conventions.

---

## Pipeline Flow

```
Pipeline.run(prompt)
    |
    + UnderstandingStage.run(prompt)
    |     + M1: parser.parse(prompt) -> ParsedScene
    |     + M2: kb_retriever.retrieve(entity) -> enriched properties
    |     + M3: planner.plan(parsed) -> PlannedScene
    |
    + MotionStage.run(parsed)
    |     + M4: motion_gen.generate(query) -> MotionClip
    |     + PhysicsSSM: refine motion with physics gate
    |     + Validator: biomechanical repair loop
    |
    + PhysicsStage.run(planned, clips, parsed)
    |     + Scene setup (ground + primitives + humanoid)
    |     + M5: simulation loop (240 Hz)
    |     + Returns PhysicsResult (skeleton poses + contacts)
    |
    + RenderingStage
          + M6: render_frames(skeleton) -> RGB via SMPL+aitviewer + export_video -> MP4
          + M7 (opt): diffusion enhancement
```

---

## Type Contracts

| Boundary | Type | Key Fields |
|----------|------|------------|
| M1 output | `ParsedScene` | `.entities: list[ParsedEntity]`, `.actions: list[ParsedAction]` |
| M3 output | `PlannedScene` | `.entities: list[PlannedEntity]` (with `.position: Position3D`) |
| M4 output | `MotionClip` | `.frames: ndarray(T,168)`, `.raw_joints: ndarray(T,21,3)`, `.fps`, `.source` |
| M5 output | `PhysicsResult` | `.skeleton_positions`, `.sim`, `.scene`, `.cam`, `.body_params` |
| M6 output | `FrameData` | `.rgb: ndarray(H,W,3)`, `.timestamp: float` |

---

## Coordinate Systems

| System | Convention | Units |
|--------|-----------|-------|
| AMASS / SMPL-X motion data | Y-up | metres |
| PyBullet simulation | Z-up | metres |
| SMPL body model | Y-up | metres |
| aitviewer rendering | Y-up | metres |

The retargeting module (`motion_retarget.py`) handles Y-up -> Z-up conversion for PyBullet.

---

## Error Handling

Each module has a fallback:

| Module | Failure Mode | Fallback |
|--------|-------------|----------|
| M1 | T5 checkpoint missing | Rules-based PromptParser |
| M2 | KB index empty | Skip enrichment |
| M4 | SSM checkpoint missing | AMASS retrieval -> procedural |
| M5 | Physics NaN | Stop early, return partial frames |
| M6 | SMPL model unavailable | Sphere-per-joint mesh |
| M7 | Diffusion OOM | Disabled (skip enhancement) |

---

## Lazy Loading (VRAM Management)

Peak VRAM usage ~3.8 GB (during M7 diffusion). Strategy:

- M1 T5 loaded first (~1.2 GB), unloaded before M7
- M4 MotionSSM is small (~0.5 GB), stays loaded
- M7 diffusion only loaded when `use_diffusion=True`
- All non-ML modules (M3, M5, M6) are CPU-only
