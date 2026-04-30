# Module Specifications

Technical spec for each of the 7 pipeline modules.

---

## M1 -- Scene Parser

| | |
|---|---|
| **Model** | Flan-T5-Small (60M params) |
| **Input** | `str` (natural language prompt) |
| **Output** | `ParsedScene` (entities, actions, relations) |
| **Code** | `src/modules/understanding/t5_parser.py` |
| **Checkpoint** | `checkpoints/scene_extractor/` |
| **Status** |  Trained (v5, eval_loss=0.062) |

Fine-tuned Seq2Seq transformer that parses text prompts into structured JSON with entities, actions, and spatial relations. Falls back to rules-based `PromptParser` when checkpoint unavailable.

**Training**: 15 epochs on 40K VG-derived samples, label-smoothed cross-entropy, AdamW lr=3e-4.

---

## M2 -- Knowledge Retriever

| | |
|---|---|
| **Model** | all-MiniLM-L6-v2 (SBERT, 22M params) |
| **Input** | `ParsedScene` from M1 |
| **Output** | `EnrichedScene` (entities + mass, friction, material) |
| **Code** | `src/modules/understanding/retriever.py` |
| **Status** |  Inference-only |

SBERT embeddings + FAISS index for semantic lookup against a curated knowledge base (12 objects with physical properties). Enriches parsed entities with real-world physical attributes.

---

## M3 -- Scene Planner

| | |
|---|---|
| **Model** | L-BFGS-B (classical optimizer, no ML) |
| **Input** | `EnrichedScene` from M2 |
| **Output** | `PlannedScene` (entities with 3D positions) |
| **Code** | `src/modules/planner/planner.py`, `constraint_layout.py` |
| **Status** |  Working |

Non-ML constrained solver that places entities in 3D space using 10 spatial constraint types (on, above, near, left_of, etc.). Falls back to row-based grid layout.

---

## M4 -- Motion Generator  (includes PhysicsSSM)

| | |
|---|---|
| **Model** | TextToMotionSSM (7.5M) + PhysicsSSM (12M) |
| **Input** | `str` (action description) |
| **Output** | `MotionClip` (T x 168 SMPL-X params + N_BODY_JOINTSx3 joints) |
| **Code** | `src/modules/motion/` (generator, ssm_generator, ssm/) |
| **Checkpoints** | `checkpoints/motion_ssm/`, `checkpoints/physics_ssm/` |
| **Status** |  Both trained |

Two-stage motion generation:
1. **TextToMotionSSM**: 4-layer Mamba SSM architecture, trained on AMASS + InterX (SMPL-X 168-dim @ 30fps, 250 epochs, val_loss=0.371)
2. **PhysicsSSM**  NOVEL: Gated SSM that refines motion with physics constraints via learned sigmoid gate: `gate = sigma(W[ssm_out; phys_embed])`. Physics-aware loss includes foot sliding penalty, ground penetration penalty, and jerk regularisation.

Retrieval cascade: semantic search (SBERT) -> SSM generation -> keyword fallback -> procedural sinusoidal.

---

## M5 -- Physics Simulator

| | |
|---|---|
| **Model** | PyBullet (deterministic, no ML) |
| **Input** | `PlannedScene` + `MotionClip` |
| **Output** | `PhysicsResult` (skeleton poses, contacts) |
| **Code** | `src/modules/physics/` (simulator, scene, humanoid) |
| **Status** |  Working |

Rigid-body physics at 240 Hz. URDF humanoid with PD joint control. SMPL-X motion retargeting from Y-up (m) to PyBullet Z-up (m). Contact detection for foot-ground and object interactions. Cinematic camera with orbit/pitch/zoom.

---

## M6 -- Render Engine

| | |
|---|---|
| **Model** | SMPL + aitviewer + OpenCV (no ML) |
| **Input** | Skeleton poses from M5 |
| **Output** | MP4 video file (1280x720) |
| **Code** | `src/modules/render/aitviewer_renderer.py`, `src/modules/render/engine.py`, `src/modules/physics/smpl_body.py` |
| **Status** |  Working |

Two-step rendering:
1. **Mesh rendering**: GPU-accelerated SMPL-X mesh rendering via aitviewer HeadlessRenderer. Maps body-joint skeleton (N_BODY_JOINTS) to SMPL-X body mesh with body-type-specific shape parameters (betas). Falls back to sphere-per-joint mesh when SMPL-X unavailable.
2. **Post-processing**: 5 cinematic effects applied in sequence: motion blur, depth-of-field, color grading, vignette, film grain. H.264 export via imageio-ffmpeg.

---

## M7 -- AI Enhancer (Optional)

| | |
|---|---|
| **Model** | SD 1.5 + ControlNet OpenPose (1.2B) / AnimateDiff (41M) |
| **Input** | Skeleton images from M6 |
| **Output** | Photorealistic frames |
| **Code** | `src/modules/diffusion/` |
| **Status** |  Optional (requires ~4 GB VRAM) |

Two modes:
- **ControlNet**: Per-frame skeleton -> photorealistic image via SD 1.5 with OpenPose conditioning
- **AnimateDiff**: Batch temporal attention for cross-frame coherence (8 frames at a time, sliding window overlap)

Disabled by default. Enabled with `--with-enhance` flag.
