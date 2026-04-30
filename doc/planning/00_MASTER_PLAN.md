Q# Physics-Constrained Video Generation -- Master Plan

**Project**: Dissertation -- Physics-Constrained Video Generation from Text  
**Author**: Catalin Butacu  
**Date**: March 2026  
**Architecture**: 7 modules, 4 pipeline stages  

---

## 1. Problem & Solution

Current text-to-video systems (Sora, Runway Gen-3) produce visually impressive but physically implausible output -- objects pass through surfaces, gravity is inconsistent, humans exhibit foot sliding and impossible joint rotations. This happens because purely data-driven diffusion models have no explicit physics reasoning.

We propose a **modular pipeline** that decomposes text-to-video into interpretable stages, each grounded in physical constraints:

```
Text Prompt
    v  M1  Scene Parser          -- T5: parse entities, actions, relations
    v  M2  Knowledge Retriever   -- SBERT: enrich entities with physical properties
    v  M3  Scene Planner         -- L-BFGS-B: compute 3D spatial layout
    v  M4  Motion Generator      -- MotionSSM + PhysicsSSM: text -> motion
    v  M5  Physics Simulator     -- PyBullet: rigid-body simulation (240 Hz)
    v  M6  Render Engine          -- SMPL + aitviewer + post-processing + MP4 export
    v  M7  AI Enhancer            -- ControlNet / AnimateDiff (optional)
    v
  MP4 Video
```

**Key Innovation**: **PhysicsSSM** -- a gated state space model that blends temporal motion predictions with physics constraint embeddings through a learned sigmoid gate: `gate = sigma(W[ssm_out; phys_embed])`. This enables physically plausible human motion while maintaining SSM efficiency.

---

## 2. Code Architecture

### 2.1 Pipeline Stages (src/stages/)

The pipeline orchestrator (`src/pipeline.py`, ~180 LOC) delegates to four stage classes:

| Stage | File | Modules Used | What It Does |
|-------|------|-------------|--------------|
| **Understanding** | `understanding.py` | M1 + M2 + M3 | Parse prompt -> enrich with KB -> plan 3D layout |
| **Motion** | `motion.py` | M4 | Generate motion clips -> PhysicsSSM refinement -> validate |
| **Physics** | `physics.py` | M5 | PyBullet simulation -> skeleton poses + contacts |
| **Rendering** | `rendering.py` | M6 + M7 | SMPL mesh -> effects -> MP4 export |

### 2.2 Module Directory (src/modules/)

| Module | Directory | Model | Params | Status |
|--------|-----------|-------|--------|--------|
| M1 Scene Parser | `understanding/` | Flan-T5-Small | 60M |  Trained (v5) |
| M2 Knowledge Retriever | `understanding/` | all-MiniLM-L6-v2 | 22M |  Inference |
| M3 Scene Planner | `planner/` | L-BFGS-B | -- |  Non-ML |
| M4 Motion Generator | `motion/` | TextToMotionSSM + PhysicsSSM | 7.5M + 12M |  Trained |
| M5 Physics Simulator | `physics/` | PyBullet | -- |  Non-ML |
| M6 Render Engine | `render/` + `physics/` | SMPL + aitviewer + OpenCV | -- |  Non-ML |
| M7 AI Enhancer | `diffusion/` | SD 1.5 + ControlNet | ~1.2B |  Optional |

### 2.3 Data Flow

```
M1:  str              -> ParsedScene        (entities, actions, relations)
M2:  ParsedScene       -> EnrichedScene      (+ physical properties from KB)
M3:  EnrichedScene     -> PlannedScene       (+ 3D positions via constraint solver)
M4:  str (action text) -> MotionClip         (T x 168 SMPL-X params + N_BODY_JOINTSx3 joints)
M5:  PlannedScene+Clip -> PhysicsResult      (skeleton poses, contacts)
M6:  skeleton poses    -> MP4               (SMPL mesh -> effects -> video export)
M7:  skeleton images   -> photorealistic     (ControlNet / AnimateDiff, optional)
```

---

## 3. Research Contributions

| # | Contribution | Novelty | Status |
|---|-------------|---------|--------|
| 1 | **PhysicsSSM**: Gated SSM with physics-aware loss (foot sliding + ground penetration + jerk) | **High** -- novel architecture |  Trained |
| 2 | **Physics-constrained pipeline**: Explicit physics simulation as structural guarantee | **Medium** -- novel integration |  Implemented |
| 3 | **T5 scene extraction**: Fine-tuned Flan-T5 for structured JSON scene parsing | **Medium** -- novel application |  Trained (v5, loss=0.062) |
| 4 | **Multi-tier motion retrieval**: Semantic -> SSM -> procedural fallback cascade | **Low-Medium** |  Implemented |

---

## 4. Datasets

| Dataset | Used By | Size | Status |
|---------|---------|------|--------|
| AMASS | Motion training (M4) | ~50 GB, 168-dim SMPL-X @ 30fps |  Ready |
| InterX | Multi-person SMPL-X (M4) | 11,388 samples @ 30fps |  Downloaded |
| PAHOI | Human-object interactions (M4) | 562 sequences @ 30fps |  Available |
| ARCTIC | Hand interactions (M4) | varies @ 30fps |  Available |
| M1 Training Set | Parser training (M1) | 40,000 JSONL (35.5 MB) |  Built |
| Physics State | PhysicsSSM (M4) | Derived from AMASS/InterX |  Ready |
| KB Index | Retriever (M2) | 12 curated objects |  Built |

---

## 5. Training Status

| Model | Module | Architecture | Checkpoint | Status |
|-------|--------|-------------|------------|--------|
| Flan-T5-Small | M1 | Seq2Seq (60M) | `checkpoints/scene_extractor/` |  Trained (eval_loss=0.062) |
| TextToMotionSSM | M4 | Mamba SSM (7.5M) | `checkpoints/motion_ssm/` |  Trained (val_loss=0.371) |
| PhysicsSSM | M4 | Gated Mamba + Physics (12M) | `checkpoints/physics_ssm/` |  Trained |

---

## 6. Goals & Status

### Must-Have (Dissertation Pass)

- [x] End-to-end pipeline: text -> MP4 with physics constraints
- [x] 2 models trained from scratch (MotionSSM + PhysicsSSM)
- [x] 1 model fine-tuned (T5)
- [ ] PhysicsSSM ablation study (gate contribution)
- [ ] Evaluation chapter with quantitative metrics
- [ ] Literature review (30+ papers)

### Should-Have (Strong Grade)

- [ ] FID + FVD metrics vs. baselines
- [ ] Physics adherence verification with statistical significance
- [ ] User study (N>=20)

---

## 7. Hardware Constraints

**Target**: NVIDIA RTX 3050 (4 GB VRAM)

- fp16 inference + CPU offload for diffusion models
- Batch processing (<=8 frames) for AnimateDiff
- Lazy loading -- only one large model in VRAM at a time
- All non-ML modules are CPU-only

---

## 8. Planning Documents

| Doc | Description |
|-----|-------------|
| [01_MODULES.md](01_MODULES.md) | Technical spec for each module (M1-M7) |
| [02_DATASETS.md](02_DATASETS.md) | Dataset inventory, formats, preprocessing |
| [03_TRAINING.md](03_TRAINING.md) | Training procedures, losses, schedules |
| [04_EVALUATION.md](04_EVALUATION.md) | Metrics, benchmarks, ablation studies |
| [05_INTEGRATION.md](05_INTEGRATION.md) | Pipeline data flow, type contracts, error handling |
