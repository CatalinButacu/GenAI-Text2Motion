# Training and Finetuning Guide

This document describes the training status and procedures for each module.

---

## Overview

| Component | Needs Training? | Status | Dataset | Script |
|-----------|:-:|--------|---------|--------|
| M1: T5 Scene Parser | Yes | **Trained** (v5, loss=0.062) | m1_training (40k) | `scripts/train_m1_t5.py` |
| M1: Prompt Parser | No | Rule-based fallback | -- | -- |
| M1: KB Retriever | No | Pretrained SBERT + FAISS | knowledge_base | -- |
| M2: Scene Planner | No | Deterministic solver | -- | -- |
| M3: Asset Generator | No | Pretrained Shap-E/TripoSR | -- | -- |
| M4: Motion SSM | Yes | **Trained** (250 ep, loss=0.37) | KIT-ML (4.9k) | `scripts/train_motion_ssm.py` |
| M4: KIT-ML Retriever | No | SBERT semantic search | KIT-ML | -- |
| **M4: PhysicsSSM** | **Yes** | **Trained** | KIT-ML + derived physics | `scripts/train_physics_ssm.py` |
| M5: Physics Engine | No | PyBullet simulation | -- | -- |
| M6: Render Engine | No | SMPL + aitviewer + OpenCV | -- | -- |
| M7: AI Enhancer | No | Pretrained SD+ControlNet | -- | -- |

---

## 1. M1 Scene Parser (T5) -- DONE

Fine-tuned `google/flan-t5-small` for text -> JSON scene extraction.

### Training History

| Version | Epochs | Eval Loss | Notes |
|---------|--------|-----------|-------|
| v1 | 3 | NaN | Failed run |
| v2 | 10 | 0.089 | First successful |
| v3 | 10 | 0.072 | More data |
| v4 | 12 | 0.068 | Hyperparameter sweep |
| **v5** | **15** | **0.062** | **Current best** |

### How to Retrain

```bash
python scripts/train_m1_t5.py \
    --epochs 15 --batch-size 4 --lr 5e-5 \
    --data-dir data/m1_training \
    --output-dir checkpoints/understanding/scene_extractor_v6
```

**Data:** `data/m1_training/` -- 40,000 train samples (.jsonl, 35.5 MB).
**Source:** Built from Visual Genome (`data/M1_VisualGenome/`, 4.85 GB) via `scripts/build_vg_dataset.py`.
**Checkpoint:** `checkpoints/understanding/scene_extractor_v5/model.safetensors` (293 MB).

---

## 2. M4 Motion SSM -- DONE

Custom `TextToMotionSSM`: SimpleTextEncoder -> 4x MambaLayer -> MotionDecoder.

### Training Details

- **Data:** AMASS + InterX -- SMPL-X 168-dim pose params at 30 fps
- **Architecture:** d_model=256, d_state=32, n_layers=4, vocab~=10k
- **Optimizer:** AdamW + OneCycleLR, lr=5e-5, weight_decay=0.01
- **Duration:** 250 epochs, ~76,500 steps
- **Final:** train_loss=0.381, val_loss=0.371
- **Checkpoint:** `checkpoints/motion_ssm/best_model.pt` (58 MB)

### How to Retrain

```bash
python scripts/train_motion_ssm.py \
    --epochs 250 --batch-size 16 --lr 5e-5 \
    --data-dir data/AMASS --device cuda
```

---

## 3. M4 PhysicsSSM -- NOVEL CONTRIBUTION

**Novel contribution:** Learned sigmoid gate blending SSM temporal modelling
with physics constraints:

```
gate = sigma(W [ssm_out ; physics_embed])
output = gate  ssm_out + (1 - gate)  constraints
```

### Architecture

Three jointly-trained components:
1. **MotionProjector** -- learned encode/decode between motion space (251-dim) and latent (256-dim)
2. **MotionSSM** -- 4-layer Mamba stack for temporal modelling
3. **PhysicsSSM** -- physics encoder + constraint projection + sigmoid gate

### Training Objective

```
L_total = L_reconstruction + lambda  L_physics

L_reconstruction = MSE(predicted_motion, target_motion)
L_physics = foot_sliding_penalty + ground_penetration + jerk_smoothness
```

### Physics State (64-dim, derived from motion data)

| Channels | Description |
|----------|-------------|
| [0] | Gravity prior (-9.81) |
| [1] | Pelvis height |
| [2] | Pelvis vertical velocity |
| [3] | Pelvis vertical acceleration |
| [4:7] | Root linear velocity (XYZ) |
| [7:10] | Root angular velocity |
| [10:12] | Estimated foot contacts (L/R) |
| [12:15] | Centre-of-mass velocity |
| [15:18] | Centre-of-mass acceleration |
| [18:21] | Root position (XYZ) |
| [21:24] | Angular momentum proxy |
| [24:64] | Reserved (zero-padded) |

### How to Train

```bash
python scripts/train_physics_ssm.py \
    --epochs 100 --batch-size 16 --lr 5e-5 \
    --lambda-physics 0.1 --device cuda
```

### Estimated Training Time

| Hardware | Epochs | Time |
|----------|--------|------|
| RTX 3050 4GB | 100 | ~60 hours |
| RTX 3080 10GB | 100 | ~15 hours |
| RTX 4090 24GB | 100 | ~5 hours |

**Checkpoint:** `checkpoints/physics_ssm/best_model.pt`
**Loaded by:** `SSMMotionGenerator._load_checkpoint()` in `ssm_generator.py`

---

## 4. Knowledge Base Retriever -- READY

Uses pretrained `all-MiniLM-L6-v2` (SBERT) + FAISS IndexFlatIP.

- **Data:** `data/knowledge_base/objects/common_objects.json` (12 curated entries)
- **Index:** Built automatically on first `setup()`, saved to `data/knowledge_base/embeddings/object_index.faiss`
- **No training needed** -- pretrained SBERT embeddings are sufficient

To expand the KB, add entries to `data/knowledge_base/objects/` following the existing JSON schema.

---

## 5. Modules Not Requiring Training

| Module | Implementation | Notes |
|--------|---------------|-------|
| M1 PromptParser | Regex + vocabulary | Fallback when T5 unavailable |
| M2 ScenePlanner | scipy L-BFGS-B solver | Deterministic constraint optimisation |
| M3 AssetGenerator | Pretrained Shap-E / TripoSR | Downloads from HuggingFace |
| M5 PhysicsEngine | PyBullet rigid-body sim | URDF humanoid + PD joint control |
| M6 RenderEngine | SMPL + aitviewer + OpenCV post-processing | Mesh rendering, motion blur, DoF, colour grade |
| M7 AIEnhancer | Pretrained SD 1.5 + ControlNet + AnimateDiff | Downloads from HuggingFace |

---

## Datasets

| Path | Size | Format | Used By |
|------|------|--------|---------|
| `data/AMASS/` | ~50 GB | `.npz` SMPL-X params | M4 training + retrieval |
| `data/inter-x/` | ~10 GB | `.npz` SMPL-X params | M4 training (multi-person) |
| `data/PAHOI/` | varies | `.npz` SMPL-X params | M4 training (HOI) |
| `data/ARCTIC/` | varies | `.npz` SMPL-X params | M4 training (hand interactions) |
| `data/m1_training/` | 35.5 MB | `.jsonl` (40k samples) | M1 T5 finetuning |
| `data/M1_VisualGenome/` | 4.85 GB | Raw VG data | M1 dataset building |
| `data/knowledge_base/` | ~1 MB | JSON + FAISS index | M1 KB retriever |

---

## Checkpoints

| Path | Size | Module | Status |
|------|------|--------|--------|
| `checkpoints/understanding/scene_extractor_v5/` | 293 MB | M1 T5 | Best (loss=0.062) |
| `checkpoints/motion_ssm/best_model.pt` | 58 MB | M4 MotionSSM | Best (loss=0.371) |
| `checkpoints/physics_ssm/best_model.pt` | TBD | M4 PhysicsSSM | **Trained** |

---

## References

1. **KIT-ML**: Plappert et al., 2016 -- <https://arxiv.org/abs/1609.04733>
2. **HumanML3D**: Guo et al., CVPR 2022 -- <https://arxiv.org/abs/2205.01061>
3. **Motion Mamba**: Zhang et al., ECCV 2024 -- <https://arxiv.org/abs/2403.07487>
4. **PINNs**: Raissi et al., JCP 2019 -- Physics-informed neural nets
5. **Neural ODEs**: Chen et al., NeurIPS 2018
