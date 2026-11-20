---
name: port-from-donor
description: >
  Selectively reuse assets from the donor project at D:\Facultate\dissertation — data, SMPL-X
  models, eval matcher, and reference loaders/renderers — into this clean repo, WITHOUT dragging
  in the model/training core that plateaued. Use whenever you need donor data paths, want to
  re-implement a loader/renderer, or are tempted to copy donor code.
---

# Porting from the donor (D:\Facultate\dissertation)

The donor is a **read-only resource**, not a codebase to fork. Reuse assets; re-implement code.

## Reuse AS-IS (point at it / copy data, don't re-derive)
- **Datasets:** `D:\Facultate\dissertation\data\{amass, humanml3d, inter-x, arctic, pahoi}` — point
  configs at these paths; do not duplicate 290 GB. (`amass` is Z-up; convert at load.)
- **SMPL-X bodies:** `data\models_smplx_v1_1.zip` — unzip to a configured `SMPLX_MODELS` path.
- **Eval matcher:** `data\t2m\text_mot_match` — the fixed FID/R-precision network. Reuse exactly.
- **Normalization stats:** `data\stats` — reuse only after confirming the representation matches.
- **Vocabulary:** `data\vocabulary\*.yaml`.

## Re-implement CLEAN (read for the contract, then write fresh)
- Loaders: `src\data\*` (AMASS/HML3D/Inter-X, Z-up→Y-up, SMPL-X packing, normalization).
- Render: `src\modules\render\*` (smplx_render, aitviewer chat_viewer).
- Streaming infra: `src\modules\runtime\*` (StreamBus).
- Eval scripts: `scripts\evaluation\*` (FID/metrics drivers).

## Do NOT port (this is what plateaued)
- `src\modules\motion\nn\*` model core, the trainer, the loss-of-only-token-CE. Read the
  **diagnosis** (`.claude\docs\TRAINING_DIAGNOSIS_AND_PHYSICS.md`) for lessons, then build new
  per the accepted ADRs.

## Method
1. `Read` the donor file to learn its I/O contract and gotchas (units, axes, shapes).
2. Write a fresh, minimal equivalent in this repo's `src/`.
3. Prove equivalence with a tiny round-trip test (same input → same shape/units).
4. Never `import` from the D: tree; copy data, not code.
