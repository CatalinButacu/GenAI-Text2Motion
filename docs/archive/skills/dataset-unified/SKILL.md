---
name: dataset-unified
description: >
  Recipe for loading the full multi-source motion corpus at scale (the 290 GB the prior run
  ignored): AMASS + HumanML3D + Inter-X, normalized, mirrored, cached, with clean splits. Use when
  building/fixing dataset loaders, adding a source, or when "only a fraction of data is used."
---

# Unified multi-source dataset

The prior run trained on ~14k HumanML3D clips while 290 GB sat unused → early bad minimum. This
recipe makes the **full mix** the default.

## Sources (donor paths; point configs here, don't duplicate)
- `D:\Facultate\dissertation\data\humanml3d` — text-paired, the Phase-1 primary corpus (263-dim).
- `D:\Facultate\dissertation\data\amass` — raw SMPL-X, **Z-up**, all subsets (151 G). Phase-2 whole-body.
- `D:\Facultate\dissertation\data\inter-x` — multi-person extension (optional, single-actor subset usable).
- `D:\Facultate\dissertation\data\stats` — precomputed normalization (reuse only if representation matches).

## Pipeline
1. **Load** each source via its own loader → the canonical representation (`motion-representation`).
   Convert AMASS **Z-up → Y-up** at load and assert it.
2. **Filter** garbage: NaN frames, T-pose-only clips, velocity outliers (port the donor's quality
   audit logic). Log how many clips each source contributes.
3. **Augment**: mirror-augmentation (left/right swap on the SMPL-X joint map). HumanML3D ships
   mirrored copies; for AMASS apply the mirror transform.
4. **Normalize** per-channel with stats computed over the *training* split only (no leakage).
5. **Split**: respect HumanML3D's official train/val/test; for AMASS-only data, split by sequence id.
6. **Cache** the preprocessed tensors to `data/.cache/` (joblib/npz, gitignored). Provide a
   `prebuild_cache` script; never re-preprocess 290 GB inside the training loop.

## Checks before declaring done
- Print per-source clip counts and total — confirm it's the full corpus, not a subset.
- A `WeightedRandomSampler` or interleave so no single source dominates if mixing.
- Round-trip a random sample through `denormalize` → aitviewer and confirm it looks like real motion.
- Phase ordering: HumanML3D-263 first (Phase 1), SMPL-X whole-body second (Phase 2).
