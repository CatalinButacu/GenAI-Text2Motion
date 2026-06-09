---
name: motion-representation
description: >
  The single canonical motion-representation contract. PRIMARY = standard HumanML3D-263 (citable
  FID track). SECONDARY = SMPL-X 168 (deferred whole-body studio demo). Use before writing any
  loader, model I/O, loss, or renderer that touches motion tensors, and whenever shapes/units/axes
  look wrong.
---

# Motion representation contract

There is **one** representation per track. Every module converts to/from it. Format/axis drift is
the #1 cause of silent breakage — never inline a second layout or guess the up-axis.

## PRIMARY — HumanML3D-263 (body-only, the citable-FID track)
Per-frame 263-dim feature in HumanML3D's **native** packing (do NOT re-pack it):
root angular velocity (1) + root linear velocity xz (2) + root height (1) + ric local joint
positions (21×3=63) + rot6d local joint rotations (21×6=126) + local joint velocities (22×3=66) +
foot contacts (4) = **263**, over the 22-joint SMPL skeleton.
- This is exactly what the reused **Guo et al. evaluator** consumes — feed it 263 **unmodified**
  (do not use the donor's 168→259 hack; that breaks comparability).
- Frame is HumanML3D's own (root-relative, Y-up in its recovered joint space). Use the **recovery**
  helper to go 263 → (T, 22, 3) joint positions for eval/viz. Do not hand-derive joints.
- Source data is ready: `donor data/humanml3d/{motion_data (263 .npy), mean_std/Mean.npy+Std.npy,
  split, texts}`. Normalize with the provided Mean/Std.

## SECONDARY — SMPL-X 168 (deferred 168 whole-body demo track)
Per-frame 168-dim, layout **matching the donor exactly** so its pretrained 168 RVQ + render reuse:

| Slice | Dims | SMPL-X param |
|---|---|---|
| `[0:3]`     | 3  | `root_orient` (axis-angle) |
| `[3:6]`     | 3  | `trans` (root translation) |
| `[6:69]`    | 63 | `body_pose` (21 joints × 3) |
| `[69:114]`  | 45 | `left_hand_pose` (15 × 3, full axis-angle, use_pca=False) |
| `[114:159]` | 45 | `right_hand_pose` |
| `[159:162]` | 3  | `jaw_pose` |
| `[162:168]` | 6  | `leye_pose` + `reye_pose` |

- **Z-up** representation (AMASS convention): `trans` index 5 = z (vertical), index 4 = y. Convert
  Z-up→Y-up **only at render** (rotation about X). If a body lies down, this is the bug.
- AMASS here is **SMPL-H** → `jaw_pose`/`eyes_pose` are zero-filled (face deferred).

## Conventions (assert at every boundary)
- Units: translation meters; 30 fps. Rotations axis-angle in the 168 layout; if a model wants 6D,
  convert at the model boundary and back — the stored contract stays as above.
- Normalization: per-channel `(x - mean) / std`. Always provide a matching `denormalize`.
  **Round-trip test mandatory:** `denorm(norm(x)) ≈ x` to 1e-5.

## Helpers to provide
- 263 track (`data/representation.py`): `recover_joints(feat263) -> (T,22,3)`, `normalize`,
  `denormalize`, `assert_263(t)` (last-dim 263, finite).
- 168 track: `to_smplx_params(vec168) -> dict`, `from_smplx_params(dict) -> vec168`, `assert_168(t)`.
Import these everywhere; never hand-slice the motion vector in model or render code.
