# CRISP-DM: M4a -- MotionSSM (Text-to-Motion Generator)

## 1. Business Understanding

**Question**: Given a text description of an action, can we generate a plausible
168-dim SMPL-X pose sequence?

**Goal**: Produce temporally coherent, text-conditioned pose sequences at 30 FPS
for downstream physics simulation and rendering.

**Success metrics**:
- FID <= 1.0 (measured against HumanML3D test distribution)
- R-Precision Top-3 >= 0.70 (text retrieval from motion)
- Foot skating ratio <= 10% of contact frames

---

## 2. Data Understanding

| Dataset | Sequences | Notes |
|---------|-----------|-------|
| **HumanML3D** | 14,616 | Body-only (22 joints), text annotations, 30 FPS |
| **AMASS** | ~40k | Full SMPL-X (168-dim), no text labels |
| **Inter-X** | ~8k | Two-person interactions, full SMPL-X, short captions |
| **PAHOI** | ~3k | Person-object interactions, full SMPL-X |

**Gaps**:
- HumanML3D has text but 263-dim representation (not SMPL-X) -> must convert
- AMASS has SMPL-X params but no text -> use filename verbs as weak labels

**Trade-off accepted**: Training on AMASS with weak (filename-derived) text labels
reduces text-motion alignment quality. Mitigated by `SBERTTextEncoder` which brings
richer semantic space to the conditioning.

---

## 3. Data Preparation

**Motion representation**: 168-dim SMPL-X axis-angle parameters:
- `[0:3]`   root_orient (global rotation)
- `[3:6]`   transl (global translation)
- `[6:69]`  pose_body (21 body joints x 3)
- `[69:114]` pose_lhand (15 joints x 3)
- `[114:159]` pose_rhand (15 joints x 3)
- `[159:162]` pose_jaw (1 joint x 3)
- `[162:168]` pose_eye (2 joints x 3)

**Normalisation**: Per-channel z-score (mean/std computed over training set, saved
as `stats.npz` alongside checkpoint).

**Sequence handling**:
- Max length: 200 frames (6.67s at 30FPS). Longer -> truncated, shorter -> zero-padded.
- `N_BODY_JOINTS` validator skips clips where the joint count != 22 (after the March 2026 fix)

---

## 4. Modelling

**Architecture**: `TextToMotionSSM` -- custom Mamba SSM with optional SBERT conditioning.

```
text -> [SBERTTextEncoder | SimpleTextEncoder] -> condition (B, d_model=256)
                                                     v
                            pos_embed(T) + condition -> x  (B, T, 256)
                                                     v
                           4x [BiMambaLayer | MambaLayer]  (residual)
                                                     v
                              MotionDecoder -> (motion (B,T,168), length (B,))
```

**Configuration flags** (all in `TrainingConfig`):
| Flag | Default | Effect |
|------|---------|--------|
| `use_sbert` | `False` | SBERT vs SimpleTextEncoder |
| `bidirectional` | `False` | BiMambaLayer vs MambaLayer |
| `d_model` | 256 | SSM hidden dim |
| `n_layers` | 4 | SSM depth |

**Novel contribution**: `PhysicsSSM` gate (see m4b_physics_ssm.md).

**Recommended training config for retraining with SBERT**:
```bash
python scripts/training/train_motion_ssm.py \
  --data-source humanml3d \
  --use-sbert \
  --bidirectional \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4 \
  --device cuda \
  --seed 42
```

---

## 5. Evaluation

| Metric | Current | Target | How to compute |
|--------|---------|--------|----------------|
| Val MSE | ~0.43 (FK-SSM best) | -- | `comparison_results.csv` |
| FID | **NOT COMPUTED** | <= 1.0 | `scripts/evaluation/compute_fid.py` \[TODO\] |
| R-Precision | **NOT COMPUTED** | Top-3 >= 0.70 | `scripts/evaluation/compute_fid.py` \[TODO\] |
| Foot skating | **NOT COMPUTED** | <= 10% | FK + contact label check |

> [!CAUTION]
> FID is the primary metric the field uses. Without it the dissertation cannot
> be compared to any published work. Implementing `compute_fid.py` is Phase 5.

---

## 6. Deployment

- Loaded via `SSMMotionGenerator(backend="ssm", config=cfg)` in `MotionStage`
- Inference: `generate(text_query, num_frames=N)` -> `MotionClip`
- Cloud training: Colab / Kaggle notebooks in `notebooks/colab_train_motion.ipynb`
