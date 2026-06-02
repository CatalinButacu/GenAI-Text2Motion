# Motion Generator Module

Text-to-motion generation. The single runtime backend is the **TextToMotionSSM**
research model defined in this repo (`nn_models.py` + `ssm/`), trained on AMASS /
HumanML3D. A trained checkpoint at `checkpoints/motion_ssm/best_model.pt` is
required -- `SSMMotionModel` raises `FileNotFoundError` if it is missing (no
silent fallback).

MoMask (CVPR 2024) was considered as a production alternative but is not
integrated; the comparison lives in the evaluation scripts, not in the pipeline.

## Output Format

`MotionClip`:

- `smplx_params`: `np.ndarray (num_frames, 168)` -- SMPL-X axis-angle (AMASS layout)
- `raw_joints`: `np.ndarray (num_frames, 22, 3)` -- Z-up metres from FK
- `fps`: int (30)
- `source`: `MotionSource`

### Motion-vector layout (168 dims)

- `[0:3]`    Root orientation (axis-angle)
- `[3:6]`    Root translation (Z-up metres)
- `[6:69]`   Body pose (21 joints x 3 axis-angle)
- `[69:159]` Hand pose (30 joints x 3 axis-angle)
- `[159:168]` Jaw + eyes (zero in AMASS)

## Files

| File           | Purpose                                                    |
|----------------|------------------------------------------------------------|
| `generator.py` | Production API; wraps `SSMMotionModel` + init-pose blend   |
| `ssm_model.py` | Checkpoint load + `generate_from_text_tokens` inference    |
| `nn_models.py` | `TextToMotionSSM` architecture                             |
| `ssm/`         | Mamba / S4 layer primitives                                |
| `clip_ops.py`  | Per-action generation, sequencing, biomechanical validation|
| `training/`    | Trainers (v1/v2/v3), dataset loaders                       |

## SSM architecture (research)

`SBERTEncoder -> FiLM -> MambaLayer x4 -> MotionDecoder`. Autoregressive inference
via `generate_ar`. ~2.8M params. Trained on AMASS (~16,884 sequences, SMPL-X
168-dim @ 30 fps). `PhysicsSSM` adds a physics-conditioned gate on top.

## References

- MoMask: Guo et al., CVPR 2024 -- <https://arxiv.org/abs/2312.00063>
- AMASS: Mahmood et al., ICCV 2019 -- <https://amass.is.tue.mpg.de/>
- SMPL-X: Pavlakos et al., CVPR 2019 -- <https://smpl-x.is.tue.mpg.de/>
- Mamba: Gu & Dao, 2024 -- <https://arxiv.org/abs/2312.00752>
