# Training Reference

Complete training strategy for each learned model in the pipeline.

---

## Data Available

| Dataset | Sequences | Format | Used by |
|---------|-----------|--------|---------|
| AMASS | ~16 900 | `.npz` SMPL-X (T, 168) @ 30-120 fps | M1 labels, M4 motion, M4 physics |
| Inter-X | ~57 000 | `.npz` SMPL-X pairs | M4 multi-person motion |
| ARCTIC | ~1 500 | `.npy` body + object trajectories | M4 interaction motion |
| PAHOI | ~6 400 | Mixed | M4 interaction motion |
| Knowledge base | ~500 entries | `.json` | M1 KB enrichment |
| T5 scene labels | generated from above | `.jsonl` | M1 fine-tuning |

All motion data is standardised to 30 fps and 168-dim SMPL-X format before training.

---

## M1 -- T5 Scene Parser

**Script:** `scripts/training/train_m1_t5.py`
**Checkpoint:** `checkpoints/understanding/scene_extractor_v5`

### Architecture
Fine-tuned `flan-T5-small` (60M params, seq2seq). Braces are replaced with
`<extra_id_0>` / `<extra_id_1>` during training because T5 tokenises `{}`
poorly; `T5SceneParser.postprocess()` reverses this.

### Training data
`.jsonl` files with `(prompt, scene_json)` pairs built from vocabulary in
`src/shared/vocabulary.py`. Run `scripts/data/build_vg_dataset.py` to
regenerate.

### Loss
Cross-entropy on decoder token predictions (masked on padding), via HuggingFace
`Seq2SeqTrainer`. Evaluated by JSON syntax rate, Entity F1, and BLEU-4.

### Key hyper-parameters
```
epochs=15  batch=4  lr=5e-5  warmup=5%  grad_accum=4
max_input=256  max_output=512  num_beams=4
```

---

## M4a -- MotionSSM

**Script:** `scripts/training/train_motion_ssm.py`
**Checkpoint:** `checkpoints/motion_ssm/best_model.pt`

### Architecture
`TextToMotionSSM`: `SimpleTextEncoder` (hash-token embedding + 2-layer
Transformer) -> condition projection -> `N` Mamba/S4 layers with pre-norm
residuals -> `MotionDecoder` (project to 168-dim + predict sequence length).

```
d_model=256  d_state=32  n_layers=4  motion_dim=168
max_motion_length=200 frames  max_text_length=64 tokens
```

### Training data
AMASS via `MotionDataset` (80/20 split). Online augmentation: temporal crop,
speed perturbation x[0.8, 1.2], Gaussian noise sigma=0.002 on joint angles
(translation channels kept clean).

### Loss
```
L = MSE(pred_motion, gt_motion)  +  lambda_len x MSE(pred_length, gt_length)
```
Both terms computed only over non-padded frames (masked by `motion_mask`).

### Key hyper-parameters
```
epochs=50  batch=16  lr=5e-5  weight_decay=0.01
warmup=1000 steps  grad_clip=1.0  early_stop_patience=20
```

---

## M4b -- PhysicsSSM  *(novel contribution)*

**Script:** `scripts/training/train_physics_ssm.py`
**Checkpoint:** `checkpoints/physics_ssm/best_model.pt`

### Architecture
Loads a frozen or jointly trained `MotionSSM` and adds a physics-constrained
gating layer:

```
ssm_out   = MotionSSM(motion_input)            # Mamba temporal features
phys_emb  = MLP(physics_state)                 # encode 64-dim physics vector
gate      = sigmoid(W [ssm_out ; phys_emb])    # learned blending gate
output    = gate * ssm_out + (1gate) * constraints
```

`constraints = Linear(physics_state)` directly projects physics into motion
space. The gate learns when to trust physics (e.g. foot-ground contact) vs
SSM motion priors (e.g. swing phase).

Parameters: `PhysicsSSM` + `MotionProjector` (168  d_model) trained jointly.

```
d_model=256  d_state=32  d_physics=64  n_layers=4
```

### Training data
Same AMASS sequences as M4a; physics state derived by `extract_physics_state()`
(root velocity/acceleration, foot contacts, pelvis height). Global normalisation
statistics computed on training split and shared with validation.

### Loss
```
L = MSE(pred_motion, gt_motion)  +  lambda_phys x physics_violation_loss
```

`physics_violation_loss` penalises three biomechanical constraints:
1. **Foot sliding** -- root velocity during contact frames should be near zero
2. **Ground penetration** -- pelvis height below floor plane (ReLU penalty)
3. **Jerk** -- third derivative of motion (smoothness regulariser, weight 0.01)

### Key hyper-parameters
```
epochs=100  batch=16  lr=5e-5  lambda_physics=0.1
warmup=500 steps  grad_clip=0.5  early_stop_patience=15  grad_accum=1
```

`grad_clip=0.5` is stricter than M4a because the composite loss landscape is
sharper when physics and reconstruction objectives disagree.

---

## Training Order

Train in this order because each stage depends on the previous.
**Step 0 (data generation) must run before M1 training.**

```
0. Generate M1 data  ->  python scripts/data/build_action_dataset.py
                         (no downloads needed; uses vocabulary.py action space)
                         Optional: also run scripts/data/build_vg_dataset.py if
                         data/M1_VisualGenome/scene_graphs.json is available.

1. M1    ->  python scripts/training/train_m1_t5.py
2. M4a   ->  python scripts/training/train_motion_ssm.py
             (train from scratch; old 251-dim checkpoints are incompatible)
3. M4b   ->  python scripts/training/train_physics_ssm.py
             (train from scratch after M4a; uses M4a checkpoint as backbone)
```

M2 (constraint layout), M5 (PyBullet), M6 (SMPL-X render) are not learned --
they use optimisation or pretrained weights respectively.

---

## Known Issues

| Model | Status | Notes |
|-------|--------|-------|
| PhysicsSSM | Shim preserved | Checkpoint pickled with old `src.modules.motion.physics_trainer` path. Shim at that path re-exports the class; transparent to users. |
| MotionProjector | **Must retrain** | Old checkpoints are 251-dim; current architecture is 168-dim (SMPL-X). `load_checkpoint()` now auto-detects and skips incompatible projector weights with a clear warning rather than crashing. |
| MotionSSM | **Must retrain** | Old checkpoints trained with 251-dim projector. Retrain M4a from scratch, then M4b. |
| M1 training data | **Fixed** | VG dataset (static images) had no motion action predicates. `build_action_dataset.py` generates action-focused pairs directly from `vocabulary.py` without any external downloads. |
