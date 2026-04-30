# Training Procedures

Training details for the three models trained from scratch / fine-tuned.

---

## Summary

| Model | Script | Dataset | Time | Status |
|-------|--------|---------|------|--------|
| Flan-T5-Small (M1) | `scripts/train_m1_t5.py` | 40K JSONL | ~3h |  Done |
| TextToMotionSSM (M4) | `scripts/train_motion_ssm.py` | KIT-ML (4,886) | ~8-12h |  Done |
| PhysicsSSM (M4) | `scripts/train_physics_ssm.py` | KIT-ML + physics | ~6-10h |  Done |

---

## M1: Flan-T5-Small (Scene Parser)

**Task**: Text -> structured JSON (entities, actions, relations)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (beta=0.9, beta=0.999) |
| Learning rate | 5e-5, linear decay with 5% warmup |
| Epochs | 15 |
| Batch size | 4 (gradient accum = 4, effective = 16) |
| Precision | FP16 (AMP) |
| Gradient clipping | 1.0 |
| Weight decay | 0.01 |

**v5 Results**: eval_loss=0.062, BLEU-4~=0.38, ROUGE-L~=0.52, Entity F1~=0.83

---

## M4: TextToMotionSSM

**Task**: Action text -> 251-dim motion feature sequence

| Parameter | Value |
|-----------|-------|
| Architecture | 4x MambaLayer, d_model=256, d_state=16 |
| Optimizer | AdamW, lr=1e-3, weight_decay=0.01 |
| Epochs | 250 |
| Loss | Masked MSE + length prediction |
| Batch size | 32 |
| Noise warm-start | sigma=0.01 for first 10 epochs |
| Data augmentation | Speed perturbation, temporal jitter |

**Result**: val_loss = 0.371 (checkpoint: `checkpoints/motion_ssm/best_model.pt`)

---

## M4: PhysicsSSM (Novel Contribution)

**Task**: Refine MotionSSM output with physics constraints

| Parameter | Value |
|-----------|-------|
| Architecture | Gated Mamba + physics encoder, d_model=256 |
| Gate mechanism | `gate = sigma(W[ssm_out; phys_embed])` |
| Optimizer | AdamW, lr=5e-4 |
| Epochs | 100 |
| d_physics | 64 (velocities, contacts, CoM) |

**Physics-aware loss**:
```
L = L_recon + lambda_slide  L_foot_slide + lambda_pen  L_ground_pen + lambda_jerk  L_jerk
```

- `L_foot_slide`: Penalise foot velocity when contact label is active
- `L_ground_pen`: Penalise pelvis Y < 0 (below ground)
- `L_jerk`: Regularise third derivative of joint trajectories

**Checkpoint**: `checkpoints/physics_ssm/`

---

## Hardware Requirements

All training done on NVIDIA RTX 3050 (4 GB VRAM):

- T5: fits in fp16 with gradient checkpointing
- MotionSSM: small model (7.5M), fits easily
- PhysicsSSM: 12M params, fits with batch 32
