# Training plateau + physics-constraint gap — diagnosis & literature

This doc answers three questions:

1. **Why does training plateau at val_ce ≈ 4.64 / top-1 ≈ 12 %?**
2. **The thesis is "physics-constrained text-to-motion" — where is the physics in our pipeline?**
3. **What architectures from the literature actually work, and which experiments failed?**

---

## 1. Diagnosis of the training plateau

### What the trainer actually optimises

`src/modules/motion/training/trainer_utils.py:300-302`:

```python
tok_loss = token_ce_loss(logits, target_tokens_cut, latent_mask_cut)   # CE on K=6 codebooks
len_loss = F.mse_loss(length_pred, batch["length"])
loss     = tok_loss + 0.1 * len_loss
```

That is the **only** signal. There is no reconstruction loss in continuous motion
space, no foot-contact loss, no velocity loss, no physical-plausibility loss.

### Why 12 % top-1 is roughly the ceiling for this setup

| Cause | Evidence in our repo | Impact |
|---|---|---|
| **Discrete CE on residual VQ codes is brutal** | `K=6` codebooks × `V=512` entries → ~5.0 nat upper bound. We sit at 4.64. | Inflated CE is normal even for SOTA — but lack of reconstruction loss means the model never learns *near misses count*. |
| **Model is small for the task** | `d_model=384`, `n_layers=6`, ~5 M params for the SSM. T2M-GPT uses ~80 M; MoMask uses ~150 M. | Capacity bottleneck. |
| **Training too short** | 24 epochs achieved; MDM and T2M-GPT both report ≥ 200 epochs on HumanML3D. | We stopped at the elbow of the learning curve, not its plateau. |
| **Single loss term** | No `foot_contact_loss`, no `position_loss`, no `velocity_loss` — `grep "_loss" src/modules/motion/` returns only `commit_loss` (VQ) and `tok_loss`. | The model is allowed to be jittery as long as token guesses are close in code-space. Token-space proximity ≠ motion-space plausibility. |
| **No EMA / model averaging** | `base_trainer.py` keeps last checkpoint, not EMA weights. MDM and MoMask both apply EMA with decay 0.999. | Final-step weights are noisy. |
| **Training data is one source, downsampled** | `unified_factory` is wired but the headline run pulls only HumanML3D (~14 k clips). MoMask trains on HumanML3D + KIT-ML + mirror augmentation. | Less data variance ⇒ early convergence to a bad minimum. |
| **Text encoder is fully frozen CLIP-B/32** | `freeze_sbert: true` in `configs/motion_ssm.yaml`. | Frozen CLIP gives a static condition signal — no gradient to refine the text-motion mapping. T2M-GPT unfreezes the last 2 layers of CLIP; MotionGPT trains its own T5 conditioner. |

**Quick wins (no architecture change):**
1. Train for 100 + epochs and apply EMA (decay 0.999).
2. Add an L1 reconstruction loss in motion space: decode `argmax(logits)` through the RVQ decoder and compute `L1(reconstructed_pose, target_pose)`. Even a `0.1 ×` weight stabilises training dramatically.
3. Enable mirror augmentation (HumanML3D already ships mirrored copies).
4. Unfreeze the last 2 CLIP layers.

**Architecture changes (next-tier):**
5. Replace `independent K-head` arch with `residual_k` (already implemented, never tested at scale). This matches MoMask's hierarchical residual transformer.
6. Use Mamba/S4 with `d_state=128` instead of 64 — Mamba scales much better past 6 layers.

---

## 2. Where is the physics in our pipeline?

### Current state (honest answer)

Our "physics-constrained" claim today lives **only in the prompt parser**:

```
src/shared/vocab_objects.py   ← knows the word "gravity"
src/modules/understanding/actions.py ← knows the words "jump", "fall", etc.
```

That is **lexical physics**, not generative physics. The model is never told that
its output should obey gravity, foot contact, or non-penetration. There is no
physics simulator in the loop, no contact loss, no plausibility post-processing.

### What "physics-constrained" means in the literature

There are three established families:

| Family | Where physics enters | Example papers |
|---|---|---|
| **A. Geometric losses in supervised training** | Foot-contact loss + position loss + velocity loss added to the standard reconstruction loss. No simulator. | MDM (foot_contact_loss only fires when GT foot velocity ≈ 0). PhysFormer. |
| **B. Physics-projection in the diffusion / sampling loop** | A physics simulator (Isaac Gym, MuJoCo) projects each denoising step back onto a feasible manifold. | **PhysDiff** (NVIDIA, ICCV 2023 Oral). |
| **C. RL with a physics simulator as environment** | The motion model becomes a policy; rewards include tracking + balance + contact. | UniHSI, **SuperPADL** (Juravsky 2024), CALM, ASE. |

### Realistic options for our thesis

Given that we already have an RVQ + SSM kinematic backbone, **family A is the
right fit**. It requires no simulator, no RL, no extra GPUs. Adding it is ~150
LOC. Family B and C are full PhDs by themselves.

**Concrete plan — add a foot-contact + velocity loss to the SSM trainer:**

```python
# pseudo-code, integrate into trainer_utils.run_train_epoch
recon_motion = rvq.decode(argmax_logits)          # (B, T, 168)
gt_motion    = mgt                                # (B, T, 168)

# L1 reconstruction (motion-space)
recon_loss = F.l1_loss(recon_motion, gt_motion)

# Foot contact (joints 7, 10 = left/right toes in SMPL-X)
foot_vel_pred = (recon_motion[:, 1:, foot_idx] - recon_motion[:, :-1, foot_idx]).norm(dim=-1)
foot_vel_gt   = (gt_motion[:, 1:, foot_idx] - gt_motion[:, :-1, foot_idx]).norm(dim=-1)
contact_mask  = (foot_vel_gt < 0.01).float()     # GT considers foot in contact
foot_contact_loss = (foot_vel_pred * contact_mask).mean()

loss = tok_loss + 0.05*recon_loss + 0.1*foot_contact_loss + 0.1*len_loss
```

This is the **MDM recipe** (Tevet et al., 2022) and is the lowest-risk way to add real physics signal to our current architecture.

---

## 3. Literature review — what works, what failed

### What is proven to work on HumanML3D

| Method | FID ↓ | Top-1 ↑ | Year | Key idea |
|---|---|---|---|---|
| MDM (Tevet et al., arXiv:2209.14916) | 0.544 | 0.611 | 2022 | Predict **the sample** (not the noise) in a diffusion transformer. Adds foot_contact / position / velocity geometric losses. **Lightweight resources.** |
| MotionDiffuse | 0.630 | 0.491 | 2022 | First text-conditioned motion diffusion. Outperformed by MDM. |
| T2M-GPT (Zhang et al., arXiv:2301.06052, CVPR 2023) | 0.116 | 0.491 | 2023 | VQ-VAE + GPT decoder. **Beats diffusion** despite simplicity. Authors note: *"dataset size is a limitation"*. |
| MoMask (Guo et al., arXiv:2312.00063, CVPR 2024) | **0.045** | **0.521** | 2024 | Residual-VQ + **two** bidirectional masked transformers (base + residual). Current SOTA on HumanML3D / KIT-ML. |
| CrossDiff (arXiv:2312.10993, ECCV 2024) | competitive | — | 2024 | Train one diffusion model in a shared 2D/3D space; cross-decode at sampling. Useful for 2D-only data. |
| **PhysDiff** (Yuan et al., arXiv:2212.02500, ICCV 2023 Oral) | comparable | comparable | 2023 | MDM + physics-projection per denoising step. **+78 % physical plausibility** over MDM. No quality regression. |
| SuperPADL (Juravsky 2024, arXiv:2407.10481) | n/a (RL metrics) | n/a | 2024 | RL + supervised distillation; 5 000 skills running real-time on a single GPU. Path to interactive control. |

### What failed (so we don't repeat)

| Approach | Why it failed | Source |
|---|---|---|
| **Direct continuous regression (text → 168-dim pose stream)** | Mode collapse + jitter; no high-frequency detail. | Pre-2022 baselines in HumanML3D paper. |
| **Naïve diffusion predicting noise on raw poses** | Hard to add geometric losses on noise-space; physical plausibility poor. **MDM's key insight is sample-prediction, not noise-prediction.** | MDM ablation. |
| **Single-codebook VQ** | Reconstruction too coarse for fine motion. **Residual VQ (≥ 4 codebooks)** is now standard. | T2M-GPT and MoMask both use multi-codebook VQ. |
| **Autoregressive token sampling with greedy decode** | Repetitive / collapses to motionless tokens. | T2M-GPT corruption strategy paper. **We currently default to `temperature=1.0, top_p=1.0` which is exactly greedy.** |
| **Frozen large language encoders without adaptation** | Text-motion alignment poor. T2M-GPT lightly fine-tunes the encoder; MotionGPT trains its own. | T2M-GPT §4.4. |
| **Pure RL on > 100 skills** | Scales poorly past hundreds of clips; sample efficiency cliff. **SuperPADL solves this with progressive distillation.** | SuperPADL §3, §5. |
| **PhysDiff without sample-prediction backbone** | Earlier physics-aware diffusion variants struggled because they tried to project in noise space. | PhysDiff §3.2. |

### Net recommendation for our thesis

1. **Keep the RVQ-SSM kinematic backbone.** This is closer to T2M-GPT / MoMask than to MDM — the strongest direction on HumanML3D.
2. **Add geometric losses (family A).** This is the MDM trick that makes the motion *look* right and lets us put "physics-constrained" in the title without bolting on a simulator.
3. **Switch the K-head arch to `residual_k`** (already in repo). This is the MoMask recipe.
4. **Apply non-greedy sampling at inference** (`temperature=1.2, top_p=0.9`) to avoid the T2M-GPT failure mode of greedy collapse.
5. **Defer PhysDiff / RL.** These are interesting *future work* paragraphs but not realistic before defense.

---

## References

- Tevet et al., *Human Motion Diffusion Model (MDM)*, arXiv:2209.14916 (2022)
- Zhang et al., *T2M-GPT*, arXiv:2301.06052 (CVPR 2023)
- Yuan et al., *PhysDiff*, arXiv:2212.02500 (ICCV 2023 Oral)
- Guo et al., *MoMask*, arXiv:2312.00063 (CVPR 2024)
- Ren et al., *CrossDiff*, arXiv:2312.10993 (ECCV 2024)
- Juravsky et al., *SuperPADL*, arXiv:2407.10481 (SIGGRAPH 2024)
- HumanML3D dataset, https://github.com/EricGuo5513/HumanML3D
