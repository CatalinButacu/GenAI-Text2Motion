---
name: t2m-losses
description: >
  The generator training recipe that escapes the prior plateau: the multi-term loss (token-CE +
  soft-decode reconstruction + velocity + foot/root), EMA, CFG dropout, text-encoder unfreezing,
  schedule, and non-greedy sampling. Use when building/tuning the generator trainer or when a run
  plateaus or looks jittery.
---

# Generator losses & training recipe (Contribution B)

The prior run died on **token-CE alone, no EMA, 24 epochs, frozen encoder, greedy sampling**. Make
each of those impossible by default.

## Loss (do NOT ship token-CE alone)
```
loss = ce + w_recon*recon + w_vel*velocity + w_foot*foot + w_root*root + w_len*length
```
- **token-CE** over the K RVQ codebooks (masked by valid frames/latents).
- **soft-decode reconstruction** (the key trick): softmax over each codebook's logits → expected
  code embedding → run through the **frozen** RVQ decoder (gradients ON, decoder weights NOT in the
  optimizer) → L1 vs GT motion. Gives motion-space signal without a non-differentiable argmax.
- **velocity**: L1 on first-order temporal differences of the soft-decoded motion.
- **foot contact**: penalise predicted foot velocity where GT foot is planted (contact mask). On 263,
  use the foot-contact channels / `recover_joints` toe velocity; on 168, use the `transl_z` floor proxy.
- **root**: keep vertical drift bounded (root height term).
- **length**: small MSE on predicted sequence length.
Start weights ~ `recon 0.5, vel 0.3, foot 0.1, root 0.3, len 0.1`; tune by the FID, not the train loss.

## Stabilisers (all on by default)
- **EMA** decay 0.999 on weights; **evaluate the EMA copy**.
- **Unfreeze the top ~2 layers** of the text encoder (not all, not none) — frozen CLIP/SBERT gave a
  static, ungradable condition last time.
- **CFG dropout**: drop the text condition ~10% of steps so classifier-free guidance works at inference.
- Cosine schedule + warmup; **100–200+ epochs**; resume-safe checkpoints; log seeds (twin must match).

## Inference
- **Non-greedy** sampling: temperature ~1.0–1.2, top-p ~0.9, CFG scale ~2–4. Greedy collapses (the
  T2M-GPT failure mode). Use the SAME sampler in `t2m-eval` and `streaming-decode`.

## Gates
- `sanity-overfit` (one batch → loss ≈ 0) before any full run — non-negotiable.
- Cheap FID checkpoint early (don't train blind). For the twin: identical loss/schedule/data/budget
  across transformer and SSM, or the comparison is invalid.
