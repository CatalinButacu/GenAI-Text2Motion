# Applied Lesson A — The loss function and the optimizer (what we actually run)

> Out of the theory sequence (1-3 = representation); this covers *how the model is trained*.
> Grounded in `losses.py`, `trainer.py`, and `shared/config.py` (TrainCfg).

## A.1 Two families of loss (this is "combining both")

The generator predicts **discrete tokens**, so the total loss mixes two kinds:

1. **Classification — token cross-entropy (CE).** "Which of the 1000 codes is correct here?" Correct
   tool for a codebook choice. Blind spot: to CE every wrong token is *equally* wrong — it cannot see
   that some wrong codes decode to nearly-identical motion. It has no notion of *motion distance*.
2. **Geometric (regression) on the decoded motion.** Fix CE's blind spot with the **soft-decode
   trick**: logits -> softmax -> *expected* codes -> **frozen decoder** -> reconstructed `(T, 263)`,
   differentiably. Then penalize geometric error. The reconstruction term is **L1 over the full
   263**, so it penalizes **positions (ric) AND rotations (rot6d) AND velocities together** — this is
   the "combining both". Foot and root get extra dedicated terms.

## A.2 The exact recipe (losses.py + TrainCfg weights)

```
total = CE
      + 0.5 * recon      # L1 over full 263 (positions + rotations + everything)
      + 0.3 * velocity   # L1 on frame-to-frame change (smoothness)
      + 0.1 * foot       # L1 on the 4 foot-contact channels (anti foot-skate)
      + 0.3 * root       # L1 on root height (global stability)
```
CE drives the discrete choice; the geometric terms make "close in motion" count, which CE alone
cannot. Weights balance token accuracy vs motion fidelity.

## A.3 Which cost function is better

- **Tokens -> cross-entropy** (classification over the codebook; not L1/L2).
- **Geometry -> L1, not L2.** L2 punishes large errors quadratically -> the model hedges toward the
  *average* of plausible motions -> **blurry, over-smoothed** output. **L1 is outlier-robust and
  yields crisp, decisive motion.** Standard choice for motion regression; used throughout.

## A.4 Which optimizer (the full stack, already in trainer.py)

- **AdamW** — per-parameter adaptive rates (gradients differ wildly across embeddings / SSM-attention
  core / heads) + **decoupled weight decay** (`weight_decay=0.01`; plain Adam+L2 does this wrong).
  lr 2e-4 (generator group) and 1e-5 (unfrozen CLIP group).
- **Warmup -> cosine** — ramp lr from ~0 (early gradients are noisy and can wreck a sequence model),
  then cosine-anneal; schedule sized to the ~60-epoch peak window.
- **EMA 0.999** — evaluate a moving average of weights, not the jittery live weights -> stabler gens.
- **Gradient clipping at 1.0** — cap gradient norm so a rare exploding gradient (common in
  recurrent/sequence models) cannot blow up training.

> **Why not SGD?** SGD needs careful per-layer lr tuning and momentum to match Adam on transformers/
> SSMs; Adam's adaptivity is why the whole field uses AdamW for these models. SGD remains competitive
> mainly for CNNs with heavy tuning.

## A.4b FK-consistency loss (proposed, data-validated 2026-06-14)

Idea (user-proposed; an established technique — "forward-kinematics consistency loss"): the 263 stores
BOTH rot6d and ric positions, so force them to agree — run FK on rot6d, compare to positions. Two
flavors: **self-consistency** `FK(rot6d) vs model ric`, and **FK-to-GT** `FK(rot6d) vs GT positions`
(anchors rotations to truth, fights kinematic-chain error accumulation). Strictly richer than a
bone-length loss (checks the whole pose, not just segment lengths), so bone-length is dropped.

**Viability check (read-only, 30 random GT clips, `recover_from_ric` vs `recover_from_rot`):**
mean per-joint L2 **0.84 mm**, median 0.00 mm, **0.05% of body height** -> NEGLIGIBLE. The data's
own ric-vs-rot floor is ~0, so the loss is clean (not chasing a HumanML3D artifact). Worst joints =
wrists/hands (FK chain accumulation, expected); root exact. Differentiable FK already exists
(`recover_from_rot` / `recover_from_ric`). Decision: viable; both flavors usable; validate impact via
a local A/B before the final run.

## A.5 What to hold onto
1. Two loss families: **CE** (discrete token choice) + **L1 geometric** (motion fidelity via
   soft-decode through the frozen decoder).
2. The recon L1 over the full 263 is where **positions and rotations are penalized together**.
3. **L1 over L2** for geometry -> no blur.
4. **AdamW + warmup/cosine + EMA + grad-clip** — the correct, adopted stack for token-AR models.
