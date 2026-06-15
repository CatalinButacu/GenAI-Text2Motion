# Lesson 6 — FSQ: turning motion into tokens with a fixed grid (our tokenizer)

> Mirror of Lesson 5, for the **Grouped-FSQ tokenizer (Contribution A)**. Same beginner framing:
> what it is, how it is trained, why it needs almost no machinery, how we apply it. Values in
> Lesson 7. Code: `tokenizer.py`, `tokenizer_trainer.py`. Research: FSQ (Mentzer), ScaMo.

## 6.1 The same problem, a different answer
We still need continuous motion -> discrete tokens. RVQ's answer was "learn a box of crayons."
**FSQ's answer: don't learn any crayons — use graph paper.** Snap each number to the nearest line on
a fixed grid. The grid never moves and is never learned.

## 6.2 Finite Scalar Quantization: rounding onto a fixed grid
Take a **small** latent vector (say 4 numbers). For each number, allow only a few fixed values — a
**level count** per dimension, e.g. levels `(8, 5, 5, 5)` means dim 1 may take 8 values, dims 2-4 may
take 5. To quantize:
1. **bound** each number into the grid's range (a squash);
2. **round** it to the nearest allowed level.
The tuple of rounded values *is* the code; its index ranges over the product of levels
(8x5x5x5 = **1000**). No nearest-neighbour search, no stored vectors — just rounding.

Analogy: RVQ picks the closest crayon from a learned box; FSQ snaps your point to the nearest
intersection on fixed graph paper. The graph paper is the same for everyone, always.

**Definitions to note**
- **Levels** — how many fixed values each latent dimension may take.
- **Implicit codebook** — the product of levels (1000 here); never stored, just *implied* by the grid.

## 6.3 How FSQ is TRAINED (and why it needs almost nothing)
Same autoencoder: encoder -> small latent -> **round to grid** -> decoder -> reconstruction.
- **Rounding is not differentiable** -> same **STE** trick as VQ (backward pretends rounding is
  identity). This is the *only* shared piece of machinery.
- **No commitment loss** — there is nothing to commit to; the grid is fixed, it can't drift.
- **No codebook EMA** — there is no codebook to update.
- **No dead-code reset** — unused grid points are just unvisited coordinates that **cost zero
  params**; nothing collapses, nothing to reinitialise.

So FSQ training is just: **reconstruct + STE**. That radical simplicity (no collapse to babysit) is
the practical argument for FSQ.

## 6.4 Grouped-FSQ (ours): partition the latent into groups
A single small FSQ latent is too low-dimensional to represent 263-D motion (we learned this the hard
way — the residual-FSQ variant collapsed at recon-FID 0.22). **Fix:** split the latent into **G = 6
groups**, FSQ each group independently -> **6 codes per token step** (the same token budget as RVQ's
6 levels, for a fair comparison; and a wide-enough 6x4 = 24-D latent).

> Contrast with RVQ (Lesson 5): RVQ's 6 codes are **sequential refinement levels** (learned,
> additive); FSQ's 6 codes are **parallel partition groups** (fixed grids, independent). One learns
> and refines, the other partitions and rounds.

## 6.5 How WE do it (applied — `tokenizer.py`)
- **6 groups x (8,5,5,5) = 1000** codes/group; **shares** the conv encoder/decoder with RVQ (same
  width/downsample/resblocks) so the comparison isolates the quantizer.
- **Training loop** (same `tokenizer_trainer.py`): encode -> round -> decode; loss = **reconstruction
  L1 (+ velocity)** only — **commit is structurally 0**; AdamW + weight-EMA (eval smoothing, generic,
  not a codebook EMA); ~500 epochs; best by downstream recon-FID.
- Logs the same two tiers; a healthy FSQ shows `recon` falling, `perplexity`/`usage_frac` high, and
  **`commit 0.0000`** (the on-screen proof of "no machinery").

## 6.6 Research lineage
FSQ (Mentzer et al., 2309.15505) showed fixed-grid quantization matches learned VQ with ~100% code
usage and none of the machinery; ScaMo (2412.14559) confirmed FSQ > VQ on motion. Our contribution is
the **Grouped-FSQ formulation on HumanML3D-263 + the controlled head-to-head vs strong RVQ**.

### Check before Lesson 7
1. FSQ skips commitment, codebook-EMA and dead-code reset. For each, say in one phrase WHY it isn't
   needed.
2. What single piece of machinery does FSQ still share with VQ, and what is it for?
3. Why grouped (6 parallel groups) instead of one small FSQ latent or a residual stack?
