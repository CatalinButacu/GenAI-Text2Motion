# Lesson 6 — FSQ: turning motion into tokens with a fixed grid (our tokenizer)

> Mirror of Lesson 5, for the **Grouped-FSQ tokenizer (Contribution A)**. Same beginner framing:
> what it is, how it is trained, why it needs almost no machinery, how we apply it. Values in
> Lesson 7. Code: `tokenizer.py`, `tokenizer_trainer.py`. Research: FSQ (Mentzer), ScaMo.

## 6.1 The same problem, a different answer
We still need continuous motion -> discrete tokens. VQ (L5) *learned centroids* and searched for the
nearest. **FSQ instead rounds onto a fixed integer lattice** — no centroids, no search, no learning
of the quantizer at all.

## 6.2 Finite Scalar Quantization: rounding onto a fixed lattice (the math)
Take a SMALL latent `z ∈ R^d` (here `d = 4` per group). Give each dimension `i` a fixed number of
**levels** `L_i` (e.g. `(8, 5, 5, 5)`). Quantize each dimension independently — bound, then round:

```
half_i  = (L_i - 1) / 2
z_hat_i = round( half_i * tanh(z_i) )        # STE on round; lands in {-half_i, ..., half_i}
idx_i   = z_hat_i + half_i                    # per-dim index in {0, ..., L_i - 1}
```

The single token index is the **mixed-radix** combination of the per-dim indices:

```
index = sum_i  idx_i * prod_{j<i} L_j         # ranges over prod_i L_i = 8*5*5*5 = 1000
```

No nearest-neighbour search and **no stored vectors** — the "codebook" is the fixed lattice
`{0..L_i-1}`, never materialised. (Implementation detail: the exact `bound` adds a small even/odd
shift so the levels straddle zero correctly — see `FSQ.bound` in `tokenizer.py`; the essence is
`tanh`-bound then `round`.)

**Definitions to note**
- **Levels `L_i`** — allowed values per latent dimension.
- **Implicit codebook** — `prod_i L_i` (= 1000); implied by the lattice, never stored.

## 6.3 Training FSQ — the math is "STE only"
Same autoencoder: encoder -> `z` -> round-to-lattice `q(z)` -> decoder. The only non-differentiable
op is `round`, handled by the **same STE** as VQ:

```
z_q = z + sg( q(z) - z )        # forward = q(z); backward = identity
L   = L_recon                   # that's the whole objective
```

Every VQ training term **vanishes**, and here is exactly why:
- **commitment** `beta*||z - sg(e)||^2` -> there is no learned `e` to commit to (the lattice is fixed);
- **codebook loss / EMA** -> there are no centroids to move;
- **dead-code reset** -> unused lattice points are coordinates with **zero parameters**; nothing
  collapses, nothing to reinitialise.

So FSQ training = **reconstruct + STE**. That radical simplicity (no collapse to babysit) is the
practical argument for FSQ.

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
