# Lesson 4 — From motion to tokens: the tokenizer (Contribution A)

> Goal: bridge from "how motion is represented" (Lessons 1-3) to "a model that generates motion".
> The key move is turning the continuous 263 vector into a few discrete integers.

## 4.1 Why not just generate the 263 floats directly?

You could try to predict the 263 numbers per frame as raw floats. The field mostly does NOT, for
three reasons:

1. **Continuous high-dim regression blurs.** Predicting 263 continuous values per frame, the model
   hedges toward the average of plausible motions -> washed-out, mushy output (the same L2-blur
   problem from the loss lesson, one level up).
2. **Autoregression wants choices, not values.** The thesis novelty is *streaming* = predict the
   next step from the past. "Predict the next *value*" is hard and unstable; "predict the next
   *token* from a fixed vocabulary" is a clean classification problem — the exact thing transformers
   and SSMs are superb at.
3. **It unlocks the LLM playbook.** Once motion is a sequence of tokens, generating motion becomes
   *literally the same problem* as generating text: predict the next token. All the machinery
   (cross-entropy, sampling, CFG, KV-cache vs SSM-state) transfers directly.

## 4.2 The core analogy: a tokenizer is BPE for motion

A text LLM never sees raw characters — a **tokenizer** (BPE) chops text into a fixed vocabulary of
sub-word tokens, and the model predicts the next token. We do the same for motion:

| Text LLM | Our motion model |
|---|---|
| characters | 263 floats per frame |
| BPE tokenizer | the **motion tokenizer** (Contribution A) |
| token vocabulary (~50k) | the **codebook** (6 x 1000 codes) |
| "predict next word" | "predict next motion token" |

Once you see this mapping, the whole generator (Lesson 5) is just "an LLM over motion words."

## 4.3 What the tokenizer is: encoder -> quantizer -> decoder

A learned compressor with three parts (all in `tokenizer.py`):

- **Encoder** — convolutions that take `(T, 263)` and produce a lower-rate latent. Ours
  **downsamples time by 4**: 4 frames in -> 1 latent step out (two stride-2 convs). So a 196-frame
  clip becomes ~49 token steps.
- **Quantizer** — snaps each latent to the nearest entry of a fixed set, turning a continuous vector
  into **discrete integer codes**. This is the FSQ step (next section).
- **Decoder** — convolutions that take codes back up to `(T, 263)`. Used to *reconstruct* motion
  (and, in the generator's soft-decode loss, to turn predicted tokens back into motion).

**Definitions to note**
- **Codebook** — the fixed vocabulary of code vectors. Each token is an index into it.
- **Downsampling factor** — frames per token step (ours = 4). Fewer tokens = shorter sequences for
  the generator = faster, but coarser.
- **Reconstruction** — encode then decode; how faithfully the motion survives the round trip. Our
  headline metric (recon-FID 0.0266).

## 4.4 The landscape of quantizers (every option, and its catch)

The quantizer is the heart of the tokenizer — turning the continuous latent into discrete codes.
There is a whole family, and they fall on a spectrum from "learned codebook" to "fixed grid":

| Quantizer | How | Catch | Cite |
|---|---|---|---|
| **Plain VQ-VAE** | one learned codebook, nearest-neighbour lookup | **codebook collapse** / dead codes — most codes go unused. T2M-GPT naive VQ recon-FID **0.492** (awful) | van den Oord 2017 (1711.00937) |
| **VQ + EMA + dead-code reset** | update codes by moving average; reinit unused ones | works, but fragile machinery; the *minimum* viable VQ. T2M-GPT: 0.070 (7x better than naive) | Jukebox, SoundStream |
| **RVQ (Residual VQ)** | a cascade: each codebook quantizes the *residual* of the previous | more capacity -> finer recon (MoMask **0.019**), but still learned codebooks + EMA/reset + commitment loss + quant-dropout | SoundStream (2107.03312), EnCodec (2210.13438), MoMask (2312.00063) |
| **Grouped / Product VQ** | split the latent into groups, quantize each separately | combinatorial vocabulary from small codebooks; still learned | — |
| **FSQ (Finite Scalar Quantization)** | **no learned codebook** — round each latent dim onto a small fixed grid | implicit vocab = product of levels; ~100% usage, **no collapse, no EMA/reset/commitment** | Mentzer 2023 (2309.15505); motion: ScaMo (2412.14559) |
| **LFQ / MAGVIT-v2** | binary per-dim quantization, huge implicit vocab | image/video; vocab too large for HumanML3D's data scale | 2310.05737 |

The trend across audio -> image -> motion: **move work out of the learned codebook and into a fixed
structure**, because learned codebooks are the thing that collapses and needs babysitting. FSQ is the
extreme: *no* codebook at all.

### RVQ mechanics (the baseline)
**Residual Vector Quantization** is a cascade of *learned* codebooks, coarse-to-fine:
1. Take the encoder latent `z`. Find its nearest code `c1` in **codebook 1**. Quantized so far = `c1`.
2. Compute the **residual** `r1 = z - c1` (the leftover error). Find its nearest code `c2` in
   **codebook 2**. Quantized = `c1 + c2`. New residual `r2 = r1 - c2`.
3. Repeat for all 6 codebooks: `z ~= c1 + c2 + ... + c6`. The **6 indices** are the token step.

So each level *refines the previous error* — early codebooks capture gross motion, later ones add
detail. The codebooks are **learned** and need the machinery: **EMA** updates the codes toward the
data, **dead-code reset** reinitialises codes that stop being used (fighting collapse), a
**commitment loss** pulls the encoder output toward its chosen codes, and **quantization dropout**
randomly truncates levels in training for robustness. Decode = sum the selected codes -> decoder.

### FSQ mechanics (ours)
**Round each latent dimension onto a small fixed grid** (levels e.g. (8,5,5,5)); the tuple of rounded
dims *is* the integer code; implicit vocab per group = product of levels (8x5x5x5 = 1000). Gradient
flows through the rounding via a straight-through estimator (`round_ste`). **No codebook is learned**,
so none of RVQ's machinery (EMA / reset / commitment) exists — nothing to collapse.

> **The structural contrast.** RVQ's 6 codes are **sequential refinement levels** (each fixes the
> last one's error, learned codebooks). FSQ's 6 codes are **parallel partition groups** (independent,
> fixed grids). Both yield 6 integers per step — but one learns and refines, the other partitions and
> rounds. That difference is the whole of Contribution A.

### Why "6 x 1000" specifically (our config)
- **6 groups** -> a token step emits **6 integers**. Chosen to (a) match MoMask RVQ's 6 levels =
  same token budget per step = fair FSQ-vs-RVQ comparison; (b) give a 6x4 = **24-D latent**, the fix
  for the residual-FSQ collapse (4-D was too small for 263-D motion).
- **1000 codes/group** = levels (8,5,5,5) = 8x5x5x5, targeting ~2^10 (~10 bits), comparable to
  standard VQ sizes. **Asymmetric levels** (more resolution on dim 1) are FSQ-recommended as more
  efficient than equal splits. Integers required, so 1000 not 1024.
- **Honesty check:** the iso-vocab ablation used (8,8,8) = **512** (matching RVQ exactly) and STILL
  won (0.0307 vs 0.0382) -> the win is the *quantizer*, not the bigger 1000 vocab.
- **Expressivity:** 1000^6 ~= 10^18 distinct token-steps, yet the generator only makes 6 independent
  1000-way choices per step (6 embedding tables, 6 heads over 1000, +1 for END = 1001).

## 4.4b Reading the training logs: is the tokenizer healthy? (FSQ vs RVQ)

The trainer logs two categories, so health is justifiable from either quantizer's point of view:

**Category 1 — common (both quantizers):**
- `recon` — reconstruction L1 (lower = more faithful round trip). The core "is it learning" signal.
- `total` — recon + (commit, RVQ only). For FSQ, total == recon.

**Category 2 — quantizer health:**
- `perplexity` — effective codes used (exp of usage entropy). Higher = richer use of the vocabulary.
- `usage_frac` — fraction of the codebook actually hit (alive). Low = **collapse**.
- `commit` — commitment loss (RVQ machinery). **Structurally 0 for FSQ** (no codebook to commit to).

**A healthy FSQ run** looks like: `recon` falling, `perplexity` rising and high, `usage_frac` high,
and **`commit 0.0000`** — because FSQ rounds to a *fixed grid*, it needs **no commitment loss, no
codebook-EMA, no dead-code reset**. Unused grid points are free (zero params), so it only needs to be
"high enough", not maximal. The literal `commit 0.0000` in the log is the on-screen proof of the
"no machinery" claim.

**A healthy RVQ run** looks like: `recon` falling, `perplexity`/`usage_frac` high *and kept high by
dead-code reset* (a learned codebook collapses without it), and `commit` **small and stable** (the
encoder stays committed to its codes). A *sick* RVQ shows `usage_frac` crashing → reset fighting it,
or `commit` blowing up → encoder/codebook drifting apart.

> **The contrast IS Contribution A.** Side by side in the same harness: FSQ logs `commit 0.0000` +
> no reset machinery; RVQ logs `commit > 0` + needs EMA + dead-code reset — at comparable recon-FID.
> Simplicity at matched quality, visible in the logs.

(Two EMAs, don't conflate: *codebook-EMA* updates VQ codebook entries — FSQ has none; *weight-EMA*
is a generic eval-stability moving-average of all weights — the tokenizer trainer applies it to both.)

## 4.4c Results to report (interpretation deferred to the paper)

Full HumanML3D test split (2,189 clips), recon-FID via the frozen Guo evaluator. Same conv
encoder/decoder, width, downsample, and 500-epoch budget for every row — only the quantizer differs.

| Tokenizer | codes/step x vocab | recon-FID ↓ | MPJPE ↓ | perplexity (used) |
|---|---|---|---|---|
| Grouped-FSQ | 6 x 1000 | **0.0266** | 119 mm | 572 / 1000 |
| Grouped-FSQ (iso-vocab) | 6 x 512 | 0.0307 | 119 mm | 340 / 512 |
| Strong-RVQ | 6 x 512 | 0.0382 | 125 mm | 328 / 512 |
| *context (published, more compute):* T2M-GPT VQ | — | 0.071 | — | — |
| *context (published, more compute):* MoMask RVQ | 6 x 512 | 0.019 | 29.5 mm | — |

Plain reading (full analysis in the paper): FSQ and our RVQ are in the **same ballpark** (~0.03,
both well under T2M-GPT's 0.071); FSQ is modestly better **and** needs no learned codebook or
collapse machinery; both our rows sit above MoMask's heavily-tuned 0.019. The honest framing to
develop in the paper: *comparable-or-better reconstruction with a simpler, collapse-free quantizer*,
not a landslide.

## 4.5 How WE got to Grouped-FSQ (the actual decision path)

This is the story to tell — it has a hypothesis, a failure, a diagnosis, and a fix:

1. **Hypothesis.** FSQ is motion-proven (ScaMo) and the residual *structure* is motion-proven
   (MoMask). The original Contribution A was their **combination: Residual-FSQ** — believed novel.
2. **Build the bar.** A **strong-RVQ baseline** (EMA + dead-code reset + commitment + quant-dropout,
   the T2M-GPT/MoMask/EnCodec recipe) to beat: recon-FID **0.0382**.
3. **The failure.** Residual-FSQ **collapsed at recon-FID 0.22** — 5x WORSE than RVQ.
4. **The diagnosis.** Residual FSQ keeps the latent at the tiny `fsq_dim` (codes summed in ~4-D);
   later residual levels collapse onto the fixed grid, and a 4-D continuous latent cannot represent
   263-D motion. The residual structure fights the fixed grid.
5. **The fix -> Grouped-FSQ.** Instead of stacking residuals in 4-D, **partition** the latent into
   6 groups (6 x 4 = 24-D), FSQ each independently. Every group is used; the latent is wide enough.
   Result: recon-FID **0.0266**, beating strong-RVQ 0.0382 — and the iso-vocab check (matched 512
   vocab -> 0.0307) confirms it is the quantizer, not codebook size. **That is Contribution A.**

So we got to FSQ for its no-collapse/no-machinery property, and to *Grouped*-FSQ specifically because
the residual variant failed and the partitioned variant fixed it. The collapse is reported as the
motivating negative result.

## 4.6 Train it first, then FREEZE it

Critical workflow point: the tokenizer is trained **on its own** (reconstruct motion well), then
**frozen**. The generator (Lesson 5) never changes the tokenizer — it only learns to predict the
frozen tokenizer's codes. Two reasons:

1. **Stable target.** If the codebook kept changing, the generator would chase a moving target.
2. **Clean separation of contributions.** Tokenizer quality (Contribution A) and generator quality
   (Contribution B) are measured independently. The tokenizer's recon-FID is the *ceiling* the
   generator can reach.

## 4.7 What to hold onto

0. Quantizers span a spectrum from **learned codebook** (VQ -> RVQ, collapse-prone, needs EMA/reset)
   to **fixed grid** (FSQ, no codebook, no collapse). The field is moving toward fixed structure.
1. We discretize because AR streaming wants **next-token classification**, not continuous regression
   (which blurs), and it unlocks the **LLM playbook**.
2. A motion tokenizer is **BPE for motion**: encoder -> quantizer -> decoder, with a **codebook** as
   the vocabulary.
3. Ours downsamples time **/4** and emits **6 FSQ codes per step**, each in 0..999.
4. **FSQ** = round onto a fixed grid (no learned codebook); **Grouped-FSQ** is Contribution A.
5. The tokenizer is **trained then frozen**; its recon-FID (0.0266) is the generator's ceiling.

---

### Questions before Lesson 5

1. Why is "predict the next token" easier for a streaming model than "predict the next 263 floats"?
2. A token step is "6 integers in 0..999". Where do the 6 and the 1000 come from?
3. Why must the tokenizer be frozen before training the generator?

### Looking ahead (Lesson 5 preview)
Lesson 5 is the generator itself (Contribution B): how text conditions it, how it predicts the next
token causally, and the Transformer-vs-Mamba twins — finally connecting "motion words" to the
streaming model that writes them.
