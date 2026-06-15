# Lesson 5 — RVQ: turning motion into tokens with a learned codebook (the baseline)

> Beginner-friendly theory of the **strong-RVQ baseline tokenizer**: what it is, how it is trained,
> why each piece exists, and how we apply it in our experiment. Numbers are confirmed later (Lesson 7).
> Code: `rvq_baseline.py`, `tokenizer_trainer.py`. Research: VQ-VAE, SoundStream/EnCodec, T2M-GPT,
> MoMask.

## 5.1 The problem (recap from Lesson 4)
Motion is **continuous** (the 263 floats per frame). To generate it with a next-token model we need
**discrete tokens** — a small set of "motion words". A *tokenizer* learns to (a) compress motion into
tokens and (b) rebuild motion from them. RVQ is one way to do the discrete part.

## 5.2 Vector Quantization (VQ): rounding to a learned crayon box
Start simple. A **codebook** is a list of `K` vectors ("crayons"), each with an index 0..K-1. To
**quantize** a continuous vector `z`:
1. find the **nearest** codebook vector `e`;
2. output `e`; the **token** is its index.

Analogy: you have a continuous color and a box of K crayons; you pick the closest crayon and write
down its number. That number is the token. The crayons (codebook) are **learned** during training.

**Definitions to note**
- **Codebook** — the learned set of K code vectors.
- **Quantize** — replace a continuous vector by its nearest codebook vector.
- **Token / index** — which codebook entry was chosen.

## 5.3 How VQ is TRAINED (and the three problems it must solve)
The tokenizer is an autoencoder: **encoder** -> continuous `z` -> **quantize** -> `e` -> **decoder**
-> reconstructed motion. Train it to reconstruct well. Three problems arise, each with a fix:

1. **"Nearest" is not differentiable.** Picking the closest crayon (an argmin) has no gradient, so
   the encoder can't learn. **Fix — Straight-Through Estimator (STE):** in the backward pass, pretend
   quantization was the identity (copy the gradient from `e` straight back to `z`). Forward is
   discrete, backward flows.
2. **Encoder and codebook can drift apart.** If `z` and its crayon `e` are far, reconstruction
   suffers. **Fix — commitment loss** (pull `z` toward `e`) plus a codebook update that pulls `e`
   toward the `z`s assigned to it. The codebook update is often done by **EMA** (exponential moving
   average of assigned encoder outputs) instead of gradient.
3. **Codebook collapse.** Often only a few crayons ever get used; the rest are dead weight. **Fix —
   dead-code reset:** periodically reinitialise unused entries to active encoder outputs.

So plain-but-healthy VQ already needs: **STE + commitment + EMA + dead-code reset**. (T2M-GPT
measured this: naive VQ recon-FID 0.49; with EMA+reset it drops to 0.07 — a 7x gap. The machinery is
not optional.)

## 5.4 RVQ = stacking VQ in residuals (coarse-to-fine)
One codebook of reasonable size can't capture all of motion. **Residual VQ** stacks several:
1. codebook 1 quantizes `z` -> `e1`; the **residual** (leftover error) is `r1 = z - e1`;
2. codebook 2 quantizes `r1` -> `e2`; residual `r2 = r1 - e2`;
3. ... repeat for L levels. Final code `~= e1 + e2 + ... + eL`; the token step is the **L indices**.

Each level **refines** the previous one's error — early levels capture gross motion, later levels add
detail. This is how audio codecs (SoundStream, EnCodec) and MoMask reach high fidelity.

**Extra trick — quantization dropout:** randomly use only the first few levels during training, so the
model is robust to using fewer levels at inference.

## 5.5 How WE do it (applied — `rvq_baseline.py`)
Our strong-RVQ baseline is the full, competitive recipe (T2M-GPT + MoMask + EnCodec defaults):
- **6 levels x 512 codes** each, code dim 512.
- **EMA 0.99** codebook updates, **dead-code reset**, **commitment 0.02**, **quant-dropout 0.2**.
- It **shares the conv encoder/decoder** with the FSQ tokenizer (same width 512, downsample 4,
  3 resblocks) — so when we compare FSQ vs RVQ, only the quantizer differs (a fair test).

**The training loop (`tokenizer_trainer.py`), one step:**
1. take a batch of normalized 64-frame motion windows;
2. encode -> quantize (the 6-level residual cascade) -> decode;
3. loss = **reconstruction L1** (+ velocity smoothness) **+ commitment**;
4. backward (STE lets gradient reach the encoder), AdamW step, EMA update of the weights;
5. log health: `recon`, `commit`, `perplexity` (effective codes used), `usage_frac` (alive fraction).
We train ~500 epochs on HumanML3D windows; best checkpoint by downstream recon-FID.

## 5.6 What "healthy RVQ" looks like (preview; values in Lesson 7)
`recon` falling, `perplexity`/`usage_frac` high **and kept high by dead-code reset**, `commit` small
and stable. A *sick* RVQ shows usage crashing (collapse) or commit blowing up (drift).

## 5.7 Research lineage (so the design is grounded)
VQ-VAE (van den Oord et al., 1711.00937) introduced learned-codebook quantization; SoundStream
(2107.03312) and EnCodec (2210.13438) made RVQ the audio-codec standard; T2M-GPT (2301.06052) and
MoMask (2312.00063) brought VQ/RVQ to motion. Our baseline follows their recipe exactly so it is a
*strong* opponent, not a strawman.

### Check before Lesson 6
1. Why can't the encoder learn through a plain "nearest crayon" step, and what fixes it?
2. What does each successive RVQ level actually quantize?
3. Name two pieces of machinery a learned codebook needs to stay healthy, and what each prevents.
