# Lesson 5 — RVQ: turning motion into tokens with a learned codebook (the baseline)

> Beginner theory of the **strong-RVQ baseline tokenizer**, with the mathematics at each step.
> Framing: quantization is how the analog world becomes digital; VQ generalizes it to vectors and is
> exactly **online k-means**. Numbers confirmed in Lesson 7. Code: `rvq_baseline.py`,
> `tokenizer_trainer.py`.

## 5.1 The problem
Motion is **continuous** (the 263 floats per frame). A next-token generator needs **discrete**
tokens. A tokenizer learns to map motion -> a few integers and back. RVQ does the discrete step.

## 5.2 Vector Quantization = nearest-centroid (the digital-signal idea, for vectors)
Recall how a continuous audio signal is digitized: each sample is replaced by the nearest value from
a finite set of **reproduction levels**. **Vector Quantization (VQ)** is the same idea for vectors:
the reproduction levels become a set of representative vectors — a **codebook** — and these are
exactly **cluster centroids** (as in k-means).

Setup and the quantization map:
- encoder produces a latent vector `z ∈ R^d`;
- codebook `C = {e_1, ..., e_K}`, each `e_k ∈ R^d` (the K centroids);
- **quantize** = pick the nearest centroid:

```
k*   = argmin_k  || z - e_k ||_2          # nearest-centroid assignment (a Voronoi cell)
q(z) = e_{k*}                             # the quantized vector
token = k*                                # the discrete index we store
```

The decoder reconstructs `x_hat = Decoder(q(z))`, trained against the input `x` with
`L_recon = || x - x_hat ||_1` (L1 — Lesson A: L1 avoids the L2 blur).

**Definitions to note**
- **Codebook** — the K learned centroids `e_k`.
- **Quantization** — assign `z` to its nearest centroid (a Voronoi partition of `R^d`).
- **Token** — the index `k*` of that centroid.

## 5.3 Training VQ — three problems, each with its math

**Problem 1 — `argmin` has no gradient.** The assignment `k*` is piecewise-constant, so
`∂q/∂z = 0` almost everywhere and the encoder can't learn. **Fix — Straight-Through Estimator
(STE):** define the quantized value as

```
z_q = z + sg(e_{k*} - z)        # sg = stop-gradient
```

Forward, `z_q = e_{k*}` (discrete). Backward, the `sg(...)` term has zero gradient, so
`∂z_q/∂z = 1` — the gradient is **copied straight through** to the encoder as if quantization were
the identity.

**Problem 2 — the centroids and the encoder must agree.** Two coupled objectives:
- move each centroid toward the latents assigned to it (the k-means update);
- move the encoder's latents toward their centroid (so it commits to discrete codes).

Written as losses (van den Oord VQ-VAE):

```
L = L_recon + || sg(z) - e_{k*} ||^2  +  beta * || z - sg(e_{k*}) ||^2
              \_____codebook loss____/     \______commitment loss______/
```

In practice the **codebook loss is replaced by an EMA update** (more stable — it *is* online
k-means). Per batch, with decay `gamma` (we use 0.99), let `n_k` = number of latents assigned to `k`:

```
N_k  <- gamma * N_k + (1 - gamma) * n_k                  # running cluster size
m_k  <- gamma * m_k + (1 - gamma) * sum_{i in k} z_i     # running sum of members
e_k  <- m_k / N_k                                        # centroid = mean of members
```

So only the **commitment** term `beta * ||z - sg(e)||^2` stays in the loss (our `beta = 0.02`); the
codebook is moved by EMA, not gradient.

**Problem 3 — codebook collapse.** With nearest-centroid + EMA, many centroids end up with `N_k ~ 0`
(never assigned) — wasted capacity. **Fix — dead-code reset:** when `N_k` drops below a threshold,
reinitialise that centroid to a random encoder output from the current batch:

```
if N_k < tau:   e_k <- z_j   for a random j in the batch
```

So a *healthy* VQ needs **STE + commitment + EMA + dead-code reset**. (T2M-GPT measured the stakes:
naive VQ recon-FID 0.49 vs EMA+reset 0.07 — a 7x gap. The machinery is mandatory.)

## 5.4 RVQ = quantize the residual, recursively (coarse-to-fine)
One codebook of moderate `K` is too coarse. **Residual VQ** applies `L` quantizers, each to the
*leftover error* of the previous:

```
r_0 = z
for l = 1..L:
    k_l   = argmin_k || r_{l-1} - e_k^{(l)} ||      # nearest centroid in codebook l
    q_l   = e_{k_l}^{(l)}
    r_l   = r_{l-1} - q_l                           # the new residual (what's still unexplained)
z_hat = sum_{l=1..L} q_l = z - r_L                  # reconstruction of z; tokens = (k_1,...,k_L)
```

Each level shrinks the residual `||r_l|| < ||r_{l-1}||` — early levels capture gross structure, later
levels add detail. The effective vocabulary is `K^L` combinations from `L` small codebooks.
**Quantization dropout:** with some probability use only the first `L' < L` levels in a step, so the
model degrades gracefully when fewer levels are used.

## 5.5 How WE do it (applied — `rvq_baseline.py`)
The strong, competitive recipe (T2M-GPT + MoMask + EnCodec defaults), so it is a real opponent:
- **L = 6 levels x K = 512** centroids, code dim 512;
- **EMA gamma = 0.99**, **dead-code reset**, **commitment beta = 0.02**, **quant-dropout 0.2**;
- **shares the conv encoder/decoder** with the FSQ tokenizer (same width 512, downsample 4,
  3 resblocks) — so FSQ-vs-RVQ differs *only* in the quantizer.

Training step (`tokenizer_trainer.py`): encode a normalized 64-frame window -> the 6-level residual
quantize -> decode; loss `L_recon + beta * commitment`; backward (STE), AdamW, EMA of the *weights*
for eval; ~500 epochs; keep the best by downstream recon-FID.

## 5.6 Health, with math (preview; values in Lesson 7)
Per codebook, with usage probabilities `p_k = n_k / sum_j n_j`:

```
perplexity = exp( - sum_k p_k * log p_k )   # = exp(entropy) = effective # of centroids in use
usage_frac = (# k with n_k > 0) / K          # fraction of centroids alive
```

Healthy RVQ: `L_recon` falling, perplexity/`usage_frac` high (held up by dead-code reset),
commitment small and stable. Collapse shows as usage crashing; drift shows as commitment growing.

## 5.7 Research lineage
VQ-VAE (van den Oord et al., 1711.00937) — learned-codebook quantization + STE + commitment;
SoundStream (2107.03312) / EnCodec (2210.13438) — RVQ as the audio-codec standard; T2M-GPT
(2301.06052), MoMask (2312.00063) — VQ/RVQ for motion. Our baseline follows them exactly.

### Check before Lesson 6
1. Write the STE identity for `z_q` and explain why its backward pass is the identity.
2. In RVQ, what exactly does codebook `l` quantize, and what happens to `||r_l||` as `l` grows?
3. The codebook is updated by EMA, not gradient. Which loss term then remains, and what is its job?
