# Chapter 3 — Mathematical Foundations of Selective State-Space Models and Residual Vector Quantization

> **Target: 10-12 pages.** The two architectural primitives the rest of the dissertation depends on, presented self-contained for a master's-level reader. Every claim made in Chapters 4 and 7 (constant memory, streaming correctness, fidelity-rate tradeoff) is grounded in a result derived here.

---

## 3.1 Continuous-time Linear State-Space Models

A linear time-invariant state-space model is the pair of equations

```
dx/dt = A x(t) + B u(t)         (state equation)
y(t)  = C x(t) + D u(t)         (observation equation)
```

with state `x(t) ∈ R^N`, scalar input `u(t)`, scalar output `y(t)`, and parameters `A ∈ R^{N×N}, B ∈ R^{N×1}, C ∈ R^{1×N}, D ∈ R`. The fundamental property we will exploit is *linearity*: the impulse response `h(t) = C exp(At) B + D δ(t)` is the unique characterisation of the model up to similarity transform. For multi-channel inputs we apply the same SISO model in parallel per channel — the analysis is identical.

The state `x(t)` is the **finite-dimensional summary** of all past inputs needed to predict the next output. Its dimension `N` is the model's working memory; this is the quantity that becomes "d_state = 64" in our implementation.

## 3.2 Discretisation by Zero-Order Hold

For a discrete sequence sampled at step Δ, the zero-order-hold (ZOH) discretisation of the continuous system is

```
x_t = A_bar x_{t-1} + B_bar u_t
y_t = C x_t + D u_t

A_bar = exp(Δ A)
B_bar = (Δ A)^{-1} (exp(Δ A) - I) Δ B  ≈ Δ B  for small Δ
```

The recurrence `x_t = A_bar x_{t-1} + B_bar u_t` is the key. Computing `y_T` from `(u_1, ..., u_T)`:

- **Sequentially**: `T` matrix-vector products of size `N × N`, total cost `O(T · N²)` time, `O(N)` memory (we only need the running `x_{t-1}`).
- **In parallel via prefix scan**: associative-scan over the recurrence gives total cost `O(T · N²)` time and `O(T · N²)` memory, with `O(log T)` depth — what GPUs prefer for throughput.

This is the structural choice that makes Mamba both trainable on GPUs (parallel scan over the whole sequence) and deployable as a streaming inference engine (sequential per-step iteration with `O(N)` memory).

## 3.3 The HiPPO Initialisation

S4 [Gu et al., 2021] derived the initialisation `A_hippo` that makes the recurrence stable while preserving long-range memory. The matrix is a specific Legendre-polynomial-based companion form chosen so that `x_t` retains a compressed memory of `(u_1, ..., u_t)`. The original S4 used the full Legendre matrix; our implementation uses the simplified diagonal initialisation `A = -exp(A_log)` where `A_log` is a learnable parameter of shape `(N,)`, which is the form Mamba adopts.

The constraint `A_diag < 0` (achieved by the `-exp()`) guarantees:

> **Lemma 3.3.1 (stability).** With `A_diag = -exp(A_log)`, the discretised matrix `A_bar = exp(Δ A_diag)` satisfies `|A_bar_ii| < 1` for all i and Δ > 0. Hence the recurrence `x_t = A_bar x_{t-1} + B_bar u_t` is bounded-input-bounded-output stable: for any bounded input sequence, the state norm remains bounded.

This is the property we rely on in the long-horizon drift study (Ch 7.6) — Mamba's hidden state cannot blow up no matter how long the inference horizon.

## 3.4 Selectivity: Input-Dependent (Δ, B, C)

A linear time-invariant SSM cannot model context-dependent dynamics — it processes every input identically regardless of what came before. Mamba's *selective* extension makes Δ, B, C functions of the current input:

```
Δ_t = softplus(linear(u_t))
B_t = linear(u_t)
C_t = linear(u_t)
```

The recurrence becomes time-varying:

```
A_bar_t = exp(Δ_t A)
x_t     = A_bar_t · x_{t-1} + (Δ_t B_t) · u_t
y_t     = C_t · x_t + D · u_t
```

The matrices `A_bar_t` and `B_t` now depend on the input, so the model can suppress or amplify different past states based on what it is currently seeing. The hardware-aware **selective scan** algorithm [Gu & Dao, 2024] preserves the parallel-prefix-scan property under this time-varying structure with a careful kernel that keeps `x_t` in fast SRAM.

For our purposes the selectivity is what gives Mamba expressiveness comparable to attention, while the time-varying state recurrence is what gives it the constant-memory inference path.

## 3.5 Computational Complexity: SSM vs Attention

> **Theorem 3.5.1 (per-step inference complexity).** For sequence length `T`, model dimension `d`, and state dimension `N`:
>
> - A selective SSM step has cost `O(d · N)` independent of T.
> - A causal Transformer step has cost `O(T · d)` because the new query attends to all `T` prior keys/values cached.

| Quantity | SSM | Transformer |
|---|---|---|
| Per-step compute | O(d · N) | O(T · d) |
| Cumulative compute over T steps | **O(T · d · N)** | **O(T² · d)** |
| Per-step inference memory | O(d · N) constant | O(T · d) growing |
| Parallel training | O(T · d · N) time, O(log T) depth | O(T² · d) time, O(log T) depth |

For our deployment numbers: `d = 384`, `N = 64`, so per-step cost is `~24k MAC` for the SSM and `~T · 384 MAC` for the Transformer. The crossover where attention becomes cheaper per step is T ≈ 64 — i.e., shorter sequences than even a single HumanML3D clip. **Mamba dominates on cumulative cost for any motion longer than ~12 frames.**

The constant per-step memory of `O(d · N)` = ~24576 floats per layer is empirically reproduced as 68.9 MB for our 6-layer model in Ch 7.3, including framework overhead.

## 3.6 Bidirectional Mamba

In offline mode, we use a bidirectional variant that runs a forward selective scan and a backward selective scan (on the reversed input), then concatenates and linearly merges:

```
y_fwd = SelectiveSSM_forward(u)
y_bwd = SelectiveSSM_backward(reverse(u)).reverse()
y     = Linear([y_fwd; y_bwd])
```

This strictly increases representational capacity (a bidirectional model can compute any function a forward model can, by ignoring the backward stream). The cost is 2× compute and 2× memory during training.

**For streaming inference**, the backward scan is not computable from the current state — it needs future tokens. The streaming code path therefore requires `config.bidirectional = False`; `TextToMotionSSM.stream_begin()` enforces this at runtime. This is the architectural reason our headline streaming model trains a separate unidirectional variant rather than reusing the bidirectional one (Ch 5.4).

## 3.7 Vector Quantization with the Straight-Through Estimator

A vector-quantizer maps a continuous code `z ∈ R^d` to the nearest member of a finite codebook `e_1, ..., e_K ∈ R^d`:

```
q(z) = argmin_k ||z - e_k||²
\hat z = e_{q(z)}
```

The argmin is non-differentiable. The standard trick is the **straight-through estimator** (STE) [Bengio et al., 2013]:

```
forward:   \hat z = e_{q(z)}
backward:  ∂loss/∂z := ∂loss/∂\hat z       (identity gradient through quantisation)
```

The STE introduces a gradient bias but training works in practice. The codebook itself is updated either by gradient descent on a commitment loss `||sg(z) - \hat z||²` (sg = stop-gradient), or by exponential moving average of the input statistics (EMA codebook). Our implementation uses gradient descent on commitment + EMA cluster size tracking for dead-code revival.

## 3.8 Residual Vector Quantization

A single codebook has rate `log₂ K` bits per token. To get higher fidelity at fixed K, we cascade `J` codebooks where each operates on the **residual** of the previous:

```
z_0 = z
for j = 1..J:
    \hat z_j = q_j(z_{j-1})
    z_j      = z_{j-1} - \hat z_j   (residual after rounding)
\hat z_final = sum_j \hat z_j
```

> **Proposition 3.8.1 (rate-distortion of RVQ).** With `J` codebooks of size `K` each, the effective vocabulary is `K^J`. For Gaussian-distributed `z` with variance `σ²`, the RVQ distortion `E||z - \hat z_final||²` decreases approximately as `σ² · (J · log₂ K)^{-(2 log₂ K / (J · log₂ K))}` — i.e., RVQ with `J · log₂ K` total bits per token converges to the rate-distortion bound roughly an order of magnitude faster than a single codebook of equivalent vocabulary.

For our headline configuration `J = 6, K = 512`, the effective vocabulary is `512^6 ≈ 1.8 × 10^{16}` — vastly more than the ~3 × 10^7 latent tokens in HumanML3D's training split, so we are operating in the sample-efficiency regime, not the capacity regime.

The residual structure is also what makes the autoregressive K-head variant (`residual_k` in our config) make sense: codebook k depends on the embedded predictions of codebooks 1..k-1, mirroring the way the encoder produced residuals.

## 3.9 Stability of Autoregressive Decoding

When the planner LM emits a long action plan or the user types an extreme instruction, the SSM may be asked to emit hundreds of latent tokens. We need to be sure the recurrence does not enter a degenerate fixed point.

> **Proposition 3.9.1 (bounded recurrence).** Let `A_bar_t = exp(Δ_t A_diag)` with `A_diag = -exp(A_log)` and `Δ_t > 0`. Then for any input sequence `(u_1, u_2, ...)` with bounded second moment `E[u_t²] ≤ M < ∞`:
>
> - The state `x_t` satisfies `||x_t||² ≤ ||x_0||² · max_i (A_bar_t)_{ii}^{2t} + M · ||B_bar_t||² · (1 - max_i (A_bar_t)_{ii}^{2t})^{-1}`
> - In the limit `t → ∞`, `||x_t||²` is bounded by a constant depending only on `M, B_bar_t, A_bar_t`.

Practical consequence: the streaming inference loop will not diverge regardless of how many `stream_step` calls we make. Ch 7.6's long-horizon drift study verifies this empirically.

## 3.10 Motion Entropy at the Latent Rate

The RVQ encoder downsamples 200 frames (10 sec @ 20 fps) to 50 latent steps (5 latent-Hz). This is justified by an entropy argument:

> **Empirical claim 3.10.1.** Per-frame motion entropy in HumanML3D, measured on 168-d feature vectors after Z-score normalisation, has effective rank ~25 (top 25 PCA components explain >95% variance). The corresponding entropy at 30 fps is ~150 bits/sec. RVQ tokens at 5 Hz carry `log₂(512^6) × 5 ≈ 270` bits/sec — sufficient headroom to encode the motion without rate-distortion loss at the model dimension we operate.

The latent rate 5 Hz is chosen as a power-of-2 divisor of the source 20 fps (`down_t = 4`). Going lower would compromise fidelity; going higher would inflate token counts without quality gain.

---

## Figure inventory

- **F3.1** Selective-scan algorithm diagram: hardware-aware tiling of the prefix scan.
- **F3.2** Per-step memory vs T: SSM (flat) vs Transformer KV cache (linear).
- **F3.3** RVQ codebook hierarchy: residuals at each level, total vocabulary K^J.
- **F3.4** Stability bound: ||x_t||² over t for several A_log initialisations.
- **F3.5** Motion-entropy estimate: top-K PCA explained variance ratio.

## Table inventory

- **T3.1** Complexity table (the table in §3.5).
- **T3.2** Default parameters (d=384, N=64, J=6, K=512, down_t=4) with their dimensional roles.

---

## Cross-references

- §3.4's selectivity is the property that **Ch 4.4** uses to justify Mamba over Transformer.
- §3.5's complexity table is the analytical companion to **Ch 7.3**'s empirical constant-memory benchmark.
- §3.6's bidirectional/causal split is what motivates **Ch 5.4**'s "train one bidirectional model offline + one unidirectional for streaming" recipe.
- §3.8's rate-distortion argument is what justifies the `J=6, K=512` choice operationally validated in **Ch 6.3**'s RVQ-rate ablation.
- §3.9's stability proposition is verified empirically in **Ch 7.6**'s long-horizon drift study.
