# Research Proposal — Streaming Text-to-Motion with a Token-Autoregressive State-Space Generator

**Author:** Cătălin Butacu · **Status:** living document (started 2026-06-10) · **Repo:** GenAI-Text2Motion @ thesis-v2

## 1. Problem statement

Text-to-motion generation maps a natural-language caption to a 3D human motion sequence. The
strongest published systems (MoMask, MMM, BAMM) are *bidirectional*: they iteratively refine entire
sequences with masked prediction, so motion exists only after a full-sequence pass. Interactive
applications (avatars, games, robotics teleoperation) instead need **streaming**: motion must be
emitted incrementally, with bounded memory and stable per-step latency, while the user watches.
Causal GPT-style generators (T2M-GPT, AttT2M, Mogo) can stream, but their attention KV-cache grows
linearly with the generated horizon — memory and per-step cost increase without bound.

**Research gap (verified against the June 2026 literature):** no published motion generator is a
*next-token, discrete, causal, fixed-recurrent-state* model. All Mamba/SSM motion work is
diffusion-based or masked-bidirectional (Motion Mamba, KMM); the streaming rival MotionStreamer is
continuous diffusion-AR. The cell "token-AR S6 for motion" is unoccupied.

## 2. Hypotheses

- **H1 (tokenizer).** A *Grouped Finite Scalar Quantization* (FSQ) motion tokenizer matches or beats
  a strong residual-VQ (EMA + dead-code reset + quantization dropout) at matched encoder capacity
  and token budget, without any codebook-collapse machinery.
  - H1a (iso-vocab): the win survives when the FSQ per-group vocabulary is reduced to RVQ's 512.
- **H2 (generator parity).** A causal token-autoregressive S6/Mamba generator reaches generation
  quality (FID, R-precision) within range of a parameter-matched causal transformer twin trained
  under identical data/seed/budget/losses.
- **H3 (streaming efficiency).** At long horizons the Mamba generator holds **O(1) state and flat
  per-step latency** while the transformer twin's KV-cache and latency grow with the horizon — the
  efficiency axis where the SSM wins *given* H2 parity.
- **H4 (emerging, registered 2026-06-10).** At matched budget the SSM twin is more sample-efficient:
  it overfits later and peaks higher on retrieval precision than the transformer twin.

## 3. Contributions

1. **Grouped-FSQ motion tokenizer** (Contribution A): FSQ is motion-proven (ScaMo) and residual
   quantization is standard (MoMask); the grouped-partition formulation and its controlled
   comparison against strong RVQ on HumanML3D-263 are ours. Includes the *negative* finding that
   naive residual-FSQ collapses (recon-FID 0.22) because residual codes summed in a 4-D lattice
   cannot span 263-D motion.
2. **First token-AR S6/Mamba motion generator** (Contribution B), evaluated as a *controlled twin
   experiment*: param-matched transformer (94.70M) vs Mamba (96.37M, +1.77%), identical tokenizer,
   data, seed, losses, schedule. Claim = H2 parity + H3 bounded streaming, not absolute SOTA.
3. **A streaming evaluation protocol**: per-step latency + exact recurrent-state bytes vs horizon
   (49 → 1024 steps), alongside the standard Guo et al. metric suite.

## 4. Method

- **Representation:** standard HumanML3D-263 (22-joint; root velocities, RIC, rot6d, local
  velocities, foot contacts), 20 fps — the citable track. SMPL-X 168 whole-body demo deferred.
- **Tokenizer:** conv encoder (width 512, stride-4) → 6 groups × FSQ(8,5,5,5) (1000/group) →
  conv decoder. Baseline: same encoder/decoder with 6×512 RVQ (EMA 0.99, dead-code reset,
  commitment 0.02, quant-dropout 0.2 — the T2M-GPT/MoMask recipe).
- **Generator:** text prefix (CLIP ViT-B/32; pooled, upgraded to 16-token prefix) + per-codebook
  embeddings → causal backbone (S6 mixer stack | transformer decoder) → per-codebook heads + END
  token. `forward` (parallel teacher-forcing) and `step` (recurrent streaming) share one recurrence;
  stream==batch parity is a unit-tested structural identity, not an approximation.
- **Training:** token-CE + soft-decode reconstruction (expected FSQ codes through the frozen
  decoder) + velocity + foot/root terms; EMA 0.999; CFG dropout 0.1; pkeep 0.8 input corruption;
  AdamW wd 0.01; warmup+cosine sized to the empirical peak window (~60 epochs); bf16.
- **Inference:** nucleus sampling + classifier-free guidance (locked on val: scale 5.0, T 1.1,
  top-p 0.9); END-token self-termination for the demo, GT-length mode for protocol comparability.

## 5. Evaluation protocol (locked)

- **Evaluator:** Guo et al. `text_mot_match` reused **unmodified**; our reimplementation reproduces
  published GT numbers exactly (R@1 0.514 vs 0.511, Diversity 9.67 vs 9.50, MM-Dist 2.98 vs 2.97) —
  the precondition for citable FID.
- **Metrics:** FID, R-precision@1/2/3, MM-Dist, Diversity, MultiModality (100 captions × 30
  samples), each as 20-repetition mean±std on the full split; plus the streaming benchmark (H3).
- **Discipline:** model selection on **val** only; **test touched once** per final table; every
  number traces to a run manifest (config + git commit + seed + versions). Published baselines are
  cited, never retrained.

## 6. Experiment plan and budget

| # | Experiment | Status | Cost |
|---|---|---|---|
| E1 | Tokenizer: Grouped-FSQ vs strong-RVQ (H1) | **done** — 0.0266 vs 0.0382 | local |
| E2 | Iso-vocab ablation (H1a) | running | local |
| E3 | 31.6M twins, matched budget (H2 pilot) | done (cloud); twin table v0 pending GPU | ~$21 |
| E4 | CFG/temperature sweep (sampling lock) | **done** — FID −44%, R@1 ×2 | local |
| E5 | 100M twins, full recipe (H2 main) | config gated + sanity-PASS; canary then run | ~$22 |
| E6 | Streaming benchmark (H3) | harness built; run on final ckpts | local |
| E7 | AMASS tokenizer pretraining (ablation) | queued | local |
| E7b | **AMASS generator pretraining**: regen+tokenize AMASS locally (tokens ≈ tens of MB), ship tokens to S3, pretrain the generator *unconditionally* (token-CE only) on the full corpus, then fine-tune conditionally on HumanML3D with the full loss. The principal anti-overfitting lever (22k captioned clips is small for ~100M params); reported as a "+pretraining" ablation row. | registered 2026-06-11 | local + ~$5-8 |
| E8 | MultiModality + final tables | harness done | local |

Total cloud budget: ≤ $80 (≈$21 spent; ~$45 reserved for E5 + contingency).

## 7. Threats to validity / limitations

- **Absolute FID gap to MoMask (0.045):** expected and acknowledged — bidirectional refinement +
  far larger compute; our claim is twin parity on the streaming axis, with published numbers cited
  as the ceiling.
- **Budget asymmetry in E3:** the pilot transformer ran 150 epochs, Mamba ~40; both demonstrably
  peak before epoch 35 (transformer ep20, mamba ep30), so best-checkpoint comparison stands; E5
  removes the asymmetry.
- **Training-time SSM cost:** our eager parallel-scan trains ~16× slower than fused attention; an
  implementation artifact (resolved by the fused mamba-ssm kernel in E5), distinct from the O(1)
  inference claim.
- **GT-length conditioning** in fixed-length eval (field-standard caveat); the END token provides
  the self-terminating alternative and both are reported.
