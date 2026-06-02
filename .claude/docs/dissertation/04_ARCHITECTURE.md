# Chapter 4 — System Architecture

> **Target: 15-18 pages.** This document is the dissertation-grade prose source for Chapter 4. Figures referenced by their identifier (F4.1, F4.2, ...) are listed at the end; tables (T4.1, T4.2, ...) likewise. Cross-references to other chapters use Ch-N notation.

---

## 4.1 End-to-End Overview

The system decomposes a free-form natural-language instruction into a streaming sequence of 3D human-pose frames. Five modules cooperate in a pipeline whose interfaces are kept narrow so each can be replaced in isolation:

```
                    ┌──────────────────────────────────────────────┐
M1 — Understanding  │  SpaCy parser → entities + spatial relations  │
                    └──────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────────────┐
M2 — Planner        │  ScenePlanner → PlannedScene (actor coords)  │
                    └──────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────────────┐
M5 — Agent          │  fine-tuned GPT-2 → action plan (until-grammar)│
                    └──────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────────────┐
M3 — Motion         │  CLIP text encoder → BiMamba SSM → RVQ head    │
                    │  streaming inference: stream_begin / stream_step│
                    │  hidden-state carryover across action transitions│
                    └──────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────────────┐
M4 — Render         │  frozen RVQ decoder → 168-d SMPL-X → mesh      │
                    └──────────────────────────────────────────────┘
```

The novel contributions concentrate in M3 (the Mamba-RVQ streaming core) and M5 (the from-scratch fine-tuned action-decomposition LM). M1, M2, M4 are conventional and serve as the boundary layers.

This chapter walks through each architectural decision in execution order. The companion implementation lives at:

- `src/modules/understanding/` (M1)
- `src/modules/planner/` (M2)
- `src/modules/motion/` (M3)
- `src/modules/render/` (M4)
- `src/modules/agent/` (M5)

---

## 4.2 The RVQ Tokenizer

The Residual Vector Quantizer (RVQ) is the bridge between continuous SMPL-X motion and discrete tokens. We use the architecture introduced by Soundstream [Zeghidour et al., 2022] and popularised in T2M by T2M-GPT and MoMask:

**Encoder.** A 1D convolutional encoder with two stride-2 downsampling blocks compresses a 200-frame motion clip (200 × 168 floats) into a sequence of 50 latent vectors (50 × 128 floats). The kernel size is 3 throughout and the SiLU non-linearity is used between layers.

**Quantizer stack.** Six independent codebooks of 512 entries each quantize the latent residual at each step. Codebook k operates on the residual error left by codebooks 0..k-1. After K codebooks, each latent step is represented by K = 6 integers in {0, ..., 511}; the total vocabulary across the K-product is 512^6 ≈ 1.8 × 10^16 distinct quantized states — far more than the 14k clips in HumanML3D, so the rate-distortion curve is dominated by sample efficiency, not capacity.

**Decoder, symmetric vs causal.** This is the architectural choice that enables streaming. The default decoder uses `ConvTranspose1d` for the two upsampling stages, which means output frame *t* sees latent tokens both before and after *t* — fine for offline reconstruction, fatal for streaming because emitting frame *t* would require knowing future tokens we have not generated yet. We introduce a **causal decoder variant** that replaces `ConvTranspose1d` with nearest-neighbor upsampling followed by causally-padded `Conv1d` (left-only padding, kernel size 3 or 4). The receptive field count is identical to the symmetric path, but output frame *t* now depends only on latent tokens 0..⌈t/4⌉. We retrain the decoder side from scratch with this flag; the encoder + codebooks transfer unchanged.

The causality property is tested structurally: zeroing the latent suffix must not change the decoded prefix (`tests/test_causal_rvq_decoder.py`).

---

## 4.3 The Text Encoder

We use the **frozen** OpenAI CLIP ViT-B/32 text encoder via the `sentence-transformers` library. The choice is motivated by two findings from our text-encoder ablation (Ch 6.5):

1. CLIP outperforms SBERT MiniLM-L6-v2 on motion-verb similarity tasks. Verbs like "stroll" and "saunter" are closer to "walk" in CLIP's embedding space than in SBERT's, because CLIP's contrastive web-scale training saw many image-caption pairs with motion words. SBERT's NLI-tuned objective optimises for semantic entailment, which is a different problem.
2. CLIP is bigger (151M parameters vs SBERT's 23M) but **frozen** during MotionSSM training, so the parameter overhead is amortised across many inference calls.

The 512-dimensional CLIP text features are projected to `d_model = 384` via a single trainable `nn.Linear(512, 384)` layer (`condition_proj` in the code).

---

## 4.4 The Bidirectional Mamba Backbone

Six stacked BiMamba layers (forward + backward selective SSM scans, summed) form the trunk that produces the per-latent-step features.

**Why Mamba, not Transformer?** Three reasons, in priority order:

1. **Linear-time per-step inference.** Mamba's recurrent hidden state is `(B, d_inner, d_state)` regardless of how many steps have been emitted. A Transformer's KV cache grows linearly with sequence length. For streaming a 30-second motion (150 latent steps), this is negligible; for streaming a 5-minute session (1500 latent steps), the KV cache exceeds the GPU budget. Our constant-memory benchmark (Ch 7.3) shows streaming Mamba flat at 68.9 MB across T ∈ {50, 250, 1000, 2000, 4000}; the offline parallel path grows to 2504 MB at T=4000.

2. **No attention window to tune.** Long-range music or video work has needed attention windowing, sparse attention, or hierarchical attention to handle long contexts. Mamba's selective scan operates on the whole sequence in linear time without windowing artifacts at boundaries.

3. **Architecture novelty for the dissertation contribution.** Motion Mamba [Zhang et al., 2024 ECCV] used bidirectional Mamba inside a diffusion U-Net — a different family entirely. Mogo [Hu et al., 2024] used causal Transformer + RVQ for streaming. The (Causal Mamba) × (RVQ) × (Streaming) intersection is unoccupied in the published literature as of 2026-05.

**Bidirectional during training, unidirectional at streaming inference.** We train with `BiMambaLayer` (both forward and backward scans, merged by a linear layer) because the offline parallel forward sees the whole latent sequence at once and a bidirectional scan strictly increases context. At streaming inference time we fall back to a **unidirectional Mamba** (the forward scan only), exposing the per-step `step()` API. This is enforced by a runtime check in `TextToMotionSSM.stream_begin()` — bidirectional models raise immediately.

**Hidden-state shape.** Each Mamba layer carries `(B, d_inner=768, d_state=64)` = ~49k floats per layer × 6 layers = ~300k floats per inference, regardless of T.

---

## 4.5 The RVQ Prediction Head

Two variants are supported; the default is the "independent" head used by T2M-GPT.

**Independent K-head.** Each of the K = 6 codebooks is predicted by an independent `nn.Linear(d_model, codebook_size)` from the same Mamba feature vector. K loss terms, summed. Trains fast, has no autoregressive dependency across codebooks.

**Residual K-head (`residual_k`).** Following the Mogo formulation, we add an autoregressive structure across codebooks: codebook k is conditioned on the embedded sum of the predictions for codebooks 0..k-1. This restores the residual semantics that the independent head loses. We include it as an opt-in variant via `config.arch = "residual_k"`; the headline run leaves it off for clean comparison with the offline T2M-GPT baseline.

**Length head.** A small MLP that regresses the clip length in raw frames from the mean of the trunk features. Supervised by `motion_mask.sum(dim=1)`; loss weight 0.1.

---

## 4.6 Streaming Inference: stream_begin and stream_step

The novelty of the inference path is that we expose Mamba's natural single-step recurrence as a first-class API.

```python
state = model.stream_begin(text="walk forward")
# state contains: cond, list[h_per_layer], list[conv_buf_per_layer], latent_step

while not termination_fired:
    logits, length_pred, state = model.stream_step(state)
    # state.latent_step += 1, hidden state mutated in place
```

The per-step computation invokes `MambaLayer.step()` once per layer, which:

1. Applies the input projection to the current frame's input
2. Carries out one ZOH discretisation: `h_t = A_bar * h_{t-1} + B * x_t`, output `y_t = C^T h_t + D x_t`
3. Maintains a rolling conv buffer of `d_conv - 1` past frames so the depthwise convolution sees the same context as the parallel forward

The implementation is verified by `tests/test_streaming_equivalence.py`: T calls to `stream_step` produce the same logits as a single `forward()` over T tokens, within 5e-4 float tolerance.

**Action transitions: state.carry_over.** When the agent advances to a new action, the runner calls `state.carry_over(new_cond)` which preserves all per-layer hidden states `layer_h` but resets `latent_step` and `cond`. This is the architectural reason the avatar's motion is continuous across instruction boundaries — the SSM does not see a "fresh start" between actions, only a new text condition.

---

## 4.7 Pose-Prefix Conditioning

A complementary continuity mechanism for training: with probability 0.5 per batch (configurable via `pose_prefix_prob`), a clip is split into prefix and suffix at a random latent step P, the prefix tokens are decoded back to a `(B, P, latent_dim)` tensor via the frozen RVQ decoder path, projected to `d_model` via a learned `nn.Linear` (`seed_projector`), and prepended to the SSM input. The model is then supervised to predict the suffix tokens.

This trains the model to handle "continue from this pose" cases — necessary for the action-transition demo where the second action begins with the avatar mid-stride. At inference time, the streaming path uses hidden-state carryover instead (which is cheaper and equivalent in distribution if the curriculum was applied at training time).

The forward API exposes the seed:

```python
logits, length_pred = model(text, motion_length=L, seed_latent=seed)
# seed_latent: (B, P, d_model) prepended to SSM input, dropped from output
```

The seed_projector adds ~50k parameters and is unused (random init, no harm) when `pose_prefix_prob = 0`.

---

## 4.8 The Agent Layer (M5)

Five components, layered:

```
                       ┌──────────────────────────────┐
                       │  ActionPlanner  (GPT-2-small) │
                       │  fine-tuned on 6k synthetic   │
                       │  (instruction, action_list)   │
                       └──────────────────────────────┘
                                       │
                                       ▼ list[PlannedAction]
                       ┌──────────────────────────────┐
                       │  conditions.parse_condition   │
                       │  duration | distance | rot…   │
                       └──────────────────────────────┘
                                       │
                                       ▼ ParsedCondition
        ┌──────────────────────────────────────────────────┐
        │                StreamingRunner                    │
        │  for action in plan:                              │
        │      stream_begin or carry_over                   │
        │      world.reset_for_new_action()                 │
        │      while not until(world):                      │
        │          stream_step → decode → emit frames       │
        │          world.update_from_frame                  │
        └──────────────────────────────────────────────────┘
                                       │
                                       ▼ raw 168-d frames
                              [renderer / mp4 / WebSocket]
```

**No hardcoded action queues anywhere.** The planner always emits the action list, even in tests (where it's mocked to return crafted lists for control-flow testing). This is a load-bearing design constraint: the dissertation's "natural-language input is real" claim depends on never bypassing the LM.

---

## 4.9 Parameter Counts

| Component | Parameters |
|---|---:|
| CLIP ViT-B/32 text encoder (frozen) | 151.2M |
| condition_proj (CLIP → d_model) | 0.2M |
| Position embedding (max_motion_length / down_t × d_model) | 0.02M |
| BiMamba × 6 (d_model=384, d_inner=768, d_state=64) | 12.5M |
| FiLM blocks × 6 | 1.0M |
| RVQ head (independent, K=6 × d_model × codebook_size) | 1.2M |
| Length head | 0.2M |
| seed_projector (rvq_latent_dim=128 → d_model=384) | 0.05M |
| RVQ tokenizer (frozen at SSM training time) | 2.1M |
| Planner LM (GPT-2-small, fine-tuned) | 124.4M |

**Trainable at SSM training time: ~15.2M.** The CLIP backbone and the frozen RVQ tokenizer dominate the on-disk weights but contribute zero gradient updates.

---

## 4.10 Why this combination is unoccupied in the literature

Quick competitive landscape (full literature scan in Ch 2):

- **Motion Mamba (ECCV 2024)** — bidirectional Mamba + diffusion U-Net. Offline.
- **Mogo (Dec 2024 / Jun 2025)** — causal Transformer + RVQ + streaming. Closest to ours but Transformer-based; KV cache grows with T.
- **MotionStreamer (ICCV 2025)** — causal Transformer + continuous diffusion latent. Also Transformer-based; no discrete tokens.
- **T2M-GPT, MoMask, MMM** — Transformer + VQ/RVQ, offline.

The (Causal Mamba) × (RVQ) × (Streaming) × (Hidden-State Carryover across instructions) cell is, to our knowledge, unaddressed.

---

## Figure inventory

- **F4.1** Block diagram of the M1-M5 pipeline (the ASCII version above, redrawn for the dissertation).
- **F4.2** RVQ tokenizer encoder/decoder, including the symmetric vs causal upsampling contrast.
- **F4.3** BiMamba selective-scan diagram with FiLM injection at every layer.
- **F4.4** State diagram of StreamingState through `stream_begin → stream_step* → carry_over → stream_step* → ...`.
- **F4.5** Hidden-state-vs-T plot (forward-reference to Ch 7.3 headline figure).

## Table inventory

- **T4.1** Parameter count by component (the table in §4.9).
- **T4.2** Architectural-choice ablation summary (forward-reference to Ch 6).

---

## Cross-references

- The empirical evidence for §4.6 lives in **Ch 7.3** (constant-memory benchmark).
- The training methodology for §4.7's pose-prefix curriculum is in **Ch 5.3**.
- §4.10's competitive positioning is expanded in **Ch 2.4**.
- §4.8's planner LM training pipeline is in **Ch 5.5**.
