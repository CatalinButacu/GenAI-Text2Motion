# Literature map: streaming, causal, endless text-to-motion (2026-11-25)

50 papers, 5 topics. Each record (id, title, authors, date, abstract) was fetched from the arXiv API on
2026-11-25. Nothing below is cited from memory.

Limits of this map:

- Claims and numbers come from the **abstracts only**. Before a number enters the thesis, check it against
  the paper's own table (status "to verify" in `docs/project/references.md`).
- The Copilot research workers could not run (monthly quota exceeded). The fetch was a keyword search
  plus 13 title searches for core papers. 13 off-topic hits of the keyword search were dropped.
- No PhD or master thesis was found on arXiv. Theses need a search in university repositories (open).
- Guo et al. 2022 (HumanML3D, the evaluator we use) was not found on arXiv by title (open).

Marks: **R** relevant, **C** to compare, **N** not relevant. Streaming: causal, AR-diff (autoregressive
diffusion), bidir (bidirectional), diff (full-sequence diffusion), n/a.

## 1. What this means for the thesis

1. **The novelty cell still looks open.** No paper in this set has a discrete, token-autoregressive,
   causal Mamba text-to-motion generator. Mamba appears only in bidirectional or non-autoregressive roles:
   Motion Mamba (diffusion U-Net with a bidirectional spatial block), Light-T2M (pseudo-bidirectional scan),
   FlashMotion (non-autoregressive BiMamba), InfiniMotion (bidirectional Mamba as memory of a transformer).
   This is a sample of 50, not a full search: keep the paper watch running.
2. **Our core question has a direct answer in language models, and it is split.**
   - For a fixed state: "Fixed State, Long Reach" (2609.11998) reports a constant Mamba cache with 4.3x lower
     latency and 11x less memory than attention at 256k tokens, and retrieval out to 8-16x the training
     length, where attention collapses at 2x.
   - Against a fixed state: "Repeat After Me" (2402.01032) proves and shows that fixed-state models are
     worse at copying from context. Echo (2605.06997) reports a pure Mamba-2 at about 3% (chance) on
     multi-query associative recall.
   - Value for us: the twins can test which effect holds for motion. Motion is mostly local, which favours
     the fixed state; exact recall of far context favours the KV cache. A **recall probe** (prompt A, then B,
     then A again: does the second A match the first?) would place our result against both camps.
3. **Transition metric to adopt.** FlowMDM (2402.15509) proposes **Peak Jerk** and **Area Under the Jerk** to
   detect abrupt transitions. Our A28-T11 joint jump in cm measures the same event. Report both, so the
   steering result is comparable with the composition literature.
4. **Closest streaming competitors** (causal or AR-diffusion, all transformers): MotionStreamer (continuous
   causal latents), CMDM (causal diffusion forcing), MOGO (causal RVQ, one pass), DartControl (real-time
   motion primitives), ActionPlan (streaming 5.25x faster, FID 18% better than the best prior, per abstract).
   None of them is a Mamba twin, and none reports a fixed-state versus KV-cache comparison.
5. **Discrete tokens are under attack.** MotionStreamer, MARDM (2411.16575) and DC-Motion (2606.14721) argue
   that discrete codes lose detail and diversity. FSQ (2309.15505) answers part of it: no codebook collapse,
   no commitment loss. We must report reconstruction error next to FID so the trade is visible.
6. **Small-model baseline for a 4-6 GB GPU.** Light-T2M reports 4.48M parameters and FID 0.040 on HumanML3D
   (abstract). It is the reference point for "quality per parameter" on small hardware.
7. **Evaluation beyond the Guo matcher.** MARDM proposes a more robust evaluation. T2MBench (2602.13751)
   gives 1,025 out-of-distribution prompts and finds most models weak on fine-grained accuracy. TMR is the
   standard retrieval model. Option: add T2MBench prompts as a qualitative stress test, keep the Guo
   matcher as the citable number.

## 2. Agreement and disagreement

- **Agree:** real-time and endless generation needs a causal model (every streaming paper above); single
  short clips are no longer enough (TEACH, FlowMDM, InfiniMotion, DartControl, MotionStreamer).
- **Disagree on the representation:** discrete tokens (T2M-GPT, MoMask, MOGO, BAMM, FSQ) versus continuous
  latents (MotionStreamer, MARDM, CMDM) versus a mix (DC-Motion, VersatileMotion).
- **Disagree on the memory:** fixed recurrent state (Fixed State Long Reach, DMT, Light-T2M) versus
  attention over history (Repeat After Me, Echo, DART, InfiniMotion uses both).
- **Disagree on the metric:** FID and R-precision with the Guo matcher (most papers) versus new metrics:
  jerk for transitions (FlowMDM), robust evaluation (MARDM), OOD prompts (T2MBench), retrieval (TMR).

## 3. The 50 papers

### Topic 1: streaming, real-time and long motion

| id         | first author, year | title                                                  | stream  | mark | said (abstract)                                                                                         | value for us                                                                         |
| ---------- | ------------------ | ------------------------------------------------------ | ------- | ---- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 2410.05260 | Zhao, 2024         | DartControl                                            | AR-diff | R    | Real-time text-driven motion from autoregressive latent diffusion over motion primitives.               | Real-time streaming baseline; primitive length is a design point for our chunk size. |
| 2602.22594 | Yu, 2026           | Causal Motion Diffusion Models (CMDM)                  | AR-diff | R    | Causal VAE plus causal diffusion forcing: streaming, long horizon, lower latency than diffusion and AR. | Direct competitor on streaming quality; continuous latents.                          |
| 2506.05952 | Fu, 2025           | MOGO                                                   | causal  | R    | Causal RVQ transformer, all token layers in one pass; real-time and streaming.                          | Closest discrete causal competitor (already in references).                          |
| 2303.17076 | Zhang, 2023        | DiffCollage                                            | diff    | N    | Parallel generation of large content from piece-wise diffusion models.                                  | Not causal.                                                                          |
| 2501.18726 | Gang, 2025         | Strong and Controllable 3D Motion Generation           | other   | C    | Targets slow generation and relative motion representation for real-time use.                           | Latency comparison.                                                                  |
| 2604.08088 | Ding, 2026         | Coordinate-Based Dual-Constrained AR Motion Generation | causal  | C    | AR on coordinates to avoid mode collapse from discretization.                                           | Counter-argument to discrete tokens.                                                 |
| 2308.01850 | Yang, 2023         | Long-Term Motions via Coherent Sampling                | diff    | C    | Coherent sampling links per-sentence motions without interpolation.                                     | Long-motion baseline.                                                                |
| 2611.13626 | Yang, 2026         | FlashMotion                                            | bidir   | R    | Few-step MeanFlow with a non-autoregressive BiMamba for long horizons.                                  | Mamba used non-causally: supports our novelty gap.                                   |
| 2604.03799 | Zheng, 2026        | MoScale (next-scale AR)                                | other   | C    | Coarse-to-fine scale autoregression instead of next-token.                                              | Alternative AR order; not frame-causal.                                              |
| 2412.05095 | Tan, 2024          | SoPo                                                   | n/a     | N    | Semi-online preference optimization for text-to-motion.                                                 | Later fine-tuning idea only.                                                         |

### Topic 2: motion tokenizers

| id         | first author, year | title                                     | stream | mark | said (abstract)                                                 | value for us                                          |
| ---------- | ------------------ | ----------------------------------------- | ------ | ---- | --------------------------------------------------------------- | ----------------------------------------------------- |
| 2504.03639 | Liao, 2025         | Shape My Moves                            | n/a    | R    | FSQ-VAE motion tokens, de-quantized with body shape.            | Evidence that FSQ works for motion.                   |
| 2411.17335 | Ling, 2024         | VersatileMotion                           | n/a    | C    | VQ-VAE tokenizer combined with flow matching in a motion LLM.   | Hybrid discrete-continuous decoder.                   |
| 2505.12774 | Geng, 2025         | UniHM                                     | n/a    | N    | Scene and object aware motion language model.                   | Out of scope (scenes).                                |
| 2301.06052 | Zhang, 2023        | T2M-GPT                                   | causal | R    | CNN VQ-VAE with EMA and code reset plus GPT; simple and strong. | Our transformer twin lineage (already in references). |
| 2308.14480 | Kong, 2023         | Priority-Centric Motion in Discrete Space | other  | C    | Discrete diffusion that generates salient motions first.        | Token-order idea.                                     |
| 2410.10010 | Javed, 2024        | InterMask                                 | bidir  | N    | 2D token maps for two-person interactions.                      | Out of scope.                                         |
| 2505.08293 | Yin, 2025          | M3G                                       | n/a    | C    | Multi-granular gesture tokens of variable length.               | Variable token granularity.                           |

### Topic 3: state space models against the KV cache

| id         | first author, year | title                                         | stream | mark | said (abstract)                                                                       | value for us                                    |
| ---------- | ------------------ | --------------------------------------------- | ------ | ---- | ------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 2608.02032 | Qian, 2026         | DART: Decoded Attention over Recurrent States | causal | R    | Reads Mamba-2 chunk states as a compressed KV cache; cuts the length-dependent cache. | Middle ground between our twins.                |
| 2605.06997 | Sridhar, 2026      | Echo                                          | causal | R    | Pure Mamba-2 near chance on associative recall; constant-memory operator fixes it.    | Names the failure mode the Mamba twin may show. |
| 2610.41850 | Bendi-Ouis, 2026   | Dynamical Memory Transformer                  | causal | R    | Fixed-size recurrent memories read by attention; between Mamba and Transformer.       | Hybrid option for future work.                  |
| 2609.11998 | Singh, 2026        | Fixed State, Long Reach                       | causal | R    | Mamba cache O(1): lower latency and memory, longer retrieval than attention.          | Strongest support for the fixed-state side.     |

### Topic 4: transitions, composition and steering

| id         | first author, year    | title                                   | stream  | mark | said (abstract)                                                               | value for us                                   |
| ---------- | --------------------- | --------------------------------------- | ------- | ---- | ----------------------------------------------------------------------------- | ---------------------------------------------- |
| 2405.11126 | Cohan, 2024           | Flexible Motion In-betweening (CondMDI) | diff    | C    | One diffusion model for keyframes plus text.                                  | Keyframe control reference.                    |
| 2510.19789 | Xu, 2025              | OmniMotion-X                            | AR-diff | C    | Autoregressive diffusion transformer for many conditions and in-betweening.   | Multi-task streaming reference.                |
| 2209.00349 | Kim, 2022             | FLAME                                   | diff    | C    | Diffusion synthesis and frame- or joint-wise editing.                         | Editing reference.                             |
| 2312.03596 | Pinyoanuntapong, 2023 | MMM                                     | bidir   | C    | Masked motion model: fast, editable, not causal (already in references).      | Not streamable.                                |
| 2502.02063 | Chang, 2025           | CASIM                                   | n/a     | R    | Composite-aware text injection instead of one fixed text vector.              | Text conditioning upgrade for prompt switches. |
| 2603.13500 | Nazarenus, 2026       | ActionPlan                              | AR-diff | R    | Frame-level text plans; streaming 5.25x faster and 18% better FID (abstract). | Strongest streaming result to compare.         |
| 2209.04066 | Athanasiou, 2022      | TEACH                                   | other   | R    | Temporal composition of a series of texts.                                    | Origin of the multi-prompt task we steer.      |
| 2304.10417 | Athanasiou, 2023      | SINC                                    | n/a     | C    | Simultaneous actions by body part.                                            | Spatial composition, later.                    |

### Topic 5: evaluation and related

| id         | first author, year | title                              | stream | mark | said (abstract)                                               | value for us                          |
| ---------- | ------------------ | ---------------------------------- | ------ | ---- | ------------------------------------------------------------- | ------------------------------------- |
| 2508.14561 | Jeong, 2025        | RVQ for expressive pose codes      | n/a    | C    | Pose codes plus residual continuous features.                 | Tokenizer detail trade.               |
| 2602.18199 | Shim, 2026         | Distortion-aware Motion Calibrator | n/a    | C    | Post-hoc fix of foot floating while keeping semantics.        | Physical plausibility post-process.   |
| 2506.10353 | Ouyang, 2025       | Motion-R1                          | n/a    | N    | Chain-of-thought and RL for motion.                           | Out of scope.                         |
| 2603.27185 | Tan, 2026          | MotionRFT                          | n/a    | N    | Reinforcement fine-tuning across representations.             | Later.                                |
| 2503.06119 | Tanaka, 2025       | MoMug                              | n/a    | N    | One LLM for diffusion motion and text tokens.                 | Out of scope.                         |
| 2606.14721 | Wang, 2026         | DC-Motion                          | n/a    | R    | Discrete tokens for structure, continuous for detail.         | Answer to the discrete-loss critique. |
| 2603.25178 | Weng, 2026         | BiHumanML3D                        | diff   | N    | Bilingual benchmark.                                          | Out of scope.                         |
| 2602.13751 | Yang, 2026         | T2MBench                           | n/a    | R    | 1,025 OOD prompts; most models weak on fine-grained accuracy. | Stress test for our twins.            |

### Core papers (title search)

| id         | first author, year    | title                        | stream  | mark | said (abstract)                                                                         | value for us                                            |
| ---------- | --------------------- | ---------------------------- | ------- | ---- | --------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| 2403.07487 | Zhang, 2024           | Motion Mamba                 | bidir   | R    | First SSM motion generator; up to 50% FID gain and 4x faster (abstract).                | Mamba in diffusion, not causal (already in references). |
| 2407.10061 | Zhang, 2024           | InfiniMotion                 | other   | R    | 1-hour motion (about 80,000 frames) with a transformer plus bidirectional Mamba memory. | Endless-motion competitor; check its causality.         |
| 2503.15451 | Xiao, 2025            | MotionStreamer               | AR-diff | R    | Streaming in a continuous causal latent space; less error accumulation.                 | Main streaming competitor (already in references).      |
| 2402.15509 | Barquero, 2024        | FlowMDM                      | diff    | R    | Seamless composition; proposes Peak Jerk and Area Under the Jerk.                       | Adopt the jerk metrics for steering.                    |
| 2312.00063 | Guo, 2023             | MoMask                       | bidir   | C    | Masked RVQ, FID 0.045 on HumanML3D (abstract).                                          | Quality ceiling (already in references).                |
| 2403.19435 | Pinyoanuntapong, 2024 | BAMM                         | bidir   | C    | Bidirectional autoregression, length prediction.                                        | Not streamable (already in references).                 |
| 2305.00976 | Petrovich, 2023       | TMR                          | n/a     | C    | Contrastive text-motion retrieval.                                                      | Second evaluator option.                                |
| 2309.15505 | Mentzer, 2023         | Finite Scalar Quantization   | n/a     | R    | FSQ: no codebook collapse, no commitment loss.                                          | Basis of our tokenizer (already in references).         |
| 2402.01032 | Jelassi, 2024         | Repeat After Me              | causal  | R    | Transformers beat fixed-state models at copying; a fundamental gap.                     | Strongest argument against the fixed state.             |
| 2411.16575 | Meng, 2024            | MARDM (Rethinking Diffusion) | other   | R    | VQ wins on metrics but loses diversity; proposes a more robust evaluation.              | Evaluation critique to answer.                          |
| 2410.03441 | Tevet, 2024           | CLoSD                        | other   | C    | Diffusion planner plus RL physics control.                                              | Physics-based studio option.                            |
| 2412.11193 | Zeng, 2024            | Light-T2M                    | bidir   | R    | 4.48M parameters, FID 0.040 (abstract); Mamba with pseudo-bidirectional scan.           | Small-hardware reference.                               |
| 2404.19759 | Dai, 2024             | MotionLCM                    | diff    | C    | Real-time control via one- or few-step latent consistency.                              | Latency reference.                                      |

## 4. Next steps (proposals)

1. Verify the 6 numbers used above against the paper tables: ActionPlan, Light-T2M, MoMask, Motion Mamba,
   InfiniMotion, Fixed State Long Reach.
2. Add Peak Jerk and Area Under the Jerk to the A28-T11 steering measure.
3. Add a recall probe (A, B, A) to the twin benchmark.
4. Search university repositories for theses on text-driven motion (not on arXiv).
