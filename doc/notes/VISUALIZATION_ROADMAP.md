# Beginner-First Dissertation Visualization Roadmap

## Goal

Make the dissertation understandable to a reader who knows basic machine learning but has never worked with motion representation, vector quantization, Mamba/S6, HumanML3D, or streaming inference.

This audit proposes **64 useful visual placements**. It does **not** recommend 64 full-page figures. The camera-ready target should be:

- **22 core numbered figures** in the main text;
- **13 compact visual tables, callouts, or inset panels** in the main text;
- **15 technical figures** in appendices;
- **14 optional presentation/demo visuals** only if page budget permits.

A visual earns a place only if it introduces a concept, explains a mechanism, proves a result, or states a limitation. Decorative visuals should be removed.

## Consistent visual language

Use the same encoding throughout:

- Grouped-FSQ: orange;
- RVQ: red;
- Mamba/SSM: green;
- Transformer/attention: blue;
- ground truth/reference: dark gray;
- favorable result: solid outline plus check mark;
- limitation/caveat: amber outline;
- invalid/non-streaming path: gray with strike-through.

Never encode meaning with color alone. Add direct labels, symbols, and line styles. Prefer SVG/PDF vector output; use 300 dpi PNG only for rendered motion/mesh images.

---

# Placement catalog

## Front matter and abstract

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 1 | Title page | One-frame graphical abstract: text → tokens → bounded-state generator → moving skeleton | What is the whole thesis about? | Full-width pipeline | Should |
| 2 | Abstract, after the first sentence | Full-sequence generation versus incremental generation timeline | What does “streaming motion” mean? | Two-row timeline | Must |
| 3 | Abstract, after Contributions A and B | Two-contribution summary: FSQ improves representation; Mamba improves runtime state | How are the two contributions different and connected? | Two-column infographic | Must |
| 4 | Abstract, before its final sentence | Four-number result strip: 0.017 recon-FID; R@1 0.255 vs 0.246; 2.68 vs 605 MB; 6.58 → 2.12 FID | What should the reader remember? | Compact result cards | Should |

## Chapter 1 — Introduction

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 5 | §1.1, after “animation, robotics and embodied agents” | Three application examples: animation, embodied agent, assistive/robotics simulation | Why does text-to-motion matter? | Three-panel pictogram | Could |
| 6 | §1.1, after “producing whole-body motion incrementally” | Prompt-arrival and motion-emission timeline, including first-output latency | What arrives when, and what is emitted when? | Annotated timeline | Must |
| 7 | §1.1, after the KV-cache sentence | Growing KV-cache stack versus one fixed recurrent-state box | Why is transformer memory unbounded with horizon? | Mechanism cartoon | Must |
| 8 | §1.1, end | One caption with three valid motions and two invalid motions | Why is text-to-motion one-to-many? | Skeleton storyboard | Should |
| 9 | §1.2, beside the thesis statement | Claim matrix: quality parity, memory complexity, what is and is not claimed | What exactly is the hypothesis? | 2×3 claim matrix | Must |
| 10 | §1.3, before the numbered list | End-to-end contribution map with chapter numbers | Where does each contribution sit in the system? | Roadmap diagram | Must |
| 11 | §1.3, item 1 | Tokenizer versus generator “do not confuse these” inset | Why are there two learned models? | Before/after analogy | Should |
| 12 | §1.3, item 3 | Pretrain → fine-tune data flow: AMASS motion-only, then HumanML3D text-motion | How can unlabeled motion help a text-conditioned model? | Two-stage flow | Should |
| 13 | §1.4 | Scope boundary map: primary HumanML3D-263; qualitative SMPL-X/hands; face deferred | What is inside and outside the thesis? | Concentric scope diagram | Must |
| 14 | §1.5 | Related-work landscape with axes: discrete/continuous × causal/bidirectional × bounded/growing state | Which research cell is claimed as new? | 2D/3D taxonomy | Must |

## Chapter 2 — Background and related work

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 15 | Start of §2.1 | Pose → sequence of poses → motion clip at 20 fps | What is a pose, frame, and motion? | Four-frame storyboard | Must |
| 16 | §2.1, beside the 263-channel list | Labeled 22-joint skeleton with root coordinate frame | What physical body is represented? | Annotated skeleton | Must |
| 17 | §2.1, after the list | HumanML3D-263 channel ledger: 4 + 63 + 126 + 66 + 4 = 263 | Where do all 263 numbers come from? | Stacked channel bar | Must |
| 18 | §2.1 | Root-relative versus world-space motion example | Why use root-invariant coordinates? | Two-coordinate-frame panel | Should |
| 19 | §2.1 | 6D rotation intuition: two vectors → orthonormal frame, contrasted with discontinuous angle wrap | Why six numbers for a 3D rotation? | Geometry inset | Should |
| 20 | §2.2, before VQ-VAE discussion | Continuous latent trajectory mapped to discrete token IDs | What does motion tokenization do? | Analogy diagram | Must |
| 21 | §2.2 | VQ versus RVQ versus FSQ in three aligned panels | How do the three quantizers differ? | Comparison flow | Must |
| 22 | §2.2, beside code collapse | Learned codebook with used and dead entries | What is codebook collapse? | Lattice/scatter cartoon | Should |
| 23 | §2.3 | Generator family taxonomy: causal AR, masked/bidirectional, diffusion, continuous streaming | How do prior generators differ? | Family tree/table | Should |
| 24 | §2.3 | Teacher forcing versus autoregressive inference | How is a next-token model trained and then sampled? | Two-row sequence diagram | Must |
| 25 | §2.4, before S4/Mamba prose | State-space recurrence loop with fixed-size state | What is a state-space model intuitively? | Recurrent cell diagram | Must |
| 26 | §2.4 | Selectivity cartoon: retain useful past, forget irrelevant past based on current input | What makes S6 “selective”? | Memory-gate storyboard | Should |
| 27 | §2.4 | Parallel scan in training versus recurrent step at inference | How can training be parallel while inference is recurrent? | Split-panel DAG | Should |
| 28 | §2.5 | Bidirectional matrix, causal attention triangle, recurrent state loop | Why does streaming require causality? | Three-panel comparison | Must |
| 29 | §2.5, end | Architecture eligibility table: future access, streaming, memory growth, use in this thesis | Why are some accurate models rejected as runtime generators? | Visual decision table | Must |

## Chapter 3 — Representation, data, and evaluation

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 30 | §3.1, before channel details | Full representation round trip: joints/pose → 263 → normalize → model → denormalize → joints | How does data move between human motion and tensors? | End-to-end flow | Must |
| 31 | §3.1 | Tensor-shape ladder with concrete example: batch, frames, 263 channels, latent steps, 8 codebooks | What do symbols such as $B,T,Q,V$ mean? | Shape table/diagram | Must |
| 32 | §3.1 | Normalization before/after distributions for representative channels | Why normalize motion features? | Small multiples | Could |
| 33 | §3.1 | Round-trip validation: original and recovered skeleton keyframes plus numerical error | How do we know the representation conversion is correct? | Overlay/storyboard | Should |
| 34 | §3.2, beginning | One HumanML3D sample: caption plus six motion keyframes | What does one training example look like? | Qualitative strip | Must |
| 35 | §3.2 | Dataset accounting: HumanML3D captions/clips/hours and AMASS sequences/hours actually used | How much data is available and used? | Bars plus exact-count table | Must |
| 36 | §3.2 | Canonical train/val/test split and “test touched once” lock | Which data can influence model choices? | Split funnel | Must |
| 37 | §3.2 | Mirror augmentation: original and mirrored motion, including left/right joint swap | What does mirror augmentation change? | Paired skeleton strip | Could |
| 38 | §3.2 | AMASS conversion path: source parameters → SMPL-X body → joints → HumanML3D-263 | How can two datasets share one representation? | Data pipeline | Should |
| 39 | §3.3, before FID equation | Frozen Guo evaluator pipeline with shared text-motion embedding space | Where are all metrics computed? | Evaluation flowchart | Must |
| 40 | §3.3, beside FID | Generated and real embedding distributions with close/far examples | What does lower FID mean? | Toy scatter/distribution | Must |
| 41 | §3.3 | R-precision retrieval pool with ranked candidate captions/motions | What does R@1/R@2/R@3 measure? | Ranked retrieval example | Must |
| 42 | §3.3 | Metric compass: FID=distribution fidelity; R-precision/MM-Dist=alignment; Diversity=global spread; MultiModality=within-caption variation | Why are several metrics necessary? | Four-quadrant explainer | Must |
| 43 | §3.3, after sample-size warning | Same generator evaluated with increasing clip counts, showing FID estimate stabilization/bias | Why must comparisons use equal sample counts? | Convergence plot | Should |
| 44 | §3.4 | Reproducibility chain: config + commit + seed → checkpoint → metrics → dissertation table | Can a reader trace every number? | Provenance flow | Should |

## Chapter 4 — Contribution A: Grouped-FSQ tokenizer

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 45 | §4.1, before the equation | Scalar input passing through bound, fixed levels, and rounding | What does FSQ do to one number? | Function plot | Must |
| 46 | §4.1, beside the STE equation | Forward path rounds; backward path bypasses rounding | How can gradients cross a discrete operation? | Forward/backward arrows | Must |
| 47 | §4.1, after implicit codebook claim | Cartesian product of scalar levels creates implicit codes without stored vectors | How can there be many codes with zero codebook parameters? | 2D lattice example | Should |
| 48 | §4.2, beginning | Encoder → grouped channels → residual FSQ stages → summed reconstruction → decoder | What is the complete tokenizer architecture? | Full architecture | Must |
| 49 | §4.2, beside residual equation | Residual refinement over stages with visibly shrinking error | What does each quantizer add? | Three-stage vector cartoon | Must |
| 50 | §4.2, beside bit-rate equation | Codes/step × bits/code accounting with one worked configuration | What does matched bit-rate mean? | Visual arithmetic | Must |
| 51 | §4.3 | Controlled tokenizer comparison checklist: same encoder, decoder, data, seed, epochs; quantizer differs | Why is FSQ versus RVQ fair? | Twin checklist | Must |
| 52 | §4.3 | RVQ maintenance machinery versus FSQ: EMA, commitment, reset, dropout, learned parameters | What complexity does FSQ remove? | Side-by-side table | Should |
| 53 | §4.4, immediately before results table | Existing tokenizer matrix bar chart, updated and captioned | Where does FSQ win at matched rates? | Data figure | Must |
| 54 | §4.4, after results table | Distortion versus quantizer parameters Pareto plot | Why is “Pareto-dominant” justified? | Scatter/Pareto frontier | Must |
| 55 | §4.4, beside honest bounds | Our controlled cells versus cited MoMask cell, with compute/protocol warning | Which cross-paper comparison is valid? | Comparison callout | Should |
| 56 | §4.5 | Existing train-versus-test recon-FID scatter | Did FSQ win by memorizing? | Data figure | Should |

## Chapter 5 — Contribution B: generator and streaming

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 57 | §5.1, beginning | Text prefix + previous motion tokens → shared front end → interchangeable mixer → 8 token heads | What enters and exits the generator? | End-to-end architecture | Must |
| 58 | §5.1 | One autoregressive generation step expanded, then looped | How are eight codebook IDs generated at every latent step? | Step sequence | Must |
| 59 | §5.1, beside CFG equation | Conditional and unconditional logits extrapolated by guidance scale | What does classifier-free guidance do? | Vector/score illustration | Should |
| 60 | §5.1, beside nucleus sampling | Probability bars showing temperature and top-$p$ cutoff | How does sampling trade reliability for diversity? | Distribution inset | Could |
| 61 | §5.1.1, optimization paragraph | Warmup + cosine LR and observed useful epoch window | Why use this schedule, and why short runs? | LR curve with overlay | Should |
| 62 | §5.1.1, decoding paragraph | Beam search converging to one mode versus stochastic samples covering valid modes | Why reject beam search? | Branching trajectories | Could |
| 63 | §5.1.1, attention paragraph | Prefix self-attention stream versus cross-attention, highlighting the shared twin-compatible design | Why is text represented as a prefix? | Two architecture sketches | Should |
| 64 | §5.1.1, positional encoding | Clip phases (start/develop/end) aligned with learned absolute positions | Why can absolute position help motion clips? | Timeline heatmap | Could |
| 65 | §5.1.1, variable-length paragraph | Right padding, causal mask, valid-loss mask, and END token | How are variable-length clips trained safely? | Token-grid diagram | Should |
| 66 | §5.1.1, normalization paragraph | Pre-norm residual block and why BatchNorm is unsuitable at batch 1 | Why RMSNorm for streaming? | Mini block diagram | Could |
| 67 | §5.1.1, SSM module paragraph | Expanded S6/Mamba block: projection → causal conv → selective scan → gate → projection | What is inside the mixer? | Technical architecture | Should |
| 68 | §5.1.1, gradient stability | Stable transition factor $e^{\Delta A}$ in $(0,1)$ and input-controlled decay | Why does recurrence not explode? | Decay curves | Could |
| 69 | §5.2 | Controlled-twin diagram with all shared components gray and only mixers colored | What exactly differs between twins? | Matched architecture | Must |
| 70 | §5.2 | Parameter-matching bars and layer counts for 34M and 100M configurations | How can different-depth networks have similar size? | Stacked parameter bars | Should |
| 71 | §5.3 | Batch/parallel path versus streaming/step path converging to the same output | What does the parity test prove? | Two-path diagram | Must |
| 72 | §5.3 | Numerical parity test result with tolerance and passed cases | Is parity merely asserted or measured? | Test-result card | Should |
| 73 | §5.4, beginning | AMASS $p(z)$ pretraining then HumanML3D $p(z\mid c)$ fine-tuning | What knowledge transfers? | Two-stage pipeline | Must |
| 74 | §5.4, after ablation table | Before/after bars for FID, R@1, and Diversity, with direction arrows | How large is the pretraining gain? | Data figure | Must |
| 75 | §5.5, before existing figure | Memory accounting cartoon: per-layer KV entries accumulate; SSM/conv state updates in place | Why do the measured lines have these shapes? | Mechanism inset | Must |
| 76 | §5.5 | Existing streaming benchmark, retaining exact caveat in caption | What is the signature efficiency result? | Data figure | Must |
| 77 | §5.5, after benchmark table | Convert token horizons into seconds/minutes at the latent rate | What does 8192 tokens mean in motion duration? | Secondary axis/table | Should |
| 78 | §5.5, caveats | Architectural result versus hardware-dependent result: memory exact, latency provisional | Which result is strongest? | Evidence-strength badge | Must |
| 79 | §5.5.1, after “the constraint” | Decoder convolution receptive field centered on a latent token | Why are future latent tokens needed? | Receptive-field strip | Must |
| 80 | §5.5.1, after “the fix” | Naive chunk decode versus ring buffer + lookahead + crop | How is streaming decode made exact? | Three-panel mechanism | Must |
| 81 | §5.5.1, beside lookahead table | Fidelity-delay Pareto curve for $r=0…4$ | How should a runtime choose lookahead? | Line/Pareto plot | Should |
| 82 | §5.6, before main table | Protocol card: frozen tokenizer, shared prior/recipe/seed/effective batch, 2189 clips × 20 reps | What makes this the citable comparison? | Experiment card | Must |
| 83 | §5.6, after main table | Metric win/loss slope chart: Mamba alignment wins; transformer fidelity/diversity wins | Does either twin dominate? | Dumbbell/slope chart | Must |
| 84 | §5.6, beside stream-state column | Quality versus streaming-state trade-space | What is gained for the quality trade? | 2D trade-off plot | Must |
| 85 | §5.6, before separate val pair | Strong visual separator labeled “not protocol-comparable” | Why must these tables not be merged? | Warning divider | Must |
| 86 | §5.6, after val pair | Confounding diagram: capacity, prefix length, and sampling all change | Why can the improvement not be attributed to scale? | Causal/confound diagram | Must |
| 87 | §5.7 | Fault-localization chain: evaluator check → tokenizer ceiling → teacher-forced diagnostics → generator | Where does the absolute FID gap originate? | Diagnostic decision tree | Should |
| 88 | §5.8, beside CE decomposition | Irreducible entropy plus optimizable KL gap | What does the equation mean? | Stacked-loss cartoon | Should |
| 89 | §5.8, after holdout table | Training CE versus held-out CE reversal across the three priors | Why is low training CE misleading? | Paired slope chart | Must |
| 90 | §5.8, beside holdout caveat | Train clips and short unseen clips length distributions | How serious is the distribution-shift caveat? | Histogram/box plot | Should |
| 91 | §5.8, stopping protocol | Observable stopping dashboard: val loss, train–val gap, downstream FID | What should practitioners monitor? | Three-signal checklist | Should |
| 92 | §5.9, matched paragraph | Batch 8 versus two batch-4 microbatches before one update | Why is gradient accumulation update-matched? | Microbatch flow | Should |
| 93 | §5.9, first asymmetry | Prior-training asymmetry and held-out result, with direction of possible bias | Does the asymmetry favor Mamba? | Bias audit diagram | Should |
| 94 | §5.9, second asymmetry | Weight decay pushing $a_{\log}$ toward shorter memory | How can one optimizer choice penalize SSM memory? | Cause-effect diagram | Should |
| 95 | §5.9, after parameter census | Parameter composition bars for both twins | Where are unmatched parameter types? | Stacked bars | Could |

## Chapter 6 — Discussion and limitations

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 96 | Start of Chapter 6 | Claimed / supported / not claimed matrix | What may the reader conclude? | Evidence matrix | Must |
| 97 | Chapter 6 threats list | Threat-to-validity map: internal, construct, external, compute | What can weaken each conclusion? | Four-quadrant map | Should |
| 98 | §6.1, after exact token budget | Frames → latent steps → 8 codebook targets → mirror factor = 6.53M targets | How was the data budget counted? | Worked visual equation | Must |
| 99 | §6.1, after Chinchilla paragraph | Actual operating points versus nominal $D^*=20N$ line, log-log axes | How far is the project from the text-derived optimum? | Scaling plot | Must |
| 100 | §6.1, after repetition formula | Effective unique-data gain saturating with repeated epochs | Why can training longer not create new information? | Saturation curve | Should |
| 101 | §6.1 | Actual, all-local-AMASS, MotionX, and nominal-required hours | How large is the capture-data gap in human units? | Log-scale bars | Should |
| 102 | §6.1, entropy paragraph | Dataset maximum entropy versus 34M/100M model storage estimates | Why is memorization plausible? | Capacity bars | Should |
| 103 | §6.1, “Why we nonetheless train at 100M” | Applicability ladder: Chinchilla insight transfers; constant 20 does not necessarily transfer | How should the scaling-law argument be interpreted cautiously? | Caveat ladder | Must |
| 104 | §6.1, bearing on claims | Shared-confound diagram showing data limitation applies equally to both twins | Why does data scarcity limit absolute FID but not automatically invalidate the twin comparison? | Causal diagram | Must |

## Chapter 7 — Conclusion and future work

| # | Insert after / beside | Proposed visual | Beginner question answered | Form | Priority |
|---:|---|---|---|---|---|
| 105 | Start of conclusion | Reprise the full pipeline, now annotated with measured outcomes at each stage | What was built and proven? | Summary pipeline | Must |
| 106 | After contribution summary | Evidence ledger linking each claim to one experiment/figure | What evidence supports each contribution? | Claim-evidence table | Must |
| 107 | Future work: causal tokenizer | Current non-causal decoder versus proposed causal decoder with zero lookahead | How could the 0.8 s delay be removed? | Before/after architecture | Should |
| 108 | Future work: whole-body | HumanML3D body joints versus SMPL-X body + hands + face scope | What is needed for whole-body generation? | Skeleton/mesh scope | Should |
| 109 | Future work: scaling/data | Prioritized roadmap: tokenize remaining AMASS → controlled prefix ablation → 100M twin → broader corpus | What should happen next, and in what order? | Milestone roadmap | Must |
| 110 | Final page | One-page take-home summary using the same four key numbers as the abstract | What should remain after the details are forgotten? | Closing infographic | Could |

---

# Recommended camera-ready subset

The catalog contains 110 placements because several dense paragraphs deserve small insets or appendix support. To avoid visual overload, use this **22-figure main-text spine**:

1. Streaming versus full-sequence timeline (#2/#6 combined).
2. Contributions and end-to-end roadmap (#3/#10 combined).
3. Scope and novelty landscape (#13/#14 combined).
4. Pose, motion, and HumanML3D-263 anatomy (#15–#17 combined).
5. VQ versus RVQ versus FSQ (#20–#22 combined).
6. Generator-family causality and runtime eligibility (#23/#28/#29 combined).
7. SSM recurrence and selective memory (#25–#27 combined).
8. Representation/data round trip and tensor shapes (#30/#31/#38 combined).
9. Dataset sample, accounting, and split discipline (#34–#36 combined).
10. Guo evaluator and metric intuition (#39–#42 combined).
11. FSQ scalar quantization and STE (#45–#47 combined).
12. Grouped-residual tokenizer and bit-rate accounting (#48–#50 combined).
13. Controlled tokenizer comparison plus measured matrix (#51–#54 combined).
14. Generator architecture and one autoregressive step (#57/#58/#67 combined).
15. Controlled transformer–Mamba twin (#69/#70 combined).
16. Batch/stream parity (#71/#72 combined).
17. AMASS pretraining mechanism and ablation (#73/#74 combined).
18. Streaming benchmark with memory mechanism (#75/#76/#78 combined).
19. Decoder receptive field and ring-buffer solution (#79–#81 combined).
20. Controlled-twin results and quality/state trade-off (#82–#86 combined).
21. Held-out CE reversal and stopping lesson (#88–#91 combined).
22. Data-scaling limitation and claim boundary (#96/#98–#104 combined).

The remaining placements should become compact visual callouts, appendix figures, or presentation material.

---

# Existing assets: what can be reused now

## Ready or nearly ready

- `paper/figures/streaming.png`: use for #76. It is the strongest existing figure. Keep the memory claim primary and label latency as hardware-sensitive/provisional.
- `paper/figures/tokenizer_matrix.png`: use for #53 after checking labels against the final 8×1024 table and adding direct numeric labels/accessibility patterns.
- `paper/figures/generalization.png`: use for #56, preferably in an appendix or as a compact supporting figure.
- `paper/figures/pilot_progress.png`: useful near #61 or #91, but its 200-clip validation gauge must be visibly labeled non-citable and not confused with the 2189-clip test result.
- `paper/figures/demo_clip.png`: useful for #8 or #34 as a qualitative example, with model/checkpoint and “illustrative, not an evaluation result” in the caption.

## Diagram sources that must be updated before reuse

Do **not** insert the current DOT diagrams unchanged:

- `paper/diagrams/tokenizer.dot` contains stale six-group/1000-vocabulary wording and old result values, while the dissertation’s selected configuration is FSQ 8×1024 with recon-FID 0.0170.
- `paper/diagrams/generator-twins.dot` contains stale six-codebook heads, old parameter values, CFG 5.0, and temperature 1.1; the final controlled test uses eight codebooks, 34M twins, CFG 6, temperature 1.0, and top-$p$ 0.9.
- `paper/diagrams/evaluation.dot` is structurally useful but should be revised to make metric semantics exact: MM-Dist is text–motion embedding distance; Diversity is spread across generated motions; MultiModality is variation among repeated generations for the same caption.
- All DOT diagrams need SVG/PDF exports and shorter labels before camera-ready use.

## Missing high-value assets

The largest current gaps are:

1. no beginner explanation of pose/motion/HumanML3D-263;
2. no visual explanation of causality and why bidirectional methods cannot stream;
3. no exact metric-intuition figure;
4. no current, accurate full generator architecture;
5. no decoder receptive-field/ring-buffer figure;
6. no visual of the controlled-twin result and its trade-off;
7. no held-out-CE reversal plot;
8. no Chinchilla/data-budget plot;
9. no explicit claimed-versus-not-claimed summary;
10. no final claim-to-evidence recap.

---

# Chapter-level beginner comprehension test

A chapter is visually complete only if a new reader can answer these questions after reading it:

- **Chapter 1:** What is streaming, why is it hard, and what are the two contributions?
- **Chapter 2:** What is a motion token, why does causality matter, and how does fixed recurrent state differ from attention history?
- **Chapter 3:** What does one data sample look like, what is the 263-vector, and what does every evaluation metric mean?
- **Chapter 4:** How does one scalar become an FSQ code, how do residual stages refine it, and where does FSQ beat RVQ fairly?
- **Chapter 5:** How does text become motion tokens one step at a time, what differs between twins, and what proves end-to-end bounded streaming?
- **Chapter 6:** What is limited by data, which claims remain valid, and where are the honest caveats?
- **Chapter 7:** What was proven, what was not proven, and what is the next experiment?

If any answer still requires rereading equations or searching another chapter, the relevant visual is missing or placed too late.

---

# Production order

## Phase 1 — make the thesis understandable

Create figures 1–12 from the 22-figure spine: motion basics, representation, quantization, causality, metrics, and architecture. These remove prerequisite gaps.

## Phase 2 — make the claims convincing

Create figures 13–20: controlled comparisons, ablations, benchmark, decoder closure, and main results.

## Phase 3 — make the limitations trustworthy

Create figures 21–22 and the conclusion evidence ledger: holdout reversal, scaling constraints, and exact claim boundaries.

## Phase 4 — polish

Add appendix technical figures, alt text, consistent captions, vector exports, and references from the prose. Remove any visual that repeats a table without adding a pattern, mechanism, or interpretation.
