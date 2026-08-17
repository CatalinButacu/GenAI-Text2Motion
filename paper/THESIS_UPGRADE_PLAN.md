# Thesis Upgrade Plan: Clean Retraining and Reevaluation

## Executive decision

The thesis already has a defensible core idea:

> A discrete Mamba/S6 motion generator trades a fixed recurrent state for a transformer's growing KV cache, while a Grouped-FSQ tokenizer provides a strong discrete motion representation.

The **bounded-memory result is currently the strongest claim**. The **quality-parity claim is not yet final evidence** because the current comparison is based on one training seed, historically unmatched pretraining, an old Mamba weight-decay defect, uncertain text-encoder pairing in evaluation, and a nonstandard interpretation of “20 repetitions.”

The correct next step is **not simply to train a larger model**. The thesis improves most by producing one clean, preregistered, replicated controlled experiment.

### Recommended experiment hierarchy

1. **Primary inferential experiment:** corrected 34M Transformer/Mamba twins, three paired training seeds.
2. **Data ablation:** HumanML3D-only versus identically AMASS-pretrained twins.
3. **Scale replication:** one matched 100M Transformer/Mamba pair after the 34M protocol is frozen.
4. **Final reevaluation:** official sample construction, complete repeated generations, paired uncertainty, and trained end-to-end streaming measurements.

If compute is limited, complete the six 34M runs before spending budget on a single 100M model. Replication and control strengthen this thesis more than an isolated lower FID.

---

# 1. Current thesis structure: evaluation

## What is already strong

### A. The contributions form one coherent system

Contribution A supplies discrete motion tokens. Contribution B predicts those tokens with a causal fixed-state mixer. The streaming decoder closes the loop from generated tokens to motion frames. This is a coherent thesis rather than two unrelated model experiments.

### B. The primary novelty is specific and testable

The work does not merely say “Mamba for motion.” It tests a specific conjunction:

- discrete motion tokens;
- text-conditioned autoregression;
- causal generation;
- recurrent fixed-size state;
- controlled comparison with a causal Transformer.

### C. Negative results and limitations are documented

The draft reports the absolute FID gap, pretraining memorization, optimizer asymmetries, data constraints, and provisional latency caveats. This honesty should be preserved.

### D. The repository has unusually strong engineering support

It already contains:

- batch-versus-step parity tests;
- bounded decoder-state tests;
- run manifests and metric logs;
- fixed Guo evaluator weights;
- tokenizer reconstruction evaluation;
- streaming state and latency instrumentation;
- a working incremental studio viewer.

## What weakens the present narrative

### A. Chapter 5 carries too many different arguments

The current generator chapter combines:

- architecture description;
- optimizer rationale;
- sampling choices;
- positional encodings;
- gradient stability;
- controlled-twin methodology;
- pretraining;
- streaming complexity;
- decoder receptive field;
- main results;
- information theory;
- fairness audits.

This makes the main claim difficult to find. The clean final chapter should follow:

1. hypothesis;
2. architecture;
3. controlled protocol;
4. training;
5. streaming system;
6. results;
7. ablations;
8. threats to validity.

Move long derivations and implementation defenses to appendices.

### B. The background and data chapters are too thin relative to Chapter 5

A beginner reaches FSQ, S6, FID, and HumanML3D-263 with insufficient scaffolding. Expand Chapters 2–3 using the visualization roadmap and reduce defensive detail in Chapter 5.

### C. The contribution hierarchy is blurred

AMASS pretraining currently appears as a third contribution in some places and as a supporting ablation elsewhere. For a sharper thesis:

- **Contribution A:** Grouped-FSQ tokenizer.
- **Contribution B:** bounded-state token-AR SSM generator versus controlled Transformer.
- **Supporting result:** leakage-safe AMASS motion-prior pretraining.
- **System result:** bounded end-to-end streaming decode and studio demonstration.

Pretraining should become a contribution only if it is rerun cleanly for both mixers and evaluated as a factorial treatment.

### D. “Parity” is not operationally defined

The thesis must define a practical non-inferiority/equivalence margin before seeing new test results. Failure to find a statistically significant difference does not prove parity.

### E. Absolute benchmark comparison is currently unsafe

The project uses fixed Guo weights, but the local evaluator/sample construction has not yet been shown output-equivalent to the released official evaluator. The current evaluator also generates one motion set and repeats only caption/retrieval regrouping. Published baseline rows should remain contextual—not directly comparable—until this is fixed.

---

# 2. Blockers before any retraining

## Gate 1 — Freeze the research question

Preregister the primary question:

> At matched trainable parameters, data, tokenizer, optimizer-step budget, text conditioning, sampler, and training seeds, is Mamba non-inferior to a causal Transformer in HumanML3D generation quality while using horizon-independent recurrent state?

Declare:

- primary quality metric: FID;
- primary alignment metric: R@1;
- primary efficiency metric: recurrent-state bytes versus horizon;
- secondary metrics: R@2/3, MM-Dist, Diversity, MultiModality, latency;
- non-inferiority/equivalence margins;
- seeds and run matrix;
- validation selection rule;
- final test protocol.

Existing test results must be labeled historical/pilot evidence.

## Gate 2 — Repair data provenance

Before training:

1. Freeze HumanML3D train/val/test lists and their hashes.
2. Assert base-ID disjointness after removing mirror prefixes.
3. Pass the configured seed into all datasets/loaders.
4. Eliminate or count invalid caption/full-motion fallback pairings.
5. Pin and hash Mean/Std and the target skeleton.
6. Build immutable manifests containing usable clips, frames, captions, mirrors, drops, and durations.
7. Operationally prevent training-time test evaluation.

For AMASS:

1. resample every source exactly to 20 fps using timestamps;
2. exclude captures underlying HumanML3D validation/test;
3. group holdout by source/subject/session, not by arbitrary window stem;
4. include short sequences and long-sequence tails;
5. report unique source duration separately from overlapping sampled windows;
6. embed tokenizer/scaler/corpus/split hashes in the token release;
7. use one identical AMASS release for both twins.

The historical pack used only a fraction of the prepared corpus and should not be called “full AMASS.”

## Gate 3 — Repair the training recipe

### Optimizer

- Reuse the corrected no-decay grouping in both fine-tuning and pretraining.
- Assert `a_log`, skip/state parameters, biases, norms, and embeddings are never decayed.
- Log parameter names and counts per optimizer group.
- Retrain both twins; do not compare a corrected Mamba to an old Transformer.

### Losses

Choose one of two honest approaches:

1. implement the documented literal losses; or
2. rename and justify the current channel-space surrogates.

Preferred final loss:

$$
\mathcal L = \mathcal L_{CE}
+ \lambda_r\mathcal L_{recon}
+ \lambda_v\mathcal L_{\Delta motion}
+ \lambda_f\mathcal L_{foot\ skating}
+ \lambda_{root}\mathcal L_{root}
+ \lambda_{end}\mathcal L_{length}.
$$

At minimum, add real temporal-difference velocity and planted-foot velocity under GT contact. Use identical weights for both twins.

### Checkpoint bundle

A final checkpoint must atomically identify:

- generator weights;
- generator EMA weights;
- co-adapted CLIP weights;
- tokenizer identity/hash;
- resolved config and actual parameter count;
- optimizer/scheduler state;
- RNG and sampler state;
- dataset/split release IDs;
- Git commit and dirty-tree state.

Evaluation should fail if the generator was trained with unfrozen CLIP but its matching text encoder is absent.

### Resume

Store Python, NumPy, CPU/CUDA Torch, DataLoader, and sampler RNG states. Record parent-run identity and accumulation/global-step state.

## Gate 4 — Enforce real-batch overfit

Both final architectures must overfit one real HumanML3D batch using the production tokenizer, text encoder, masks, losses, optimizer groups, and precision.

Acceptance criteria:

- token accuracy at least 99%;
- CE near zero;
- geometric/reconstruction losses reduced at least 95%;
- nonzero gradients in intended generator/text layers;
- no tokenizer gradients;
- decoded result visually matches the input;
- curve, manifest, and rendered clip are archived.

The current synthetic sanity script is insufficient.

## Gate 5 — Validate the evaluator

Before new headline numbers:

1. reproduce embeddings from the released evaluator on a fixed input fixture;
2. reproduce the official GT metric row;
3. use the official motion-text sample/segment construction;
4. freeze exact scored IDs and captions;
5. integrate paired metric deltas and confidence intervals;
6. ensure both twins score exactly the same examples;
7. save per-item outcomes and embeddings.

Replace “unmodified evaluator” with “fixed published weights in a validated local reimplementation” unless output equivalence is proven.

---

# 3. Controlled retraining matrix

## Stage A — Canary recipe validation

Use corrected 34M models, seed 2026, prefix 16, no AMASS prior initially.

| Backbone | Seed | Maximum | Purpose |
|---|---:|---:|---|
| Transformer | 2026 | 20–30 epochs | Validate data, loss, EMA, CLIP bundle, FID trend |
| Mamba | 2026 | 20–30 epochs | Same under identical protocol |

Use a fixed validation set and fixed generation seeds. If either architecture requires a recipe change, change it for both and discard/restart both canaries.

Suggested promotion gate by epoch 15–20:

- FID below a preregistered threshold based on current pilots;
- R@1 materially above current pilot trajectory;
- no diversity collapse;
- no widening train/validation gap while validation quality reverses;
- optimizer-step and token exposure counts identical.

## Stage B — Primary six-run experiment

| Backbone | Approximate size | Training seeds |
|---|---:|---|
| Transformer | 33.9M | 2026, 2027, 2028 |
| Mamba/S6 | 34.1M | 2026, 2027, 2028 |

This is the smallest defensible replicated architecture comparison.

Lock across all six runs:

- FSQ 8×1024 tokenizer checksum;
- canonical HumanML3D data release;
- prefix length 16;
- same CLIP initialization/unfreezing;
- same loss and weights;
- same corrected decay groups;
- same effective batch and optimizer-step count;
- same warmup and scheduler by optimizer step;
- same EMA policy;
- same validation IDs and generation seeds;
- same checkpoint-selection rule;
- same sampler in final comparison;
- same GT-length mode for literature metrics;
- same precision policy.

Different physical microbatches are acceptable only with matched effective batch and optimizer steps. Describe this as **accumulation-matched**, not “gradient-identical.”

## Stage C — AMASS factorial ablation

The cleanest thesis design separates architecture from auxiliary data:

| Backbone | HumanML3D-only | Corrected AMASS prior + HumanML3D |
|---|---:|---:|
| Transformer | ✓ | ✓ |
| Mamba | ✓ | ✓ |

Run the complete $2\times2$ matrix at seed 2026 first. Promote AMASS pretraining into the three-seed primary matrix only if it benefits both architectures without introducing architecture-specific tuning.

Pretraining fairness requires identical:

- corpus release and group split;
- effective batch;
- optimizer steps;
- token exposure;
- schedule;
- decay policy;
- validation selection/stopping rule.

If this cannot be guaranteed, use HumanML3D-only for the main architecture claim and report AMASS as exploratory support.

## Stage D — Focused ablations

Run only after the main recipe is frozen, at 34M and seed 2026 unless elevated to a main claim.

1. **Prefix length:** 1 versus 16 for both twins.
2. **Mamba state dimension:** 8, 16, 32, preserving parameter-match tolerance where possible.
3. **Loss ablation:** CE-only versus complete motion-aware loss for one backbone first; repeat on both only if the effect is architecture-sensitive.
4. **Tokenizer downstream A/B:** identical generator on FSQ versus strong-RVQ tokens. This would turn Contribution A from reconstruction-only evidence into downstream generation evidence.
5. **Shared sampler versus per-model val-tuned sampler:** shared is the headline controlled result; individual tuning is secondary best-achievable performance.

Do not perform a broad hyperparameter sweep. Every ablation must answer a thesis question.

## Stage E — 100M scale replication

After Stage B is complete, train one matched pair:

| Backbone | Approximate resolved size | Seed |
|---|---:|---:|
| Transformer | 98M | 2026 |
| Mamba/S6 | 100M | 2026 |

This is a **single-seed scale replication**, not the primary inferential result. Add more 100M seeds only after the six 34M runs are complete.

Do not compare a new 100M Mamba against an old Transformer checkpoint. Both must share the frozen corrected protocol.

---

# 4. Training and model-selection protocol

## Budget definition

Match models by:

- trainable parameter count within a preregistered tolerance, preferably 2%;
- unique examples and motion frames;
- optimizer steps;
- effective batch;
- data order policy;
- validation frequency;
- total maximum training tokens.

Wall-clock time need not match because efficiency is part of the architectural difference.

## Validation selection

Use two levels:

1. **Trend evaluation:** fixed 600 validation clips at frequent intervals.
2. **Candidate confirmation:** all available validation base clips over at least five fixed generation seeds.

Select by lowest mean validation FID with guardrails:

- R@1 cannot degrade beyond a preregistered tolerance;
- diversity cannot collapse;
- tie-break by earlier epoch to avoid overfitting.

Use a maximum of roughly 120–160 epochs with early stopping. Do not assume more epochs improve the model.

Suggested stopping logic:

- never stop before epoch 30 unless a hard failure occurs;
- evaluate every five epochs after the initial dense checks;
- patience of six evaluations;
- require an improvement larger than estimated validation noise;
- stop a pilot when train CE falls while FID/R@1 consistently worsen.

## Sampler selection

Tune only on validation:

- temperature: 0.9, 1.0, 1.1;
- top-$p$: 0.9 fixed initially;
- CFG: 2, 3, 4, 5, and 6 only if justified by the pilot.

Choose one shared headline setting using a preregistered joint validation objective. Freeze it before final test generation.

---

# 5. Final reevaluation protocol

## Complete repetitions

For every selected checkpoint, perform **20 complete generation/evaluation repetitions**, not 20 regroupings of one generated set.

Each repetition must freshly generate every test motion and recompute:

- FID;
- R@1/2/3;
- MM-Dist;
- Diversity;
- MultiModality.

Pair Transformer and Mamba on:

- training seed;
- generation seed;
- prompt order;
- caption/segment choice;
- retrieval grouping;
- MultiModality prompt list.

## Statistical unit and reporting

Training seed is the unit for architecture generalization. For each model:

- report each seed explicitly;
- report mean and 95% interval over seed-level means;
- report within-seed sampling variability separately;
- report paired per-seed Transformer–Mamba deltas.

Use a hierarchical paired bootstrap over training seed, generation repetition, and test item. Use at least 10,000 draws for final intervals.

For parity/non-inferiority, use a preregistered equivalence test. If the confidence interval crosses the practical margin, report the comparison as inconclusive—not as parity.

## Canonical versus private holdout

The canonical test set has already been inspected historically. Disclose this. Preserve canonical evaluation for comparison with prior work, but add a private untouched holdout or robustness subset if feasible. The private subset strengthens internal validity but does not replace canonical benchmark numbers.

## Tokenizer reevaluation

For the final tokenizer table report:

- reconstruction FID with uncertainty;
- MPJPE with root-alignment convention;
- root trajectory error;
- joint velocity and acceleration/jerk error;
- foot-contact and foot-skating metrics;
- stage-wise entropy/perplexity and usage;
- bits/frame and bits/second;
- parameters and decode latency;
- downstream generator A/B for FSQ versus RVQ if compute permits.

The last item is the strongest possible upgrade to Contribution A.

## Streaming reevaluation

Separate three claims:

1. **Architectural state complexity:** exact state bytes versus horizon.
2. **Generator runtime:** trained checkpoint token/chunk latency.
3. **End-to-end system latency:** prompt → first displayed frame and sustained playback.

For each benchmark record hardware, driver, precision, kernel, checkpoint, model size, batch/concurrency, commit, and command.

Measure:

- prompt-to-first-token;
- token and chunk latency;
- tokenizer decode latency;
- lookahead fill latency;
- first-frame latency;
- chunk-to-viewer availability;
- sustained real-time factor;
- p50/p90/p95/p99 latency and jitter;
- exact recurrent-state bytes;
- peak allocated/reserved GPU memory;
- concurrency 1, 2, 4, and maximum feasible.

Use at least 30 independent timing trials per condition. Distinguish actual trained-horizon benchmarks from synthetic 8192-token asymptotic stress tests.

Replace “bit-exact decoder parity” with “numerically equivalent within tolerance” unless every bit is actually equal.

## Qualitative study

Freeze a prompt suite before viewing outputs. Include:

- locomotion and direction;
- speed modifiers;
- transitions such as sit/stand;
- cyclic and asymmetric actions;
- compositional prompts;
- rare verbs;
- long continuation;
- ambiguous and out-of-distribution prompts;
- known failure cases.

Generate both twins with matched seeds, randomize/blind display order, preserve all videos and raw arrays, and show failures as well as successes. If possible, collect blinded ratings for text alignment, naturalness, smoothness, and preference.

---

# 6. What would produce a materially better thesis?

## Highest-value upgrades

### 1. A corrected three-seed twin result

This is the single largest improvement. It converts a one-off pilot into evidence about an architecture class.

### 2. A verified official evaluator

Without this, better FID numbers remain difficult to compare to the field. Evaluator equivalence and GT-oracle reproduction are more important than another training trick.

### 3. A real downstream FSQ-versus-RVQ generator comparison

Current Contribution A proves reconstruction superiority. Training the same generator recipe on matched FSQ and RVQ tokenizers would show whether better tokenization improves actual text-to-motion generation.

### 4. HumanML-only and AMASS-pretrained factorial results

This cleanly separates the architecture effect from the auxiliary-data effect and turns pretraining into a rigorous supporting result.

### 5. End-to-end streaming evidence

The fixed recurrent state is already strong. Measuring time-to-first-frame, decoder delay, viewer delivery, and sustained real-time factor would elevate the work from an architecture benchmark to a complete streaming system thesis.

### 6. A stronger chapter structure

Reorganize the final dissertation:

1. **Introduction:** problem, hypothesis, contributions, scope.
2. **Foundations and related work:** motion, 263 representation, tokenization, causal generation, SSMs.
3. **Data and evaluation:** exact samples, splits, evaluator validation, metrics, statistical protocol.
4. **Grouped-FSQ tokenizer:** mechanism, controlled baseline, reconstruction and downstream results.
5. **Controlled generator study:** architecture, preregistered protocol, training, twin results, ablations.
6. **Streaming system:** recurrent state, latency, decoder lookahead, viewer, end-to-end results.
7. **Discussion and threats:** data scale, test reuse, evaluator limitations, whole-body scope.
8. **Conclusion:** claim-to-evidence ledger and future work.
9. **Appendices:** optimizer rationale, gradient stability, parameter census, detailed configs, extended plots.

This gives the streaming system enough importance to stand as a thesis contribution rather than a subsection inside model training.

---

# 7. Go/no-go sequence

## No-go conditions before compute

Do not launch if any of these remain true:

- pretraining still decays Mamba state parameters;
- generator and co-adapted CLIP cannot be bundled unambiguously;
- HumanML/AMASS releases lack split and provenance hashes;
- real-batch overfit has not passed for both twins;
- official evaluator equivalence and GT checks are absent;
- parity margin and final protocol are not frozen;
- test access remains available for checkpoint selection;
- optimizer-step exposure differs between twins.

## Promotion sequence

1. Static data/optimizer/provenance audit passes.
2. Full tests pass.
3. Real-batch Transformer and Mamba overfit passes.
4. One-seed 34M canary pair passes.
5. Corrected AMASS $2\times2$ ablation determines the main initialization policy.
6. Complete six-run 34M matrix.
7. Freeze shared sampler on validation.
8. Optionally complete one 100M pair.
9. Run locked final canonical evaluation.
10. Run trained and end-to-end streaming benchmarks.
11. Freeze tables/figures and rewrite claims to exactly match the evidence.

---

# 8. Minimum, recommended, and ideal plans

## Minimum defensible thesis

- repair evaluator and checkpoint provenance;
- corrected 34M twins, three paired seeds;
- HumanML3D-only main comparison;
- one shared validation-selected sampler;
- complete repeated final evaluation;
- trained streaming state/latency benchmark;
- honest disclosure of prior test reuse.

## Recommended thesis

Everything above, plus:

- corrected AMASS-pretraining $2\times2$ factorial ablation;
- downstream FSQ-versus-RVQ generator A/B;
- one matched 100M scale pair;
- end-to-end decoder/viewer latency;
- frozen qualitative prompt suite.

## Ideal thesis

Everything above, plus:

- three 100M seeds per backbone;
- blinded human evaluation;
- causal-tokenizer variant removing 0.8 s lookahead;
- broader leakage-safe motion pretraining corpus;
- final whole-body SMPL-X qualitative system demonstration.

---

# Final recommendation

Prioritize the work in this order:

1. **validity fixes;**
2. **three-seed 34M controlled twins;**
3. **official reevaluation;**
4. **FSQ downstream A/B;**
5. **AMASS factorial ablation;**
6. **one 100M replication;**
7. **end-to-end streaming and qualitative presentation.**

A thesis with FID 1–3 but airtight controls, replicated architecture effects, and a complete bounded-streaming demonstration is stronger than a thesis with one FID below 1 obtained from a confounded, single-seed run.
