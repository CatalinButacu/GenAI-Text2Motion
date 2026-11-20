# R7 decision table and conditional P2 proposal

Date: 2026-11-08. Measurements completed; policy and training approvals remain open.

The strongest objection is that these screens do not prove generalization or naturalness.
They use frozen, validation-selected checkpoints and one training seed. Shorter history
changes inference behavior, not learned capacity. The precision screen uses synthetic
motion and precomputed text, not complete training epochs.

## Verified switch-history results

Both width320 twins use frozen FSQ8x1024, seed2026, TF32 and the same sampler. Each cohort
contains32 caption pairs from64 distinct validation recordings absent from TRAIN. The
two cohorts share no recordings. They are not subject-disjoint or unused selection data.

Only the reset at token48 changes retained history. Other resets remain every8 tokens
with capacity16. Discarded tokens cannot return; new-action history subsequently regrows.
The shared generator interface supports this bounded callback; production remains16.

R@1 is the fraction of captions retrieving their matching motion first in a fixed32-pair
pool. Early action2 covers96 frames, or4.8 seconds at20 FPS.

| Twin | Cohort | Keep16 early R@1 | Keep8 | Keep0 | Keep0 minus16, paired95% interval |
|---|---:|---:|---:|---:|---|
| Mamba | 0 | 0.25000 | 0.25000 | 0.50000 | +0.25000 [0.00000;0.50000] |
| Mamba | 64 | 0.40625 | 0.53125 | 0.59375 | +0.18750 [-0.06250;0.40625] |
| Transformer | 0 | 0.28125 | 0.43750 | 0.53125 | +0.25000 [0.03125;0.46875] |
| Transformer | 64 | 0.40625 | 0.34375 | 0.56250 | +0.15625 [-0.06250;0.37500] |

Zero history improves the early own-caption distance margin in all four cells:
Mamba +4.75036 [2.64607;7.07174] and +4.73035 [2.72818;6.92111];
Transformer +3.29510 [1.46879;5.36224] and +2.69026 [1.28438;4.04859].
The margin is old-caption distance minus new-caption distance; larger favors the new
caption. These unadjusted exploratory intervals do not establish universal improvement.
Keep8 does not replicate its retrieval benefit across twins and cohorts.

Physical results prevent automatic adoption. AUJ, the discrete area of deviation from
the cohort's ground-truth jerk level, changes as follows:

| Twin | Cohort | Keep16 AUJ | Keep0 AUJ | Difference95% interval |
|---|---:|---:|---:|---|
| Mamba | 0 | 0.77908 | 0.68848 | [-0.24573;0.07423] |
| Mamba | 64 | 0.42369 | 0.40262 | [-0.17687;0.17436] |
| Transformer | 0 | 0.41891 | 0.56737 | [-0.06151;0.25813] |
| Transformer | 64 | 0.48216 | 0.47332 | [-0.17754;0.17893] |

For example, Transformer cohort0 AUJ rises35.4% at the point estimate. Its interval
includes both directions; that is uncertainty, not proof of acceptable motion.
Conditional skate comparisons preserve jointly contacting pairs: Mamba has31, Transformer32.
No acceptable physical-regression threshold was approved. Lower peak jerk alone is not
naturalness evidence; excessive smoothing can also differ from ground-truth motion.

The adapted jerk definition follows [FlowMDM](https://arxiv.org/abs/2402.15509), but these
validation-cohort values are not comparable to its published test table.
[DartControl](https://arxiv.org/abs/2410.05260) is a published causal text-driven motion
control reference, not evidence that our conditioning change will work.

## Three options for R7

| Option | Measured evidence and source | Additional proposed GPU budget | Protocol effect | Decision |
|---|---|---|---|---|
| Keep current restart8/retain16 | Early R@1 values above; [history CSV](../paper/pics/switch_history_validation.csv) and [earlier caption screen](../paper/pics/caption_switch_validation.csv) | 0 h | Existing bounded policy and sampler unchanged; report delayed adaptation | Current default, not a claim that16 is optimal |
| Review switch-only retain0 | Early R@1 improves0.15625-0.25000 across four cells; AUJ trade-offs above. Follow-up reversed Mamba switches also increase contacting-foot skate | Accounted review842.3s within15min; controls1021.1s within20min; no retraining | Caption-switch history changes, ordinary resets do not. Same noncausal tokenizer, with4-token lookahead | Measured response candidate; adoption remains blocked by physical trade-offs and pending human ratings |
| Conditional per-step-text P2 | [Shared implementation](../../src/text2motion/generation/nn_generator.py), [training curves](../paper/pics/training_curves.csv). P1 clean validation CE4.5806 Transformer/4.4755 Mamba; no measured P2 gain | Up to4 h pilot plus up to1 h overfit gates, proposed only | Retrain both twins with the same conditioning intervention, tokenizer, loss, precision and selection rule; needs matched controls | Defer unless frozen-policy quality/control remains inadequate |

The current recommendation is to rate the completed blinded retain0/16 review before
paying for new training. The controls do not justify universal adoption of retain0.
Confidence: medium that it improves early caption response under this protocol;
low that its physical quality is acceptable for deployment.
The previous council recommended frozen diagnostics before retraining. It was not
rerun here; no fresh council verdict or new model calls are claimed.

## Saved-motion audit and exploratory control follow-up

The [paired diagnostic CSV](../paper/pics/switch_review_pairs.csv) contains128 rows:
both twins, two32-pair cohorts, keep0 versus16. Independent verification reproduced
root steps, per-pair jerk peaks, saved physical/semantic fields, quantiles, paired
intervals, copied-array hashes and46 unique blinded cases. Of these,32 cases are
randomly selected from16 shared pairs;14 are additional purposive outliers.
The random and outlier groups must not be pooled into a human failure-rate estimate.
The policy-to-panel mapping is stored separately and has not been disclosed.

The initial full-sequence renders were stopped after22 GIFs because the skeletons
were too small to judge. The owner approved focused transition clips within the
remaining allowance. The corrected renderer uses a fixed common orthographic camera,
thicker dark bones, explicit camera-matrix updates and a checked horizontal-arm SMPL-X
T-pose. The T-pose, known clip and sample montages were visually inspected.

All46 focused GIFs and46 montages completed and were independently checked for hashes,
dimensions and duration:113 source frames,176 through288, or5.65 seconds at20 FPS.
Raw arrays remain unchanged; only static panel offsets are applied for display.
The retry took333.3s; including the first logged509s gives842.3s accounted wall time,
below the15-minute allowance. This is not measured GPU utilization. Partial initial
media remain preserved. The review pack is in
`artifacts/outputs/qa/switch_review_focus_media_20261108`; human ratings remain pending.
These are recovered-joint visuals, not fitted surfaces or a naturalness pass.

## Counterfactual caption controls

The strongest new objection is that Mamba gains equally at the point estimate when
the caption does not change, and it shows a contacting-foot skate regression.
Zero history is not justified as a universal switch-only improvement.

The control screen reuses the same probe for A-to-A, A-to-B and B-to-A, keep16/0, on both twins.
Original A/B sampling identities preserve common random uniforms, which are archived
with token indices. A-to-B/16 must reproduce frozen R6 raw motion. A-to-A intentionally
uses32 distinct recordings; the other scenarios use64.

All12 cells completed in1021.1s,17.0 minutes, within the20-minute allowance.
Independent verification reproduced actual uniforms, prefix/cut token identities,
canonical recovery, physical diagnostics, semantic arrays and paired intervals.
A-to-B raw features reproduce the prior cohort0 for BOTH policies and twins:
these replay rows are not an independent replication.
See [control CSV](../paper/pics/switch_controls_validation.csv) and
`artifacts/outputs/qa/switch_controls_verification_20261108.json`.

Early R@1, retrieval against each scenario's actual second caption:

| Twin | Scenario | Keep16 | Keep0 | Difference, paired95% interval |
|---|---|---:|---:|---|
| Mamba | A-to-A | .34375 | .59375 | +.25000 [.00000;.50000] |
| Mamba | A-to-B | .25000 | .50000 | +.25000 [.00000;.50000] |
| Mamba | B-to-A | .31250 | .56250 | +.25000 [.06250;.43750] |
| Transformer | A-to-A | .56250 | .59375 | +.03125 [-.15625;.21875] |
| Transformer | A-to-B | .28125 | .53125 | +.25000 [.03125;.46875] |
| Transformer | B-to-A | .28125 | .56250 | +.28125 [.09375;.46875] |

The switch-specific interaction is
\[
  (\mathrm{keep0}-\mathrm{keep16})_{A\rightarrow B}
  -(\mathrm{keep0}-\mathrm{keep16})_{A\rightarrow A}.
\]
Both terms are scored against the **same original A/B caption embeddings**. Comparing
own-caption margins would be misleading: the A-to-A margin is always zero. Reversed
B-to-A switches provide a direction check, not a matched interaction without B-to-B.

With fixed A/B targets, early margin interactions are Mamba+6.0872
[3.8719;8.3671] and Transformer+3.8078 [2.0736;5.8514]. Corresponding R@1
interactions are+.21875 [-.03125;.46875] and+.25000 [.03125;.46875].
This supports directional response under this fixed protocol, not exclusive benefit
at caption switches. A descriptive contrast of each scenario's own-target R@1 gains
is Mamba0 [-.40625;.37500] and Transformer+.21875 [-.06250;.50078].
Those targets use different caption pools; neither interval establishes a universal
switch-specific gain or equivalence.

Physical results still block automatic adoption. Mamba contacting-foot skate increases
by+.04667m/s [.02190;.07079] for A-to-A and+.05606m/s [.02428;.08288] for B-to-A.
Both use31 jointly contacting pairs at the common Y=0 floor. These are exploratory
proxy regressions, not human judgments. Mamba B-to-A AUJ decreases, so a single jerk
metric would miss the foot trade-off. Transformer A-to-B retains its prior AUJ concern.

Assumptions: medium confidence that keep0 redirects these frozen validation motions
toward changed captions; low confidence that its physical quality is acceptable.
One training seed, already-selected validation data and unadjusted intervals remain.
Production retain16, width320 and TF32 are unchanged; P2 is still a proposal.
The next falsification test is a blinded rating of the balanced random review cases,
with a physical guard chosen before unsealing labels. Outliers remain separate.
Do not fund training or optional GPU screens from an assumed remaining A1 allowance;
the full earlier cost has not been reconstructed.

## K8 precision screen

One process compared strictFP32, TF32 and BF16 on both twins. Batch8,196 frames,
one warmup and five timed updates per cell. Generator initial weights and clean-CE
targets match across modes. The Mamba training core uses the compiled scan.

| Twin | Mode | Median ms/update | Peak allocated GB | Clean-CE gradient relative L2 error vs FP32 |
|---|---|---:|---:|---:|
| Transformer | FP32 | 80.77 | 0.559 | 0 |
| Transformer | TF32 | 69.63 | 0.564 | 0.000669 |
| Transformer | BF16 | 114.17 | 0.557 | 0.007257 |
| Mamba | FP32 | 123.72 | 0.676 | 0 |
| Mamba | TF32 | 129.96 | 0.681 | 0.000655 |
| Mamba | BF16 | 143.38 | 0.622 | 0.008109 |

BF16 is64.0% slower for Transformer and10.3% slower for Mamba than TF32 here.
Mamba allocated memory falls8.7%. No BF16 speed gain was demonstrated.
Five repeats in a fixed order are not a cross-session performance study.

Timing includes live tokenizer encoding, configured reconstruction losses,
corruption/dropout and generator optimizer/EMA updates. It excludes CLIP
encoding/backprop, data loading and validation. Numerical gradients cover
generator-only clean token cross-entropy, not the full multi-term training loss.
The numerical results therefore do not certify BF16 convergence.
See [precision CSV](../paper/pics/precision_costs.csv). Keep existing TF32 unchanged.

## Conditional P2 protocol, not an approved run

If retain0 cannot meet the owner's physical guard, the simplest existing training
candidate is `generator.per_step_text: true`. It reuses the projected caption vector
at each token and adds no parameters. CPU construction verified identical counts with
the flag off/on: Transformer15,300,040 and Mamba15,553,480. Both twins must change together.
Confidence in a generalization gain is low until a matched experiment measures it.

Two tests:

1. First compare the frozen retain0/16 archives for motion failures and the recorded
   semantic/physical trade-off. An acceptable no-training policy is cheaper than P2.
2. If training is still justified, compare prefix-only and per-step text for each twin:
   four fresh cells, seed2026, width320, frozen tokenizer, unchanged loss and TF32.
   Pass the existing single-batch overfit gate before each cell. Proposed cap:
   one hour per cell plus at most15 minutes per gate, five GPU hours total.
   Match optimizer-update milestones and record elapsed cost, clean validation CE,
   CE gap, FID, and replicated switch metrics. A capped falling curve is partial,
   not failed convergence or proof of generalization.

Prediction: per-step conditioning may improve early caption response while retaining
history16; it may not improve FID. Failure to reproduce its semantic benefit, or
unacceptable physical regression, would contradict this proposal.

A convergence-scale comparison would need a separate budget. P1 cost about17 GPU hours
for one pair; two fully matched pairs have a historical cost scale of about34 hours,
not a forecast or approved cap. Apply six consecutive non-improving clean-validation-CE
epochs to every new cell. Changing patience or precision only for the candidate would
confound attribution against historical P1.

More AMASS data is a separate hypothesis. Motion-only windows are not paired caption
examples; do not count them as supervised text augmentation without a verified annotation
and split protocol. More epochs alone are not evidence of better generalization.

Before committing to training, obtain owner approval for the intervention and budget,
and use the approved non-Claude council if a consequential choice remains:
three fresh `gpt-6-luna` members plus one `gemini-4-argon`, eight calls including review.
No new training, production-policy adoption, precision switch, commit or push occurred.

## Evidence audit

The history audit independently reproduced504 semantic metric arrays, physical scores,
paired intervals, canonical skeleton recovery, source hashes, baseline16/R6 parity and
the12-row CSV. The precision audit recomputed numerical errors from saved arrays,
matched gradient coverage and weights, verified all30 saved timings and the six-row CSV.
Raw evidence is under `artifacts/outputs/qa/`, which is git-ignored; open explicit paths.

Reports: [history](../../artifacts/outputs/qa/switch_history_20261108_retry1.json),
[history audit](../../artifacts/outputs/qa/switch_history_verification_20261108.json),
[precision](../../artifacts/outputs/qa/k8_precision_20261108.json),
[precision audit](../../artifacts/outputs/qa/k8_precision_verification_20261108.json).
Focused validation:107 tests passed, one skipped; Ruff/type checks report0 problems.
