# Lessons roadmap -- the whole thesis, as a grounded teaching path

These lessons explain the thesis from its basic concepts to its defense before an examiner.
The system produces motion from text in real time and displays an SMPL-X body.
The thesis has two contributions:

- (A) a Grouped-FSQ motion tokenizer compared with a strong-RVQ baseline.
- (B) the first token-autoregressive S6/Mamba motion generator compared with a transformer twin with a matched parameter count.

Each lesson names its source file and uses LaTeX mathematics.
Most lessons include a Mermaid diagram called "The picture".
Read the lessons in the order below.
The lesson numbers reflect their original order.
The groups define the current order.

## Foundations -- representation and data
| Lesson | Covers | Diagram |
|---|---|---|
| 01 -- pose, joints, rotations | positions and rotations; the skeleton; FK and IK | yes (tree, FK/IK) |
| 02 -- pose to motion | the frame rate; velocity; the choice of 20 fps | yes (sampling, velocity) |
| 03 -- the 263 vector | the root, RIC, rot6d, velocity, and foot channels; each slice and its checks | yes (layout + consumers) |
| 04 -- from motion to tokens | the cost of 263 floats per frame; the conversion to discrete tokens | yes (BPE) |
| 04a -- the data pipeline | normalization; windows; the mirror-symmetry map; the splits; data checks | yes (shapes) |

## Part I -- the tokenizer (Contribution A)
| Lesson | Covers | Diagram |
|---|---|---|
| 05 -- RVQ tokenizer | VQ with a learned codebook; STE, commitment, EMA, and dead-code reset; the residual stack | yes (residual cascade) |
| 06 -- FSQ tokenizer | quantization on a fixed grid; Grouped-FSQ; sec. 6.6a discrete capacity: bits, rate-distortion, and data flow with shapes | yes (group split, RVQ-vs-FSQ, capacity) |
| 07 -- tokenizer results | the matched matrix; recon-FID; code usage; reproducibility; sec. 7.0a the formal comparison: controls and rate-distortion dominance | yes (experiment structure) |

## Part II -- the generator (Contribution B)
| Lesson | Covers | Diagram |
|---|---|---|
| 08 -- the autoregressive objective | $p_\theta(z\mid c)$; the text prefix; R parallel heads; CE; token selection; CFG | yes (objective) |
| 11 -- the transformer twin | causal attention; the positional embedding; the KV-cache and its growth before a restart | yes (cache growth) |
| 12 -- the S6/Mamba generator | the continuous SSM; the ZOH recurrence; selectivity; the state with a fixed size | yes (mixer + recurrence) |
| 13 -- streaming and complexity | the state size in bytes; $O(1)$/$O(L)$ vs $O(L)$/$O(L^2)$ without a restart; both twins have bounded state with a service restart; the positional limit | yes (fixed state vs cache growth without a restart; both bounded with a restart) |
| 14 -- evaluation metrics | FID; R-precision; MM-Dist; Diversity; MultiModality; the GT-oracle | yes (shared space) |
| 09 -- generator training trace | each value and shape: frozen tokens; CLIP; teacher forcing; the losses | yes (shapes) |
| 10 -- generator inference trace | each value and shape: stream_step; bounded state; CFG; END | yes (shapes) |

## Training methodology
| Lesson | Covers | Diagram |
|---|---|---|
| A -- loss and optimizer | CE split by term; geometry from soft-decode; L1; AdamW; warmup/cosine; EMA; gradient clipping | |
| B -- escaping the plateau | the checklist for a loss plateau; CFG dropout; pkeep; text encoder updates; reproducibility; the gate learning rate | yes (training step) |
| C -- contracts and configuration correctness | the ADR 0004 contracts between stages; four corrected defects: RVQ vocabulary size, gate dropout comparison, absent DataLoader workers, duplicated `num_quantizers`; fixes and applicable regression tests | yes (dependency rule) |

## System, framing, future work
| Lesson | Covers | Diagram |
|---|---|---|
| 15 -- streaming decode and demo | the live pipeline with bounded state; the producer; analytical SMPL-X pose fit from joints; rotation-SLERP display in aitviewer | yes (pipeline shapes) |
| 20 -- long streams and the restart (`20-long-streams-and-the-restart.md`) | how the prompt stays active; restart policy B: every 8 tokens with 16 tokens of context; the state range; per-step text; progress counters | |
| 16 -- related work and positioning | the position of contributions A and B; gaps in prior work; citations and differences | yes (landscape) |
| 17 -- limitations and conclusion | established and pending claims; the scope; the table of threats to validity | yes (status map) |
| 18 -- scaling the tokenizer with data | AMASS pretraining; the two rules: a versioned rebuild and a citable evaluation; the protocol | yes (versioned pipeline) |
| 19 -- results tables and figures | the 3 tables; the plan for the stream figure; seeded entries in Table 2 for Contribution A | |

## What remains (finishing, not new theory)

> Status (2026-11-04): the seed-2026 Mamba e60 evaluation used the epoch-30 EMA; the transformer e60 evaluation used 60 epochs.
> These evaluations do not form a comparison with a matched epoch budget.
> The EMA stayed frozen after resume, so the e100 results do not establish convergence or deterioration from further training.
>
> The reference initialization and embedding-only dropout changes require fresh final runs for both twins.
> The earlier conclusions remain provisional.
>
> The retained service policy B restarts every 8 tokens with 16 tokens of context.
> The supplied November 3 measurements report Mamba R@3 of 0.47–0.69 with policy B and a flat state.
> They report transformer R@3 of 0.63–0.66.
> Without a restart, Mamba R@3 falls from 0.72 to 0.09 at 60 seconds.
> Its state RMS grows by a factor of 3–5, with a peak factor of 13.
> Per-step text alone gives R@3 of 0.13 and does not resolve this failure.
>
> A gate learning rate of 0.0003 passes all four arms: Mamba prefix, Mamba per-step, transformer prefix, and transformer per-step.
> A rate of 0.003 fails the transformer per-step arm.
> The default change remains pending.
>
> The live analytical SMPL-X pose fit and rotation-SLERP display are implemented.
> The random-weight timing measures the model alone, not the service.
> The earlier comparison of graphed Mamba with an eager transformer does not provide a fair timing result.

The earlier completion record lists these items:

- The seeded full matrix and the Lesson 7 rewrite.
- All diagrams and checks for numeric consistency.
- The generalization audit (`outputs/generalization.md`: no overfitting, gap <= 0 for 13/15 cells).

The 100M twin run remains unmeasured.
Its results will supply the generator rows in Table 1 of Lesson 19.
A random-weight benchmark exists at the 100M scale and does not require a trained checkpoint.
The final stream figure requires a fair comparison and a clear distinction between model timing and service timing.
