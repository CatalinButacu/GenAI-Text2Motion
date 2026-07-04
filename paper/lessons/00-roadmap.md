# Lessons roadmap -- the whole thesis, as a grounded teaching path

A complete beginner-to-committee path through the system: real-time, incremental text -> SMPL-X motion,
with two contributions -- (A) a Grouped-FSQ motion tokenizer vs a strong-RVQ baseline, and (B) the first
token-autoregressive S6/Mamba motion generator vs a parameter-matched transformer twin. Every lesson is
grounded in the actual code (file named in its header), uses LaTeX math, and most carry a Mermaid
"The picture" diagram. Read in the order below; the numbering is historical, the **grouping** is the
reading order.

> Status (2026-06-17): compute is RUNNING (tokenizer matrix finishing locally; the 100M twin run is the
> next cloud step). The lesson spine is complete.

## Foundations -- representation and data
| Lesson | Covers | Diagram |
|---|---|---|
| **01 -- pose, joints, rotations** | positions vs rotations, skeleton, FK vs IK | yes (tree, FK/IK) |
| **02 -- pose to motion** | fps, velocity, why 20 fps | yes (sampling, velocity) |
| **03 -- the 263 vector** | root / ric / rot6d / vel / foot, slice by slice; how we know it's correct | yes (layout + consumers) |
| **04 -- from motion to tokens** | why 263 floats/frame is too much; the discretization bridge | yes (BPE analogy) |
| **04a -- the data pipeline** | normalization, windows, the mirror-symmetry map, splits, hygiene | yes (shapes) |

## Part I -- the tokenizer (Contribution A)
| Lesson | Covers | Diagram |
|---|---|---|
| **05 -- RVQ tokenizer** | learned-codebook VQ -> STE/commitment/EMA/dead-code reset -> residual stack | yes (residual cascade) |
| **06 -- FSQ tokenizer** | fixed-grid quantization, no machinery, Grouped-FSQ; **sec. 6.6a discrete capacity** (bits, rate-distortion, shape-annotated dataflow) | yes (group split, RVQ-vs-FSQ, capacity) |
| **07 -- tokenizer results** | the matched matrix, recon-FID, health, reproducibility; **sec. 7.0a the comparison formally** (controlled experiment + rate-distortion dominance) | yes (experiment structure) |

## Part II -- the generator (Contribution B)
| Lesson | Covers | Diagram |
|---|---|---|
| **08 -- the autoregressive objective** | $p_\theta(z\mid c)$, text prefix, R parallel heads, CE, sampling, CFG | yes (objective) |
| **11 -- the transformer twin** | causal attention, positional embedding, the growing KV-cache | yes (cache growth) |
| **12 -- the S6/Mamba generator** | continuous SSM -> ZOH recurrence, selectivity, fixed-size state | yes (mixer + recurrence) |
| **13 -- streaming and complexity** | bounded state in bytes, $O(1)$/$O(L)$ vs $O(L)$/$O(L^2)$, the signature plot | yes (bounded vs growing) |
| **14 -- evaluation metrics** | FID, R-precision, MM-Dist, Diversity, MultiModality; the GT-oracle | yes (shared space) |
| **09 -- generator training trace** | value-by-value: frozen tokens + CLIP -> teacher forcing -> losses | yes (shapes) |
| **10 -- generator inference trace** | value-by-value: stream_step, bounded state, CFG, END | yes (shapes) |

## Training methodology
| Lesson | Covers | Diagram |
|---|---|---|
| **A -- loss and optimizer** | term-split CE + soft-decode geometric, L1, AdamW + warmup/cosine + EMA + clip | |
| **B -- escaping the plateau** | the plateau-avoidance checklist, CFG dropout, pkeep, text-unfreeze, reproducibility | yes (training step) |

## System, framing, future work
| Lesson | Covers | Diagram |
|---|---|---|
| **15 -- streaming decode and demo** | the bounded end-to-end live pipeline -> aitviewer (producer done, render Phase 5) | yes (pipeline shapes) |
| **16 -- related work and positioning** | where A and B sit; the unoccupied cells; cite-and-distinguish | yes (landscape) |
| **17 -- limitations and conclusion** | established vs pending, scope, threats-to-validity table | yes (status map) |
| **18 -- scaling the tokenizer with data** | AMASS pretraining: the two rules (versioned redo; citable eval) + protocol | yes (versioned pipeline) |
| **19 -- results tables and figures** | the 3 tables + streaming-figure scaffold; Table 2 (Contribution A) seeded-filled | |

## What remains (finishing, not new theory)
- **DONE:** seeded full matrix + Lesson 7 rewrite; all diagrams; numeric consistency; **generalization
  audit** (`outputs/generalization.md`: no overfitting, gap <= 0 for 13/15 cells).
- The **100M twin run** -> Table 1's generator rows + the streaming figure (Lesson 19). [cloud-gated]
