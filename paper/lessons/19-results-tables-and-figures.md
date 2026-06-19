# Lesson 19 — The results we will report: three tables and the streaming figure

> The dissertation's results scaffold: the exact tables and figure the committee reads, with columns
> defined (Lesson 14), Contribution A filled from the seeded matrix (Lesson 7), and the generator rows
> marked **PENDING** until the 100M twin run. Honesty rule (the plan): every number carries split,
> #clips, #reps, CFG scale, length mode, and a run-dir id; the claim is **parity + bounded streaming
> cost**, not beating MoMask.

## Table 1 — Main twin table (test, 20-rep, mean +/- std)

Columns from Lesson 14 + the streaming state size (Lesson 13). Lower FID/MM-Dist better; higher
R-precision/Diversity better; Diversity/MultiModality calibrate to GT.

| Row | FID | R@1/2/3 | MM-Dist | Diversity | MModality | params | stream state |
|---|---|---|---|---|---|---|---|
| Real (GT) | -- | **0.514**/.. | 2.977 | 9.67 | -- | -- | -- |
| Tokenizer ceiling (FSQ 8x1024 recon) | 0.0170* | -- | -- | -- | -- | -- | -- |
| Transformer-100M (twin) | PENDING | PENDING | PENDING | PENDING | PENDING | ~100M | O(L) growing KV |
| **Mamba-100M (ours)** | PENDING | PENDING | PENDING | PENDING | PENDING | ~100M | **O(1) ~0.6 MB** |
| Transformer-31.6M (pilot, scale-ablation) | ~3.55 (val) | ~0.22 (val) | -- | -- | -- | 31.6M | O(L) |
| *cited:* T2M-GPT | 0.116 | -- | -- | -- | -- | -- | KV |
| *cited:* Mogo | 0.079 | -- | -- | -- | -- | -- | KV |
| *cited:* MoMask (not streamable) | 0.045 | -- | -- | -- | -- | -- | bidir. |

\*reconstruction FID (the generator's ceiling), not a generation number. GT row reproduces published
(R@1 0.514 vs 0.511) -> harness check (Lesson 14.7). Pilot row is val/300-clips, different protocol —
a *scale* ablation, not the headline.

## Table 2 — Contribution A (fillable NOW, seeded matrix)

Recon-FID on HumanML3D-263 test, matched bits/step, shared enc/dec, seed 2026 (Lesson 7.2).

| Tokenizer | matched cell | recon-FID | codebook params |
|---|---|---|---|
| **Grouped-FSQ (best)** | 8 x 1024 | **0.0170** | **0** |
| Grouped-FSQ | 6 x 1000 | 0.0274 | 0 |
| Grouped-FSQ (iso-vocab) | 6 x 512 | 0.0305 | 0 |
| Strong-RVQ (best) | 8 x 1024 | 0.0221 | ~3M |
| Strong-RVQ | 6 x 512 | 0.0342 | ~1.6M |
| *cited:* MoMask RVQ | 6 x 512 | 0.019 (more compute) | learned |

**Result:** FSQ dominates RVQ at all six matched cells with 0 codebook params; best cell beats MoMask.
**Pending columns:** train/val/test generalization gap (`eval_generalization.py`) and downstream
gen-FID (FSQ-tokens vs RVQ-tokens generator A/B).

## Table 3 — Ablations

| Ablation | Status | Finding |
|---|---|---|
| CFG scale curve (val, pilot) | have | FID 6.14 -> 3.55 at s=5 (-42%), R@1 x2.1 (Lesson 8.9) |
| temperature / top-p | have | locked (1.1, 0.9) |
| iso-vocab (FSQ 512 vs RVQ 512) | have | FSQ still wins -> the win is the quantizer, not vocab size |
| pkeep / weight-decay on/off | partial | the 06-09 run is the "without" row (Lesson B) |
| length: GT-length vs END-self-terminated | have (both modes) | report both (Lesson 8.8) |
| tokenizer generalization gap | have | gap = test - train <= 0 for 13/15 cells -> no overfitting (`generalization.md`) |
| AMASS-pretrained tokenizer | future | Lesson 18 protocol |

## Figure — the streaming signature (MEASURED, 100M twins, to horizon 8192)

Recurrent-state size and per-step latency vs horizon, both backbones
(`outputs/streaming_bench*.json`, random-weight 100M twins). **Measured:** Mamba state **2.68 MB flat**
at every horizon (64 -> 8192); transformer KV-cache **5.9 -> 605 MB** (O(L), ~225x at 8192). Latency
**crosses over near ~4096 steps**: below it the shallower transformer is faster (~12 vs ~30 ms); at
**8192 the transformer is 264 ms/step vs Mamba's 53 ms (~5x)** as quadratic attention dominates. Memory
favors Mamba everywhere; latency too past ~4k. Figure **done**, no checkpoint needed.

## Interpretation discipline (write into every caption)

- The claim is **twin parity on quality at matched params/data/seed + bounded-memory streaming**, not
  beating MoMask; **state the compute gap** when citing published rows.
- Every number: split, #clips, #reps, CFG scale, length mode, run-dir id. **Val selected, test once.**
- Negative/neutral results reported (the residual-FSQ collapse is already framed as a finding; if
  Mamba needs more epochs to match at fixed budget, that is itself a thesis-relevant SSM-sample-
  efficiency result, not a failure to hide).

> **Bottom line:** Table 2 (Contribution A) is **done and seeded**; Table 1's generator rows and the
> streaming figure are the remaining deliverables — Table 1 gated on the 100M twin run, the figure
> runnable from shapes today. This scaffold is the dissertation's results chapter with the empties
> labelled, so filling it is mechanical once the runs land.
