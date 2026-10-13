# Lesson 19 -- The results we will report: three tables and the streaming figure

> The dissertation's results scaffold: the exact tables and figure the committee reads, with columns
> defined (Lesson 14), Contribution A filled from the seeded matrix (Lesson 7), and Contribution B's
> 34M twin filled from the completed, epoch-budget-robust test-split runs (sec. 19, Table 1) -- the
> 100M generator rows remain **PENDING**, not yet measured. Honesty rule (the plan): every number
> carries split, #clips, #reps, CFG scale, length mode, and a run-dir id; the claim is **the SSM leads
> at a converged budget + bounded streaming cost**, not beating MoMask.
>
> **Status (2026-09-27):** Table 1's 34M rows are filled at a converged 60-epoch budget -- the Mamba
> leads on every metric measured, including FID, reversing the earlier 18-epoch reading (sec. 7.4's
> tokenizer A/B showed the same 18-vs-60-epoch reversal pattern first). A further 40-epoch extension
> to 100 total confirmed this is not itself undertraining: the Mamba's checkpoint and result are
> unchanged (still epoch 30), and the transformer's val-selected checkpoint at 100 epochs generalizes
> *worse* to test than at 60, widening the Mamba's lead rather than closing it. See the row notes below.

## Table 1 -- Main twin table (test, 20-rep, mean +/- std)

Columns from Lesson 14 + the streaming state size (Lesson 13). Lower FID/MM-Dist better; higher
R-precision/Diversity better; Diversity/MultiModality calibrate to GT.

| Row | FID | R@1/2/3 | MM-Dist | Diversity | MModality | params | stream state |
|---|---|---|---|---|---|---|---|
| Real (GT) | -- | 0.51/--/-- | -- | 9.5 | -- | -- | -- |
| Tokenizer ceiling (FSQ 8x1024 recon) | 0.017* | -- | -- | -- | -- | -- | -- |
| Transformer-34M (ours, test, 60 ep) | 0.132 | 0.395/0.570/0.669 | 3.796 | 9.362 | 1.758 | 34M | O(L) growing KV |
| **Mamba-34M** (ours, test, 60 ep) | **0.121** | **0.418/0.594/0.693** | **3.677** | **9.437** | 1.822 | 34M | **O(1) 2.68 MB** |
| Transformer-34M (ours, test, 100 ep) | 0.163 (worse) | 0.391/0.566/0.667 | 3.823 | 9.373 | 1.718 | 34M | O(L) growing KV |
| Mamba-34M (ours, test, 100 ep) | 0.121 (unchanged) | same as 60 ep -- checkpoint never improved past epoch 30 | | | | 34M | O(1) 2.68 MB |
| Transformer-100M (scale ablation, val only) | 1.63 (val) | 0.323/0.478/0.580 (val) | 4.43 | 8.79 | -- | 98M | O(L) growing KV |
| Mamba-100M (twin) | not yet measured | | | | | ~100M | O(1) |
| *cited:* T2M-GPT | 0.116 | 0.417/--/-- | -- | -- | -- | ~230M | KV |
| *cited:* Mogo | 0.079 | -- | -- | -- | -- | -- | KV |
| *cited:* MoMask (not streamable) | 0.045 | 0.521/--/-- | -- | -- | -- | -- | bidir. |

Both 34M twins were fine-tuned to a converged 60 epochs for the headline row, then both extended 40
more epochs (100 total) as a robustness check: the Mamba plateaued at epoch 30 the entire time (its
60-epoch and 100-epoch rows are the identical checkpoint), while the transformer found a new best-by-val
checkpoint at epoch 100 that scores *worse* on the full test split than its epoch-48 checkpoint did --
a val/test disagreement, not a bug (dissertation sec. 5.6a). Training loss decreased smoothly and
monotonically for both models across all 100 epochs in both cases, so this is read as the
data-constrained regime (Lesson 17 / dissertation sec. 6.1) rather than an optimization failure.

\*reconstruction FID (the generator's ceiling), not a generation number. GT row reproduces published
R@1 (0.51 vs 0.511) -> harness check (Lesson 14.7). The Transformer-100M row is val/730-clips, a
different protocol and split from the 34M test rows -- a *scale* ablation, not the headline.

## Table 2 -- Contribution A (fillable NOW, seeded matrix)

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
**Done since drafting:** train/val/test generalization gap (no overfitting, 13/15 cells <= 0, sec. 7.2)
and downstream gen-FID (FSQ-tokens vs RVQ-tokens generator, converged 60ep -- FSQ ahead on all three
axes measured, sec. 7.4). **Pending:** multi-seed variance bars for both.

## Table 3 -- Ablations

| Ablation | Status | Finding |
|---|---|---|
| CFG scale curve (val, pilot) | have | FID 6.14 -> 3.55 at s=5 (-42%), R@1 x2.1 (Lesson 8.9) |
| temperature / top-p | have | locked (1.1, 0.9) |
| iso-vocab (FSQ 512 vs RVQ 512) | have | FSQ still wins -> the win is the quantizer, not vocab size |
| pkeep / weight-decay on/off | partial | the 06-09 run is the "without" row (Lesson B) |
| length: GT-length vs END-self-terminated | have (both modes) | report both (Lesson 8.8) |
| tokenizer generalization gap | have | gap = test - train <= 0 for 13/15 cells -> no overfitting (`generalization.md`) |
| AMASS-pretrained tokenizer | future | Lesson 18 protocol |

## Figure -- the streaming signature (MEASURED, 100M twins, to horizon 8192)

Recurrent-state size and per-step latency vs horizon, both backbones
(`scripts/figures/plot_streaming_bench.py` from `eval.streaming_bench`, random-weight 100M twins).
**Measured:** Mamba state **2.68 MB flat** at every horizon; transformer KV-cache grows **76.7 MB @
1024 -> 303 MB @ 4096 -> 605 MB @ 8192** (O(L), ~225x at 8192). Latency: transformer per-step cost is
flat (~12 ms) while linear projections dominate, then rises as O(L) attention takes over -- 16.5 ms @
4096, 23.6 ms @ 6144 -- crossing the SSM's flat ~26 ms near L ~ 6500, beyond which the SSM is faster
even without its CUDA kernel (unavailable on the Windows test machine; eager-mode carries a roughly
constant 2x per-step penalty). Memory favors Mamba everywhere and carries no such caveat; the latency
magnitude past L~6000 is amplified by the 4 GB test GPU nearing its memory limit, so it is reported as
a corroborating trend rather than an exact figure. Figure **done**, no checkpoint needed.

## Interpretation discipline (write into every caption)

- The claim is **the SSM leads on quality at matched params/data/seed once converged, plus
  bounded-memory streaming**, not beating MoMask; **state the compute gap** when citing published rows.
- Every number: split, #clips, #reps, CFG scale, length mode, run-dir id. **Val selected, test once** --
  even when, as at 100 epochs, the val-selected checkpoint tests worse than an earlier one; that
  disagreement is reported, not resolved by picking the better-testing checkpoint instead.
- Negative/neutral results reported (the residual-FSQ collapse is already framed as a finding; the
  100-epoch extension finding no improvement for either twin is reported as evidence of the
  data-constrained regime, not a failure to hide).

> **Bottom line:** Table 2 (Contribution A) is **done and seeded**; Table 1 is filled at the 34M pilot
> scale (both backbones, full test split, converged at 60 epochs and confirmed robust to a further
> 40-epoch extension) and the streaming figure is done. Remaining: the 100M Mamba row (not yet measured,
> and not currently planned -- sec. 6.1's data-constrained diagnosis argues against it) and multi-seed
> variance bars throughout.
