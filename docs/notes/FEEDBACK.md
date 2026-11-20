# Dissertation draft v0.2 -- notes for advisor feedback

Prepared 2026-07-04. Companion to `paper/dissertation_draft.md`. This lists **what is now measured**,
**what is still pending**, and the **specific decisions I want feedback on**. Every number below traces
to a run directory under `artifacts/outputs/runs/` (seed 2026, manifests record config + git + versions).

## Update (2026-09-27) -- Contribution B's main twin table: the same undertraining pattern, now closed

Section 1's original twin table (FID 3.30 transformer / 4.06 Mamba, "parity with the transformer ahead
on FID") was measured at **18 fine-tune epochs**, the same short-schedule risk that turned out to have
flipped the tokenizer downstream-generation result below. Both twins were retrained matched to a
converged **60 epochs**, then extended a further 40 (**100 total**) specifically to check whether 60
was itself converged.

**60-epoch result (the new headline):** the Mamba **leads on every metric measured**, including FID:

| 34M twin (test, 60ep) | FID | R@1 | R@2 | R@3 | MM-Dist | Div | MModality |
|---|---|---|---|---|---|---|---|
| Transformer | 0.132 | 0.395 | 0.570 | 0.669 | 3.796 | 9.362 | **1.758** |
| Mamba | **0.121** | **0.418** | **0.594** | **0.693** | **3.677** | **9.437** | 1.822 |

This is a full reversal of the 18-epoch reading, and both converged FIDs now sit close to T2M-GPT's
cited 0.116 despite T2M-GPT's ~230M parameters -- a strong sign the 18-epoch table was measuring
undertraining, not a property of either architecture.

**100-epoch robustness check (does the reversal hold, or is 60 itself undertrained?):** extending both
twins another 40 epochs changed nothing for the Mamba (best-by-val checkpoint never left epoch 30 --
val FID oscillated 0.24-0.34 the rest of the way, never beating 0.183) and made the transformer's
*citable* number worse: its running best-by-val genuinely improved twice more (epoch 72: 0.242, epoch
100: 0.230), but that epoch-100 checkpoint scores FID **0.163** on the full test split -- worse than
the epoch-48 checkpoint's 0.132, despite the better validation score. This is not cherry-picked in
either direction: best-by-val tracked continuously across the full 100 epochs for both twins, and the
protocol (val-selected, test touched once) says the epoch-100 checkpoint is the correct number to
report at that budget, even though an earlier checkpoint in the same trajectory tests better. Net
effect: the Mamba's lead **widens**, from ~8% relative at 60 epochs to ~26% relative at 100. Training
loss fell smoothly and monotonically for both models across all 100 epochs in both runs, so this reads
as the data-constrained regime already argued in sec. 6.1 (a 34M model can already memorize the
~6.53M-token fine-tune budget), not an optimization bug -- confirmed directly, not just predicted.

**Decision: no mamba-100M run.** Scaling capacity further on the same fixed data budget is expected to
push deeper into the same over-parameterised regime this update just observed directly, not away from
it (sec. 6.1: 100M twins are already ~18x past compute-optimal vs 34M's ~6x). With Azure abandoned, a
local 100M run is a multi-week commitment the thesis's own diagnosis argues against. If Contribution B
is revisited further, more/more-diverse fine-tune data is the indicated lever, not more parameters or
more epochs. Full provenance and the mechanistic aside on why Mamba trains ~6x slower per epoch (not a
computation flaw -- a genuine parallel scan, but no fused kernel plus gradient checkpointing to fit
memory) are in `.claude/docs/STATUS.md`'s 2026-09-27 entry; the dissertation's sec. 5.6/5.6a/5.9 and
lessons 08/11/12/17/19/00-roadmap have been updated to carry this as the final result.

## Update (2026-09-21) -- the 2026-09-20 downstream-generation result was an undertraining artifact

The 2026-09-20 entry below reported RVQ beating FSQ on downstream generation at 18 epochs and flagged
it explicitly as needing care ("single seed... worth understanding before trusting the magnitude").
That caution was warranted. Both twins were retrained from scratch, matched recipe, to a converged 60
epochs (CLAUDE.md's own golden rule: 100-200+ epochs, far short at 18) -- see
`.claude/docs/STATUS.md`'s 2026-09-20/21 entries and `scripts/training/run_matched_twin_60ep_local.ps1`.
**The direction reversed:**

| | 18 epochs | 60 epochs (converged) |
|---|---|---|
| FSQ test FID | 3.30 | **0.132** |
| RVQ test FID | 0.832 | 0.294 |
| Winner | RVQ (~4x) | **FSQ (~2.2x)** |
| FSQ R@1 | 0.246 | 0.395 |
| RVQ R@1 | 0.351 | 0.392 (statistically tied) |

At a converged epoch budget, FSQ wins downstream generation FID and the two are statistically
indistinguishable on R@1. This resolves the 2026-09-20 puzzle rather than deepening it: FSQ now leads
on **all three** axes measured -- reconstruction (0.0170 vs 0.0221), codebook health (sec. 7.2a), and
downstream generation -- a coherent story instead of an unexplained contradiction. The lesson: an
18-epoch pilot comparison between two conditions can flip sign entirely once both are actually
converged; treat any single-seed, short-schedule downstream comparison as provisional until re-checked
at a realistic epoch budget. Still single seed at 60 epochs too -- that caveat doesn't go away, it just
moves.

## Update (2026-09-20) -- downstream generation A/B measured, contracts audited

- **The downstream generation A/B (FSQ-tokens vs RVQ-tokens generator) is now measured**, locally, no
  cloud needed. Transformer+RVQ, full 2189-clip test split, same protocol as the FSQ twin table: FID
  **0.832** vs FSQ's 3.30, R@1 0.351 vs 0.246, R@2 0.534 vs 0.386, R@3 0.646 vs 0.485 -- RVQ leads on
  every metric. This is the opposite direction from tokenizer reconstruction (FSQ 0.0170 beats RVQ
  0.0221), so it does not touch Contribution A's closed reconstruction claim, but it is a genuinely
  surprising new signal that needs care before it goes in the draft: single seed both sides, and the
  val-to-test generalization direction flipped between the two conditions (FSQ's test FID was worse
  than its val gauge; RVQ's test FID came in better than its own val FID at the same checkpoint) --
  worth understanding before trusting the magnitude.
- **Four real contract-boundary defects found and fixed** while producing that result -- none were
  model or math bugs, all were a value with two possible sources of truth and no check that they
  agreed (an RVQ vocabulary size silently computed from the wrong tokenizer's formula, the mandatory
  overfit gate comparing a value it should have ignored, a DataLoader worker setting present on one
  training contract and missing from its sibling, and a quantizer-count duplicated across two
  dataclasses with no cross-check). Two now have regression tests; the other two are structural fixes
  (branch on tokenizer kind; fail loud at config-load time) verified by direct re-run rather than a
  persisted test. Full write-up, each with file/line and the general pattern behind all four:
  `paper/lessons/C-contracts-and-configuration-correctness.md`.

## Update (2026-09-15) -- status of section 4's pending items

Nothing below is a rewrite of the 2026-07-04 notes -- they stay as originally written. This is what
has changed since, checked against the actual run directories, not assumed:

- **100M Mamba twin**: still cloud-gated, unchanged in substance. What's new: the Azure platform
  (resource group, storage, three least-privilege managed identities, key vault) is fully provisioned
  and verified, and a GPU quota increase has been requested -- self-service is refused for this SKU
  (`QuotaNotAvailableForResource` in every region tried), so it needs human review. All six runs
  (pretrain+gate+finetune x transformer/FSQ, mamba/FSQ, transformer/RVQ) are launch-ready via
  `infra/azure/run_twins.sh`, guarded by max-hours budgets and auto-shutdown. Quota is the only
  remaining gate on this row.
- **Downstream generation A/B (FSQ-tokens vs RVQ-tokens generator)**: setup is now complete --
  `configs/generator/gen_pilot_rvq8x1024.yaml` and a matching AMASS RVQ token pack exist, mirroring the
  FSQ config exactly (seed/width/downsample/resblocks/8 quantizers/1024 vocab) so the comparison
  isolates the tokenizer alone. The run itself (`transformer_rvq` in the six-run campaign) is blocked
  on the same Azure quota as the row above -- **still not measured**, do not cite a result yet.
- **Multi-seed variance bars, camera-ready streaming figure, prose expansion / LaTeX port**: unchanged,
  still open. (The fused-kernel path the streaming figure needs now has a preflight gate and a
  Kaggle-based build canary to de-risk it before the paid cloud run, but the figure itself isn't produced.)
- **One number in this document was re-verified, not corrected**: the strong-RVQ figure in secs. 1 and
  4.6 (0.0221, `rvq_l8_1024`) was checked directly against
  `artifacts/outputs/runs/tokenizer/20260617T102700Z_tokenizer_rvq_l8_1024/metrics.jsonl` on the full
  2189-clip test split. It is correct as written -- a separate summary in `.claude/docs/STATUS.md`
  states 0.0205 for the same cell, which does not match the run artifact and should not be trusted
  over it.

## 1. What changed since v0.1
- **Contribution A (tokenizer) is complete.** The full 3x5 FSQ-vs-RVQ matrix is measured (sec. 4.4):
  Grouped-FSQ beats the strong RVQ at all six matched bit-rate cells, best cell recon-FID 0.0170
  (< MoMask's released RVQ 0.019), with 0 codebook parameters. Single seed per cell.
- **Contribution B (twin) is complete at the 34M pilot scale.** Both mixers were trained under the
  matched recipe and scored once on the full test split (2189 clips, 20-rep). Table in sec. 5.6:

  | 34M twin (test) | FID | R@1 | R@2 | R@3 | MM-Dist | Div | MModality |
  |---|---|---|---|---|---|---|---|
  | Transformer | **3.30** | 0.246 | 0.386 | 0.485 | 5.26 | **8.55** | **3.36** |
  | Mamba | 4.06 | **0.255** | **0.398** | **0.498** | **5.00** | 8.03 | 2.88 |

  Read: **parity** -- Mamba leads all R-precision ranks (~1.5 sigma) and MM-Dist; transformer leads FID
  (~23% rel.) and diversity. Neither dominates. This supports the thesis claim (matched-budget parity +
  bounded streaming), not a SOTA claim.
- **Streaming result** unchanged and strong: Mamba recurrent state constant 2.68 MB vs transformer
  KV-cache 605 MB at L=8192 (225x), architectural and exact (sec. 5.5).

## 2. The three questions I most want feedback on
1. **Is the parity framing the right headline?** The SSM does not win FID, it wins R-precision/MM-Dist
   and ties overall. I frame Contribution B as "matches the transformer at matched budget while
   streaming in O(1) memory." Is that the correct, defensible claim for a master's thesis, or should I
   foreground the streaming/memory result (which is unambiguous and exact) and treat quality as
   supporting?
2. **Is the 34M pilot table sufficient, or is the 100M twin mandatory for the defense?** The complete,
   both-sides-measured comparison is at 34M. The 100M is single-sided (transformer, val only) because
   the Mamba-100M needs cloud budget (~$20, g5.xlarge). Does the pilot twin carry the contribution, with
   100M as "scales as expected," or is the 100M Mamba a hard requirement?
3. **The one fairness asymmetry (sec. 5.9).** The fine-tune is fully matched (effective batch 8; the
   Mamba uses bs4 x grad-accum 2, a gradient-identical step). But the *AMASS prior* was trained at
   different batch sizes (transformer 64, Mamba 8 -- the 4 GB memory limit), so the Mamba prior saw more
   optimizer steps and started fine-tuning from a lower CE. I disclose this and note it cuts toward the
   SSM on the metrics where it leads. Is disclosure enough, or should I re-run a batch-matched prior
   (cloud) before the numbers are quotable? A fully symmetric from-scratch no-prior pair also exists as
   a cross-check.

## 3. Honest caveats already written into the draft
- Absolute FID (3.30/4.06) is far above cited baselines because these are **34M pilots** vs 100M-class
  systems; stated in sec. 5.6 and sec. 5.7. The gap is capacity, localized (recon-through-eval 0.059
  keeps the tokenizer clean; the shortfall is the generator).
- The 200-clip in-training **val gauge was optimistic** (~1-1.6) vs the 2189-clip test (3.30/4.06);
  same streaming-at-GT-length protocol, so the difference is sample size + split. Test is the citable
  number; val is selection-only.
- Our strong-RVQ (0.0221) is weaker than MoMask's tuned RVQ (0.019, far more compute); the FSQ-beats-
  0.019 cell is on our controlled budget, not a claim to beat the best-ever tuned RVQ.
- Single seed per config (tokenizer + generator) -- no variance bars yet beyond the 20-rep R-precision std.
- Streaming latency magnitude past L~6000 is inflated by the 4 GB card near its memory limit; the
  **memory** result carries no such caveat, so the efficiency argument leads with memory (sec. 5.5).

## 4. Still pending (not blocking feedback)
- 100M Mamba twin (cloud) -> completes the scale row.
- Downstream generation A/B (FSQ-tokens vs RVQ-tokens generator) -> makes Contribution A airtight
  beyond reconstruction.
- Multi-seed variance bars.
- Camera-ready streaming figure on a memory-unconstrained GPU with the mamba-ssm CUDA kernel.
- Prose expansion (sec. 2-3, sec. 4.1-4.3, sec. 5.1-5.3) from `paper/lessons/` to reach 30-45 pp; LaTeX/Overleaf port.

## 5. Where to read the evidence
- Twin test: `artifacts/outputs/runs/20260704T044653Z_evaluate_test` (transformer),
  `artifacts/outputs/runs/20260704T050118Z_evaluate_test` (mamba). Command:
  `python -m text2motion.app.cli evaluate --backbone <b> --ckpt <ckpt> --split test --cfg_scale 6.0 --temperature 1.0 --length_mode fixed --mm_clips 100 --mm_repeats 30`.
- Tokenizer matrix: `artifacts/outputs/runs/*tokenizer_<stem>/metrics.jsonl` (e.g. `*tokenizer_fsq_g8_v1024` = 0.0170).
- 100M transformer val: `artifacts/outputs/runs/evaluate/20260624T062231Z_evaluate_val`.
- Streaming bench: `scripts/figures/plot_streaming_bench.py` from `eval.streaming_bench` -> `paper/figures/streaming.png`.
