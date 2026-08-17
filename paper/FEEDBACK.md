# Dissertation draft v0.2 -- notes for advisor feedback

Prepared 2026-07-04. Companion to `paper/dissertation_draft.md`. This lists **what is now measured**,
**what is still pending**, and the **specific decisions I want feedback on**. Every number below traces
to a run directory under `outputs/runs/` (seed 2026, manifests record config + git + versions).

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
- Twin test: `outputs/runs/20260704T044653Z_evaluate_test` (transformer),
  `outputs/runs/20260704T050118Z_evaluate_test` (mamba). Command:
  `python -m text2motion.app.cli evaluate --backbone <b> --ckpt <ckpt> --split test --cfg_scale 6.0 --temperature 1.0 --length_mode fixed --mm_clips 100 --mm_repeats 30`.
- Tokenizer matrix: `outputs/runs/*tokenizer_<stem>/metrics.jsonl` (e.g. `*tokenizer_fsq_g8_v1024` = 0.0170).
- 100M transformer val: `outputs/runs/evaluate/20260624T062231Z_evaluate_val`.
- Streaming bench: `scripts/figures/plot_streaming_bench.py` from `eval.streaming_bench` -> `paper/figures/streaming.png`.
