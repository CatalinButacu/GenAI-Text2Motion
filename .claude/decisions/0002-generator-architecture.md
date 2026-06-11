# ADR 0002 — Generator architecture

**Status:** Accepted (direction, 2026-06-03) — implementation-gated by a small-config FID sanity vs
the transformer twin before scaling. Decided on a literature survey with cross-checked FID numbers
(see `.claude/docs/references.md`), not preference.

## Context
Hard gate: the runtime generator must decode **causally + incrementally** (left-to-right, bounded
memory) to stream. We also build a controlled twin baseline. Survey findings:
- **Streamable, proven FID:** T2M-GPT (0.116, arXiv:2301.06052), AttT2M (0.112, arXiv:2309.00796),
  **Mogo** (0.079, causal RVQ, explicitly streaming, arXiv:2412.07797 / 2506.05952).
- **Not streamable** (bidirectional/masked → reference ceiling only): MoMask (0.045, arXiv:2312.00063),
  MMM (0.080), BAMM (0.055, refinement pass is bidirectional).
- **The gap:** no published **token-autoregressive Mamba/S6** motion generator exists — all Mamba
  motion work is diffusion- or masked-bidirectional (Motion Mamba 2403.07487, T2M Mamba 2602.01352,
  KMM 2411.06481). Token-AR Mamba is proven in image/video but not motion.

## Decision
**Generator = a causal, token-autoregressive S6/Mamba over RVQ/FSQ motion tokens** (Mogo-mold:
causal, single forward step per token, long-sequence). **Twin = a causal-AR Transformer** in the
T2M-GPT mold (KV-cache). Both are causal/streamable; they differ in the runtime memory profile.

- **Novelty:** first token-AR S6 motion generator (the unoccupied literature cell).
- **Efficiency claim (honest):** Mamba's **fixed-size recurrent state** vs the Transformer twin's
  **growing KV-cache** → bounded memory + flat per-step latency at long horizons (the `t2m-eval`
  streaming benchmark), at matched FID/R-precision.
- **Reference ceiling:** MoMask 0.045 / Mogo 0.079, reported with the caveat that they used more
  compute; we are not obligated to beat them — the claim is the streaming axis.

## Validation gate (before scaling)
Small-config run of both backbones on the same tokens → `sanity-overfit`, then a short train, then a
real FID from `t2m-eval`. Proceed to full scale only if the S6 generator is within range of the twin.
A superseding ADR is required to change backbone, not a silent edit.

## Gate evidence (2026-06-11) — **PASSED**
31M pilot twins (identical tokenizer/data/seed/losses; g4dn cloud run), best-by-eval checkpoints
(transformer ep20 of 150, mamba ep30 of ~42), scored with the 20-rep full-test protocol
(2,189 clips, CFG 5.0 / T 1.1 / top-p 0.9, `text2motion.eval.evaluate`, run-manifest logged):

| backbone | FID | R@1 | R@2 | R@3 | MM-Dist | Diversity | MModality |
|---|---|---|---|---|---|---|---|
| transformer 31.6M | **3.310** | **0.217±.008** | 0.354 | 0.455 | 5.260 | 7.975 | 3.747 |
| mamba 31.8M | 3.928 | 0.168±.006 | 0.287 | 0.381 | 5.644 | 7.888 | **4.059** |

**FID ratio 1.19× — well inside the ≤1.5× gate → proceed to the 100M run (E5).** Notes recorded:
(1) budget asymmetry favors the transformer (150 vs ~42 epochs; both peak early, so best-ckpt
comparison is meaningful but E5 removes the asymmetry); (2) mamba overfit a decade later than the
twin (ep30 vs ep20 peak — H4); (3) mamba leads MultiModality; transformer leads retrieval precision
at this scale; (4) both rows pull the in-train trend numbers into citable form for the first time.

## Consequences
- Unlocks `src/text2motion/model/generator.py` (S6 backbone + the transformer twin behind one
  interface) after the gate. Streaming `stream_step` contract per `streaming-decode`.
- Depends on Contribution A tokens (ADR 0001 / `motion-tokenizer`). Losses per `t2m-losses`.
