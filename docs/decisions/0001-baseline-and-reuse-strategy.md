# ADR 0001 — Baseline & from-scratch-vs-reuse strategy

**Status:** Accepted (2026-06-02)

## Context
Master's thesis on streaming text → SMPL-X motion. The prior attempt trained *everything* from
scratch (tokenizer + generator) on ~14k clips and plateaued (top-1 ~12%, FID never computed). A
thesis needs a credible baseline and a defensible, achievable contribution.

## Decision
Train from scratch **only what we claim novelty on**; reuse everything else so numbers stay
comparable to the field.

| Component | Decision |
|---|---|
| Eval matcher (`text_mot_match`) | **Reuse** Guo et al. (donor `data\t2m`). Never retrain — keeps FID comparable. |
| Baseline (MoMask / T2M-GPT) | **Reuse** published checkpoint + numbers as reference ceiling. Do not retrain a baseline. |
| Motion tokenizer (RVQ) | **Contribution A** — build a *better* RVQ on 263; benchmark against a vanilla RVQ and MoMask's released RVQ. Not inherited. |
| Generator (SSM/streaming head) | **Contribution B** — train from scratch. |

**Two contributions:** (A) a better RVQ tokenizer (reconstruction + downstream-FID gains — the
pivot away from the plateau); (B) a streaming SSM generator. The headline claim is on the
**streaming/efficiency axis** (FID parity vs a matched non-streaming transformer twin at bounded
memory/latency), with a citable absolute FID from the standard 263 + unmodified Guo evaluator.
The twin uses identical tokenizer + data + budget.

**Staging:** primary HumanML3D-263 now (citable FID/R-precision); SMPL-X 168 whole-body demo
deferred (streaming + qualitative; face deferred; standard FID not defined there).

## Consequences
- `eval-metrics-engineer` ports the fixed matcher and reports published baselines as a reference row.
- `motion-model-architect` builds the controlled twin under matched conditions.
- Avoids the prior plateau trap of training all components from scratch on limited compute.
