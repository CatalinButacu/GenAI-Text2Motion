---
name: arch-decision
description: >
  Run a structured architecture debate for the motion generator and record the outcome as an
  ADR. Use before writing any model code, when comparing candidate architectures (autoregressive
  token / residual-masked / SSM-Mamba-S6 / physics-SSM / diffusion-baseline), or when a major
  modeling trade-off must be settled with evidence rather than preference.
---

# Architecture decision (ADR) recipe

Goal: pick the generator on evidence, not vibes, and leave a paper-trail your thesis can cite.

## 1. Enumerate candidates
List every realistic option with its lineage and a public reference:
autoregressive-token (T2M-GPT), residual-masked (MoMask), SSM/Mamba-S6, physics-constrained SSM,
diffusion (MDM/Motion Mamba — baseline only). Pull a dossier per candidate from `research-scout`.

## 2. Score against our HARD requirements (columns)
Build a matrix. Columns are the non-negotiables from CLAUDE.md:

| Candidate | Causal streaming / bounded mem | SMPL-X whole-body capable | Trainable from scratch on our compute | Novelty on streaming axis | FID feasibility |
|---|---|---|---|---|---|

- **Streaming is a hard gate.** A "no" here means the candidate can only be a *baseline*, never the
  runtime generator. Full-sequence diffusion fails this gate.
- Mark each cell ✓ / ✗ / ~ with a one-line justification + citation. No empty cells.

## 3. Shortlist and adjudicate with a real number
Take the top 2 that pass the gate. Run each as a **tiny fast config** (small d_model, few epochs,
subset data) through `sanity-overfit` then a short train, and get a real **FID** from
`eval-metrics-engineer`. Decide on the number + the streaming benchmark, not the matrix alone.

## 4. Write the ADR
Create `.claude/decisions/0002-generator-architecture.md`:
- **Status:** proposed → accepted (date).
- **Context:** the requirements + the prior plateau lessons.
- **Options considered:** the matrix.
- **Decision:** chosen generator + the controlled twin (ADR 0001) it will be compared against.
- **Consequences:** what code gets written, what's reused, what's deferred to future work.

## 5. Gate
Until ADR 0002 is `accepted`, no architecture-specific code is blessed. After it's accepted,
`motion-model-architect` implements; revisiting requires a superseding ADR, not a silent edit.
