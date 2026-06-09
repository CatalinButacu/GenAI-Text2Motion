---
name: t2m-eval
description: >
  The text-to-motion evaluation recipe — FID, R-precision (top-1/2/3), Diversity, MultiModality,
  MM-Dist with the fixed Guo et al. matcher, plus the streaming latency/memory benchmark. Use
  when building the eval harness, scoring a checkpoint, or adjudicating the architecture debate.
---

# Text-to-motion evaluation

The prior project never computed FID and flew blind. The harness exists **before** the first real
training run, and every checkpoint gets a number.

## The fixed matcher (reuse, never retrain)
Use Guo et al.'s `text_mot_match` (port from `donor data\t2m\text_mot_match`): a co-trained motion
encoder + text encoder. All metrics are computed in **its** embedding space — retraining it makes
numbers incomparable to MoMask/T2M-GPT. Load it frozen.

## Metrics (HumanML3D test split)
- **FID** — Fréchet distance between matcher-embeddings of generated vs GT motions. Primary number.
- **R-precision (top-1/2/3)** — retrieval accuracy of matching generated motion to its text among
  distractors. The headline "does it follow the prompt" metric.
- **MM-Dist** — mean text↔motion embedding distance.
- **Diversity** — average pairwise distance across many generations.
- **MultiModality** — average pairwise distance across generations *for the same prompt*.

## Protocol (match the literature so numbers are citable)
- Generate with the **inference sampler** you'll ship (non-greedy: temperature/top-p, CFG scale),
  not greedy — greedy collapses and reports misleadingly.
- Repeat the eval over several seeds; report mean ± 95% CI (the field reports ±).
- Evaluate the **EMA** weights, on the **test** split only. Guard against train/test leakage.
- For the controlled twin (ADR 0001): identical matcher, split, sampler, and seeds for both
  transformer and SSM. Tabulate side by side.

## Streaming benchmark (the thesis efficiency claim)
- Peak GPU memory vs sequence length T (e.g. T = 100…4000); expect SSM ≈ flat, transformer ≈ growing.
- Latency per generated chunk; time-to-first-frame. Plot both. These figures back "real-time / bounded memory".

## Reporting
Always show published baselines (MoMask 0.045, T2M-GPT 0.116) as a reference row, with the honest
caveat about their compute/data. Phase 2 (SMPL-X whole-body) has no standard FID — report streaming
+ reconstruction + qualitative and say so.
