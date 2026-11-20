# ADR 0003 — Text encoder: reuse CLIP ViT-B/32 (minimally unfrozen)

**Status:** Accepted (2026-06-20; documents the choice implemented since project start, previously
recorded only in `.claude/docs/references.md`).

## Context
The generator conditions on a caption, so a text encoder is required. Its choice affects two things:
(1) **comparability** of our FID/R-precision to the field, and (2) text-following quality. Under
ADR 0001 (train only what we claim; reuse everything else), the text encoder is a **reuse** decision,
not a contribution. The text-to-motion baselines we compare against — **T2M-GPT, MoMask, Mogo** — all
use the *identical* CLIP ViT-B/32 text encoder (verified in their repos).

## Decision
Reuse the pretrained **CLIP ViT-B/32 text encoder** — HF `openai/clip-vit-base-patch32`
(`CLIPTextModelWithProjection`, projection dim 512; `openai/` is just the HF org hosting the original
Radford et al. 2021 weights, not an API). Use only the text tower. Field-standard recipe: the pooled,
projected 512-d sentence vector (extended to a 16-token prefix in the 100M config = pooled + first 15
token hidden states), linear-projected into the generator and consumed as a **prefix** (no
cross-attention). Implemented in `src/text2motion/generation/nn_text.py`.

**One deliberate deviation** from the field (which fully freezes CLIP): per the prior-plateau lesson
("don't fully freeze the text encoder"), unfreeze only the **last transformer layer + final
layer-norm + the text projection** (~3M params) at a low LR (`TrainCfg.text_encoder_lr`); freeze the
rest.

**Rejected alternatives:**
- **T5** (MotionGPT, AnyMo arXiv:2605.29488) — changes the text-feature space → breaks FID/R-precision
  comparability with the T2M-GPT/MoMask baselines.
- **A larger / newer text encoder** — same comparability break; the text encoder is not our
  contribution, so a fancier one buys nothing citable.
- **Fully frozen CLIP** (the field default) — the prior plateau was blamed partly on a frozen encoder
  capping text grounding; hence the minimal unfreeze above.

## Consequences
- Keeps FID/R-precision **comparable** to T2M-GPT / MoMask / Mogo (same text features) — the same
  reuse discipline as the unmodified Guo evaluator and the 263 representation (ADR 0001).
- Only ~3M params (last layer + LN + projection) are fine-tuned, so results stay close to
  frozen-CLIP baselines while gaining some motion adaptation.
- **Invariant:** `text_encoder.prefix_len` must equal `generator.text_prefix_len` (asserted at
  construction). The 100M config (`final100m_fsq8x1024.yaml`) uses 16; the dataclass default is 1.
- A one-time pretrained-weight download (`openai/clip-vit-base-patch32`) is a build dependency;
  weights are cached locally — no runtime/API dependency.

> Note: ADRs live under `.claude/decisions/`, which is now local-only (the `.claude/` tree was pulled
> from the published repo). If the architectural-decision record should ship with the dissertation,
> mirror `decisions/` into the committed `paper/` tree.
