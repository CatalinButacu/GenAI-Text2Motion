# 10 — Text encoder choice and CFG: both kept for discussion

Status: **active**. Last reviewed 2026-05-22.

## What's in the code right now

Two text encoders are wired in and selectable at runtime via the same CLI flag.
Both ride the same pipeline (FiLM modulation, RVQ codebook head). The choice is
a single command-line argument.

```bash
# SBERT baseline (default; matches the historical training runs)
python scripts/training/train_motion_ssm.py --use-sbert --text-encoder sbert-small

# CLIP (the candidate from the 2026-05 technique audit)
python scripts/training/train_motion_ssm.py --use-sbert --text-encoder clip-b
```

Underneath, both call `PretrainedTextEncoder` from
`src/modules/motion/nn_models.py`, which:
- loads any `sentence-transformers` model by name,
- probes the output dim at construction (no hardcoded 384),
- projects to `d_model` via a learnable `Linear + LayerNorm`,
- freezes the pretrained backbone by default (the projection stays trainable).

## CLI alias map

| Alias | sentence-transformers model | Dim | Params | Notes |
|---|---|---|---|---|
| `sbert-small` | `all-MiniLM-L6-v2` | 384 | 22M | Project default. Distilled BERT, general NLP. |
| `sbert-mpnet` | `all-mpnet-base-v2` | 768 | 110M | Larger SBERT. Modest motion-domain gain. |
| `clip-b` | `clip-ViT-B-32` | 512 | ~150M | OpenAI CLIP. **Trained on action-rich image captions.** |
| `clip-l` | `clip-ViT-L-14` | 768 | ~430M | Larger CLIP. Better but heavy. |

## Classifier-free guidance at inference (--cfg-scale)

CFG is wired through the inference path:

```bash
python main.py "a person walks forward" --cfg-scale 4.0
```

What it does at sample time:

```
guided_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
```

The uncond pass uses an empty prompt `""`. **CFG only works when:**
- the encoder produces a meaningful empty-prompt embedding (SBERT and CLIP do; the
  fallback `SimpleTextEncoder` token-id path does not, so CFG is silently a no-op there);
- the model was trained with `cfg_dropout_prob > 0` (default 0.1) so the empty
  conditional path was actually exercised.

`cfg_scale=1.0` is a hard pass-through — sampling is identical to the
non-CFG path. `cfg_scale=2..4` is the typical literature range; higher values
push harder toward text adherence at the cost of motion diversity.

## Why this matters for the dissertation

The 2026-05 technique audit (full report at
`scripts/maintenance/validate_clip_swap.py` results + research session)
identified three concrete plateau-attacks ranked by impact-per-hour:

1. **SBERT → CLIP** for text conditioning. Single biggest paper-aligned gap
   vs MoMask. Expected 0.3–0.8 nats on val_ce at scale.
2. **CFG at inference + autoregressive K-head sampling.** Both training-free
   quality bumps. CFG alone is ~0.05–0.1 FID per MoMask's ablation.
3. **`torch.compile` (Windows) or `mamba_ssm.selective_scan_fn` (Linux/CUDA)**
   for 3–5× training throughput so we can run more epochs in the same budget.

Items 1 and 2 are now both in the codebase. **They are kept side-by-side
deliberately** so the dissertation chapter can present:
- a controlled SBERT vs CLIP ablation on the same RVQ tokenizer,
- a CFG-scale sweep at inference on the chosen encoder.

The validation harness lives at
`scripts/maintenance/validate_clip_swap.py`. The first ~5-epoch comparison
showed CLIP-b ahead of SBERT by 0.05 nats — small (5-epoch noise floor is
large) but directionally consistent with the literature. A defensible
ablation needs full HumanML3D with 3+ seeds and ~50–100 epochs.

## What's NOT yet ported from the local-WIP stash

The pre-snake_case-migration stash on the main checkout
(`git stash list` → "pre-snake-sync 2026-05-22") contained two additional
features that are NOT yet on `main`:

- **Stats loading + auto-denormalization at inference.** The `loadRvqStats`
  helper read motion stats from the RVQ checkpoint's `config` field and
  applied inverse z-score on the way out. Useful so inference produces
  real-scale SMPL-X poses instead of normalised values. Worth porting once
  the eval harness needs real-scale output.
- **`generateBothActors` method** for K≥2 multi-actor models trained on
  `interx_paired`. Returns `(motion_p1, motion_p2)`. Only meaningful once
  multi-actor training runs land.

Both are still recoverable from `stash@{0}` on the main checkout if needed.

## Open question for discussion

Which encoder is the dissertation's headline run? Options:

- **SBERT-small** — historical default; all the prior training runs and
  ablations were under this encoder, so it's the cleanest comparison to
  prior results. Cheap, fast, baseline.
- **CLIP-b** — best paper alignment; validation hints at gains; small enough
  that a full training run is affordable.
- **CLIP-l** — biggest expected gains; heaviest. Only run this if cloud
  budget permits.

This page deliberately doesn't pick. The code supports all three.
