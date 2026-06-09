---
name: sanity-overfit
description: >
  The mandatory pre-flight before any real training run: overfit a single batch to near-zero loss
  to prove the model+data+loss+optimizer path is wired correctly. Use before every full run, when
  a new architecture lands, and as step 3 of the architecture debate.
---

# Single-batch overfit (do this before any real run)

If a model can't memorize one batch, the bug is structural — more epochs/data won't help. This one
check would have caught the prior project's plateau on day one.

## Procedure
1. Grab **one** small batch (e.g. 4 sequences). Disable shuffling, augmentation, dropout, and EMA.
2. Train on only that batch for ~300–1000 steps with a slightly high LR.
3. **Expected:** total loss drops toward ~0; token-CE accuracy → ~100%; reconstruction L1 → tiny.
4. Decode the overfit batch through the tokenizer/renderer and eyeball it in aitviewer — it should
   reproduce the input motion almost exactly.

## If it does NOT overfit, diagnose in this order (do not add epochs)
- **Shape/contract mismatch** — assert against `motion-representation`; check masks aren't hiding the loss.
- **Targets/inputs misaligned** — off-by-one in teacher forcing / token shift; labels not matching logits.
- **Normalization mismatch** — model sees normalized input but loss compares denormalized (or vice versa).
- **Loss masked to nothing** — frame/latent mask all-zero or inverted.
- **Gradient flow broken** — frozen params that should train; `requires_grad` off; soft-decode path
  detached. Print grad norms per module.
- **LR/optimizer** — only suspect this last.

## Gate
A passing single-batch overfit is a precondition for starting a full run. Record it (loss curve +
the decoded clip) so regressions are caught. `training-engineer` must not launch without it.
