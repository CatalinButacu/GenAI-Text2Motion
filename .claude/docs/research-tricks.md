# Field tricks for better text-to-motion results — status map

What the strong token-AR papers (T2M-GPT, MoMask, Mogo, AttT2M, MMM, MotionStreamer) do beyond the
basics, mapped to this repo. Status: ✅ in place · 🔜 planned/next · 💡 optional, do if a run stalls.
Numbers must follow the CLAUDE.md logging rule (val for selection, `_twin_eval` for test).

## Sampling / inference (free — no retraining)
- ✅ **Classifier-free guidance** at decode (`stream(cfg_scale=)`), trained via cfg_dropout 0.1.
  MoMask/MMM-standard; sweep scale on val (see `outputs/cfg_sweep.log`).
- ✅ Nucleus sampling (top_p 0.9, temperature) — greedy collapses (T2M-GPT failure mode).
- 💡 **Temperature/top_p joint sweep** per checkpoint (cheap; do after the CFG scale is fixed).
- 💡 Repetition penalty on recent tokens if generated motion freezes/loops at long horizons.

## Training the generator
- ✅ `pkeep` input corruption 0.8 (T2M-GPT uses 0.5) — fights memorization + exposure bias.
- ✅ AdamW weight_decay 0.01, dropout 0.1, EMA 0.999, warmup+cosine sized to the real peak window
  (ep ~20–40 at 31M/22k clips → 60-epoch runs).
- ✅ Multi-term loss: token-CE + soft-decode recon + velocity + foot/root (`t2m-losses`).
- 🔜 **Scale 31.6M → 50–150M** (the plateau lesson; T4 fits ~100M at bs 32–64 fp16). Highest-leverage
  single change once the recipe is stable.
- 💡 Label smoothing 0.1 on the token-CE; stochastic depth if depth grows past ~12 blocks.

## Text conditioning (R-precision lever)
- ✅ CLIP pooled vector as prefix, last layer + projection unfrozen (field standard, T2M-GPT mold).
- 🔜 **Token-level text conditioning**: feed CLIP's 77-token sequence as a multi-token prefix (or
  cross-attention) instead of one pooled vector — AttT2M shows fine-grained text attention is worth
  several R-precision points. Cheap for the transformer twin; for Mamba, prefix-tokens keep the
  bounded-state story intact (prefix is consumed once).
- 💡 Caption paraphrase augmentation with an LLM (2024+ papers); keep the official captions for eval.

## Length / streaming
- 🔜 **End-token (or length predictor)** so generation stops itself — removes the GT-length crutch
  (currently an honest caveat in our eval) and is REQUIRED for the live demo.
- 🔜 Long-horizon stitching: continue generation across prompts (re-condition on new text, keep or
  reset state) — MotionStreamer is the rival to cite; our causal AR makes this natural but
  mid-stream prompt switches are out-of-distribution until trained (concat-clip training, later).

## Data
- ✅ Full official HumanML3D + mirrors, truncate-not-drop, caption-segment pairing fixed (2026-06-10).
- 🔜 **AMASS tokenizer pretraining** (no text needed; regen pipeline exists) → report as an ablation.
  Keeps generator data protocol-comparable while exploiting the 290 GB donor corpus.
- 💡 Speed augmentation (resample 0.9–1.1×) for the tokenizer only; never for eval data.

## Tokenizer (Contribution A — already strong: recon-FID 0.0266)
- ✅ Grouped-FSQ beats strong-RVQ at matched capacity; no dead codes (perplexity 572/1000).
- 🔜 Iso-vocab ablation (per-group product ≈ 512) — closes the codebook-size confound.
- 💡 Wider receptive field / deeper decoder if MPJPE (119mm vs MoMask 29.5mm) matters for the demo.

## Evaluation hygiene (comparability)
- ✅ Guo matcher unmodified; GT repro matches published; 20-rep full-split protocol (`_twin_eval`).
- ✅ Selection on val, test touched once; run manifests (`shared/run_log.py`).
- 🔜 Report **MultiModality** (k samples per caption) once the final checkpoints exist — the one
  standard metric still missing from the table.
- 🔜 The streaming benchmark: latency + peak memory vs horizon, Mamba state vs KV-cache — the
  thesis's signature plot, pure inference, runs locally.
