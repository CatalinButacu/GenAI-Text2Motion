---
name: motion-tokenizer
description: >
  Build and evaluate the RVQ motion tokenizer — Contribution A (the plateau pivot). Covers the
  baseline RVQ, the improvement levers (codebook EMA + dead-code reset, usage balancing,
  factorized codes, FSQ/LFQ), and the reconstruction + downstream-FID protocol that proves the gain.
  Use before/while building or training the tokenizer.
---

# Motion tokenizer (RVQ) — Contribution A

The tokenizer discretizes motion into tokens the generator predicts and the streaming decoder emits.
The prior project plateaued partly here, so this is a **named contribution**: show a *better* RVQ.

## Architecture (263 track)
- Encoder: 1D-conv (temporal) over the (T, 263) feature → latent (T/d, C), downsample factor `d`
  (donor used 4; codebook 6×512, latent 128). Residual VQ over `K` codebooks. Conv decoder back to 263.
- Keep encoder/decoder symmetric; assert the round-trip shape against `motion-representation`.

## Baselines to beat (measure first — make them STRONG, not strawmen)
1. **RVQ + EMA + code-reset** (the T2M-GPT recipe), NOT naive VQ: naive 0.492 vs EMA+reset 0.070
   recon FID (T2M-GPT arXiv:2301.06052, Tbl 3). A naive-VQ baseline is a strawman — don't ship it.
2. **MoMask's released 263 RVQ** — the field's bar: single-VQ 0.091 → RVQ 0.019 recon FID
   (arXiv:2312.00063, Tbl 2; EMA + reset + quantization dropout).
Record reconstruction (per-joint position error via `recover_joints`, and feature-L2) **and the
downstream FID of reconstructions** (decode GT tokens → `t2m-eval` FID vs GT). Downstream FID is the
number that matters — it caps what any generator on these tokens can achieve.

## Chosen direction: Residual-FSQ (proven + novel combination)
FSQ (finite scalar quantization, arXiv:2309.15505) gives ~100% codebook usage, no collapse, and no
EMA/reset/commitment machinery — and it is **motion-proven**: ScaMo (arXiv:2412.14559) shows FSQ > VQ
on HumanML3D; ST-Multi-Scale (arXiv:2508.08991) reports recon FID 0.037 / gen FID 0.063 beating
MoMask. Residual structure is motion-proven (MoMask). Their **combination — residual FSQ +
quantization dropout on the 263 feature — is unestablished**, so it is both supported and novel.
- Size the FSQ implicit codebook to match MoMask's effective capacity (FSQ trails VQ only at <2^10).
- Optional cheap lever if recon stalls: low-dim + L2-normalized factorized codes (ViT-VQGAN,
  arXiv:2110.04627, image-proven). Skip LFQ (arXiv:2310.05737, image/video-only; vocab too large for HML3D).

## Protocol
1. `sanity-overfit`: one batch reconstructs near-perfectly before any full run.
2. Train on the full 263 corpus (`dataset-unified`), log reconstruction + codebook perplexity per epoch.
3. Compare baseline vs improved on **reconstruction + downstream FID** on the test split.
4. Freeze the winner; its decoder is reused (frozen) by the generator's soft-decode loss (`t2m-losses`)
   and by `streaming-decode`. Record the comparison table — this is Contribution A's evidence.
