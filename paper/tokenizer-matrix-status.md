# Tokenizer matrix — status note (2026-06-17)

Reconstruction-FID on the HumanML3D-263 val split (2189 clips), seed 2026, 500 epochs,
batch 128, eval every 25. Lower is better. Each cell traces to `outputs/sweep_<name>.log`
and `checkpoints/<name>.pt`.

| codes/step | RVQ-512 | RVQ-1024 | FSQ-512 | FSQ-1024 | FSQ-1000 |
|---|---|---|---|---|---|
| 4 | 0.0626 | 0.0558 | 0.0612 | 0.0527 | 0.0514 |
| 6 | 0.0342 | 0.0310 | 0.0305 | 0.0283 | 0.0274 |
| 8 | 0.0277 | 0.0221 | 0.0196 | **0.0170** | 0.0199 |

**COMPLETE (2026-06-18).** All 15 cells seed-2026, 500-epoch, test recon-FID from run manifests. FSQ
dominates RVQ at every matched cell; best FSQ 8x1024 = 0.0170 beats MoMask's 0.019. Remaining: run
`scripts/eval_generalization.py` (GPU free) for the train/val/test generalization gap.

Matched-bits head-to-head reads **down each column**: RVQ-1024 vs FSQ-1024 are bit-for-bit
matched (40 / 60 / 80 bits/step for 4 / 6 / 8 codes), same for the 512 columns. FSQ-1000 is the
round-vocab FSQ-native variant ((8,5,5,5)=1000, ~0.3% fewer bits/group).

Trend so far: **FSQ beats RVQ at every completed matched cell**, with zero codebook parameters,
and improves monotonically with codes/step. Best so far FSQ 8x512 = 0.0196 (~= MoMask 0.019).

## Remaining work — 5 runs (configs all exist, validated)

| run | config | mech | ckpt | state |
|---|---|---|---|---|
| RVQ 8x1024 | configs/tokenizer/rvq_l8_1024.yaml | rvq | rvq_l8_1024.pt | fresh |
| FSQ 8x1024 | configs/tokenizer/fsq_g8_v1024.yaml | fsq | fsq_g8_v1024.pt | fresh |
| FSQ 8x1000 | configs/tokenizer/tok_g8_v1000.yaml | fsq | fsq_g8_v1000.pt | fresh |
| FSQ 4x1024 | configs/tokenizer/fsq_g4_v1024.yaml | fsq | fsq_g4_v1024.pt | resume ep286->500 |
| FSQ 4x1000 | configs/tokenizer/tok_g4_v1000.yaml | fsq | fsq_g4_v1000.pt | resume ep72->500 |

Run them with `scripts/matrix_remaining.ps1` (single-instance, sequential, resume-safe).

## Incident (2026-06-17) — why fsq_g4_v1024 crashed

Two driver instances of the matrix sweep ran at once (a leftover pre-session background task at
08:01 + a duplicate launched this session). Both did `--resume` on the same checkpoints; on the
4GB GPU two trainers OOM'd and the CUDA context died, killing fsq_g4_v1024 at ep286. Its
`_last.pt` (saved at the ep286 eval, before the crash) is intact, so the run resumes cleanly.
**Lesson: never run the sweep as a harness background task** (it gets supervised + respawned) and
never launch a second instance — `matrix_remaining.ps1` now guards against a concurrent trainer.
