# Kernel canary on free GPU (Kaggle) — FALLBACK VENUE

Primary venue is now Azure A10 — see `infra/azure/README.md`. Use this page only if the Azure
GPU quota request is refused or still pending after ~3 days. Nothing here is wasted either way:
`infra/kernel_canary.sh` and `infra/make_bundle.sh` are shared by both venues.

Answers one question: **does the fused `mamba-ssm` selective-scan kernel build and import
against our pinned torch, and what does it buy?** Two AWS attempts failed on an ABI mismatch
(`.claude/docs/STATUS.md:780`) because the Deep Learning AMI's torch outran `mamba-ssm`.
Building against our own environment controls both sides of the ABI.

Phase 1 costs **nothing** and needs **no data** — the probes build models from config with
random weights.

## Why Kaggle, not Colab

Kaggle gives 30 GPU-h/week, 9-hour sessions, and **keeps running after the tab closes**.
Colab free needs the tab open (background execution is Pro+).

**Pick the T4 accelerator, not P100.** P100 is sm_60 and is not a supported `mamba-ssm`
target; the canary aborts early with a clear message if it sees one. T4 is sm_75, which has
**no bf16 and no TF32** — so that box can measure the fused-vs-eager ratio, but cannot
reproduce any precision comparison.

## Run it

1. `bash infra/make_bundle.sh` locally -> `dist/bundle/thesis_code.tar.gz` (~380 KB).
   It bundles from the **working tree, not git** — `src/text2motion/app/cli.py` is untracked
   and a `git clone` would ship the old, deleted package layout.
2. New Kaggle notebook, Accelerator = **GPU T4**, Internet = **on** (the build downloads
   sources).
3. Upload the tarball as notebook input, then:

```bash
!mkdir -p /kaggle/working/thesis && tar -xzf /kaggle/input/<your-dataset>/thesis_code.tar.gz -C /kaggle/working/thesis
!cd /kaggle/working/thesis && bash infra/kernel_canary.sh
```

The build takes ~35 minutes from source. That is expected — the AWS idle watchdog was raised
90 -> 150 min for exactly this.

## What the three gates mean

| gate | question | reading |
|---|---|---|
| 1 | does the kernel build **and import**? | the import line is where both AWS attempts died; a failure here ends the plan |
| 2 | training step, eager vs fused | the number that decides whether a retrain is affordable |
| 3 | rollout latency, eager vs fused | **expect ~1.00x** |

Gate 3 is not a bug. `use_kernel` only affects `MambaMixer.forward`
(`src/text2motion/generation/model.py:142`), the training path. Streaming uses
`MambaMixer.step` (`:109`), which has **no fused path at all**. Making the *latency*
comparison fair additionally needs `mamba_ssm.ops.triton.selective_state_update` wired into
`step()` — that is not implemented, and gate 3 documents the gap.

Gate 2 confound worth stating in any write-up: `checkpoint_blocks = not cfg.use_kernel`
(`model.py:190`), so the fused run also drops gradient checkpointing. That is the real deployed
difference, not an isolated kernel swap.

## Decision table for gate 2

Baselines measured locally on an RTX 3050 (mamba 937 ms/step, transformer 51.2 ms/step, 31.8M,
batch 8; 22,418 train clips = 2,802 steps/epoch). T4 absolute numbers will differ — use the
**ratio**.

| fused mamba step | 60-epoch retrain | verdict |
|---|---|---|
| ~250 ms | ~12 h | fits Kaggle's weekly quota; retrain both pilot twins free |
| ~500 ms | ~23 h | one twin per week |
| no gain | ~44 h | keep existing checkpoints and report eager honestly |

## If you go on to train (Phase 3)

Upload the data listed by `make_bundle.sh` as one private Kaggle Dataset (4.66 GB), preserving
paths. Sessions are 9 h and a run is longer, so rely on `--resume`: `train-generator` writes
`checkpoints/<stem>_last.pt` every epoch with generator + text encoder + optimizer + EMA +
scheduler + epoch + best_fid. Push it to a Dataset version at session end, pull it back at
session start.

Two traps: the `--overfit_gate` JSON must be **byte-identical** across sessions
(`OverfitGate.verify_setup` re-checks the sha256 of both the config and the tokenizer, and
fails loud on any edit), and the *best* weights live only in the separate bundle at
`checkpoint_path` — carry both files.
