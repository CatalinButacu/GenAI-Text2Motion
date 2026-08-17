# Azure A10 runbook — twin fairness on the $200 / 30-day credit

Answers one question: **does the fused `mamba-ssm` selective-scan kernel build and import
against our pinned torch, and what does it buy?** Two AWS attempts failed on an ABI mismatch
(`.claude/docs/STATUS.md:780`) because the Deep Learning AMI's torch outran `mamba-ssm`.
Building against our own environment controls both sides of the ABI.

## Why this box

`Standard_NV36ads_A10_v5` — a **full** A10, 24 GB, sm_86 (Ampere).

The NV-A10 family is sliced: `NV6ads` is 1/6 of the card (4 GB), `NV12ads` 1/3 (8 GB),
`NV18ads` 1/2 (12 GB), `NV36ads` the whole card (24 GB). Only the full card is worth taking —
a 4 GB slice is the laptop we are trying to escape.

Ampere is the point, not the speed. sm_86 has **bf16 and TF32**; the free Kaggle T4 (sm_75) has
neither, so a T4 can only produce a fused-vs-eager ratio. On the A10 the precision results
transfer, and the 96M twin pair becomes trainable.

Spot is ~$0.591/h, on-demand ~$3.20/h. The canary is ~2 h and the pilot twins ~8 h, so **the
30-day clock binds long before the credit does.**

## The credit

The $200 is retained for the balance of the original 30 days **after upgrading to
pay-as-you-go**, and upgrading is what unblocks GPU SKUs — a trial subscription cannot allocate
them at all. Quota starts at **zero** per region and must be requested. That request costs
calendar, not money, so it goes in first.

## Phase A — account and quota (day 1, blocking, free)

Run these one at a time and read each result before the next.

1. Upgrade the trial subscription to pay-as-you-go.
2. Choose **one region** and stay in it — quota is per-region and cross-region egress is billed.
   Confirm the region actually offers NVadsA10v5 before committing.
3. Request quota: `Standard NVADSA10v5 Family vCPUs` = **36** (one full NV36ads_A10_v5).
4. Request the **Spot** quota as a separate item. On-demand and spot quotas are distinct, and
   every long run here is meant to be spot.
5. Create a budget on the subscription with alerts at **$50 / $100 / $150**.
6. On the VM, enable the built-in **auto-shutdown schedule**. This is a platform-level stop,
   strictly harder than any watchdog script, and it is the direct fix for the $18 idle burn on
   the previous AWS attempt.

Write down the quota request date. If it is refused or still pending after ~3 days, switch to
`infra/kaggle/` without further deliberation — the clock does not pause.

## Phase 1 — kernel canary (~2 h, ~$1.20 at spot)

Needs **no data**. The probes build models from config with random weights.

Locally:

    bash infra/make_bundle.sh

That writes `dist/bundle/thesis_code.tar.gz` (~380 KB) from the **working tree, not git** —
`src/text2motion/app/cli.py` is untracked and a `git clone` would ship the old, deleted package
layout.

Upload it to Blob in the VM's region with `azcopy`, pull it down on the box, then:

    tar -xzf thesis_code.tar.gz -C ~/thesis
    bash ~/thesis/infra/kernel_canary.sh

`TORCH_CUDA_ARCH_LIST` does **not** need setting. The script reads
`torch.cuda.get_device_capability()` and defaults the arch list to the detected GPU, so it
targets 8.6 on the A10 and 7.5 on a T4 without being told. Override it only to cross-compile.

Install `torch 2.12.0+cu132` per `uv.lock` before running — building `mamba-ssm` against the
distro's torch instead of ours is precisely the failure mode being tested. The from-source build
takes ~35 minutes; that is expected, and is why the AWS idle watchdog was raised 90 -> 150 min.

## What the three gates mean

| gate | question | reading |
|---|---|---|
| 1 | does the kernel build **and import**? | the import line is where both AWS attempts died; failure here ends the plan |
| 2 | training step, eager vs fused | the number that decides whether a retrain is affordable |
| 3 | rollout latency, eager vs fused | **expect ~1.00x** |

Gate 3 is not a bug. `use_kernel` only affects `MambaMixer.forward`
(`src/text2motion/generation/model.py:142`), the training path. Streaming uses
`MambaMixer.step` (`:109`), which has **no fused path at all**. Making the *latency* comparison
fair additionally needs `mamba_ssm.ops.triton.selective_state_update` wired into `step()` —
that is not implemented, and gate 3 documents the gap.

Gate 2 confound worth stating in any write-up: `checkpoint_blocks = not cfg.use_kernel`
(`model.py:190`), so the fused run also drops gradient checkpointing. That is the real deployed
difference, not an isolated kernel swap.

## Decision table for gate 2

Baselines measured locally on an RTX 3050 (mamba 937 ms/step, transformer 51.2 ms/step, 31.8M,
batch 8; 22,418 train clips = 2,802 steps/epoch). The A10 is itself 2-3x the 3050 before any
kernel gain, so these hours are an upper bound — use the **ratio**, not the absolute.

| fused mamba step (3050-equivalent) | 60-epoch retrain | verdict |
|---|---|---|
| ~250 ms | ~12 h | retrain both pilot twins, then attempt the 96M pair |
| ~500 ms | ~23 h | pilot twins only |
| no gain | ~44 h | keep existing checkpoints and report eager honestly |

## Phase 3 — training

Stage the 4.66 GB listed by `make_bundle.sh` into one Blob container in the VM's region,
preserving paths.

Spot eviction and Kaggle's 9-hour session cap are the same problem with the same fix:
`--resume` is epoch-granular and reads `checkpoints/<stem>_last.pt`, which `train_generator`
writes **every epoch unconditionally** (`generation/trainer.py:389`) with generator + text
encoder + optimizer + EMA + scheduler + epoch + best_fid. Sync that file to Blob each epoch and
an eviction costs at most one epoch.

Two traps: the `--overfit_gate` JSON must be **byte-identical** across sessions
(`OverfitGate.verify_setup` re-checks the sha256 of both the config and the tokenizer, and fails
loud on any edit), and the *best* weights live only in the separate bundle at `checkpoint_path`
— carry both files.

Keep the watchdog as well as auto-shutdown. `train-generator` writes
`outputs/runs/<stamp>_generator_<bb>/heartbeat` every 100 steps, but `train-tokenizer` and
`pretrain` write no heartbeat — for those, watch the run dir's newest file with a 45 min stale
window, as `scripts/training/run_3day_local.ps1` already does.

## Not this

- **No on-demand A10 for long runs** — $3.20/h vs $0.591/h spot. On-demand only for the short
  canary, and only if spot capacity is unavailable.
- **No multi-region sprawl** — one region, for quota and egress.
- **Nothing permanent on this subscription.** The credit dies at day 30; a lasting demo API
  belongs on OCI Always Free (`VM.Standard.A1.Flex`, 4 ARM OCPUs + 24 GB, $0 indefinitely).
