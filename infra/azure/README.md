# Azure GPU runbook — twin fairness on the $200 / 30-day credit

Answers one question: **does the fused `mamba-ssm` selective-scan kernel build and import
against our pinned torch, and what does it buy?** Two AWS attempts failed on an ABI mismatch
(`.claude/docs/STATUS.md:780`) because the Deep Learning AMI's torch outran `mamba-ssm`.
Building against our own environment controls both sides of the ABI.

## Why this box

`Standard_NC24ads_A100_v4` — one A100 **80 GB**, sm_80 (Ampere), 24 vCPUs.

### Eliminated — cannot run both twins fairly

A card qualifies only if it runs **both** twins in their intended form: the transformer on fused
SDPA, Mamba on the fused `mamba-ssm` selective scan, both in bf16. That requires **sm_80+** (bf16
and TF32 are Ampere-and-later) and a `mamba-ssm` build target that is actually proven.

| rejected | why |
|---|---|
| `NC4as_T4_v3` (T4, sm_75) | no bf16, no TF32 — same ceiling as free Kaggle, precision results would not transfer |
| `NC*_RTXPRO6000BSE_v6` (Blackwell, sm_120) | bf16 fine, but `mamba-ssm` from source on sm_120 is unproven; if it fails, Mamba falls back to eager and we are back to the exact unfairness we are paying to fix |
| `NV6/12/18ads_A10_v5` | vGPU slices of 4-12 GB — the laptop we are escaping |
| `ND*_MI300X_v5` (AMD) | ROCm; the `mamba-ssm` CUDA kernel does not exist there at all |

### Survivors — price vs power

Prices from the Azure retail API, 2026-08-18, Germany West Central, Linux spot. TFLOPS are
approximate vendor bf16-dense figures; bandwidth is the more predictive number at our model size.

| SKU | GPU | VRAM | spot/h | $/TFLOP | $/(GB/s) | hours per $200 |
|---|---|---|---|---|---|---|
| `NV36ads_A10_v5` | A10, sm_86 | 24 GB | $0.769 | 0.00615 | 1.28 | 260 |
| **`NC24ads_A100_v4`** | **A100, sm_80** | **80 GB** | **$0.882** | **0.00283** | **0.43** | **227** |
| `NC40ads_H100_v5` | H100, sm_90 | 94 GB | $1.548 | 0.00185 | 0.40 | 129 |

The A10 is the worst buy on every efficiency axis — 2.2x the A100's cost per TFLOP and 3x its
cost per unit bandwidth — for only 15% less per hour. Drop it.

The H100 wins on paper. It is **not** the pick, because at 31.8M parameters and batch 8 the
bottleneck is kernel-launch and memory latency, not FLOPs — an H100 would sit mostly idle while
billing 1.75x. Its $/bandwidth is within 8% of the A100 anyway, so the paper advantage is
largely FLOPs we cannot use. Reconsider it only if the 96M twins run at large batch.

**`NC24ads_A100_v4` is the pick:** 3.4x the A10's bandwidth (what actually limits small-model
training), `mamba-ssm`'s native and best-tested target — which directly de-risks the one
question this trip exists to answer — 80 GB, and a *smaller* quota ask (24 vCPUs vs 36).

## The credit

The $200 is retained for the balance of the original 30 days **after upgrading to
pay-as-you-go**, and upgrading is what unblocks GPU SKUs — a trial subscription cannot allocate
them at all. Quota starts at **zero** per region and must be requested. That request costs
calendar, not money, so it goes in first.

### There is no one-month commitment — and we do not want one

Azure Reserved VM Instances are **1-year or 3-year only**, exactly like AWS Reserved Instances.
Azure Savings Plans are likewise 1 or 3 years. Nothing shorter exists.

It would be the wrong instrument three times over. Measured 2026-08-18 from the retail API, a
1-year `NC24ads_A100_v4` reservation in germanywestcentral is **$27,343 upfront** (northeurope
$25,240, switzerlandnorth $30,077; 3-year runs $43k-51k).

- **The credit cannot pay for it.** On a pay-as-you-go subscription bought through Azure.com the
  upfront payment is charged to the **card on file**; only Enterprise Agreement customers draw
  reservations from prepayment. And $200 is **0.73%** of $27,343 — it would not cover three days.
- **Spot beats the reserved rate outright.** $27,343 / 8,760 h = **$3.12/h effective**, only 29%
  off the $4.408 on-demand price. Spot is **$0.882/h — 3.5x cheaper than reserving.**
- **Utilization would be 0.46%.** Committing 8,760 hours to run ~40.

Commitment discounts are the right tool for sustained predictable load. This is 40 hours of
bursty batch work inside a 30-day window — the exact shape spot exists to serve. Eviction is
already handled: `--resume` is epoch-granular.

**What $200 actually buys:** 227 hours of A100 spot ≈ **9.5 days continuous**. A full 30 days of
continuous runtime would cost ~$635 at spot, ~$3,174 on-demand — out of reach and unnecessary.
The whole job (canary ~2 h, pilot twins ~10 h, 96M pair ~30 h) is **roughly 40 hours of runtime
spread across 30 days of calendar**. The credit covers that five times over. Buy runtime, not
uptime: shut the box down between runs and the calendar is never the binding cost.

## Phase A — account and quota (day 1, blocking, free)

Run these one at a time and read each result before the next.

State measured on the subscription 2026-08-18 (`5b993c40`): `quotaId = FreeTrial_2014-09-01`,
`spendingLimit = On`, `Microsoft.Compute = NotRegistered`. So it is the **standard free trial**,
not an Azure Pass sponsorship — the upgrade path below applies and the credit is retained.

1. **Register the compute provider.** Free, reversible, and it must come first: with
   `Microsoft.Compute` unregistered, `az vm list-usage` returns **zero rows** and the portal's
   quota page shows no GPU families at all, so there is nothing to request against.

       az provider register --namespace Microsoft.Compute

2. Upgrade the trial subscription to pay-as-you-go. Portal only, no CLI equivalent.
   **This removes `spendingLimit: On`.** While that limit is active the subscription is disabled
   rather than billed when credit runs out — i.e. overspend is currently impossible. After the
   upgrade, anything past the $200 lands on the card. The guards below stop being good practice
   and start being the only thing between you and a real bill.
3. Set the budget **immediately after step 2**, not later: alerts at **$50 / $100 / $150**.
4. Choose **one region** and stay in it — quota is per-region and cross-region egress is billed.
   Default is **germanywestcentral**: measured fastest from the operator's machine (43-92 ms TCP
   connect vs 59 ms polandcentral, 137 ms switzerlandnorth, 187 ms uksouth) *and* it actually
   offers the SKU. Latency barely matters for batch training, but here it costs nothing.

   **Always cross-check price against availability — they disagree.** The retail API quotes
   northeurope cheapest for the A100 at $0.815/h, but `list-skus` does not offer NC24ads there
   at all; the same trap hides `NV36ads` from westeurope, and `ukwest` publishes a $0.002/h A10
   spot price that is a placeholder, not a bargain. Cheapest *available* European A100 spot:
   francecentral and uksouth at $0.848, then germanywestcentral, italynorth, polandcentral at
   $0.882. francecentral and italynorth both failed a connectivity probe twice. Re-check before
   committing:

       az vm list-skus --size Standard_NC24ads_A100_v4 --query "[].locationInfo[0].location" -o tsv

5. Request quota: `Standard NCADS_A100_v4 Family vCPUs` = **24** (one NC24ads_A100_v4). Then
   raise `Total Regional vCPUs` to **24** as a second request — it is an umbrella cap that
   silently blocks the VM even after the family quota is approved. Measured 2026-08-18 in
   germanywestcentral: family limit **0**, regional limit **10**.
6. Request the **Spot** quota as a separate item. On-demand and spot quotas are distinct, and
   every long run here is meant to be spot.
7. On the VM, enable the built-in **auto-shutdown schedule**. This is a platform-level stop,
   strictly harder than any watchdog script, and it is the direct fix for the $18 idle burn on
   the previous AWS attempt.

Write down the quota request date. If it is refused or still pending after ~3 days, switch to
`infra/kaggle/` without further deliberation — the clock does not pause.

## Phase 1 — kernel canary (~2 h, ~$1.80 at spot)

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
targets 8.0 on the A100, 8.6 on an A10, 7.5 on a T4 without being told. Override it only to
cross-compile.

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
batch 8; 22,418 train clips = 2,802 steps/epoch). The A100 is itself several times the 3050 before any
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
