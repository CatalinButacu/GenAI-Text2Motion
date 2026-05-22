# 12 — Cloud-run recipe: the single headline run + free CFG sweep

Status: **revised 2026-05-22**. Strategy shifted from "cloud experiments" to "local development + one cloud shot for the headline number." The Tier A/B/C/D/E nomenclature is retired; runs are now named by what they do.

Related reading: [11_REFERENCES.md](11_REFERENCES.md), [10_TEXT_ENCODER_AND_CFG.md](10_TEXT_ENCODER_AND_CFG.md).

## TL;DR

- **Account state**: clean. ~$0.05 spend over 90 days (S3 only). No active EC2.
- **Pre-warmed assets**: RVQ tokenizer + unified buffer cache + prior MotionSSM checkpoints already in `s3://dissertation-motion-cache-91340264/`.
- **Strategy**: develop the streaming code locally at tiny scale (configs/smoke_tiny.yaml). When the architecture is locked, fire ONE cloud run with `configs/motion_ssm.yaml` for the headline number.
- **Free CFG sweep first**: pull a prior best_model.pt from S3 and sweep cfg_scale at inference. Costs $0, may improve the FID number we report without retraining anything.
- **Before launching**: create a $20 monthly budget alarm so a forgotten instance can't burn the month silently.

## Pre-flight: $20 budget alarm

Free, idempotent. Run once and forget:

```bash
python scripts/cloud/aws/create_budget_alarm.py --email butacu.catalin@yahoo.com
```

Sets a `dissertation-training` budget. You'll get email at 50% / 90% actual and 100% forecasted. AWS budgets are **alert-only** — they don't stop services. Treat the alarm as "go look now," not "we're safe."

## What we know about cost

Pricing in eu-west-1 (2026-05):

| Instance | GPU | VRAM | On-demand | Spot (when avail.) |
|---|---|---|---|---|
| `g4dn.xlarge`  | 1× T4   | 16 GB        | **$0.526/hr** | ~$0.16/hr |
| `g6.xlarge`    | 1× L4   | 24 GB        | $0.805/hr     | n/a in eu-west-1 |
| `g5.xlarge`    | 1× A10G | 24 GB        | $1.006/hr     | ~$0.30/hr |
| `g5.12xlarge`  | 4× A10G | 96 GB total  | $5.672/hr     | ~$1.70/hr |

Current `terraform.tfvars` uses `g4dn.xlarge` **on-demand** (spot quota issue per the comment). $0.526/hr is the working assumption.

Calibration from prior runs (per S3 checkpoint timestamps):

| Run | Config | Epochs | Wall (est.) | Cost (est.) |
|---|---|---|---|---|
| 20260507-175255 all3rd-50e | d=384 n=6 SBERT BiMamba unified | up to ep20 saved | ~12-14 hr | ~$6-7 |
| 20260511-082827 (final) | d=384 n=6 SBERT BiMamba unified | up to ep30 saved | ~16-18 hr | ~$8-9 |

So **a 20-epoch HumanML3D run on g4dn.xlarge runs you about $5-7**. Same hyperparams. That's our budget unit.

The previous "aborted at epoch 4 = $6.95" incident (per memory file) suggests the cost model was off in that earlier run — likely an unrelated config bloated the build. With the current `startup.sh` + self-terminate trap, the cost is predictable.

## Runs, ranked by cost

### CFG inference sweep — $0, do this first

(Formerly "Tier A".)

You already have MotionSSM checkpoints with `cfg_dropout_prob=0.1` (the trainer's default). CFG at inference is **training-free**. Use it.

```bash
# Pick any prior best_model.pt that was trained with cfg_dropout_prob > 0.
# 20260511-082827 in the S3 listing is your most-trained run.
aws s3 sync s3://dissertation-motion-cache-91340264/checkpoints/motion_ssm/20260511-082827/ \
            ./checkpoints/motion_ssm/baseline/

python main.py "a person walks forward" --cfg-scale 1.0 --name walk_cfg1
python main.py "a person walks forward" --cfg-scale 2.0 --name walk_cfg2
python main.py "a person walks forward" --cfg-scale 4.0 --name walk_cfg4
python main.py "a person walks forward" --cfg-scale 7.0 --name walk_cfg7
```

Run `scripts/evaluation/compute_fid.py` once per scale for each prompt; pick the cfg_scale that minimises FID. The MoMask paper finds the sweet spot around 4. This is a **before-anything-else** result — answers "does CFG help with the current model" in zero training time.

Cost: $0.

### Headline run — the single cloud shot

(Formerly "Tier B + C + D" rolled into one.) Strategy: every architectural ablation (CLIP swap, AR-K-head, optionally compile) runs locally at tiny scale first. When the streaming code + chosen config are locked, fire ONE g5.xlarge run with `configs/motion_ssm.yaml`. ~$10-15 budget for 30 epochs.

Edit `scripts/cloud/aws/terraform.tfvars` to:

```hcl
text_encoder = "clip-b"     # 512-d CLIP text features (vs SBERT's 384-d)
ar_k_head    = false        # keep the legacy independent head this round
compile_model = false       # skip until we benchmark; first-batch compile is 30-90s
epochs_ssm   = 20           # tight cap; expect val_ce signal by epoch 5-10
batch_size   = 64
data_source  = "unified"    # uses the cached unified buffer
```

Launch:

```bash
cd scripts/cloud/aws
./run.sh up    # or: terraform init && terraform apply
```

Monitor via wandb (online mode is already wired). The startup script self-terminates the EC2 on completion or crash. Estimated wall: 11-13 hr, cost ~$5.80-6.85. Stop early if val_ce hasn't improved over SBERT baseline by epoch 12.

Decision signal: by epoch 5-8 you should see val_ce trending toward < SBERT-baseline-4.60. If it's clearly higher, kill the run early via:

```bash
aws ec2 terminate-instances --instance-ids $(terraform output -raw instance_id)
```

### CLIP + AR-K-head retraining (deprecated as standalone)

(Formerly "Tier C".) Merged into the headline run above — running two confounded variables as a separate experiment is no longer worth the burn. Kept for historical reference.

Stacks both architectural changes the audit recommended. Higher expected gain, but two confounded variables — if it wins vs Tier B, you can't attribute. Run only after Tier B has shown CLIP wins.

```hcl
text_encoder = "clip-b"
ar_k_head    = true          # ResidualKHead -- forces training from scratch
compile_model = true         # finally let's benchmark torch.compile at scale
epochs_ssm   = 30
batch_size   = 64
data_source  = "unified"
```

The AR-K-head requires training from scratch (the legacy head's weights don't map). The `--compile` flag is enabled here so we **finally** get the throughput measurement we owe ourselves (closes the "3a: flag shipped, not benchmarked at scale" item).

Estimated wall: 18-25 hr at 1× throughput. If `torch.compile` delivers the 3-5× the research predicted, more like 6-12 hr. Cost worst case ~$13, best case ~$3.50.

### 50-epoch belt-and-suspenders run (deferred)

(Formerly "Tier D".)

Don't do this yet. Tier B + Tier C give you the data to pick the right config. **Then** do a full 50-epoch run with the winning config + foot-skating metric + a real FID number for the dissertation table. That's ~$15-25 and worth the headline.

### 4×A10G multi-GPU speed run (only if wallclock matters)

(Formerly "Tier E".)

When you have the right config and want **headline-fast** turnaround. The trainer auto-wraps in `nn.DataParallel` across all visible CUDA devices when `torch.cuda.device_count() > 1`. No code change needed at the call site.

```hcl
instance_type = "g5.12xlarge"   # 4× A10G, 96 GB total VRAM
batch_size    = 256             # 4× the single-GPU batch -- DP splits per device
epochs_ssm    = 30
text_encoder  = "clip-b"
ar_k_head     = false           # AR-K-head + DP not validated; leave off here
compile_model = false           # torch.compile is auto-skipped under DP anyway
single_gpu    = false           # default; explicit for clarity
```

Constraints baked into the code (see `src/modules/motion/training/base_trainer.py`):
- `self.model` stays **unwrapped** -- state_dict save/load uses real keys (no `module.` prefix), so checkpoints are bit-compatible with single-GPU runs.
- `self.train_model = nn.DataParallel(self.model)` is what `train_epoch` / `validate` / `run_final_test` actually call.
- `--compile` under DP is **auto-skipped with a warning** -- the two interact badly. Run single-GPU + compile if you need both.
- Set `single_gpu = true` in tfvars to bypass auto-wrap (debugging / single-GPU baselining on a multi-GPU box).

Wall estimate: a 20-epoch run that takes ~12 hr on 1× T4 should land at ~3 hr on 4× A10G (roughly 4× throughput from DP, partly offset by A10G's own ~1.5× per-GPU edge over T4 and DP's ~80% scaling efficiency). Hard cap via the wallclock failsafe in `startup.sh`: instance auto-shuts down at 3 hr regardless.

Multi-GPU verification status:
- Decision logic (when to wrap, when not to, `--single-gpu` override): **unit-tested** in `tests/test_multi_gpu_wrap.py` (5/5 passing).
- Real multi-GPU forward+backward: only verified **on cloud** — can't be reproduced on a single-GPU dev box. First Tier E run is the smoke for DP itself; budget alarm at $20 will catch a runaway.

## What "monitoring" should look like

For any tier above:

1. **Live W&B**: `wandb_api_key` is already in `tfvars` (online mode). Loss curve appears on `wandb.ai` within ~30 sec of training start.
2. **SSH tail**: `ssh ubuntu@$(terraform output -raw public_ip) 'tail -f /var/log/user-data.log'`
3. **Per-epoch checkpoint upload**: every save lands in S3. Re-attach a new instance to resume if needed.
4. **Self-terminate on exit**: handled by the `cleanupOnExit` trap in `startup.sh` ([commit 0e5aab5](https://github.com/CatalinButacu/genAI-Text2Motion/commit/0e5aab5)). Crashed, OOM-killed, completed — all terminate.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Quota refused (g4dn.xlarge spot) | Already on on-demand. If on-demand quota also refuses, fall back to a smaller instance type with the same script — `g4dn.2xlarge` or skip to `g5.xlarge`. |
| EC2 boots, training fails to start | `cleanupOnExit` trap terminates within seconds. You'll see "Boot started" + error in `/var/log/user-data.log` in S3. |
| Run looks good but blows past budget | $20 budget alarm fires email at 50% / 90%. Manually `terraform destroy` if you ignore the alarm. |
| Spot interrupted (when quota allows spot) | Last best_model.pt is in S3. Re-launch with `resume_flag = "--resume latest"` (already set in `startup.sh`). |
| Wandb API key leaked in logs | xtrace is disabled around the export ([startup.sh:35-49](../../scripts/cloud/aws/startup.sh)). Sensitive in terraform. |
| The new flags break something nobody tested at scale | Smoke-tested locally on a 132-sample / 5-epoch slice (see `scripts/maintenance/validate_clip_swap.py`). Real risk is fine-grained numerical instability on a long run — that's the *point* of Tier B as the first real test. |
| Run hangs / wedges without firing cleanupOnExit trap | `startup.sh` arms a background `shutdown -h now` after `MAX_RUN_HOURS=3`. Hard cap on burn even if the python process or trap deadlocks. Bump the constant for >3-hr runs. |
| DataParallel mismatch on g5.12xlarge (load `module.` keys) | Trainer keeps `self.model` unwrapped; checkpoints save without `module.` prefix. A 1-GPU resume of a DP-trained checkpoint works directly. Verified by `tests/test_multi_gpu_wrap.py::TestCheckpointStateDictUsesUnwrappedModel`. |

## After the run

Pull artifacts:

```bash
aws s3 sync s3://dissertation-motion-cache-91340264/checkpoints/motion_ssm/$(date +%Y%m%d)-* \
            ./checkpoints/motion_ssm/latest/
```

Then for the dissertation table:

```bash
python scripts/evaluation/compute_fid.py \
    --checkpoint ./checkpoints/motion_ssm/latest/best_model.pt \
    --data-dir data/humanml3d --split test --use-sbert
```

(replace `--use-sbert` with the right text-encoder flag for the trained run).

## Cleanup checklist

After each run, before walking away:

- [ ] `terraform output instance_state` is `terminated` (or instance ID gone)
- [ ] `aws ec2 describe-instances --filters Name=instance-state-name,Values=running` returns no entries
- [ ] `terraform destroy` removes IAM role + security group
- [ ] Budget alarm did NOT fire (look in console)
- [ ] Checkpoints + training.log present in S3 under `checkpoints/motion_ssm/`
- [ ] W&B run marked `finished` (not `running` or `crashed`)

## Decision: what to run first

1. **CFG inference sweep on the existing best_model.pt** — $0, today. Answers whether CFG improves FID. If yes, the headline run inherits the winning cfg_scale.
2. **Lock the streaming code locally** via `configs/smoke_tiny.yaml` + `tests/test_streaming_equivalence.py`. No cloud cost.
3. **Fire ONE headline run** with `configs/motion_ssm.yaml` (CLIP + AR-K-head bundled). $10-15. Single shot, on-demand, NOT spot.

Everything else in this document is historical reference for what NOT to do — running 3-4 confounded cloud experiments is how the May 13 run burned $6.95 on a 4-epoch abort.

## Open question

Spot quota: the `terraform.tfvars` says `use_spot = false` because "spot quota issue." If you want to push to spot for the headline run, request a quota increase via the AWS console (Service Quotas → EC2 → "All G and VT Spot Instance Requests"). Free to request, takes 1-3 business days. Drops your cost by ~70%.
