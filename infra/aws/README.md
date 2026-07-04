# AWS GPU trainer -- Terraform (cost-guarded)

One command up, one command down. Three independent "never pay for a forgotten box" guards:
1. **`terraform destroy`** -- manual teardown.
2. **idle-GPU watchdog** -- terminates ~`idle_shutdown_minutes` after the GPU goes idle (i.e. training
   finished). Arms only after it first sees the GPU busy, so it won't kill you mid-setup.
3. **hard max-lifetime** -- self-terminates after `max_hours` regardless.

Spot by default; `instance_initiated_shutdown_behavior = terminate` so any shutdown releases the EBS.

IPv6-only client -> we use **S3 for transfer** and **SSM for the shell** (both dual-stack); no inbound
SSH needed. (Portability already done: WordVectorizer is vendored, configs/aws.yaml has relative paths.)

## Phase-3 FINAL run -- FSQ 8x1024, 100M twins (current; do this)

Frozen tokenizer **`checkpoints/tokenizer/fsq_g8_v1024.pt`** (recon-FID 0.0170). Config
**`configs/generator/final100m_fsq8x1024.yaml`** (transformer 98.1M / Mamba 99.7M param-matched, 8 codebooks x
1024, 16-token prefix, END, fused kernel, bf16). `run_canary.sh` (gate) and `run.sh` (both twins,
60 ep, val-select, locked CFG 5.0/temp 1.1) are already wired to these.

```powershell
# 1) Bundle from repo root -- note the FSQ-8x1024 tokenizer + the run scripts:
$b = "thesis-t2m-913402647373"
tar -cf bundle.tar src configs checkpoints/tokenizer/fsq_g8_v1024.pt infra/aws/run.sh infra/aws/run_canary.sh `
  data/HumanML3D_official data/eval_stats data/official_evaluator data/t2m_glove
aws s3 cp bundle.tar s3://$b/bundle.tar
# 2) Launch (cost guards arm automatically):
cd infra/aws; terraform init; terraform apply
# 3) SSM in, pull bundle, then CANARY first (gate ~30-60 min, ~$2):
#      bash run_canary.sh   -> PASS = kernel parity OK AND 5-ep timing projects < 20 h
#      (CANARY_DONE -> s3://$b/results/canary/)
# 4) Full run only if the canary passed:
#      bash run.sh          -> both twins 60 ep, S3 ckpt/log sync; ALL_DONE when finished;
#                              idle watchdog auto-terminates the box.
# 5) Harvest + Table 1:
aws s3 cp s3://$b/results/ . --recursive
#   then: python -m text2motion.eval.evaluate --split test  (20-rep, both backbones)
```

### Mamba-only 100M (transformer twin already done) -- cheaper, do THIS one

The transformer 100M is already trained (`generator_transformer_100m`, val FID 1.63), so only the
Mamba half is missing. `run_mamba100m.sh` trains it through the **same two-stage pipeline the
transformer used**: an AMASS motion-prior **pretrain** (from random init, 30 ep bs32) then a captioned
**fine-tune** initialised from that prior -- so the twins share the treatment, not just the mixer.
Fused kernel ON, spot-survivable (resumes at either stage from the last S3 checkpoint). One backbone
-> **~half the cost** (~$14-18, 10-14 h). The completed twin table then reruns from
`generator_mamba_100m.pt` + the local `generator_transformer_100m*.pt`.

```powershell
# 1) Bundle (adds run_mamba100m.sh + the AMASS token pack for the pretrain; the transformer ckpt is
#    NOT needed on the box -- the final eval is local):
$b = "thesis-t2m-913402647373"
tar -cf bundle.tar src configs checkpoints/tokenizer/fsq_g8_v1024.pt data/amass_tokens_fsq8x1024.npz `
  infra/aws/run_canary.sh infra/aws/run_mamba100m.sh `
  data/HumanML3D_official data/eval_stats data/official_evaluator data/t2m_glove
aws s3 cp bundle.tar s3://$b/bundle.tar
# 2) Launch + SSM in (cost guards arm automatically):
cd infra/aws; terraform init; terraform apply
aws ssm start-session --target (terraform output -raw instance_id)
# 3) On the box: pull bundle, CANARY (kernel-parity gate ~$2), then the Mamba run (pretrain+finetune):
#      cd /opt/thesis && aws s3 cp s3://<b>/bundle.tar . && tar -xf bundle.tar
#      bash run_canary.sh        # PASS = kernel parity OK AND 5-ep timing projects < 20 h
#      bash run_mamba100m.sh     # STAGE A pretrain -> STAGE B finetune; MAMBA100M_DONE -> s3://<b>/results/
# 4) Harvest + finish the twin table:
aws s3 cp s3://$b/results/generator_mamba_100m.pt checkpoints/generator/ --region eu-north-1
python -m text2motion.eval.evaluate --backbone mamba --ckpt checkpoints/generator/generator_mamba_100m.pt `
  --config configs/generator/final100m_fsq8x1024.yaml --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt `
  --split test --cfg_scale 6.0 --temperature 1.0 --length_mode fixed --mm_clips 100 --mm_repeats 30
```

---

## One-time setup
```powershell
winget install Hashicorp.Terraform Amazon.AWSCLI
aws configure                       # Access Key + Secret (IAM), default region e.g. eu-central-1
aws s3 mb s3://thesis-t2m-<name>    # globally-unique bucket for transfer
ssh-keygen -y -f dissertation_v2.pem | Set-Content -Encoding ascii dissertation_v2.pub
Copy-Item terraform.tfvars.example terraform.tfvars   # edit: region, bucket
```

## Build the data bundle + upload to S3 (run from repo root)
```powershell
$b = "thesis-t2m-<name>"
tar -cf bundle.tar src configs checkpoints/tokenizer/fsq_g8_v1024.pt `
  data/HumanML3D_official data/eval_stats data/official_evaluator data/t2m_glove
# add the local resume checkpoints if continuing: checkpoints/generator_*_last.pt
aws s3 cp bundle.tar s3://$b/bundle.tar
```

## Launch the instance
```powershell
cd infra/aws; terraform init; terraform apply    # prints instance_id + cost_guards
```

## Connect via SSM, pull bundle, train (detached so it survives a dropped session)
```powershell
aws ssm start-session --target (terraform output -raw instance_id)
```
On the box (Deep Learning AMI has CUDA+PyTorch+tmux+awscli):
```bash
mkdir -p ~/thesis && cd ~/thesis
aws s3 cp s3://thesis-t2m-<name>/bundle.tar . && tar -xf bundle.tar
pip install -q transformers   # plus anything else requirements lists
export PYTHONPATH=src
tmux new -s train             # detach with Ctrl-b d; reattach: tmux attach -t train
python -u -m text2motion.train.train_generator --config configs/aws.yaml --backbone transformer --epochs 150 --batch_size 64 &> outputs/tf.log
python -u -m text2motion.train.train_generator --config configs/aws.yaml --backbone mamba       --epochs 150 --batch_size 64 &> outputs/mm.log
# results back to S3 when done:
aws s3 cp checkpoints/ s3://thesis-t2m-<name>/results/ --recursive --exclude "*" --include "generator_*.pt"
aws s3 cp outputs/ s3://thesis-t2m-<name>/results/ --recursive
```

## Pull results locally, then tear down
```powershell
aws s3 cp s3://thesis-t2m-<name>/results/ ..\..\ --recursive
terraform destroy            # or let the idle watchdog auto-terminate ~20m after training ends
```
