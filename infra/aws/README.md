# AWS GPU trainer — Terraform (cost-guarded)

One command up, one command down. Three independent "never pay for a forgotten box" guards:
1. **`terraform destroy`** — manual teardown.
2. **idle-GPU watchdog** — terminates ~`idle_shutdown_minutes` after the GPU goes idle (i.e. training
   finished). Arms only after it first sees the GPU busy, so it won't kill you mid-setup.
3. **hard max-lifetime** — self-terminates after `max_hours` regardless.

Spot by default; `instance_initiated_shutdown_behavior = terminate` so any shutdown releases the EBS.

IPv6-only client -> we use **S3 for transfer** and **SSM for the shell** (both dual-stack); no inbound
SSH needed. (Portability already done: WordVectorizer is vendored, configs/aws.yaml has relative paths.)

## Phase-3 FINAL run — FSQ 8x1024, 100M twins (current; do this)

Frozen tokenizer **`checkpoints/tokenizer/fsq_g8_v1024.pt`** (recon-FID 0.0170). Config
**`configs/generator/final100m_fsq8x1024.yaml`** (transformer 98.1M / Mamba 99.7M param-matched, 8 codebooks x
1024, 16-token prefix, END, fused kernel, bf16). `run_canary.sh` (gate) and `run.sh` (both twins,
60 ep, val-select, locked CFG 5.0/temp 1.1) are already wired to these.

```powershell
# 1) Bundle from repo root — note the FSQ-8x1024 tokenizer + the run scripts:
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
