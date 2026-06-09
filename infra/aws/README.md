# AWS GPU trainer — Terraform (cost-guarded)

One command up, one command down. Three independent "never pay for a forgotten box" guards:
1. **`terraform destroy`** — manual teardown.
2. **idle-GPU watchdog** — terminates ~`idle_shutdown_minutes` after the GPU goes idle (i.e. training
   finished). Arms only after it first sees the GPU busy, so it won't kill you mid-setup.
3. **hard max-lifetime** — self-terminates after `max_hours` regardless.

Spot by default; `instance_initiated_shutdown_behavior = terminate` so any shutdown releases the EBS.

IPv6-only client -> we use **S3 for transfer** and **SSM for the shell** (both dual-stack); no inbound
SSH needed. (Portability already done: WordVectorizer is vendored, configs/aws.yaml has relative paths.)

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
tar -cf bundle.tar src configs checkpoints/tokenizer_fsq.pt `
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
