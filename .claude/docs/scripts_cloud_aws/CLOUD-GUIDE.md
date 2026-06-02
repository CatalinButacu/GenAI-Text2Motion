# AWS GPU Training — How It All Fits Together

A plain-English guide to what happens when you run `./run.sh up`. Read this
once and the scripts will make sense.

---

## The big picture in 6 steps

```
1. terraform apply  ──►  AWS spawns:
                         • IAM role  (gives EC2 permission to read/write S3)
                         • Security group  (opens SSH port 22)
                         • Spot EC2 instance  (the actual GPU machine)

2. EC2 boots  ────────►  Cloud-init runs startup.sh as root

3. startup.sh  ───────►  • Installs Python deps
                         • Clones your GitHub repo
                         • Downloads training cache from S3
                         • Runs train_motion_ssm.py
                         • Uploads checkpoint back to S3
                         • Self-terminates the EC2 (stops billing)

4. terraform destroy  ►  Cleans up IAM role + security group
                         (EC2 already gone from step 3)

5. aws s3 sync  ──────►  Downloads checkpoint from S3 to your laptop

6. Done. Bill: ~$1.50 for ~6 hours of GPU training.
```

---

## What each AWS thing actually is

### EC2 instance

A Linux VM in AWS's data center. You pick the hardware (CPU, GPU, RAM, disk),
AWS spins it up, you SSH in like any other Linux box. When you delete it, it's
gone — no recovery. Billing is per-second while it runs.

For us: **`g4dn.xlarge`** = 4 vCPUs, 16 GB RAM, **NVIDIA T4 GPU with 16 GB VRAM**.

### Spot vs on-demand

- **On-demand** (~$0.53/hr for g4dn.xlarge in eu-west-1): guaranteed availability,
  AWS will never kick you off. What you'd use for production.
- **Spot** (~$0.28/hr): same hardware, but AWS can reclaim it with 2 minutes
  warning if someone's willing to pay full price. About 50% cheaper.

For training jobs that can resume from a checkpoint, spot is the obvious choice.
We use spot. If we get interrupted, the next run picks up from S3.

### S3 (Simple Storage Service)

Object storage. Think "Dropbox for code, but cheaper and infinite." We use it for:
- Training data cache (so EC2 doesn't redownload AMASS every time)
- Trained model checkpoints (so they survive after EC2 dies)
- HumanML3D text annotations

Costs ~$0.023/GB-month. For our ~3 GB it's $0.07/month — basically free.

### IAM role

"Identity and Access Management." AWS's permission system. The EC2 instance
itself needs permission to read/write our S3 bucket and to terminate itself
when training is done. The IAM role grants exactly those permissions and
nothing else. Defined in `main.tf`.

### Security group

A firewall rule. Ours opens port 22 (SSH) from anywhere, and allows all
outbound traffic. Defined in `main.tf`.

### EC2 key pair

An SSH key pair AWS generated for you when you clicked "Create key pair" in
the console. The `.pem` file lives on your laptop at
`~/.ssh/dissertation-eu-west-1.pem`. Without it, you cannot SSH to the EC2.

### Terraform

Infrastructure-as-code tool. Instead of clicking around in the AWS console
to create the IAM role + security group + EC2, you write `.tf` files
declaring what you want, then run `terraform apply` and it talks to AWS
APIs to create everything. `terraform destroy` deletes everything cleanly
in reverse order.

The state of "what was created" lives in `terraform.tfstate` (local file).
**Don't delete `terraform.tfstate`** — if you do, Terraform forgets what
it created and you have to clean up manually in the AWS console.

---

## The lifecycle script: `run.sh`

A small bash wrapper that calls Terraform + AWS CLI in a sane order.

| command | what it does |
|---|---|
| `./run.sh up` | `terraform apply` → provisions EC2 spot |
| `./run.sh status` | `aws ec2 describe-instances` → shows state |
| `./run.sh logs` | `ssh ... 'tail -f /var/log/user-data.log'` |
| `./run.sh monitor` | polls every 5 min and prints latest training log line |
| `./run.sh wait` | blocks until EC2 self-terminates, then syncs + destroys |
| `./run.sh sync` | `aws s3 sync` → downloads checkpoints |
| `./run.sh down` | `terraform destroy` → emergency cleanup |

Typical happy-path:
```
./run.sh up                       # ~30 sec, asks confirmation
./run.sh monitor                  # status updates every 5 min, hours
                                  #   ...training runs...
                                  #   ...EC2 self-terminates...
./run.sh wait                     # syncs checkpoints + destroys IAM/SG
```

---

## The startup script: `startup.sh`

This is the file AWS runs as root inside the EC2 the moment it boots. It's
specified via the `user_data` field in `main.tf:113`. AWS runs it ONCE,
right after first boot. Output goes to `/var/log/user-data.log`.

Each section has comments explaining what's happening. The summary:

```
1. Activate the Python environment that PyTorch lives in
   (DLAMI ships PyTorch in /opt/pytorch/, NOT conda anymore as of 2025)

2. Install our extra Python deps with `uv` (sentence-transformers, spacy, etc.)
   uv is a Rust-based pip replacement, ~10x faster.

3. git clone the repo from GitHub

4. aws s3 cp the cached training data from our S3 bucket
   (so we don't redownload 150 GB of AMASS)

5. Check S3 for an existing RVQ checkpoint — if present, skip RVQ training.
   Otherwise, train RVQ from scratch (50 epochs, ~1 hour).

6. Train MotionSSM (200 epochs, ~6 hours).

7. aws s3 sync the trained checkpoints back to S3.

8. aws ec2 terminate-instances (the instance terminates itself!)
```

The IAM role attached to the EC2 (created by Terraform in `main.tf`) is what
allows step 8 to work — without that role, the EC2 wouldn't have permission
to terminate itself or write to S3.

---

## Cost model — what does each thing cost?

| Item | Rate | For one full training run |
|---|---|---|
| g4dn.xlarge spot | $0.28/hr | $1.70 (6 hours) |
| EBS gp3 volume (80 GB) | $0.08/GB-month | $0.02 (6 hours pro-rated) |
| S3 storage (3 GB) | $0.023/GB-month | $0.01 (negligible) |
| Data transfer (S3→EC2 same region) | free | $0 |
| Data transfer (S3→laptop, ~100 MB) | $0.09/GB | $0.01 |
| **Total per training run** | | **~$1.74** |

Your $119.98 in credits = ~68 training runs. You will not exhaust this budget.

---

## What can go wrong and how to recover

### "InvalidParameterCombination: not eligible for Free Tier"

Your AWS account is on Free Plan. Click "Upgrade plan" in Billing console.
Credits are preserved.

### "InsufficientInstanceCapacity"

Spot capacity for g4dn.xlarge ran out in your AZ right now. Either wait
5 min and retry, or temporarily set `use_spot = false` in `terraform.tfvars`
to use on-demand (more expensive but always available).

### "/opt/conda not found"

The DLAMI version we pulled changed its Python install location. Find the
real path with `find / -name conda.sh -o -name activate -path '*/bin/*'`
and update startup.sh accordingly.

### Spot interruption mid-training

AWS gives 2 min warning. The EC2 dies. Checkpoints saved before the
interruption are in S3 (because we save every epoch). Run `./run.sh up`
again — the startup script auto-resumes from the latest S3 checkpoint.

### "No SSH access"

Check `~/.ssh/dissertation-eu-west-1.pem` exists. Check the EC2's public IP
with `terraform output public_ip`. Make sure the security group still has
port 22 open from your IP (look in `main.tf`).

### Worst case: forgot to destroy, EC2 keeps running

Quick check: `aws ec2 describe-instances --region eu-west-1 --query 'Reservations[*].Instances[*].[InstanceId,State.Name]' --output text`

If you see anything not "terminated", run:
`./run.sh down`

Costs: max ~$7/day if you forget for a full day.

---

## Files in this directory

| file | purpose |
|---|---|
| `main.tf` | Terraform: declares IAM, security group, EC2 |
| `variables.tf` | Configurable knobs (region, instance type, etc.) |
| `terraform.tfvars` | The actual values (your bucket name, region, etc.) |
| `startup.sh` | Runs ON the EC2 at boot (training pipeline) |
| `run.sh` | Lifecycle wrapper (up, status, logs, monitor, wait, down) |
| `terraform.tfstate` | Terraform's memory of what it created (don't delete) |
| `CLOUD-GUIDE.md` | This file |
