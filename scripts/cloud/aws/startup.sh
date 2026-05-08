#!/bin/bash
# =============================================================================
# startup.sh — runs ONCE on the EC2 instance the moment it boots
# =============================================================================
#
# AWS calls this "user data". It's plain bash, executed as root by cloud-init
# right after first boot. AWS picks it up because main.tf line 113 has:
#     user_data = templatefile("$${path.module}/startup.sh", { ... })
# and Terraform substitutes $${variables} below before sending it to AWS.
#
# Where to find the live output once the EC2 is running:
#     ssh ubuntu@<ip> "sudo tail -f /var/log/user-data.log"
#
# This file does 8 things in order:
#   1. Activate the EC2's pre-installed PyTorch environment
#   2. pip install our extra Python deps (sentence-transformers, etc.)
#   3. git clone our code from GitHub
#   4. aws s3 sync our cached training data (so we don't redownload AMASS)
#   5. Reuse our existing RVQ checkpoint from S3 (or train one if missing)
#   6. Train the MotionSSM model on the GPU (the actual long step, ~6 hours)
#   7. aws s3 sync the trained checkpoints back to S3
#   8. aws ec2 terminate-instances on ourselves -> EC2 dies, billing stops
#
# Important: the EC2 self-terminates at the end. That's why this works as
# "fire and forget" — you launch it, walk away, and ~6 hours later your
# checkpoint is in S3 and the instance is gone.
# =============================================================================

# `set -e`: exit immediately if any command fails (catches bugs early)
# `set -u`: error on unset variables (catches typos)
# `set -x`: print every command before running (helpful in /var/log/user-data.log)
# `set -o pipefail`: in a pipe, fail if ANY command fails, not just the last
set -euxo pipefail

# Redirect everything to /var/log/user-data.log so SSH-tail can read it later
exec > /var/log/user-data.log 2>&1

echo "=== Boot started: $(date) ==="

# -----------------------------------------------------------------------------
# Configuration — Terraform substitutes the $${...} placeholders below from
# values in terraform.tfvars *before* this file is sent to AWS. The double
# dollar ($$) you see here in the comment is just to escape the syntax so
# Terraform's templater leaves it alone in this comment block.
# -----------------------------------------------------------------------------
REPO_DIR=/home/ubuntu/repo            # where to clone the code
S3_BUCKET="${s3_bucket}"              # e.g. dissertation-motion-cache-91340264
S3_CACHE_URI="${s3_cache_uri}"        # e.g. s3://.../cache/
REGION="${region}"                    # e.g. eu-west-1
REPO_URL="${repo_url}"                # GitHub HTTPS URL
DATA_SOURCE="${data_source}"          # 'unified' (uses pre-built cache)
EPOCHS_RVQ="${epochs_rvq}"            # ignored if S3 already has best_model.pt
EPOCHS_SSM="${epochs_ssm}"            # how long to train SSM (typically 200)
BATCH_SIZE="${batch_size}"            # depends on GPU VRAM; 64 for T4 16GB

# -----------------------------------------------------------------------------
# Step 1: activate the Python environment that has PyTorch installed
# -----------------------------------------------------------------------------
# AWS Deep Learning AMIs (DLAMI) ship PyTorch in a venv at /opt/pytorch/.
# `source <venv>/bin/activate` puts that venv on the $PATH, so when we run
# `python` later it's the one with PyTorch + CUDA configured.
#
# (Old DLAMIs used conda at /opt/conda/. AWS dropped that in 2025.)
#
# Two pre-flight tweaks needed because the venv's activate script was written
# without `set -u` in mind:
#   1. LD_LIBRARY_PATH=""  -- activate references this var; if unset, set -u
#                             aborts the whole script
#   2. set +u around the source -- belt-and-suspenders in case other vars
#                                  (PYTHONPATH, PS1, etc.) are also missing
# -----------------------------------------------------------------------------
export LD_LIBRARY_PATH="$${LD_LIBRARY_PATH:-}"
set +u
source /opt/pytorch/bin/activate
set -u

# Sanity-check that GPU + PyTorch are wired up
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# -----------------------------------------------------------------------------
# Step 2: install our extra deps with `uv` (10x faster than pip)
#
# uv (Astral) is a Rust-based pip replacement. Same API as pip but parallel
# downloads + hard-linking from cache + a much faster dependency resolver.
# Installing 6 packages with pip = ~60-90 sec, with uv = ~10 sec.
#
# We bootstrap with one pip install (5 sec) then use uv for everything else.
# `uv pip install` installs into the currently-activated venv (/opt/pytorch).
# -----------------------------------------------------------------------------
pip install --quiet uv

uv pip install --quiet \
  sentence-transformers \
  scipy \
  spacy \
  wandb \
  beartype \
  joblib

# spacy model for the M1 scene parser (used at inference, not for SSM training,
# but the imports fail without it)
python -m spacy download en_core_web_sm --quiet

# -----------------------------------------------------------------------------
# Step 3: clone the code from GitHub
# `--depth=1` = shallow clone, only the latest commit (faster, no history)
# -----------------------------------------------------------------------------
git clone --depth=1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

# -----------------------------------------------------------------------------
# Step 4: download our prebuilt training data cache from S3
#
# We do NOT download raw AMASS (150 GB — wouldn't fit on the EC2 anyway).
# Instead, locally we ran scripts/data/prebuild_unified_cache.py once to
# build a single ~60 MB joblib that contains the parsed motion+text data,
# then we uploaded that to S3. The EC2 just downloads that one file.
# -----------------------------------------------------------------------------
CACHE_DIR="$REPO_DIR/data/.cache"
mkdir -p "$CACHE_DIR"
mkdir -p "$REPO_DIR/data/AMASS"        # trainer's prereq check expects this dir
mkdir -p "$REPO_DIR/data/humanml3d"

echo "--- Downloading data from S3 ---"

# `aws s3 sync` = like rsync but to/from S3. Only copies new/changed files.
# `--exclude "*" --include "*.joblib"` = filter trick: exclude everything,
# then re-include only joblib files. Picks up unified_buf_<hash>.joblib.
aws s3 sync "$S3_CACHE_URI" "$CACHE_DIR/" --exclude "*" --include "*.joblib"

# HumanML3D text annotations + train/val/test splits (~52 MB total).
# `|| true` = don't crash if missing (this is optional data).
aws s3 sync "s3://$S3_BUCKET/data/humanml3d" "$REPO_DIR/data/humanml3d" || true

# Z-score normalization stats + per-source translation stats (~3 KB total).
aws s3 sync "s3://$S3_BUCKET/data/stats" "$REPO_DIR/data/stats" || true

# Vocabulary YAML files (action + object definitions, ~34 KB).
# These are gitignored locally but the trainer's imports pull them in.
aws s3 sync "s3://$S3_BUCKET/data/vocabulary" "$REPO_DIR/data/vocabulary" || true

echo "Cache ready: $(du -sh $CACHE_DIR 2>/dev/null | cut -f1)"

# `chown` = make the `ubuntu` user own the files (we cloned as root, but
# training runs as ubuntu in step 6).
chown -R ubuntu:ubuntu "$REPO_DIR"

# -----------------------------------------------------------------------------
# Step 5: get the RVQ tokenizer checkpoint
#
# The RVQ tokenizer is the ~10 MB encoder/decoder we trained locally (`all3rd`
# = best val_recon 0.0946). The MotionSSM in step 6 needs it as a frozen
# component. So we either:
#   (a) Reuse from S3 if `best_model.pt` is already there  -> ~5 sec download
#   (b) Train a fresh one if missing                       -> ~1 hour on T4
#
# In our setup we always go path (a) because we uploaded all3rd already.
# -----------------------------------------------------------------------------
RVQ_CKPT_DIR="$REPO_DIR/checkpoints/rvq_tokenizer"
mkdir -p "$RVQ_CKPT_DIR"

# `aws s3 ls > /dev/null 2>&1` = "does this object exist?" (silently)
if aws s3 ls "s3://$S3_BUCKET/checkpoints/rvq_tokenizer/best_model.pt" > /dev/null 2>&1; then
  echo "=== Reusing existing RVQ checkpoint from S3 ==="
  aws s3 cp "s3://$S3_BUCKET/checkpoints/rvq_tokenizer/best_model.pt" \
            "$RVQ_CKPT_DIR/best_model.pt"
else
  echo "=== No RVQ checkpoint in S3 — training one (~1 hour on T4) ==="
  # `sudo -u ubuntu` switches to the ubuntu user (so wandb/cache files are
  # owned by ubuntu, not root). The whole training is in a single quoted block.
  sudo -u ubuntu bash -c "
    source /opt/pytorch/bin/activate
    export WANDB_MODE=offline WANDB_SILENT=true
    cd $REPO_DIR
    python scripts/training/train_rvq_tokenizer.py \
      --epochs       $EPOCHS_RVQ \
      --batch-size   $BATCH_SIZE \
      --device       cuda \
      --num-workers  4
  "
  # Upload the trained checkpoint to S3 (so future runs reuse it)
  aws s3 sync "$RVQ_CKPT_DIR" "s3://$S3_BUCKET/checkpoints/rvq_tokenizer" \
    --storage-class STANDARD_IA
fi

# Re-fix ownership: the mkdir + S3 cp above ran as root, so checkpoints/
# is now owned by root. The SSM trainer below runs as the ubuntu user via
# `sudo -u ubuntu` and needs write access to create motion_ssm/<run_id>/.
chown -R ubuntu:ubuntu "$REPO_DIR/checkpoints"

# -----------------------------------------------------------------------------
# Step 6: train MotionSSM (THE long step — ~6 hours on T4)
#
# This is the actual model: text -> motion. Predicts RVQ codebook indices.
# If a previous run was interrupted (spot reclaimed by AWS), there'll be a
# partial checkpoint in S3 — pick it up and resume from where we left off.
# -----------------------------------------------------------------------------
echo "=== MotionSSM training started: $(date) ==="
SSM_CKPT_BASE="$REPO_DIR/checkpoints/motion_ssm"

# Find the most recent SSM run dir in S3 (if any). Bash plumbing:
#   `aws s3 ls`    -> lists immediate dir contents
#   `grep PRE`     -> keep only "directory" entries (S3 marks them with PRE)
#   `awk '{print $2}'`  -> extract the dir name (column 2)
#   `sort | tail -1`    -> alphabetic sort, last = most recent (our IDs are
#                          timestamped like 20260507-145715, so alpha = chrono)
#   `tr -d '/'`    -> strip the trailing slash from the dir name
LATEST_S3_RUN=$(aws s3 ls "s3://$S3_BUCKET/checkpoints/motion_ssm/" 2>/dev/null | \
  grep PRE | awk '{print $2}' | sort | tail -1 | tr -d '/' || true)

if [ -n "$LATEST_S3_RUN" ]; then
  echo "Resuming from S3 run: $LATEST_S3_RUN"
  mkdir -p "$SSM_CKPT_BASE/$LATEST_S3_RUN"
  aws s3 sync "s3://$S3_BUCKET/checkpoints/motion_ssm/$LATEST_S3_RUN" \
              "$SSM_CKPT_BASE/$LATEST_S3_RUN"
  RESUME_FLAG="--resume latest"
else
  echo "Fresh SSM training (no prior run in S3)"
  RESUME_FLAG=""
fi

# Architecture flags chosen based on what we smoke-tested locally:
#   --use-sbert         use SentenceTransformer for text (frozen)
#   --bidirectional     fwd+bwd Mamba scan (Motion Mamba ECCV 2024 style)
#   --use-film          inject text at every layer (FiLM)
#   --gradient-checkpointing  trade compute for VRAM (lets us use bigger batches)
sudo -u ubuntu bash -c "
  source /opt/pytorch/bin/activate
  export WANDB_MODE=offline WANDB_SILENT=true
  cd $REPO_DIR
  python scripts/training/train_motion_ssm.py \
    --data-source              $DATA_SOURCE \
    --sources                  humanml3d \
    --use-sbert \
    --bidirectional \
    --use-film \
    --gradient-checkpointing \
    --d-model                  384 \
    --d-state                  64 \
    --n-layers                 6 \
    --max-motion-length        200 \
    --batch-size               $BATCH_SIZE \
    --lr                       1e-4 \
    --epochs                   $EPOCHS_SSM \
    --num-workers              4 \
    --checkpoint-dir           checkpoints/motion_ssm \
    --rvq-checkpoint           checkpoints/rvq_tokenizer/best_model.pt \
    --device                   cuda \
    $RESUME_FLAG
"
echo "=== MotionSSM training done: $(date) ==="

# -----------------------------------------------------------------------------
# Step 7: upload all checkpoints to S3 (so they survive instance termination)
# `--storage-class STANDARD_IA` = cheaper "infrequent access" tier
# (~50% cheaper than standard, fine for things you read once per week)
# -----------------------------------------------------------------------------
echo "--- Uploading checkpoints to S3 ---"
aws s3 sync "$REPO_DIR/checkpoints" "s3://$S3_BUCKET/checkpoints" \
  --storage-class STANDARD_IA
echo "Upload complete"

# -----------------------------------------------------------------------------
# Step 8: self-terminate this EC2 (stops the per-second billing)
#
# 169.254.169.254 = AWS instance metadata service. From inside any EC2 you can
# query it for info about yourself. Here we ask "what's my own instance ID?"
# then pass it to the terminate-instances API call.
#
# The IAM role attached to this EC2 (defined in main.tf) gives us permission
# to terminate instances tagged Project=dissertation. Without that policy,
# this call would fail with AccessDenied.
# -----------------------------------------------------------------------------
echo "Training complete -- terminating instance to stop billing"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "=== All done: $(date) ==="
