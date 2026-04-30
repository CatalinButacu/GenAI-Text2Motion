#!/bin/bash
# =============================================================================
# EC2 User Data — MotionSSM training (RVQ tokenizer → MotionSSM)
# =============================================================================
# Logs: /var/log/user-data.log  (standard AWS location)
# Steps:
#   1. Install Python deps
#   2. Clone repo
#   3. Download data cache from S3
#   4. Train RVQ tokenizer  (prerequisite)
#   5. Train MotionSSM      (uses frozen RVQ)
#   6. Upload checkpoints to S3
#   7. Self-terminate instance (stops billing)
# =============================================================================

set -euxo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== Boot started: $(date) ==="

REPO_DIR=/home/ubuntu/repo
S3_BUCKET="${s3_bucket}"
S3_CACHE_URI="${s3_cache_uri}"
REGION="${region}"
REPO_URL="${repo_url}"
DATA_SOURCE="${data_source}"
EPOCHS_RVQ="${epochs_rvq}"
EPOCHS_SSM="${epochs_ssm}"
BATCH_SIZE="${batch_size}"

# ---------------------------------------------------------------------------
# 1. Activate the pre-installed PyTorch conda env
# ---------------------------------------------------------------------------
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# ---------------------------------------------------------------------------
# 2. Install extra deps (sentence-transformers, spacy, wandb, beartype, joblib)
# ---------------------------------------------------------------------------
pip install --quiet \
  sentence-transformers \
  scipy \
  spacy \
  wandb \
  beartype \
  joblib

python -m spacy download en_core_web_sm --quiet

# ---------------------------------------------------------------------------
# 3. Clone repository
# ---------------------------------------------------------------------------
git clone --depth=1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 4. Download preprocessed training cache from S3
#    Cache holds the joblib-serialised unified motion buffer (~1-2 GB)
#    and the HumanML3D texts/splits so we skip 170 GB raw AMASS download.
# ---------------------------------------------------------------------------
CACHE_DIR="$REPO_DIR/data/.cache"
mkdir -p "$CACHE_DIR"
mkdir -p "$REPO_DIR/data/AMASS"     # trainer expects the directory to exist
mkdir -p "$REPO_DIR/data/humanml3d"

echo "--- Downloading data cache from S3: $(date) ---"
aws s3 cp "$S3_CACHE_URI" "$CACHE_DIR/" --recursive 2>/dev/null || \
  aws s3 cp "$S3_CACHE_URI" "$CACHE_DIR/$(basename $S3_CACHE_URI)"

# Also sync humanml3d texts + splits if stored under s3://<bucket>/data/humanml3d/
aws s3 sync "s3://$S3_BUCKET/data/humanml3d" "$REPO_DIR/data/humanml3d" || true

echo "Cache ready: $(du -sh $CACHE_DIR 2>/dev/null | cut -f1)"
chown -R ubuntu:ubuntu "$REPO_DIR"

# ---------------------------------------------------------------------------
# 5. Train RVQ tokenizer  (all subsequent training depends on this)
# ---------------------------------------------------------------------------
echo "=== RVQ training started: $(date) ==="
RVQ_CKPT_DIR="$REPO_DIR/checkpoints/rvq_tokenizer"

# Check S3 for an existing RVQ checkpoint to skip retraining
if aws s3 ls "s3://$S3_BUCKET/checkpoints/rvq_tokenizer/best_model.pt" > /dev/null 2>&1; then
  echo "Reusing existing RVQ checkpoint from S3"
  mkdir -p "$RVQ_CKPT_DIR"
  aws s3 cp "s3://$S3_BUCKET/checkpoints/rvq_tokenizer/best_model.pt" \
            "$RVQ_CKPT_DIR/best_model.pt"
else
  sudo -u ubuntu bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch
    export WANDB_MODE=offline WANDB_SILENT=true
    cd $REPO_DIR
    python scripts/training/train_rvq_tokenizer.py \
      --epochs     $EPOCHS_RVQ \
      --batch-size $BATCH_SIZE \
      --device     cuda \
      --num-workers 4
  "
  echo "=== RVQ training done: $(date) ==="
  aws s3 sync "$RVQ_CKPT_DIR" "s3://$S3_BUCKET/checkpoints/rvq_tokenizer" \
    --storage-class STANDARD_IA
fi

# ---------------------------------------------------------------------------
# 6. Train MotionSSM  (uses frozen RVQ from step 5)
# ---------------------------------------------------------------------------
echo "=== MotionSSM training started: $(date) ==="
SSM_CKPT_BASE="$REPO_DIR/checkpoints/motion_ssm"

# Resume from S3 if a prior run exists (spot interruption recovery)
LATEST_S3_RUN=$(aws s3 ls "s3://$S3_BUCKET/checkpoints/motion_ssm/" | \
  grep PRE | awk '{print $2}' | sort | tail -1 | tr -d '/')
if [ -n "$LATEST_S3_RUN" ]; then
  echo "Resuming: downloading run $LATEST_S3_RUN from S3"
  mkdir -p "$SSM_CKPT_BASE/$LATEST_S3_RUN"
  aws s3 sync "s3://$S3_BUCKET/checkpoints/motion_ssm/$LATEST_S3_RUN" \
              "$SSM_CKPT_BASE/$LATEST_S3_RUN"
  RESUME_FLAG="--resume latest"
else
  RESUME_FLAG=""
fi

sudo -u ubuntu bash -c "
  source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch
  export WANDB_MODE=offline WANDB_SILENT=true
  cd $REPO_DIR
  python scripts/training/train_motion_ssm.py \
    --data-source          $DATA_SOURCE \
    --use-sbert \
    --bidirectional \
    --use-film \
    --gradient-checkpointing \
    --d-model              384 \
    --d-state              64 \
    --n-layers             6 \
    --max-motion-length    200 \
    --batch-size           $BATCH_SIZE \
    --lr                   3e-4 \
    --epochs               $EPOCHS_SSM \
    --num-workers          4 \
    --checkpoint-dir       checkpoints/motion_ssm \
    --rvq-checkpoint       checkpoints/rvq_tokenizer/best_model.pt \
    --device               cuda \
    $RESUME_FLAG
"
echo "=== MotionSSM training done: $(date) ==="

# ---------------------------------------------------------------------------
# 7. Upload all checkpoints to S3
# ---------------------------------------------------------------------------
echo "--- Uploading checkpoints to S3: $(date) ---"
aws s3 sync "$REPO_DIR/checkpoints" "s3://$S3_BUCKET/checkpoints" \
  --storage-class STANDARD_IA
echo "Upload complete"

# ---------------------------------------------------------------------------
# 8. Self-terminate instance (stops billing)
# ---------------------------------------------------------------------------
echo "Training complete -- terminating instance to stop billing"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "=== All done: $(date) ==="
