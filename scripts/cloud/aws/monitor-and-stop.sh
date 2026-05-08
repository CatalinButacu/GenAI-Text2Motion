#!/usr/bin/env bash
# =============================================================================
# monitor-and-stop.sh — watch training, stop at target epoch, sync + terminate
# =============================================================================
#
# Usage:   ./monitor-and-stop.sh <ip> <target_epoch>
# Example: ./monitor-and-stop.sh 3.248.229.130 50
#
# Polls /home/ubuntu/training.log every 2 min. Emits one stdout line per new
# completed epoch (so caller can show progress). When target_epoch is reached:
#   1. SIGTERM the training python process
#   2. aws s3 sync checkpoints/ to S3
#   3. terminate the EC2 instance (stops billing)
# =============================================================================

set -eu

IP="$1"
TARGET="$2"
KEY="$HOME/.ssh/dissertation-eu-west-1.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"
S3_BUCKET="dissertation-motion-cache-91340264"

last_seen=0

while true; do
  # What's the highest completed epoch in the log right now?
  current=$(ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" \
    "grep -oE 'epoch=[0-9]+/' /home/ubuntu/training.log 2>/dev/null \
     | grep -oE '[0-9]+' | sort -n | tail -1" 2>/dev/null || echo "")

  # Did training crash?
  err=$(ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" \
    "grep -E 'Traceback|FileNotFoundError|RuntimeError|out of memory' /home/ubuntu/training.log 2>/dev/null | head -1" \
    2>/dev/null || echo "")

  if [ -n "$err" ]; then
    echo "TRAINING CRASHED: $err"
    exit 1
  fi

  # Got a new epoch?
  if [ -n "$current" ] && [ "$current" -gt "$last_seen" ]; then
    line=$(ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" \
      "grep 'epoch=$current/' /home/ubuntu/training.log | tail -1" 2>/dev/null)
    echo "epoch $current done: $line"
    last_seen=$current

    # Hit our target?
    if [ "$current" -ge "$TARGET" ]; then
      echo "Target epoch $TARGET reached — stopping training, syncing checkpoints, terminating EC2..."

      ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" "
        pkill -SIGTERM -f train_motion_ssm.py 2>/dev/null || true
        sleep 5
        aws s3 sync /home/ubuntu/repo/checkpoints s3://$S3_BUCKET/checkpoints --storage-class STANDARD_IA
        IID=\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
        aws ec2 terminate-instances --instance-ids \$IID --region eu-west-1
      "

      echo "DONE — checkpoints in s3://$S3_BUCKET/checkpoints, EC2 terminating"
      exit 0
    fi
  fi

  sleep 120
done
