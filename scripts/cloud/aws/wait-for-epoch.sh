#!/usr/bin/env bash
# =============================================================================
# wait-for-epoch.sh — block until a specific epoch completes on the EC2
# =============================================================================
#
# Usage:   ./wait-for-epoch.sh <ip> <epoch_number>
# Example: ./wait-for-epoch.sh 3.248.229.130 1
#
# Polls /home/ubuntu/training.log (the manually-started training) every 60s.
# Exits when the completion line "epoch=N/M" is found (success) or when a
# Traceback/Error appears (failure).
# =============================================================================

set -eu

IP="$1"
N="$2"
KEY="$HOME/.ssh/dissertation-eu-west-1.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"

echo "Waiting for epoch $N to complete on $IP (polling every 60s)..."

while true; do
  # Look for the epoch-completion log line: "epoch=N/M train=... val_ce=..."
  done_line=$(ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" \
    "grep -E 'epoch=${N}/[0-9]+ train=' /home/ubuntu/training.log 2>/dev/null | head -1" \
    2>/dev/null || true)

  if [ -n "$done_line" ]; then
    echo "EPOCH $N COMPLETE"
    echo "$done_line"
    exit 0
  fi

  # Look for any error
  err=$(ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" \
    "grep -E 'Traceback|FileNotFoundError|RuntimeError|out of memory' /home/ubuntu/training.log 2>/dev/null | head -1" \
    2>/dev/null || true)

  if [ -n "$err" ]; then
    echo "FAILED:"
    echo "$err"
    ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" "tail -25 /home/ubuntu/training.log" 2>/dev/null
    exit 1
  fi

  sleep 60
done
