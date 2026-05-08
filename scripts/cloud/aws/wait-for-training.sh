#!/usr/bin/env bash
# =============================================================================
# wait-for-training.sh — block until MotionSSM training actually starts
# =============================================================================
#
# Usage:   ./wait-for-training.sh <ec2-public-ip>
# Example: ./wait-for-training.sh 108.131.96.144
#
# Polls /var/log/user-data.log on the EC2 every 30 sec and exits when:
#   - "MotionSSM training started" appears  -> exit 0 (success)
#   - any failure marker appears            -> exit 1 (failure)
# =============================================================================

set -eu

IP="$1"
KEY="$HOME/.ssh/dissertation-eu-west-1.pem"

# What we're waiting to see (good signs and bad signs)
# GOOD: an actual epoch line like "epoch=1/200" — not just the launch echo
GOOD="epoch=[0-9]+/[0-9]+"
BAD="Traceback|unbound variable|fatal:|No such file|FAILED|FileNotFoundError"

# SSH options reused on every poll
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"

echo "Polling /var/log/user-data.log on $IP every 30s..."

while true; do
  # Pull the last 200 lines of the boot log
  LOG=$(ssh $SSH_OPTS -i "$KEY" "ubuntu@$IP" "sudo tail -200 /var/log/user-data.log" 2>/dev/null || true)

  # Did training actually start?
  if echo "$LOG" | grep -qE "$GOOD"; then
    echo "TRAINING STARTED"
    echo "$LOG" | grep -E "$GOOD"
    exit 0
  fi

  # Did anything obvious go wrong?
  if echo "$LOG" | grep -qE "$BAD"; then
    echo "FAILED — last 25 lines of boot log:"
    echo "$LOG" | tail -25
    exit 1
  fi

  # Otherwise, keep waiting
  sleep 30
done
