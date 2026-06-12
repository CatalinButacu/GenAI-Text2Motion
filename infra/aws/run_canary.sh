#!/bin/bash
# Phase-2 CANARY (g5.xlarge, ~30-60 min, ~$2): validate before the paid 100M run. Fail-loud:
# any gate failing exits nonzero, training never starts, the idle watchdog reclaims the box.
#   gate 1: mamba-ssm + causal-conv1d install and the kernel parity test passes on the A10G
#   gate 2: 5 epochs per backbone at configs/final100m.yaml complete; wall-time per epoch logged
#           (abort the full run if it projects > 20 h)
set -euo pipefail
cd /opt/thesis
export PYTHONPATH=/opt/thesis/src
B=thesis-t2m-913402647373
PY=/opt/pytorch/bin/python
mkdir -p outputs

( while true; do aws s3 cp outputs "s3://$B/results/canary" --recursive --region eu-north-1 >/dev/null 2>&1; sleep 120; done ) &
SYNC=$!

echo "=== gate 1: fused kernel ===" | tee outputs/canary_gate1.log
$PY -m pip install -q --no-build-isolation causal-conv1d mamba-ssm pytest 2>&1 | tail -2 | tee -a outputs/canary_gate1.log
$PY -m pytest tests/test_generator_upgrades.py::test_kernel_matches_eager_scan -q 2>&1 | tee -a outputs/canary_gate1.log

echo "=== gate 2: 5-epoch twins at 100M ===" | tee outputs/canary_timing.log
for bb in transformer mamba; do
  start=$(date +%s)
  $PY -u -m text2motion.train.train_generator --config configs/final100m.yaml --backbone "$bb" \
    --epochs 5 --batch_size 64 --eval_every 5 --cfg_scale 5.0 --temperature 1.1 \
    > "outputs/canary_$bb.log" 2>&1
  echo "$bb total_seconds $(( $(date +%s) - start ))" | tee -a outputs/canary_timing.log
done

kill $SYNC 2>/dev/null || true
aws s3 cp outputs "s3://$B/results/canary" --recursive --region eu-north-1
echo done > outputs/CANARY_DONE
aws s3 cp outputs/CANARY_DONE "s3://$B/results/canary/CANARY_DONE" --region eu-north-1
