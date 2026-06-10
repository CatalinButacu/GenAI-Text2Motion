#!/bin/bash
# Remote training driver (runs on the EC2 box under SSM). Both twins fresh, 150 epochs, bs 64;
# logs synced to S3 every 3 min; checkpoints + logs pushed at the end. When it finishes the GPU goes
# idle and the user_data idle-watchdog auto-terminates the instance.
cd /opt/thesis
export PYTHONPATH=/opt/thesis/src
B=thesis-t2m-913402647373
PY=/opt/pytorch/bin/python
mkdir -p outputs

# Sync BOTH logs and checkpoints while training runs: a lifetime/idle kill must never lose weights
# (the 2026-06-09 run nearly lost all mamba epochs — only outputs/ was synced).
( while true; do aws s3 cp outputs s3://$B/results --recursive --region eu-north-1 >/dev/null 2>&1; sleep 180; done ) &
SYNC=$!
( while true; do aws s3 sync checkpoints "s3://$B/results/checkpoints" --region eu-north-1 >/dev/null 2>&1; sleep 600; done ) &
CKPT_SYNC=$!

# 60 epochs, not 150: the 2026-06-09 transformer peaked at ep ~20-40 then degraded (overfit);
# the cosine schedule must decay inside the useful window.
for bb in transformer mamba; do
  $PY -u -m text2motion.train.train_generator --config configs/aws.yaml --backbone "$bb" --epochs 60 --batch_size 64 > "outputs/$bb.log" 2>&1
  aws s3 cp "checkpoints/generator_$bb.pt" "s3://$B/results/" --region eu-north-1 2>/dev/null
  aws s3 cp "checkpoints/generator_${bb}_last.pt" "s3://$B/results/" --region eu-north-1 2>/dev/null
done

kill $SYNC $CKPT_SYNC 2>/dev/null
aws s3 cp outputs s3://$B/results --recursive --region eu-north-1
echo done > outputs/ALL_DONE
aws s3 cp outputs/ALL_DONE s3://$B/results/ALL_DONE --region eu-north-1
