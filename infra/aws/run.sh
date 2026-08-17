#!/bin/bash
cd /opt/thesis
export PYTHONPATH=/opt/thesis/src
B=thesis-t2m-913402647373
PY=/opt/pytorch/bin/python
mkdir -p outputs

sync_outputs() {
    while true; do
        aws s3 cp outputs "s3://$B/results" --recursive --region eu-north-1 >/dev/null 2>&1
        sleep 180
    done
}

sync_checkpoints() {
    while true; do
        aws s3 sync checkpoints "s3://$B/results/checkpoints" --region eu-north-1 >/dev/null 2>&1
        sleep 600
    done
}

sync_outputs &
SYNC=$!
sync_checkpoints &
CKPT_SYNC=$!

for bb in transformer mamba; do
    GATE="outputs/gates/${bb}_100m_overfit.json"
    mkdir -p outputs/gates
    [ -f "$GATE" ] || $PY -u -m text2motion.app.cli sanity-overfit \
        --config configs/generator/final100m_fsq8x1024.yaml \
        --backbone "$bb" \
        --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt \
        --out "$GATE" \
        > "outputs/gate_$bb.log" 2>&1
    $PY -u -m text2motion.app.cli train-generator \
        --config configs/generator/final100m_fsq8x1024.yaml \
        --backbone "$bb" \
        --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt \
        --overfit_gate "$GATE" \
        --epochs 60 \
        --batch_size 64 \
        --eval_every 5 \
        --cfg_scale 5.0 \
        --temperature 1.1 \
        > "outputs/$bb.log" 2>&1
    aws s3 cp "checkpoints/generator_$bb.pt" "s3://$B/results/" --region eu-north-1 2>/dev/null
    aws s3 cp "checkpoints/generator_${bb}_last.pt" "s3://$B/results/" --region eu-north-1 2>/dev/null
done

kill "$SYNC" "$CKPT_SYNC" 2>/dev/null
aws s3 cp outputs "s3://$B/results" --recursive --region eu-north-1
echo done > outputs/ALL_DONE
aws s3 cp outputs/ALL_DONE "s3://$B/results/ALL_DONE" --region eu-north-1
