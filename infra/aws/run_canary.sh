#!/bin/bash
set -euo pipefail
cd /opt/thesis
export PYTHONPATH=/opt/thesis/src
B=thesis-t2m-913402647373
PY=/opt/pytorch/bin/python
mkdir -p outputs

sync_canary() {
    while true; do
        aws s3 cp outputs "s3://$B/results/canary" --recursive --region eu-north-1 >/dev/null 2>&1
        sleep 120
    done
}

sync_canary &
SYNC=$!

echo "=== gate 1: fused kernel ===" | tee outputs/canary_gate1.log
TORCH_CU=$($PY -c "import torch; print(torch.version.cuda)")
if [ -d "/usr/local/cuda-$TORCH_CU" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CU"
elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
else
    echo "no CUDA toolkit matching torch $TORCH_CU" | tee -a outputs/canary_gate1.log
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
$PY -m pip install -q --no-build-isolation --no-cache-dir causal-conv1d mamba-ssm pytest 2>&1 | tail -2 | tee -a outputs/canary_gate1.log
$PY -m pytest tests/test_generator_upgrades.py::test_kernel_matches_eager_scan -q 2>&1 | tee -a outputs/canary_gate1.log

echo "=== gate 2: 5-epoch twins at 100M ===" | tee outputs/canary_timing.log
for bb in transformer mamba; do
    start=$(date +%s)
    $PY -u -m text2motion.train.train_generator \
        --config configs/generator/final100m_fsq8x1024.yaml \
        --backbone "$bb" \
        --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt \
        --epochs 5 \
        --batch_size 64 \
        --eval_every 5 \
        --cfg_scale 5.0 \
        --temperature 1.1 \
        > "outputs/canary_$bb.log" 2>&1
    elapsed=$(( $(date +%s) - start ))
    echo "$bb total_seconds $elapsed" | tee -a outputs/canary_timing.log
done

kill "$SYNC" 2>/dev/null || true
aws s3 cp outputs "s3://$B/results/canary" --recursive --region eu-north-1
echo done > outputs/CANARY_DONE
aws s3 cp outputs/CANARY_DONE "s3://$B/results/canary/CANARY_DONE" --region eu-north-1
