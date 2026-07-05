#!/bin/bash
set -euo pipefail
cd /opt/thesis
export PYTHONPATH=/opt/thesis/src
B=thesis-t2m-913402647373
REGION=eu-north-1
PY=/opt/pytorch/bin/python
mkdir -p outputs

CFG=configs/generator/final100m_fsq8x1024.yaml
TOK=checkpoints/tokenizer/fsq_g8_v1024.pt
PACK=data/amass_tokens_fsq8x1024.npz
PRIOR=checkpoints/generator_mamba_100m_pretrained.pt   # matches the transformer prior's naming
FT=generator_mamba_100m.pt                             # best-by-val fine-tune; resume: *_last.pt

echo "=== gate: fused selective-scan kernel ===" | tee outputs/mamba100m_gate.log
WHEELDIR=/tmp/wheels
mkdir -p "$WHEELDIR"

have_cached_wheels() {
    aws s3 cp "s3://$B/wheels/" "$WHEELDIR/" --recursive --region "$REGION" 2>/dev/null || return 1
    ls "$WHEELDIR"/*.whl >/dev/null 2>&1
}

if have_cached_wheels; then
    echo "using cached kernel wheels from s3://$B/wheels" | tee -a outputs/mamba100m_gate.log
    $PY -m pip install -q "$WHEELDIR"/*.whl pytest
else
    echo "no cached wheels -> one-time source build (then cached)" | tee -a outputs/mamba100m_gate.log
    TORCH_CU=$($PY -c "import torch; print(torch.version.cuda)")
    if [ -d "/usr/local/cuda-$TORCH_CU" ]; then
        export CUDA_HOME="/usr/local/cuda-$TORCH_CU"
    elif [ -d /usr/local/cuda ]; then
        export CUDA_HOME=/usr/local/cuda
    else
        echo "no CUDA toolkit matching torch $TORCH_CU" | tee -a outputs/mamba100m_gate.log
        exit 1
    fi
    export PATH="$CUDA_HOME/bin:$PATH"
    export MAMBA_FORCE_BUILD=TRUE
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    export TORCH_CUDA_ARCH_LIST=8.6
    export MAX_JOBS=4
    echo "building against $CUDA_HOME (torch cuda $TORCH_CU)" | tee -a outputs/mamba100m_gate.log
    $PY -m pip wheel --no-deps --no-build-isolation -w "$WHEELDIR" causal-conv1d mamba-ssm 2>&1 | tail -3 | tee -a outputs/mamba100m_gate.log
    aws s3 cp "$WHEELDIR/" "s3://$B/wheels/" --recursive --exclude "*" --include "*.whl" --region "$REGION"
    $PY -m pip install -q "$WHEELDIR"/*.whl pytest
fi

$PY -m pytest tests/test_generator_upgrades.py::test_kernel_matches_eager_scan -q 2>&1 | tee -a outputs/mamba100m_gate.log

sync_outputs() {
    while true; do
        aws s3 cp outputs "s3://$B/results" --recursive --region "$REGION" >/dev/null 2>&1
        sleep 180
    done
}

sync_checkpoints() {
    while true; do
        aws s3 sync checkpoints "s3://$B/results/checkpoints" --region "$REGION" >/dev/null 2>&1
        sleep 600
    done
}

sync_outputs &
SYNC=$!
sync_checkpoints &
CKPT_SYNC=$!

for f in "$PRIOR" "${PRIOR%.pt}_last.pt" "$PRIOR.done" "checkpoints/$FT" "checkpoints/${FT%.pt}_last.pt"; do
    aws s3 cp "s3://$B/results/checkpoints/$(basename "$f")" "$(dirname "$f")/" --region "$REGION" 2>/dev/null || true
done

if [ ! -f "$PRIOR.done" ]; then
    echo "=== STAGE A: 100M mamba AMASS pretrain ===" | tee -a outputs/mamba100m.log
    $PY -u -m text2motion.train.train_pretrain \
        --config "$CFG" \
        --backbone mamba \
        --token_pack "$PACK" \
        --epochs 30 \
        --batch_size 32 \
        --out "$PRIOR" \
        --resume \
        >> outputs/mamba100m_pretrain.log 2>&1
else
    echo "STAGE A skipped: $PRIOR.done present" | tee -a outputs/mamba100m.log
fi

echo "=== STAGE B: 100M mamba fine-tune (from prior) ===" | tee -a outputs/mamba100m.log
$PY -u -m text2motion.train.train_generator \
    --config "$CFG" \
    --backbone mamba \
    --tokenizer_ckpt "$TOK" \
    --init_ckpt "$PRIOR" \
    --ckpt_name "$FT" \
    --epochs 60 \
    --batch_size 64 \
    --eval_every 5 \
    --cfg_scale 5.0 \
    --temperature 1.1 \
    --resume \
    >> outputs/mamba100m.log 2>&1

aws s3 cp "checkpoints/$FT" "s3://$B/results/" --region "$REGION" 2>/dev/null || true
aws s3 cp "checkpoints/${FT%.pt}_last.pt" "s3://$B/results/" --region "$REGION" 2>/dev/null || true
kill "$SYNC" "$CKPT_SYNC" 2>/dev/null || true
aws s3 cp outputs "s3://$B/results" --recursive --region "$REGION"
echo done > outputs/MAMBA100M_DONE
aws s3 cp outputs/MAMBA100M_DONE "s3://$B/results/MAMBA100M_DONE" --region "$REGION"
