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

# EAGER training (config use_kernel: false): the fused mamba-ssm kernel fails ABI-link on this DLAMI
# (torch 2.7 / cu128, undefined symbol c10_cuda_check_implementation). The eager parallel-scan trains
# the identical weights; the kernel only affects inference latency, benchmarked separately. So there
# is no kernel-build gate here -- training goes busy within minutes of boot.

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
    echo "=== STAGE A: 100M mamba AMASS pretrain (eager) ===" | tee -a outputs/mamba100m.log
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

echo "=== STAGE B: 100M mamba fine-tune from prior (eager) ===" | tee -a outputs/mamba100m.log
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
