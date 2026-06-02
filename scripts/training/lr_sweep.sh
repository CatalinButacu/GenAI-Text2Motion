#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

RVQ_CK="checkpoints/rvq_tokenizer/20260517-233216/best_model.pt"
COMMON=(
    --data-source unified --sources interx_paired
    --num-actors 2 --cross-actor-attention
    --max-samples 1500 --epochs 20 --batch-size 8
    --weight-decay 0.1 --d-model 256 --d-state 64 --n-layers 4
    --use-sbert --bidirectional --gradient-checkpointing --use-film
    --max-motion-length 200 --velocity-loss-weight 0.5
    --separation-loss-weight 1.0 --device cuda
    --rvq-checkpoint "$RVQ_CK"
)

OUT="lr_sweep_index.tsv"
echo -e "arch\tlr\trun_dir\tbest_val\ttest_ce\ttest_top1" > "$OUT"

run_cell() {
    local arch=$1 lr=$2
    local stamp; stamp=$(date +%Y%m%d-%H%M%S)
    local out_log="ssm_${arch}_lr${lr}_${stamp}.out"
    local err_log="ssm_${arch}_lr${lr}_${stamp}.err"
    echo "=== [$arch lr=$lr] start $stamp ==="
    python scripts/training/train_motion_ssm.py \
        --arch "$arch" --lr "$lr" "${COMMON[@]}" \
        > "$out_log" 2> "$err_log"
    # Most recent run dir for this arch since stamp.
    local run_dir; run_dir=$(ls -td checkpoints/motion_ssm/${stamp:0:8}-* 2>/dev/null | head -1)
    local best_val; best_val=$(grep -oP 'best_val_loss=\K[0-9.]+' "$err_log" | tail -1)
    local test_ce; test_ce=$(grep -oP 'TEST  ce=\K[0-9.]+' "$err_log" | tail -1)
    local test_top1; test_top1=$(grep -oP 'top1=\K[0-9.]+' "$err_log" | tail -1)
    echo -e "$arch\t$lr\t$run_dir\t$best_val\t$test_ce\t$test_top1" >> "$OUT"
    echo "=== [$arch lr=$lr] done: best_val=$best_val test_ce=$test_ce test_top1=$test_top1 ==="
}

# Transformer arm (cheap, ~25 min/cell)
for lr in 5e-5 1e-4 2e-4 5e-4; do
    run_cell transformer "$lr"
done

# Mamba arm (~2.3 h/cell)
for lr in 5e-5 1e-4 2e-4; do
    run_cell mamba "$lr"
done

echo "=== ALL DONE ==="
cat "$OUT"
