#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python}"
EPOCHS="${EPOCHS:-100}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
BATCH="${BATCH:-64}"
PATIENCE="${PATIENCE:-6}"
EVAL_EVERY="${EVAL_EVERY:-5}"
CFG_SCALE="${CFG_SCALE:-6.0}"
MAX_HOURS="${MAX_HOURS:-14}"
STALL_MINUTES="${STALL_MINUTES:-45}"
SYNC_MINUTES="${SYNC_MINUTES:-10}"
BLOB="${BLOB:-}"

cd "$REPO"
export PYTHONPATH="$REPO/src"
mkdir -p artifacts/outputs/gates artifacts/logs/cli artifacts/logs/train artifacts/checkpoints/generator

FSQ_CONFIG=configs/generator/gen_pilot_fsq8x1024.yaml
RVQ_CONFIG=configs/generator/gen_pilot_rvq8x1024.yaml
FSQ_TOKENIZER=artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt
RVQ_TOKENIZER=artifacts/checkpoints/tokenizer/rvq_l8_1024.pt
FSQ_PACK=data/amass_tokens_fsq8x1024.npz
RVQ_PACK=data/amass_tokens_rvq8x1024.npz

preflight() {
    local missing=0
    for required in \
        "$FSQ_CONFIG" "$RVQ_CONFIG" \
        "$FSQ_TOKENIZER" "$RVQ_TOKENIZER" \
        "$FSQ_PACK" "$RVQ_PACK" \
        data/HumanML3D_official data/official_evaluator/extracted data/eval_stats data/t2m_glove
    do
        if [ ! -e "$required" ]
        then
            echo "MISSING: $required" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]
    then
        echo "preflight failed -- stage the payload before training" >&2
        exit 1
    fi
    $PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'"
    $PY -c "import torch; print('gpu', torch.cuda.get_device_name(0), 'sm', torch.cuda.get_device_capability())"

    if grep -q "use_kernel: true" "$FSQ_CONFIG"
    then
        if ! $PY -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null
        then
            echo "FAILED: $FSQ_CONFIG sets use_kernel: true but the fused mamba kernel does not import." >&2
            echo "Run infra/kernel_canary.sh first, or drop use_kernel from the config to train eager." >&2
            exit 1
        fi
        echo "fused mamba selective-scan kernel: OK"
    fi
}

record_kill() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $1  $2" >> artifacts/outputs/GUARD_KILL.txt
    echo "GUARD KILL: $1 -- $2" >&2
}

guarded() {
    local label="$1"
    shift
    local log="artifacts/outputs/logs/$label.log"
    local started=$SECONDS
    "$@" > "$log" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null
    do
        sleep 60
        if [ $(( SECONDS - started )) -ge $(( MAX_HOURS * 3600 )) ]
        then
            record_kill "$label" "max-hours budget ${MAX_HOURS}h exceeded"
            kill -9 "$pid" 2>/dev/null || true
            return 1
        fi
        if [ -z "$(find "$log" artifacts/outputs/runs -newermt "-${STALL_MINUTES} minutes" -print -quit 2>/dev/null)" ]
        then
            record_kill "$label" "no run-dir or log write in ${STALL_MINUTES}min"
            kill -9 "$pid" 2>/dev/null || true
            return 1
        fi
    done
    wait "$pid"
}

stage() {
    local label="$1"
    shift
    local marker="artifacts/outputs/logs/$label.done"
    if [ -f "$marker" ]
    then
        echo "skip $label (already complete)"
        return 0
    fi
    echo "=== $label ==="
    if ! guarded "$label" "$@"
    then
        echo "=== $label KILLED -- not marked complete; rerun this script to resume ===" >&2
        return 1
    fi
    touch "$marker"
    echo "=== $label complete ==="
}

train_pair() {
    local name="$1"
    local backbone="$2"
    local config="$3"
    local tokenizer="$4"
    local pack="$5"

    stage "pretrain_$name" \
        $PY -u -m text2motion.app.cli pretrain \
        --config "$config" --backbone "$backbone" --token_pack "$pack" \
        --epochs "$PRETRAIN_EPOCHS" --batch_size "$BATCH" \
        --out "artifacts/checkpoints/generator/pre_$name.pt" --resume

    stage "gate_$name" \
        $PY -u -m text2motion.app.cli sanity-overfit \
        --config "$config" --backbone "$backbone" --tokenizer_ckpt "$tokenizer" \
        --out "artifacts/outputs/gates/$name.json"

    stage "train_$name" \
        $PY -u -m text2motion.app.cli train-generator \
        --config "$config" --backbone "$backbone" --tokenizer_ckpt "$tokenizer" \
        --overfit_gate "artifacts/outputs/gates/$name.json" \
        --init_ckpt "artifacts/checkpoints/generator/pre_$name.pt" \
        --ckpt_name "generator_$name.pt" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --patience "$PATIENCE" \
        --eval_every "$EVAL_EVERY" --cfg_scale "$CFG_SCALE" --resume
}

preflight

if [ -n "$BLOB" ]
then
    (
        while sleep $(( SYNC_MINUTES * 60 ))
        do
            azcopy sync artifacts/checkpoints "$BLOB/checkpoints" --recursive > /dev/null 2>&1 || true
            azcopy sync artifacts/outputs "$BLOB/outputs" --recursive > /dev/null 2>&1 || true
        done
    ) &
    SYNC_PID=$!
    trap 'kill "$SYNC_PID" 2>/dev/null || true' EXIT
    echo "blob sync every ${SYNC_MINUTES}min -> $BLOB"
fi

train_pair transformer_fsq transformer "$FSQ_CONFIG" "$FSQ_TOKENIZER" "$FSQ_PACK"
train_pair mamba_fsq mamba "$FSQ_CONFIG" "$FSQ_TOKENIZER" "$FSQ_PACK"
train_pair transformer_rvq transformer "$RVQ_CONFIG" "$RVQ_TOKENIZER" "$RVQ_PACK"

if [ -n "$BLOB" ]
then
    azcopy sync artifacts/checkpoints "$BLOB/checkpoints" --recursive
    azcopy sync artifacts/outputs "$BLOB/outputs" --recursive
fi

echo "ALL TWINS COMPLETE -- best bundles in artifacts/checkpoints/generator/"
echo "final table: $PY -m text2motion.app.cli evaluate --split test"
