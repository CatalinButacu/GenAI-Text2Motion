#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python}"
LOG="$REPO/outputs/canary"

cd "$REPO"
export PYTHONPATH="$REPO/src"
mkdir -p "$LOG" outputs

capability=$($PY -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))")
name=$($PY -c "import torch; print(torch.cuda.get_device_name(0))")
ARCH="${TORCH_CUDA_ARCH_LIST:-$capability}"
echo "gpu $name  sm_${capability//./}  requested TORCH_CUDA_ARCH_LIST=$ARCH" | tee "$LOG/gpu.log"

if [ "${capability%%.*}" -lt 7 ]; then
    echo "ABORT: mamba-ssm needs sm_70 or newer; this GPU is sm_${capability//./}." | tee -a "$LOG/gpu.log"
    echo "Azure: NV36ads_A10_v5 is sm_86.  Kaggle: pick T4 (sm_75), not P100 (sm_60)." | tee -a "$LOG/gpu.log"
    exit 1
fi

echo "=== gate 1: build the fused kernel against THIS torch ===" | tee "$LOG/gate1.log"
TORCH_CU=$($PY -c "import torch; print(torch.version.cuda)")
TORCH_VER=$($PY -c "import torch; print(torch.__version__)")
echo "torch $TORCH_VER  cuda $TORCH_CU" | tee -a "$LOG/gate1.log"

if [ -d "/usr/local/cuda-$TORCH_CU" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CU"
elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
else
    echo "no CUDA toolkit matching torch $TORCH_CU -- cannot build from source" | tee -a "$LOG/gate1.log"
    exit 1
fi
echo "CUDA_HOME=$CUDA_HOME" | tee -a "$LOG/gate1.log"

export PATH="$CUDA_HOME/bin:$PATH"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS="${MAX_JOBS:-2}"

$PY -m pip install -q --no-build-isolation --no-cache-dir causal-conv1d mamba-ssm pytest 2>&1 \
    | tail -5 | tee -a "$LOG/gate1.log"

echo "--- import check (this is where both AWS attempts died) ---" | tee -a "$LOG/gate1.log"
if ! $PY -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('import OK')" \
        2>&1 | tee -a "$LOG/gate1.log"; then
    echo "GATE 1 FAILED: kernel import error above (the ABI wall)." | tee -a "$LOG/gate1.log"
    exit 1
fi

echo "--- parity: fused kernel vs eager scan ---" | tee -a "$LOG/gate1.log"
$PY -m pytest tests/test_generator_upgrades.py::test_kernel_matches_eager_scan -q 2>&1 \
    | tee -a "$LOG/gate1.log"
echo "GATE 1 PASSED" | tee -a "$LOG/gate1.log"

echo "=== gate 2: training step, eager vs fused (interleaved A/B) ===" | tee "$LOG/gate2.log"
$PY -u scripts/eval/perf_probe.py --probe kernel --config "${CANARY_CONFIG:-configs/generator/gen_pilot_fsq8x1024.yaml}" \
    --batch_size "${CANARY_BATCH:-8}" --repeats 6 2>&1 | tee -a "$LOG/gate2.log"

echo "=== gate 3: rollout latency, eager vs fused ===" | tee "$LOG/gate3.log"
$PY -u scripts/eval/perf_probe.py --probe kernel --rollout_only \
    --config "${CANARY_CONFIG:-configs/generator/gen_pilot_fsq8x1024.yaml}" --steps 49 2>&1 \
    | tee -a "$LOG/gate3.log"

echo "CANARY COMPLETE -- logs in $LOG"
