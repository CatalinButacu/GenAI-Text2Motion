#!/bin/bash
# Mamba-only 100M twin -- the transformer 100M twin is already done (val FID 1.63). This trains the
# missing Mamba half through the SAME two-stage pipeline the transformer used: an AMASS motion-prior
# PRETRAIN (from-scratch / random init, 30 ep bs32 -> generator_transformer_100m_pretrained.pt used
# CE 1.54) THEN a captioned FINE-TUNE initialised from that prior. Fused CUDA selective-scan kernel
# ON, spot-survivable (resume from the last S3 checkpoint at either stage).
# ~half the cost of run.sh (one backbone): g5.xlarge, ~10-14 h, ~$14-18.
#
# Run AFTER run_canary.sh passes (kernel parity + timing gate). On the box:  bash run_mamba100m.sh
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

# --- gate: install + BUILD the fused kernel against the box torch, verify it matches the eager path.
# The plan forbids training mamba on cloud WITHOUT the kernel (14x slower). Self-contained so this is
# safe even if the canary was skipped; cached after the canary -> a fast re-check then. (== canary gate 1.)
echo "=== gate: fused selective-scan kernel ===" | tee outputs/mamba100m_gate.log
if [ ! -d /usr/local/cuda-13.0 ]; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-0
fi
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
$PY -m pip install -q --no-build-isolation --no-cache-dir causal-conv1d mamba-ssm pytest 2>&1 | tail -2 | tee -a outputs/mamba100m_gate.log
$PY -m pytest tests/test_generator_upgrades.py::test_kernel_matches_eager_scan -q 2>&1 | tee -a outputs/mamba100m_gate.log

# --- sync loops: a spot-kill or the idle/lifetime watchdog must never lose weights or logs ---
( while true; do aws s3 cp outputs "s3://$B/results" --recursive --region $REGION >/dev/null 2>&1; sleep 180; done ) &
SYNC=$!
( while true; do aws s3 sync checkpoints "s3://$B/results/checkpoints" --region $REGION >/dev/null 2>&1; sleep 600; done ) &
CKPT_SYNC=$!

# --- pull any prior-run state from S3 so a relaunched spot box continues instead of restarting ---
for f in "$PRIOR" "${PRIOR%.pt}_last.pt" "$PRIOR.done" "checkpoints/$FT" "checkpoints/${FT%.pt}_last.pt"; do
  aws s3 cp "s3://$B/results/checkpoints/$(basename "$f")" "$(dirname "$f")/" --region $REGION 2>/dev/null || true
done

# --- STAGE A: AMASS motion prior, from random init, matched to the transformer (30 ep, bs32). Skipped
# once its .done marker exists (so a resumed box jumps straight to fine-tune). --resume continues a
# half-finished pretrain from its own *_last.pt.
if [ ! -f "$PRIOR.done" ]; then
  echo "=== STAGE A: 100M mamba AMASS pretrain ===" | tee -a outputs/mamba100m.log
  $PY -u -m text2motion.train.train_pretrain --config "$CFG" --backbone mamba \
    --token_pack "$PACK" --epochs 30 --batch_size 32 --out "$PRIOR" --resume \
    >> outputs/mamba100m_pretrain.log 2>&1
else
  echo "STAGE A skipped: $PRIOR.done present" | tee -a outputs/mamba100m.log
fi

# --- STAGE B: captioned fine-tune from the prior, matched recipe to the transformer twin (60 ep,
# bs64, val-select, in-train eval CFG 5.0 / temp 1.1), kernel ON (config use_kernel: true).
# --init_ckpt seeds the weights from the prior; --resume overrides with *_last.pt if a killed
# fine-tune left one, so passing both is correct on a fresh start AND on a spot relaunch.
echo "=== STAGE B: 100M mamba fine-tune (from prior) ===" | tee -a outputs/mamba100m.log
$PY -u -m text2motion.train.train_generator --config "$CFG" --backbone mamba \
  --tokenizer_ckpt "$TOK" --init_ckpt "$PRIOR" --ckpt_name "$FT" \
  --epochs 60 --batch_size 64 --eval_every 5 --cfg_scale 5.0 --temperature 1.1 --resume \
  >> outputs/mamba100m.log 2>&1

# --- final push + done marker (the idle watchdog reclaims the box once the GPU goes idle) ---
aws s3 cp "checkpoints/$FT" "s3://$B/results/" --region $REGION 2>/dev/null || true
aws s3 cp "checkpoints/${FT%.pt}_last.pt" "s3://$B/results/" --region $REGION 2>/dev/null || true
kill $SYNC $CKPT_SYNC 2>/dev/null || true
aws s3 cp outputs "s3://$B/results" --recursive --region $REGION
echo done > outputs/MAMBA100M_DONE
aws s3 cp outputs/MAMBA100M_DONE "s3://$B/results/MAMBA100M_DONE" --region $REGION
