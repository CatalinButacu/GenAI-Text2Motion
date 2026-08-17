#!/bin/bash
set -euo pipefail

OUT="${1:-dist/bundle}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT"

CODE="$OUT/thesis_code.tar.gz"
tar --force-local -czf "$CODE" \
    src configs scripts/eval pyproject.toml uv.lock \
    tests/test_generator_upgrades.py tests/test_generator.py tests/test_streaming.py
echo "code bundle  $CODE  $(du -h "$CODE" | cut -f1)"

DATA="$OUT/thesis_data"
mkdir -p "$DATA"
for item in \
    "data/HumanML3D_official" \
    "data/official_evaluator/extracted" \
    "data/eval_stats" \
    "data/t2m_glove" \
    "checkpoints/tokenizer/fsq_g8_v1024.pt"
do
    if [ ! -e "$item" ]; then
        echo "MISSING: $item -- the training bundle is incomplete" >&2
        exit 1
    fi
    printf '%-46s %s\n' "$item" "$(du -sh "$item" | cut -f1)"
done

echo
echo "code bundle is ready: azcopy it to Blob (Azure), or attach it to a notebook (Kaggle)."
echo "For the data, upload these paths preserving the layout (one Blob container, or one private Kaggle Dataset):"
echo "  data/HumanML3D_official/  data/official_evaluator/extracted/"
echo "  data/eval_stats/  data/t2m_glove/  checkpoints/tokenizer/fsq_g8_v1024.pt"
echo
echo "Phase 1 (the kernel canary) needs ONLY the code bundle -- the probes use random weights."
