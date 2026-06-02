#!/usr/bin/env bash
# Evaluate f40 BC/buffer checkpoints vs v20.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

echo "=== f40 eval vs v20 ==="

BC_CKPT="${BC_CKPT:-ckpt_bc_f40/ckpt_final.pkl}"
if [ -f "$BC_CKPT" ]; then
  bash scripts/quick_replay.sh "$BC_CKPT" "v11_f40_bc_seed"
else
  echo "[skip] $BC_CKPT not found"
fi

for U in 99 199 299 499 999; do
  CKPT="ckpt_multi_action_v11_f40_buffer/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  bash scripts/quick_replay.sh "$CKPT" "v11_f40_buffer_u${U}"
done

echo ""
echo "=== gate summaries ==="
for TAG in v11_f40_bc_seed v11_f40_buffer_u99 v11_f40_buffer_u199 \
           v11_f40_buffer_u299 v11_f40_buffer_u499 v11_f40_buffer_u999; do
  SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
