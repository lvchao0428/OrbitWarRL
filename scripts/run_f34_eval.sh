#!/usr/bin/env bash
# Post-train f34 eval: replay @399/@599/@799 vs v20 + f33/f29 baselines.
#
# Usage:
#   bash scripts/run_f34_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

CKPT_DIR="${CKPT_DIR:-ckpt_multi_action_v11_f34}"
F33_CKPT="${F33_CKPT:-ckpt_multi_action_v11_f33/ckpt_000399.pkl}"
F29_CKPT="${F29_CKPT:-ckpt_multi_action_v11_f29/ckpt_000599.pkl}"

echo "=== f34 eval: $CKPT_DIR vs v20 ==="
bash scripts/check_training_health.sh logs/v11_f34.log || true

for U in 399 599 799; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  TAG="v11_f34_u${U}"
  bash scripts/quick_replay.sh "$CKPT" "$TAG"
done

echo ""
echo "=== f33@399 pre-collapse baseline ==="
if [ -f "$F33_CKPT" ]; then
  bash scripts/quick_replay.sh "$F33_CKPT" "v11_f33_u399"
else
  echo "[skip] $F33_CKPT not found"
fi

echo ""
echo "=== f29@599 baseline ==="
if [ -f "$F29_CKPT" ]; then
  bash scripts/quick_replay.sh "$F29_CKPT" "v11_f29_u599_baseline"
else
  echo "[skip] $F29_CKPT not found"
fi

echo ""
echo "=== gate summaries ==="
for TAG in v11_f34_u399 v11_f34_u599 v11_f34_u799 v11_f33_u399 v11_f29_u599_baseline; do
  SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
