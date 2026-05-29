#!/usr/bin/env bash
# Post-train f31 eval: replay @299/@599 vs v20 + gate table.
#
# Usage (5090 or local after ckpts exist):
#   bash scripts/run_f31_eval.sh
#   CKPT_DIR=ckpt_multi_action_v11_f31 bash scripts/run_f31_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

CKPT_DIR="${CKPT_DIR:-ckpt_multi_action_v11_f31}"
F29_CKPT="${F29_CKPT:-ckpt_multi_action_v11_f29/ckpt_000599.pkl}"

echo "=== f31 eval: $CKPT_DIR vs v20 ==="
for U in 299 599; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  bash scripts/quick_replay.sh "$CKPT" "v11_f31_u${U}"
done

echo ""
echo "=== f29@599 baseline (comparison) ==="
if [ -f "$F29_CKPT" ]; then
  bash scripts/quick_replay.sh "$F29_CKPT" "v11_f29_u599_baseline"
else
  echo "[skip] $F29_CKPT not found"
fi

echo ""
echo "=== seed0 HTML replay (manual) ==="
echo "  python -m orbit_wars_rl.scripts.replay_dump --submission submission_rl_v11_f31_u599.py ..."
echo "  See scripts/quick_replay.sh output for submission path."
