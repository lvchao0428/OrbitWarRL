#!/usr/bin/env bash
# Evaluate f43 checkpoints vs v20.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

echo "=== f43 eval vs v20 ==="

for U in 49 99 149 199 249 299 349 399 449 499; do
  CKPT="ckpt_multi_action_v11_f43/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  bash scripts/quick_replay.sh "$CKPT" "v11_f43_u${U}"
done

echo ""
echo "=== gate summaries ==="
for U in 49 99 149 199 249 299 349 399 449 499; do
  SUM="logs/replay_analyze/v11_f43_u${U}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
