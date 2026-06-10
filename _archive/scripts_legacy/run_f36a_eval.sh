#!/usr/bin/env bash
# Post-train f36a eval: replay dense early checkpoints vs v20.
#
# Priority is the Day9 short-run window: @149/@199/@249/@299/@349.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

CKPT_DIR="${CKPT_DIR:-ckpt_multi_action_v11_f36a}"
F29_CKPT="${F29_CKPT:-ckpt_multi_action_v11_f29/ckpt_000599.pkl}"

echo "=== f36a eval: $CKPT_DIR vs v20 ==="
echo "=== training log health ==="
bash scripts/check_training_health.sh logs/v11_f36a.log || true

for U in 149 199 249 299 349; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  TAG="v11_f36a_u${U}"
  EMIT_HARD_STOP_MIN_STEP=2 bash scripts/quick_replay.sh "$CKPT" "$TAG"
done

echo ""
echo "=== f29@599 baseline ==="
if [ -f "$F29_CKPT" ]; then
  bash scripts/quick_replay.sh "$F29_CKPT" "v11_f29_u599_baseline"
else
  echo "[skip] $F29_CKPT not found"
fi

echo ""
echo "=== gate summaries ==="
for TAG in v11_f36a_u149 v11_f36a_u199 v11_f36a_u249 v11_f36a_u299 v11_f36a_u349 v11_f29_u599_baseline; do
  SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
