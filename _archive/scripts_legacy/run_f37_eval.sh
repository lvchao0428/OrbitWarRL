#!/usr/bin/env bash
# Post-train f37 eval: replay checkpoints vs v20.
#
# Key checkpoints: @199/@499/@999/@1999/@2999

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

CKPT_DIR="${CKPT_DIR:-ckpt_multi_action_v11_f37}"

echo "=== f37 eval: $CKPT_DIR vs v20 ==="
echo "=== training log health ==="
bash scripts/check_training_health.sh logs/v11_f37.log || true

for U in 199 499 999 1999 2999; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  TAG="v11_f37_u${U}"
  bash scripts/quick_replay.sh "$CKPT" "$TAG"
done

echo ""
echo "=== gate summaries ==="
for TAG in v11_f37_u199 v11_f37_u499 v11_f37_u999 v11_f37_u1999 v11_f37_u2999; do
  SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
