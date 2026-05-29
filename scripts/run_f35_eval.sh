#!/usr/bin/env bash
# Post-train f35 eval: replay @199/@399/@599/@799 vs v20 + f29 baseline.
#
# Watch the src-quality metrics in the summary:
#   - one_ship_rate should DROP well below f33's 31%
#   - min_game_spf should rise above f33's 1.13
#   - spf/garr should rise without the bin7-only inflation
#
# Usage:
#   bash scripts/run_f35_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

CKPT_DIR="${CKPT_DIR:-ckpt_multi_action_v11_f35}"
F29_CKPT="${F29_CKPT:-ckpt_multi_action_v11_f29/ckpt_000599.pkl}"

echo "=== f35 eval: $CKPT_DIR vs v20 ==="
bash scripts/check_training_health.sh logs/v11_f35.log || true

for U in 199 399 599 799; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  TAG="v11_f35_u${U}"
  bash scripts/quick_replay.sh "$CKPT" "$TAG"
  if [ "$U" = "399" ] || [ "$U" = "799" ]; then
    SUB="submission_rl_${TAG}.py"
    HTML_DIR="logs/replay_html/${TAG}_seed0"
    if [ -f "$SUB" ]; then
      echo ""
      echo "=== seed0 HTML: $TAG vs v20 ==="
      PY="${PYTHON:-python3}"
      "$PY" -m orbit_wars_rl.scripts.replay_html \
        --agent-a "$SUB" \
        --agent-b submission_v20_0513.py \
        --seed 0 \
        --out-dir "$HTML_DIR"
      echo "  open $HTML_DIR/replay.html"
    fi
  fi
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
for TAG in v11_f35_u199 v11_f35_u399 v11_f35_u599 v11_f35_u799 v11_f29_u599_baseline; do
  SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
