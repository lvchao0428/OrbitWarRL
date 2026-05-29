#!/usr/bin/env bash
# Post-train f32 eval: replay @299/@599 vs v20 + gate table.
#
# Usage (5090 or local after ckpts exist):
#   bash scripts/run_f32_eval.sh
#   CKPT_DIR=ckpt_multi_action_v11_f32 bash scripts/run_f32_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

CKPT_DIR="${CKPT_DIR:-ckpt_multi_action_v11_f32}"
F31_CKPT="${F31_CKPT:-ckpt_multi_action_v11_f31/ckpt_000599.pkl}"
F29_CKPT="${F29_CKPT:-ckpt_multi_action_v11_f29/ckpt_000599.pkl}"

echo "=== f32 eval: $CKPT_DIR vs v20 ==="
for U in 099 299 599; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  TAG="v11_f32_u${U}"
  bash scripts/quick_replay.sh "$CKPT" "$TAG"
  if [ "$U" = "599" ]; then
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
echo "=== f31@599 baseline ==="
if [ -f "$F31_CKPT" ]; then
  bash scripts/quick_replay.sh "$F31_CKPT" "v11_f31_u599_baseline"
else
  echo "[skip] $F31_CKPT not found"
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
for TAG in v11_f32_u099 v11_f32_u299 v11_f32_u599; do
  SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
