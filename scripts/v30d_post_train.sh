#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python3}" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1
CKPT_DIR="ckpt_multi_action_v30d_opening"
mkdir -p logs/replay_analyze logs/replay_html
U3999=3999
CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U3999").pkl"
[ -f "$CKPT" ] || CKPT="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U3999").pkl"
[ -f "$CKPT" ] || CKPT="$(ls -t "$CKPT_DIR"/ckpt_*.pkl 2>/dev/null | head -1)"
if [ -n "${CKPT:-}" ] && [ -f "$CKPT" ]; then
  TAG="v30d_u$(basename "$CKPT" .pkl | sed 's/ckpt_//')"
  bash scripts/quick_replay.sh "$CKPT" "$TAG"
  OUT="logs/replay_html/${TAG}_s0"
  SUB="submission_rl_${TAG}.py"
  if [ -f "$SUB" ] && [ ! -f "$OUT/replay.html" ]; then
    "$PYTHON" -m orbit_wars_rl.scripts.replay_html \
      --agent-a "$SUB" --agent-b submission_v20_0513.py --seed 0 --out-dir "$OUT"
  fi
fi
echo "=== v30d post-train done ==="
