#!/usr/bin/env bash
# v30b post-train: h2h + seed=0 HTML
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python3}" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1
CKPT_DIR="ckpt_multi_action_v30b_econ"
mkdir -p logs/replay_analyze logs/replay_html logs/v30b_h2h
U1599=1599
CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U1599").pkl"
[ -f "$CKPT" ] || CKPT="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U1599").pkl"
[ -f "$CKPT" ] || CKPT="$(ls -t "$CKPT_DIR"/ckpt_*.pkl 2>/dev/null | head -1)"
if [ -n "${CKPT:-}" ] && [ -f "$CKPT" ]; then
  TAG="v30b_u$(basename "$CKPT" .pkl | sed 's/ckpt_//')"
  bash scripts/quick_replay.sh "$CKPT" "$TAG"
  OUT="logs/replay_html/${TAG}_s0"
  SUB="submission_rl_${TAG}.py"
  if [ -f "$SUB" ] && [ ! -f "$OUT/replay.html" ]; then
    "$PYTHON" -m orbit_wars_rl.scripts.replay_html \
      --agent-a "$SUB" --agent-b submission_v20_0513.py --seed 0 --out-dir "$OUT"
  fi
fi
echo "=== v30b post-train done ==="
