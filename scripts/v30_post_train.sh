#!/usr/bin/env bash
# v30 post-train: h2h + seed=0 HTML
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python3}" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1
CKPT_DIR="ckpt_multi_action_v30_econ"
mkdir -p logs/replay_analyze logs/replay_html logs/v30_h2h
U3999=3999
CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U3999").pkl"
[ -f "$CKPT" ] || CKPT="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U3999").pkl"
if [ -f "$CKPT" ]; then
  bash scripts/quick_replay.sh "$CKPT" v30_u3999
  OUT="logs/replay_html/v30_u3999_s0"
  SUB="submission_rl_v30_u3999.py"
  if [ -f "$SUB" ] && [ ! -f "$OUT/replay.html" ]; then
    "$PYTHON" -m orbit_wars_rl.scripts.replay_html \
      --agent-a "$SUB" --agent-b submission_v20_0513.py --seed 0 --out-dir "$OUT"
  fi
fi
echo "=== v30 post-train done ==="
