#!/usr/bin/env bash
# v30b opening BC warmstart (econ head only, ROI teacher)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export ORBITWARS_SKIP_PARITY=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi
INIT="./ckpt_multi_action_v29_aim/ckpt_003999.pkl"
[ -f "$INIT" ] || INIT="./ckpt_multi_action_v29_aim/ckpt_latest.pkl"
REPLAY="${REPLAY:-logs/replay_html/v29_u3999_s0/replay.json}"
echo "=== v30b warmstart ==="
echo " init=$INIT replay=$REPLAY"
"$PY" -m orbit_wars_rl.bc.warmstart_opening_econ \
  --init "$INIT" \
  --replay "$REPLAY" \
  --out-dir ./ckpt_multi_action_v30b_warm \
  --steps 400 \
  --batch-size 32 \
  --lr 0.001
