#!/usr/bin/env bash
# v11_f36b: Day9 route 3B — training distribution delta only.
# Single delta family vs f33: short run + fixed f29 strong anchor.
#
# Usage:
#   bash scripts/run_v11_f36b.sh
#   tail -f logs/v11_f36b.log
#   bash scripts/check_training_health.sh logs/v11_f36b.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    PY=python3
  fi
else
  PY="$PYTHON"
fi

STRONG="ckpt_multi_action_v11_f29/ckpt_000599.pkl"
if [ ! -f "$STRONG" ]; then
  echo "[v11_f36b] ERROR: strong ckpt not found: $STRONG" >&2
  exit 1
fi

JAX_FRAC="${JAX_FRAC:-0.85}"
CFG="orbit_wars_rl/configs/multi_action_v11_f36b.yaml"
LOG="logs/v11_f36b.log"
TB="logs/v11_f36b"

COMMON_SHAPING=(
  "ORBITWARS_SHAPING_EMIT_LOG=0.0"
  "ORBITWARS_SHAPING_EMIT_GATED=0"
  "ORBITWARS_SHAPING_PLANET_SHARE=0.005"
  "ORBITWARS_SHAPING_PROD_SHARE=0.0"
  "ORBITWARS_SHAPING_FLEET_LOG=0.0"
  "ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0"
  "ORBITWARS_SHAPING_RELEASE=0.05"
  "ORBITWARS_SHAPING_RELEASE_K=20.0"
  "ORBITWARS_SHAPING_CAPTURE=0.02"
)

mkdir -p logs

echo "[v11_f36b] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f36b] single delta family: strong_ratio=0.35, frozen=0.25, 350 upd"
echo "[v11_f36b] strong ckpt: $STRONG"

echo "[v11_f36b] parity smoke"
if ! JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.inference.test_parity \
    --num-states 16 --emit-hard-stop --flip-hard-mask; then
  echo "[v11_f36b] parity FAILED — aborting" >&2
  exit 1
fi

env "${COMMON_SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
echo "monitor: bash scripts/check_training_health.sh $LOG"
