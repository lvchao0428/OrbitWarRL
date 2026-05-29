#!/usr/bin/env bash
# v11_f32: speed-aware pair features + relaxed masks (bin3) + fleet-size shaping, fresh 600 upd.
#
# Usage:
#   bash scripts/run_v11_f32.sh
#   tail -f logs/v11_f32.log
#
# After training:
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f32/ckpt_000299.pkl v11_f32_u299
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f32/ckpt_000599.pkl v11_f32_u599

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

JAX_FRAC="${JAX_FRAC:-0.85}"
CFG="orbit_wars_rl/configs/multi_action_v11_f32.yaml"
LOG="logs/v11_f32.log"
TB="logs/v11_f32"

# f32 shaping: fleet-size reward + release bonus (both very small).
COMMON_SHAPING=(
  "ORBITWARS_SHAPING_EMIT_LOG=0.0"
  "ORBITWARS_SHAPING_EMIT_GATED=0"
  "ORBITWARS_SHAPING_PLANET_SHARE=0.005"
  "ORBITWARS_SHAPING_PROD_SHARE=0.0"
  "ORBITWARS_SHAPING_FLEET_LOG=0.005"
  "ORBITWARS_SHAPING_FLEET_LOG_REF=500.0"
  "ORBITWARS_SHAPING_FLEET_LOG_FLOOR=0.3"
  "ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0"
  "ORBITWARS_SHAPING_RELEASE=0.003"
  "ORBITWARS_SHAPING_RELEASE_K=20.0"
  "ORBITWARS_SHAPING_CAPTURE=0.02"
)

mkdir -p logs

echo "[v11_f32] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f32] fresh train 600 upd  emit_hard_stop=true  flip_hard_mask=true (bin3)"
echo "[v11_f32] shaping: FLEET_LOG=0.005  RELEASE=0.003  CAPTURE=0.02"

echo "[v11_f32] parity smoke (fresh-init, f32 masks)"
if ! JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.inference.test_parity \
    --num-states 16 --emit-hard-stop --flip-hard-mask; then
  echo "[v11_f32] parity FAILED — aborting" >&2
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
