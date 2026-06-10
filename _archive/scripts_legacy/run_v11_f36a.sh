#!/usr/bin/env bash
# v11_f36a: Day9 route 3A — action mechanism delta only.
# Single delta vs f33: delay emit_hard_stop until step>=2.
#
# Usage:
#   bash scripts/run_v11_f36a.sh
#   tail -f logs/v11_f36a.log
#   bash scripts/check_training_health.sh logs/v11_f36a.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
else
  PY="$PYTHON"
fi

JAX_FRAC="${JAX_FRAC:-0.85}"
CFG="orbit_wars_rl/configs/multi_action_v11_f36a.yaml"
LOG="logs/v11_f36a.log"
TB="logs/v11_f36a"

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

echo "[v11_f36a] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f36a] single delta: emit_hard_stop_min_step=2 (keep PPO/features/selfplay frozen)"
echo "[v11_f36a] short run: 350 upd, ckpt_every=50"

echo "[v11_f36a] parity smoke"
if ! JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.inference.test_parity \
    --num-states 16 \
    --emit-hard-stop \
    --emit-hard-stop-min-step 2 \
    --flip-hard-mask; then
  echo "[v11_f36a] parity FAILED — aborting" >&2
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
