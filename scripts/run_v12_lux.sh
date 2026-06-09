#!/usr/bin/env bash
# v12_lux — Frog Parade reference: bigger model + symmetric self-play + sparse reward
#
# Key design (see docs/DAY12_LUX_REFERENCE.zh.md):
#   d_model=256, n_layers=4, ff_dim=1024 (~3-5M params)
#   Pure symmetric self-play (no frozen/strong/buffer)
#   gamma=0.9999 (Frog Parade used 0.9999-1.0)
#   Sparse +1/-1 reward only (ALL shaping = 0)
#   10000 updates (~100M+ steps, ~days of training)
#
# Usage:
#   bash scripts/run_v12_lux.sh
#   FOREGROUND=1 bash scripts/run_v12_lux.sh
#   NUM_UPDATES=5 bash scripts/run_v12_lux.sh   # smoke test

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

CFG="${CFG:-orbit_wars_rl/configs/multi_action_v12_lux.yaml}"
RESUME="${RESUME:-}"
NUM_UPDATES="${NUM_UPDATES:-}"
JAX_FRAC="${JAX_FRAC:-0.90}"
LOG="${LOG:-logs/v12_lux.log}"
TB="${TB:-logs/v12_lux}"
FOREGROUND="${FOREGROUND:-0}"

mkdir -p logs

ARGS=(--config "$CFG" --log-dir "$TB")
if [ -n "$RESUME" ]; then
  ARGS+=(--resume-from "$RESUME")
fi
if [ -n "$NUM_UPDATES" ]; then
  ARGS+=(--num-updates "$NUM_UPDATES")
fi

# ALL shaping = 0. Pure sparse +1/-1 terminal reward.
TRAIN_ENV=(
  ORBITWARS_SHAPING_SCALE=0.0
  ORBITWARS_SHAPING_CAPTURE=0.0
  ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE=0.0
  ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0
  ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.0
  ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.0
  ORBITWARS_SHAPING_MULTI_EMIT=0.0
  ORBITWARS_SHAPING_EMIT_LOG=0.0
  ORBITWARS_SHAPING_EMIT_GATED=0
  ORBITWARS_SHAPING_PLANET_SHARE=0.0
  ORBITWARS_SHAPING_PROD_SHARE=0.0
  ORBITWARS_SHAPING_FLEET_LOG=0.0
  ORBITWARS_SHAPING_RELEASE=0.0
  ORBITWARS_SHAPING_KEEP_HOME=0.0
  ORBITWARS_SHAPING_FLEET_SIZE=0.0
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC"
)

echo "[v12_lux] py=$PY cfg=$CFG log=$LOG"
echo "[v12_lux] model: d=256 layers=4 heads=8 ff=1024"
echo "[v12_lux] training: symmetric self-play, sparse ±1 only, gamma=0.9999"
echo "[v12_lux] schedule: 10000 updates, warmup=100 vs random"
if [ -n "$RESUME" ]; then
  echo "[v12_lux] resume: $RESUME"
fi
if [ "$FOREGROUND" = "1" ]; then
  env "${TRAIN_ENV[@]}" \
    "$PY" -m orbit_wars_rl.scripts.train "${ARGS[@]}" \
    2>&1 | tee -a "$LOG"
else
  env "${TRAIN_ENV[@]}" \
    "$PY" -m orbit_wars_rl.scripts.train "${ARGS[@]}" \
    >> "$LOG" 2>&1 &
  echo "pid=$!"
  echo "tail -f $LOG"
fi
