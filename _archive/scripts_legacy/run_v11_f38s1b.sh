#!/usr/bin/env bash
# v11_f38s1b: Path B retry — stronger CAPTURE bootstrap (0.10 vs s1 0.05).
#
# Triggered when f38 s1 @499 fails replay gate but @199 shows partial improvement.
# Usage:
#   bash scripts/run_v11_f38s1b.sh
#   tail -f logs/v11_f38s1b.log

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
LOG="logs/v11_f38s1b.log"
TB="logs/v11_f38s1b"
CFG="orbit_wars_rl/configs/multi_action_v11_f38s1b.yaml"

mkdir -p logs

echo "[f38s1b] CAPTURE=0.10, PROD_SHARE_DELTA=0.02, 500 upd"
echo "[f38s1b] py=$PY  cfg=$CFG  log=$LOG"

env \
  ORBITWARS_SHAPING_CAPTURE=0.10 \
  ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.02 \
  ORBITWARS_SHAPING_EMIT_LOG=0.0 \
  ORBITWARS_SHAPING_EMIT_GATED=0 \
  ORBITWARS_SHAPING_PLANET_SHARE=0.0 \
  ORBITWARS_SHAPING_PROD_SHARE=0.0 \
  ORBITWARS_SHAPING_FLEET_LOG=0.0 \
  ORBITWARS_SHAPING_RELEASE=0.0 \
  ORBITWARS_SHAPING_RELEASE_K=20.0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
  >> "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
echo "monitor: bash scripts/check_training_health.sh $LOG"
