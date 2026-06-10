#!/usr/bin/env bash
# v11_f30: f29 @599 + 800 upd extension (pure signals, no buffer).
#
# Prereq (5090):
#   ckpt_multi_action_v11_f29/ckpt_000599.pkl
#
# Usage:
#   bash scripts/run_v11_f30.sh
#   tail -f logs/v11_f30.log
#
# After training:
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f30/ckpt_000299.pkl v11_f30_u299
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f30/ckpt_000599.pkl v11_f30_u599
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f30/ckpt_000799.pkl v11_f30_u799

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
CFG="orbit_wars_rl/configs/multi_action_v11_f30.yaml"
LOG="logs/v11_f30.log"
TB="logs/v11_f30"
RESUME_CKPT="${RESUME_CKPT:-ckpt_multi_action_v11_f29/ckpt_000599.pkl}"

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

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[v11_f30] ERROR: $RESUME_CKPT not found." >&2
  exit 1
fi

echo "[v11_f30] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f30] resume=$RESUME_CKPT  lr_peak=8e-5  no buffer  emit_hard_stop=false"

env "${COMMON_SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
    --resume-from "$RESUME_CKPT" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
