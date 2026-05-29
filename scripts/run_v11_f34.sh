#!/usr/bin/env bash
# v11_f34: f33 arch+hparams + f29 strong opponent (anti self-play inflation).
#
# Single delta vs f33: strong_ratio=0.35, frozen_ratio=0.25, 800 upd.
#
# Usage:
#   bash scripts/run_v11_f34.sh
#   tail -f logs/v11_f34.log
#
# Requires f29 @599 ckpt at ckpt_multi_action_v11_f29/ckpt_000599.pkl
#
# After training:
#   bash scripts/run_f34_eval.sh

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

STRONG="${STRONG:-ckpt_multi_action_v11_f29/ckpt_000599.pkl}"
if [ ! -f "$STRONG" ]; then
  echo "[v11_f34] ERROR: strong ckpt not found: $STRONG" >&2
  echo "  set STRONG=/path/to/ckpt_000599.pkl or sync f29 ckpt first" >&2
  exit 1
fi

JAX_FRAC="${JAX_FRAC:-0.85}"
CFG="orbit_wars_rl/configs/multi_action_v11_f34.yaml"
LOG="logs/v11_f34.log"
TB="logs/v11_f34"

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

echo "[v11_f34] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f34] f33 arch + f29 strong anchor (ratio=0.35, frozen=0.25), 800 upd"
echo "[v11_f34] strong ckpt: $STRONG"

echo "[v11_f34] parity smoke"
if ! JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.inference.test_parity \
    --num-states 16 --emit-hard-stop --flip-hard-mask; then
  echo "[v11_f34] parity FAILED — aborting" >&2
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
