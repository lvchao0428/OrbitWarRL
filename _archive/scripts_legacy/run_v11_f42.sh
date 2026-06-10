#!/usr/bin/env bash
# v11_f42: Resume from f41 @49 with fleet-scaled capture bonus.
#
# Key deltas vs f41:
#   +CAPTURE_FLEET_SCALE 0.10  (large-fleet capture >> trickle capture)
#   ONE_SHIP_PENALTY 0.01->0.005 (soften to preserve e2+)
#   Resume: f41 @49 (best flip=7.19%)
#
# Usage:
#   bash scripts/run_v11_f42.sh
#   RESUME=path/to/ckpt.pkl bash scripts/run_v11_f42.sh

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

CFG="${CFG:-orbit_wars_rl/configs/multi_action_v11_f42.yaml}"
RESUME="${RESUME:-ckpt_multi_action_v11_f41/ckpt_000049.pkl}"
NUM_UPDATES="${NUM_UPDATES:-}"
JAX_FRAC="${JAX_FRAC:-0.85}"
LOG="${LOG:-logs/v11_f42.log}"
TB="${TB:-logs/v11_f42}"
FOREGROUND="${FOREGROUND:-0}"
CKPT_DIR="${CKPT_DIR:-}"
CKPT_EVERY="${CKPT_EVERY:-}"

mkdir -p logs

if [ ! -f "$RESUME" ]; then
  echo "[f42] ERROR: resume ckpt not found: $RESUME" >&2
  echo "[f42] Expecting f41 @49: ckpt_multi_action_v11_f41/ckpt_000049.pkl" >&2
  exit 1
fi

if [ ! -f "data/f40_mixed_states.npz" ]; then
  echo "[f42] ERROR: missing data/f40_mixed_states.npz" >&2
  exit 1
fi

ARGS=(--config "$CFG" --log-dir "$TB" --resume-from "$RESUME")
if [ -n "$NUM_UPDATES" ]; then
  ARGS+=(--num-updates "$NUM_UPDATES")
fi
if [ -n "$CKPT_DIR" ]; then
  ARGS+=(--ckpt-dir "$CKPT_DIR")
fi
if [ -n "$CKPT_EVERY" ]; then
  ARGS+=(--ckpt-every "$CKPT_EVERY")
fi

TRAIN_ENV=(
  ORBITWARS_SHAPING_CAPTURE="${ORBITWARS_SHAPING_CAPTURE:-0.05}"
  ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE="${ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE:-0.10}"
  ORBITWARS_SHAPING_PROD_SHARE_DELTA="${ORBITWARS_SHAPING_PROD_SHARE_DELTA:-0.005}"
  ORBITWARS_SHAPING_ONE_SHIP_PENALTY="${ORBITWARS_SHAPING_ONE_SHIP_PENALTY:-0.005}"
  ORBITWARS_SHAPING_ONE_SHIP_THRESH="${ORBITWARS_SHAPING_ONE_SHIP_THRESH:-3.0}"
  ORBITWARS_SHAPING_HIGH_PROD_CAPTURE="${ORBITWARS_SHAPING_HIGH_PROD_CAPTURE:-0.02}"
  ORBITWARS_SHAPING_HIGH_PROD_THRESH="${ORBITWARS_SHAPING_HIGH_PROD_THRESH:-3.0}"
  ORBITWARS_SHAPING_EMIT_LOG=0.0
  ORBITWARS_SHAPING_EMIT_GATED=0
  ORBITWARS_SHAPING_PLANET_SHARE=0.0
  ORBITWARS_SHAPING_PROD_SHARE=0.0
  ORBITWARS_SHAPING_FLEET_LOG=0.0
  ORBITWARS_SHAPING_RELEASE=0.0
  ORBITWARS_SHAPING_RELEASE_K=20.0
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC"
)

echo "[f42] py=$PY cfg=$CFG resume=$RESUME log=$LOG foreground=$FOREGROUND"
echo "[f42] CAPTURE=0.05 CAPTURE_FLEET_SCALE=0.10 ONE_SHIP_PEN=0.005"
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
