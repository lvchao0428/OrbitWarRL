#!/usr/bin/env bash
# v11_f40_buffer: PPO from BC seed with buffer curriculum and f29 anchor.
#
# Usage:
#   bash scripts/run_v11_f40_buffer.sh
#   RESUME=ckpt_bc_f40/ckpt_final.pkl NUM_UPDATES=100 bash scripts/run_v11_f40_buffer.sh

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

CFG="${CFG:-orbit_wars_rl/configs/multi_action_v11_f40_buffer.yaml}"
RESUME="${RESUME:-ckpt_bc_f40/ckpt_final.pkl}"
NUM_UPDATES="${NUM_UPDATES:-}"
JAX_FRAC="${JAX_FRAC:-0.85}"
LOG="${LOG:-logs/v11_f40_buffer.log}"
TB="${TB:-logs/v11_f40_buffer}"
FOREGROUND="${FOREGROUND:-0}"
CKPT_DIR="${CKPT_DIR:-}"
CKPT_EVERY="${CKPT_EVERY:-}"

mkdir -p logs

if [ ! -f "$RESUME" ]; then
  echo "[f40-buffer] ERROR: resume ckpt not found: $RESUME" >&2
  exit 1
fi

if [ ! -f "data/f40_mixed_states.npz" ]; then
  echo "[f40-buffer] ERROR: missing data/f40_mixed_states.npz" >&2
  echo "Run: bash scripts/collect_f40_expert_data.sh" >&2
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
  ORBITWARS_SHAPING_CAPTURE="${ORBITWARS_SHAPING_CAPTURE:-0.02}"
  ORBITWARS_SHAPING_PROD_SHARE_DELTA="${ORBITWARS_SHAPING_PROD_SHARE_DELTA:-0.005}"
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

echo "[f40-buffer] py=$PY cfg=$CFG resume=$RESUME log=$LOG foreground=$FOREGROUND"
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
  echo "monitor: bash scripts/check_training_health.sh $LOG"
fi
