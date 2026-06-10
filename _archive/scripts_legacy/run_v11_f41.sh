#!/usr/bin/env bash
# v11_f41: Resume from f40 @99 with stronger capture/anti-trickle shaping
#          and adjusted opponent distribution.
#
# Key deltas vs f40:
#   CAPTURE       0.02 -> 0.05   (stronger flip reward)
#   ONE_SHIP_PEN  0    -> 0.01   (penalize <=3-ship trickle)
#   HIGH_PROD_CAP 0    -> 0.02   (bonus for capturing factories)
#   buffer_rollout 0.30 -> 0.40  (more expert-state learning)
#   strong_ratio   0.20 -> 0.25  (more vs f29 anchor)
#   frozen_ratio   0.30 -> 0.25
#   random         0.20 -> 0.10
#
# Usage:
#   bash scripts/run_v11_f41.sh
#   RESUME=path/to/ckpt.pkl NUM_UPDATES=200 bash scripts/run_v11_f41.sh

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

CFG="${CFG:-orbit_wars_rl/configs/multi_action_v11_f41.yaml}"
RESUME="${RESUME:-ckpt_multi_action_v11_f40_buffer/ckpt_000099.pkl}"
NUM_UPDATES="${NUM_UPDATES:-}"
JAX_FRAC="${JAX_FRAC:-0.85}"
LOG="${LOG:-logs/v11_f41.log}"
TB="${TB:-logs/v11_f41}"
FOREGROUND="${FOREGROUND:-0}"
CKPT_DIR="${CKPT_DIR:-}"
CKPT_EVERY="${CKPT_EVERY:-}"

mkdir -p logs

if [ ! -f "$RESUME" ]; then
  echo "[f41] ERROR: resume ckpt not found: $RESUME" >&2
  echo "[f41] Expecting f40 @99: ckpt_multi_action_v11_f40_buffer/ckpt_000099.pkl" >&2
  exit 1
fi

if [ ! -f "data/f40_mixed_states.npz" ]; then
  echo "[f41] ERROR: missing data/f40_mixed_states.npz" >&2
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
  ORBITWARS_SHAPING_PROD_SHARE_DELTA="${ORBITWARS_SHAPING_PROD_SHARE_DELTA:-0.005}"
  ORBITWARS_SHAPING_ONE_SHIP_PENALTY="${ORBITWARS_SHAPING_ONE_SHIP_PENALTY:-0.01}"
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

echo "[f41] py=$PY cfg=$CFG resume=$RESUME log=$LOG foreground=$FOREGROUND"
echo "[f41] CAPTURE=0.05 ONE_SHIP_PEN=0.01 HIGH_PROD_CAP=0.02 PROD_SHARE_DELTA=0.005"
echo "[f41] distro: strong=0.25 frozen=0.25 buffer=0.40 random=0.10"
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
