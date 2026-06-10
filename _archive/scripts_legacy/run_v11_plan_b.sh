#!/usr/bin/env bash
# Plan B: v20 state-buffer curriculum for v11 K=8 (R4 OFF).
#
# Usage:
#   bash scripts/run_v11_plan_b.sh scratch   # from scratch, buffer 30%
#   bash scripts/run_v11_plan_b.sh from4k    # resume 4k ckpt, buffer 50%
#
# After 800 upd:
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_buf_from4k/ckpt_000799.pkl v11_buf_from4k

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-from4k}"
PY="${PYTHON:-python}"
JAX_FRAC="${JAX_FRAC:-0.85}"

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

if [ ! -f "data/v20_states_200g.npz" ]; then
  echo "[plan_b] ERROR: data/v20_states_200g.npz missing. Run collect_states first." >&2
  exit 1
fi

case "$MODE" in
  scratch)
    CFG="orbit_wars_rl/configs/multi_action_v11_buf.yaml"
    LOG="logs/v11_buf_run1.log"
    TB="logs/v11_buf_run1"
    RESUME=()
    ;;
  from4k)
    CKPT="ckpt_multi_action_v11_k8_4k/ckpt_003199.pkl"
    if [ ! -f "$CKPT" ]; then
      echo "[plan_b] ERROR: $CKPT not found (sync from 5090 first)." >&2
      exit 1
    fi
    CFG="orbit_wars_rl/configs/multi_action_v11_buf_from4k.yaml"
    LOG="logs/v11_buf_from4k.log"
    TB="logs/v11_buf_from4k"
    RESUME=(--resume-from "$CKPT")
    ;;
  *)
    echo "Usage: $0 {scratch|from4k}" >&2
    exit 1
    ;;
esac

echo "[plan_b] mode=$MODE  cfg=$CFG  log=$LOG"
if [ ${#RESUME[@]} -gt 0 ]; then
  echo "[plan_b] resume=${RESUME[*]}"
fi

env "${COMMON_SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
    "${RESUME[@]}" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
