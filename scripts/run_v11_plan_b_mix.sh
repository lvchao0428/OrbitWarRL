#!/usr/bin/env bash
# Plan B mix: v20 + top10 balanced buffer, resume from buf_from4k @799.
#
# Usage:
#   bash scripts/build_mixed_buffer.sh          # once
#   bash scripts/run_v11_plan_b_mix.sh          # train 800 upd
#
# After training:
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_buf_mix/ckpt_000799.pkl v11_buf_mix

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
JAX_FRAC="${JAX_FRAC:-0.85}"
MIXED_NPZ="${MIXED_NPZ:-data/mixed_v20_top10.npz}"
RESUME_CKPT="${RESUME_CKPT:-ckpt_multi_action_v11_buf_from4k/ckpt_000799.pkl}"
CFG="orbit_wars_rl/configs/multi_action_v11_buf_mix.yaml"
LOG="logs/v11_buf_mix.log"
TB="logs/v11_buf_mix"

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

if [ ! -f "$MIXED_NPZ" ]; then
  echo "[plan_b_mix] ERROR: $MIXED_NPZ not found. Run: bash scripts/build_mixed_buffer.sh" >&2
  exit 1
fi

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[plan_b_mix] ERROR: $RESUME_CKPT not found." >&2
  exit 1
fi

echo "[plan_b_mix] buffer=$MIXED_NPZ  resume=$RESUME_CKPT  log=$LOG"

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
