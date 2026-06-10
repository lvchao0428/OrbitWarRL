#!/usr/bin/env bash
# v11_f28: f27 @799 + mixed_v20_top10 buffer + anti-spam knobs.
#
# Prereqs (5090):
#   * ckpt_multi_action_v11_f27/ckpt_000799.pkl
#   * data/mixed_v20_top10.npz   (already built by scripts/build_mixed_buffer.sh)
#
# Usage:
#   bash scripts/run_v11_f28.sh
#   tail -f logs/v11_f28.log
#
# After training:
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_f28/ckpt_000099.pkl v11_f28_u099
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_f28/ckpt_000299.pkl v11_f28_u299

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
CFG="orbit_wars_rl/configs/multi_action_v11_f28.yaml"
LOG="logs/v11_f28.log"
TB="logs/v11_f28"
RESUME_CKPT="${RESUME_CKPT:-ckpt_multi_action_v11_f27/ckpt_000799.pkl}"
MIXED_NPZ="${MIXED_NPZ:-data/mixed_v20_top10.npz}"

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
  echo "[v11_f28] ERROR: $RESUME_CKPT not found. f28 must resume f27." >&2
  exit 1
fi
if [ ! -f "$MIXED_NPZ" ]; then
  echo "[v11_f28] ERROR: $MIXED_NPZ not found. Run scripts/build_mixed_buffer.sh." >&2
  exit 1
fi

echo "[v11_f28] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f28] resume=$RESUME_CKPT  buffer=$MIXED_NPZ"
echo "[v11_f28] knobs: buffer_reset_ratio=0.80 frozen_ratio=0.20 ent_coef_pct=0.012 lr_peak=7e-5"

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
