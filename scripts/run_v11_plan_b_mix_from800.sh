#!/usr/bin/env bash
# Plan B mix from800: resume from k8_no_emit @800 (pct-healthiest ckpt).
#
# Hypothesis: k8_no_emit @800 has bin0=25.4% (uncollapsed pct head);
# mix buffer lifts spf/garr while preserving pct distribution.
#
# Usage:
#   bash scripts/run_v11_plan_b_mix_from800.sh
#
# Prereq:
#   data/mixed_v20_top10.npz
#   ckpt_multi_action_v11_k8_no_emit/ckpt_000799.pkl
#
# After 800 upd (~4h):
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_buf_mix_from800/ckpt_000799.pkl v11_buf_mix_from800

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  elif [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  else
    PY=python3
  fi
else
  PY="$PYTHON"
fi

JAX_FRAC="${JAX_FRAC:-0.85}"
MIXED_NPZ="${MIXED_NPZ:-data/mixed_v20_top10.npz}"
RESUME_CKPT="${RESUME_CKPT:-ckpt_multi_action_v11_k8_no_emit/ckpt_000799.pkl}"
CFG="orbit_wars_rl/configs/multi_action_v11_buf_mix_from800.yaml"
LOG="logs/v11_buf_mix_from800.log"
TB="logs/v11_buf_mix_from800"

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
  echo "[plan_b_mix_from800] ERROR: $MIXED_NPZ not found." >&2
  exit 1
fi

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[plan_b_mix_from800] ERROR: $RESUME_CKPT not found." >&2
  exit 1
fi

echo "[plan_b_mix_from800] py=$PY  buffer=$MIXED_NPZ  resume=$RESUME_CKPT  log=$LOG"
echo "[plan_b_mix_from800] cfg=$CFG (ent_coef_pct=0.006 frozen=0.30 reset=0.70)"

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
