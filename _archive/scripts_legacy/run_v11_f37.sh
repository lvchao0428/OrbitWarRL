#!/usr/bin/env bash
# v11_f37: strengthen fundamental action signals + pure terminal reward.
#
# Deltas vs f35/f33:
#   - GLOBAL_FEAT_DIM 17->18 (+min_effective_fleet_norm)
#   - EMIT_PAIR_DIM 4->6 (+feasible_target_count_norm, +surplus_ratio)
#   - All shaping = 0 (pure +1/-1 terminal)
#   - num_updates=3000, ckpt_every=100
#
# Usage:
#   bash scripts/run_v11_f37.sh
#   tail -f logs/v11_f37.log
#   bash scripts/check_training_health.sh logs/v11_f37.log

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
CFG="orbit_wars_rl/configs/multi_action_v11_f37.yaml"
LOG="logs/v11_f37.log"
TB="logs/v11_f37"

COMMON_SHAPING=(
  "ORBITWARS_SHAPING_EMIT_LOG=0.0"
  "ORBITWARS_SHAPING_EMIT_GATED=0"
  "ORBITWARS_SHAPING_PLANET_SHARE=0.0"
  "ORBITWARS_SHAPING_PROD_SHARE=0.0"
  "ORBITWARS_SHAPING_FLEET_LOG=0.0"
  "ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0"
  "ORBITWARS_SHAPING_RELEASE=0.0"
  "ORBITWARS_SHAPING_RELEASE_K=20.0"
  "ORBITWARS_SHAPING_CAPTURE=0.0"
)

mkdir -p logs

echo "[v11_f37] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f37] pure terminal reward: all shaping=0"
echo "[v11_f37] new feats: GLOBAL_FEAT_DIM=18, EMIT_PAIR_DIM=6"
echo "[v11_f37] long run: 3000 upd, ckpt_every=100"

echo "[v11_f37] parity smoke"
if ! JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.inference.test_parity \
    --num-states 16 \
    --emit-hard-stop \
    --flip-hard-mask; then
  echo "[v11_f37] parity FAILED — aborting" >&2
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
