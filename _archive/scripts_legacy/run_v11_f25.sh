#!/usr/bin/env bash
# v11_f25: feature-engineering iteration, from scratch.
# planet 22 -> 28, fleet 8 -> 10, global 14 -> 17.
#
# Avg-stack pct signals (planet 22-24):
#   [22] flip_cost_ratio    - target_garr / my_avg_garrison
#   [23] friendly_surplus   - (friendly_inbound - garrison) / garrison
#   [24] capturable_bin3    - bin3 (40%) from avg planet flips this
# Max-stack big-fleet signals (planet 25-26):
#   [25] needed_pct_norm    - target_garr / my_MAX_garrison
#   [26] capturable_bin5    - bin5 (70%) from MAX planet flips this
# Multi-route signal (planet 27):
#   [27] weak_target_score  - per-planet softness for enemies/neutrals
# Fleet (8-9):
#   [8]  target_dist_norm   - fleet to inferred target / BOARD
#   [9]  target_garrison_norm - log1p(target garrison) / 8
# Global (14-16):
#   [14] max_garr_norm                  - my strongest stack
#   [15] n_weak_targets_norm            - how many soft enemy/neutrals exist
#   [16] ships_to_capture_all_weak_norm - total cost to flip them all
#
# Usage:
#   bash scripts/run_v11_f25.sh
#
# After training (~800 upd, ~2h):
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_f25/ckpt_000799.pkl v11_f25_u799

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    PY=python3
  fi
else
  PY="$PYTHON"
fi

JAX_FRAC="${JAX_FRAC:-0.85}"
CFG="orbit_wars_rl/configs/multi_action_v11_f25.yaml"
LOG="logs/v11_f25.log"
TB="logs/v11_f25"

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

echo "[v11_f25] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f25] training from scratch (new arch: planet=25, fleet=10)"
echo "[v11_f25] ent_coef_pct=0.006  frozen_ratio=0.40  seed=250"

env "${COMMON_SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
