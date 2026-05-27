#!/usr/bin/env bash
# v11_f26: f25 features + K-loop pair head augmentations, from scratch.
#
# Same encoder dims as f25 (planet=28, fleet=10, global=17) but heads now
# consume per-step pair features:
#   DstHead  += 4 (dist_src_dst, sun_risk, ships_needed, pair_flip_bin5)
#               + hard sun_block_mask (logit -inf on src->dst paths through sun)
#   EmitHead += 4 (n_feasible_pairs, best_margin, home_remain, total_remain)
#   PctHead  += 2 (pair_needed_pct, pair_flip_bin5_pair)
#
# Motivation: f25 fixed bin0 but produced 1-ship multi-route spam and home
# garrison drainage with fleets flying into the sun. f26 hands every head a
# direct geometric/feasibility/budget prior, so the policy stops needing
# random exploration to learn "don't shoot through the sun" or "you have 3
# ships left, send 100%".
#
# Usage:
#   bash scripts/run_v11_f26.sh
#
# After training:
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_f26/ckpt_000299.pkl v11_f26_u299

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
CFG="orbit_wars_rl/configs/multi_action_v11_f26.yaml"
LOG="logs/v11_f26.log"
TB="logs/v11_f26"

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

echo "[v11_f26] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f26] training from scratch (f25 dims + pair head augmentations)"
echo "[v11_f26] ent_coef_pct=0.004  frozen_ratio=0.40  seed=260"

env "${COMMON_SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
