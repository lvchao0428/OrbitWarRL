#!/usr/bin/env bash
# v18: multi-emit unlock + mixed opponent + constant capture + soft flip
#
# Key P0 changes:
#   - emit_hard_stop_min_step=2 (t=1 allowed even without worth-it pair)
#   - ent_coef_emit=1e-3 (10x v17; promotes multi-route exploration)
#   - MULTI_EMIT=0.02 bonus (gated: needs >=1 fleet with >=8 ships)
#   - flip_hard_mask=false (soft exploration of all targets)
#   - capture 0.02 constant (no anneal-to-zero)
#
# P1: selfplay vs v16a u3199 (strong, shape-adapted) + pool
#
# Usage:
#   bash scripts/v18_multi_emit.sh                          # default: v17 u2199
#   bash scripts/v18_multi_emit.sh <ckpt.pkl> [num_updates]
set -euo pipefail

# --- Shaping: constant capture + multi-emit bonus ---
export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_ANTI_HOARD=0.03
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.6
export ORBITWARS_SHAPING_CAPTURE=0.02
export ORBITWARS_SHAPING_CAPTURE_START=0.02
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999
export ORBITWARS_SHAPING_DEFENSE_EMPTY=0.01
export ORBITWARS_SHAPING_KEEP_HOME=0.0
export ORBITWARS_SHAPING_FLEET_SIZE=0.0
export ORBITWARS_SHAPING_PROD_SHARE=0.0
export ORBITWARS_SHAPING_PLANET_SHARE=0.0
export ORBITWARS_SHAPING_FLEET_LOG=0.0
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_EMIT_GATED=0.0
export ORBITWARS_SHAPING_RELEASE=0.0
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.0
export ORBITWARS_SHAPING_MULTI_EMIT=0.02
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_SHIPS=8

CKPT_DIR_V17="${CKPT_DIR_V17:-./ckpt_multi_action_v17_frog_hist50}"
DEFAULT_RESUME="${CKPT_DIR_V17}/ckpt_002199.pkl"
RESUME="${1:-$DEFAULT_RESUME}"
NUM_UPDATES="${2:-8000}"

if [ ! -f "$RESUME" ]; then
  echo "ERR: resume ckpt not found: $RESUME" >&2
  exit 1
fi

LOGDIR="logs/v18_multi_emit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "[v18] resume=$RESUME  num_updates=$NUM_UPDATES"
echo "[v18] capture=$ORBITWARS_SHAPING_CAPTURE (constant)  multi_emit=$ORBITWARS_SHAPING_MULTI_EMIT"
echo "[v18] emit_hard_stop_min_step=2  ent_emit=1e-3  flip_hard_mask=false"
echo "[v18] selfplay: strong=v16a_u3199 @0.2  frozen_pool @0.3  symmetric_self @0.5"
echo "[v18] logdir=$LOGDIR"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v18_multi_emit.yaml \
  --log-dir "$LOGDIR" \
  --resume-from "$RESUME" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
