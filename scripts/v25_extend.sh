#!/usr/bin/env bash
# v25: extend v24 u3999 (seed=0 win style), ~8h anti-collapse PPO run.
#
# Usage:
#   bash scripts/v25_extend.sh [num_updates]
set -euo pipefail

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_CAPTURE=0.02
export ORBITWARS_SHAPING_CAPTURE_START=0.02
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999
export ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE=0.01
export ORBITWARS_SHAPING_ANTI_HOARD=0.04
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.55
export ORBITWARS_SHAPING_MULTI_EMIT=0.02
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_SHIPS=10
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_HOME_GARR=30

export ORBITWARS_SHAPING_DEFENSE_EMPTY=0.0
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
export ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.0

export ORBITWARS_SKIP_PARITY=1

NUM_UPDATES="${1:-11000}"

LOGDIR="logs/v25_extend_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "═══════════════════════════════════════════════════"
echo " v25 — extend v24 u3999 (anti-collapse, ~8h)"
echo "═══════════════════════════════════════════════════"
echo " init     : ckpt_multi_action_v24_bc_ft/ckpt_003999.pkl"
echo " anchor   : same ckpt, KL 0.04 -> 0.02 over 8000u"
echo " opp      : strong(BC) 45% + frozen 30% + rand 25%"
echo " reward   : v24 + multi_emit home_garr>=30 min_ships=10"
echo " lr       : 2e-5 (gentler than v24)"
echo " updates  : $NUM_UPDATES (~8h @ v24 speed)"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v25_extend.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
