#!/usr/bin/env bash
# v29: pair ROI dim6 + dst dedup PPO extend from v28 u3999 (~3.5h).
#
# Usage:
#   bash scripts/v29_extend.sh [num_updates]
set -euo pipefail

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_CAPTURE=0
export ORBITWARS_SHAPING_CAPTURE_ROI=0.025
export ORBITWARS_SHAPING_CAPTURE_START=0
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999
export ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE=0.005
export ORBITWARS_SHAPING_FRIENDLY_SHUFFLE=0.01
export ORBITWARS_SHAPING_ANTI_HOARD=0.02
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.55
export ORBITWARS_SHAPING_MULTI_EMIT=0.02
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_SHIPS=10
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_HOME_GARR=30
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.006

export ORBITWARS_SHAPING_DEFENSE_EMPTY=0.0
export ORBITWARS_SHAPING_KEEP_HOME=0.0
export ORBITWARS_SHAPING_FLEET_SIZE=0.0
export ORBITWARS_SHAPING_PROD_SHARE=0.0
export ORBITWARS_SHAPING_PLANET_SHARE=0.0
export ORBITWARS_SHAPING_FLEET_LOG=0.0
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_EMIT_GATED=0.0
export ORBITWARS_SHAPING_RELEASE=0.0
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.0
export ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.0

export ORBITWARS_SKIP_PARITY=1

NUM_UPDATES="${1:-4000}"

LOGDIR="logs/v29_extend_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

RESUME="./ckpt_multi_action_v28_roi/ckpt_003999.pkl"
if [ ! -f "$RESUME" ] && [ -f "./ckpt_multi_action_v28_roi/ckpt_latest.pkl" ]; then
  RESUME="./ckpt_multi_action_v28_roi/ckpt_latest.pkl"
fi

echo "═══════════════════════════════════════════════════"
echo " v29 — pair ROI dim6 + dst dedup from v28 u3999"
echo "═══════════════════════════════════════════════════"
echo " init     : $RESUME"
echo " anchor   : v28 u3999, KL 0.04 -> 0.015 over 4000u"
echo " opp      : strong(v28) 40% + frozen 30% + rand 30%"
echo " reward   : capture_roi payback=0.025 shuffle=0.01"
echo " no BC    : PPO-only pipeline"
echo " updates  : $NUM_UPDATES"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v29_aim_dedup.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
