#!/usr/bin/env bash
# v27: frontier + anti-shuffle PPO extend from v26 u7999 (~2.5h PPO).
#
# Usage:
#   bash scripts/v27_extend.sh [num_updates]
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

LOGDIR="logs/v27_extend_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

RESUME="./ckpt_multi_action_v26_roi/ckpt_007999.pkl"
if [ ! -f "$RESUME" ] && [ -f "./ckpt_multi_action_v26_roi/ckpt_latest.pkl" ]; then
  RESUME="./ckpt_multi_action_v26_roi/ckpt_latest.pkl"
fi
if [ ! -f "$RESUME" ] && [ -f "./ckpt_multi_action_v25_extend/ckpt_009599.pkl" ]; then
  RESUME="./ckpt_multi_action_v25_extend/ckpt_009599.pkl"
fi

echo "═══════════════════════════════════════════════════"
echo " v27 — frontier fast extend from v26 u7999 (~3.5h total)"
echo "═══════════════════════════════════════════════════"
echo " init     : $RESUME"
echo " anchor   : same ckpt, KL 0.04 -> 0.015 over 10000u"
echo " opp      : strong(BC v27 epw30) 50% + frozen 30% + rand 20%"
echo " reward   : capture_roi=0.025 shuffle=0.01 prod_delta=0.006"
echo " buffer   : top10_winner_states 20% reset"
echo " eval     : 20g vs-v20 / 200u, eval_ckpt_keep"
echo " updates  : $NUM_UPDATES"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v27_frontier.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
