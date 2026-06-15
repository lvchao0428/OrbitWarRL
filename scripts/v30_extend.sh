#!/usr/bin/env bash
# v30: econ dst head + unified ROI + aux CE; extend from v29 u3999 (~3.5h).
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
LOGDIR="logs/v30_extend_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

RESUME="./ckpt_multi_action_v29_aim/ckpt_003999.pkl"
[ -f "$RESUME" ] || RESUME="./ckpt_multi_action_v29_aim/ckpt_latest.pkl"

echo "═══════════════════════════════════════════════════"
echo " v30 — econ dst + unified ROI + aux CE from v29 u3999"
echo "═══════════════════════════════════════════════════"
echo " init     : $RESUME"
echo " updates  : $NUM_UPDATES"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v30_econ_dst.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
