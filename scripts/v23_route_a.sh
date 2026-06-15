#!/usr/bin/env bash
# v23 Route A: fresh train — v21 features + mixed opp + flip-gated reward (no buffer)
#
# Usage:
#   bash scripts/v23_route_a.sh
#   bash scripts/v23_route_a.sh [num_updates]
set -euo pipefail

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_CAPTURE=0.02
export ORBITWARS_SHAPING_CAPTURE_START=0.02
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999
export ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE=0.01
export ORBITWARS_SHAPING_ANTI_HOARD=0.04
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.55
export ORBITWARS_SHAPING_MULTI_EMIT=0.02
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_SHIPS=8

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

NUM_UPDATES="${1:-8000}"

LOGDIR="logs/v23_route_a_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "═══════════════════════════════════════════════════"
echo " v23 Route A — FRESH START"
echo "═══════════════════════════════════════════════════"
echo " features : planet=63 global=427 (v21 full stack)"
echo " selfplay : strong 25% + frozen 25% + rand 50%"
echo " reward   : flip-gated capture=0.02 fleet_scale=0.01"
echo "            multi_emit=0.02 anti_hoard=0.04"
echo " policy   : ent_emit=1e-3 min_pct_bin=5 emit_min_step=2"
echo " eval     : vs-v20 every 100u x5 (SKIP_PARITY=1)"
echo " updates  : $NUM_UPDATES"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v23_route_a.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
