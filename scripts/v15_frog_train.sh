#!/usr/bin/env bash
# v15: Frog Parade style — pure terminal ±1 per match, anti-hoard only
# Best-of-3 multi-match series (wins_needed=2), 10000 updates.
#
# All reward shaping disabled except ANTI_HOARD (structural collapse prevention).
set -euo pipefail

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_ANTI_HOARD=0.03
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.6
export ORBITWARS_SHAPING_KEEP_HOME=0.0
export ORBITWARS_SHAPING_FLEET_SIZE=0.0
export ORBITWARS_SHAPING_PROD_SHARE=0.0
export ORBITWARS_SHAPING_PLANET_SHARE=0.0
export ORBITWARS_SHAPING_FLEET_LOG=0.0
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_EMIT_GATED=0.0
export ORBITWARS_SHAPING_RELEASE=0.0
export ORBITWARS_SHAPING_CAPTURE=0.0

LOGDIR="logs/v15_frog_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

python -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v15_frog.yaml \
  --log-dir "$LOGDIR" \
  --num-updates 10000 \
  2>&1 | tee "$LOGDIR/train.log"
