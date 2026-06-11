#!/usr/bin/env bash
# v17 extend: pure sparse phase after initial 15000u run (capture already annealed).
# Usage: bash scripts/v17_frog_hist50_extend.sh ckpt_multi_action_v17_frog_hist50/ckpt_0014999.pkl
set -euo pipefail

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_ANTI_HOARD=0.03
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.6
export ORBITWARS_SHAPING_CAPTURE=0.0
export ORBITWARS_SHAPING_CAPTURE_START=0.0
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=1
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

LOGDIR="logs/v17_frog_hist50_extend_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

RESUME="${1:?usage: $0 <ckpt.pkl>}"
NUM_UPDATES="${2:-14000}"
PY="${PYTHON:-python}"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v17_frog_hist50.yaml \
  --log-dir "$LOGDIR" \
  --resume-from "$RESUME" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
