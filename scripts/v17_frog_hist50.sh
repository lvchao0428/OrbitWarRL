#!/usr/bin/env bash
# v17 overnight: hist=50 + ETA-lead dst + Frog curriculum (capture anneal -> sparse)
set -euo pipefail

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_ANTI_HOARD=0.03
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.6
export ORBITWARS_SHAPING_CAPTURE=0.03
export ORBITWARS_SHAPING_CAPTURE_START=0.03
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=8000
export ORBITWARS_SHAPING_DEFENSE_EMPTY=0.02
export ORBITWARS_SHAPING_KEEP_HOME=0.0
export ORBITWARS_SHAPING_FLEET_SIZE=0.0
export ORBITWARS_SHAPING_PROD_SHARE=0.0
export ORBITWARS_SHAPING_PLANET_SHARE=0.0
export ORBITWARS_SHAPING_FLEET_LOG=0.0
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_EMIT_GATED=0.0
export ORBITWARS_SHAPING_RELEASE=0.0

LOGDIR="logs/v17_frog_hist50_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

RESUME="${1:-./ckpt_multi_action_v15_frog/ckpt_009999.pkl}"
PY="${PYTHON:-/home/charlie/anaconda3/bin/python}"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v17_frog_hist50.yaml \
  --log-dir "$LOGDIR" \
  --resume-from "$RESUME" \
  2>&1 | tee "$LOGDIR/train.log"
