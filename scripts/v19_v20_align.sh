#!/usr/bin/env bash
# v19: v20 distribution alignment (buffer + v16a strong/frozen pool)
#
# Usage:
#   bash scripts/v19_v20_align.sh
#   bash scripts/v19_v20_align.sh <ckpt.pkl> [num_updates]
set -euo pipefail

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_ANTI_HOARD=0.05
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.55
export ORBITWARS_SHAPING_CAPTURE=0.02
export ORBITWARS_SHAPING_CAPTURE_START=0.02
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999
export ORBITWARS_SHAPING_DEFENSE_EMPTY=0.015
export ORBITWARS_SHAPING_KEEP_HOME=0.0
export ORBITWARS_SHAPING_FLEET_SIZE=0.0
export ORBITWARS_SHAPING_PROD_SHARE=0.0
export ORBITWARS_SHAPING_PLANET_SHARE=0.0
export ORBITWARS_SHAPING_FLEET_LOG=0.0
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.005
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_EMIT_GATED=0.0
export ORBITWARS_SHAPING_RELEASE=0.0
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.0
export ORBITWARS_SHAPING_MULTI_EMIT=0.02
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_SHIPS=8

CKPT_DIR_V18="${CKPT_DIR_V18:-./ckpt_multi_action_v18_multi_emit}"
DEFAULT_RESUME="${CKPT_DIR_V18}/ckpt_001799.pkl"
RESUME="${1:-$DEFAULT_RESUME}"
NUM_UPDATES="${2:-6000}"

if [ ! -f "$RESUME" ]; then
  echo "ERR: resume ckpt not found: $RESUME" >&2
  exit 1
fi

BUFFER="${BUFFER_NPZ:-data/mixed_v20_top10.npz}"
if [ ! -f "$BUFFER" ]; then
  echo "ERR: v20 buffer not found: $BUFFER" >&2
  exit 1
fi

LOGDIR="logs/v19_v20_align_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "[v19] resume=$RESUME  num_updates=$NUM_UPDATES"
echo "[v19] buffer=$BUFFER reset=50% rollout=20%"
echo "[v19] opp mix: strong v16a@25%  frozen pool@30%  buf@20%  rand@25%"
echo "[v19] anti_hoard=0.05  prod_share_delta=0.005  capture=0.02 constant"
echo "[v19] logdir=$LOGDIR"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v19_v20_align.yaml \
  --log-dir "$LOGDIR" \
  --resume-from "$RESUME" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
