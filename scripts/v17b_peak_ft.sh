#!/usr/bin/env bash
# v17b: fine-tune from v17 peak-flip ckpt (default u2199).
# Fixes overnight regression: capture stays on (no anneal-to-zero sparse washout).
#
# Usage:
#   bash scripts/v17b_peak_ft.sh
#   bash scripts/v17b_peak_ft.sh ckpt_multi_action_v17_frog_hist50/ckpt_001399.pkl 3000
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

# Constant mild shaping — do NOT anneal capture to zero.
export ORBITWARS_SHAPING_CAPTURE=0.025
export ORBITWARS_SHAPING_CAPTURE_START=0.025
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999
export ORBITWARS_SHAPING_DEFENSE_EMPTY=0.015

CKPT_DIR="${CKPT_DIR:-./ckpt_multi_action_v17_frog_hist50}"
DEFAULT_RESUME="${CKPT_DIR}/ckpt_002199.pkl"
RESUME="${1:-$DEFAULT_RESUME}"
NUM_UPDATES="${2:-5000}"

if [ ! -f "$RESUME" ]; then
  echo "ERR: resume ckpt not found: $RESUME" >&2
  exit 1
fi

LOGDIR="logs/v17b_peak_ft_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "[v17b] resume=$RESUME  num_updates=$NUM_UPDATES"
echo "[v17b] capture=$ORBITWARS_SHAPING_CAPTURE  defense=$ORBITWARS_SHAPING_DEFENSE_EMPTY  (constant)"
echo "[v17b] logdir=$LOGDIR"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v17b_peak_ft.yaml \
  --log-dir "$LOGDIR" \
  --resume-from "$RESUME" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
