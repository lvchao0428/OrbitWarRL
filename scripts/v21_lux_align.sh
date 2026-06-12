#!/usr/bin/env bash
# v21: Lux-aligned — from scratch, sparse reward, symmetric self-play
#
# Key: extreme reward simplification (terminal + capture only).
# All dense shaping terms are OFF. Let the model learn from wins.
#
# Usage:
#   bash scripts/v21_lux_align.sh
#   bash scripts/v21_lux_align.sh [num_updates]
set -euo pipefail

# ── Reward: ONLY terminal ±1 + capture ──
export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_CAPTURE=0.02
export ORBITWARS_SHAPING_CAPTURE_START=0.02
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999

# Anti-hoard as a minimal safety net (very light).
export ORBITWARS_SHAPING_ANTI_HOARD=0.02
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.60

# Everything else explicitly zeroed.
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
export ORBITWARS_SHAPING_MULTI_EMIT=0.0
export ORBITWARS_SHAPING_MULTI_EMIT_MIN_SHIPS=0

NUM_UPDATES="${1:-15000}"

LOGDIR="logs/v21_lux_align_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "═══════════════════════════════════════════════════"
echo " v21 Lux-Aligned — FROM SCRATCH"
echo "═══════════════════════════════════════════════════"
echo " reward   : terminal ±1 + capture=0.02"
echo " selfplay : symmetric (same model = both players)"
echo " features : ETA-lead→all heads, planet hist, future garr"
echo " updates  : $NUM_UPDATES (~4h budget)"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v21_lux_align.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
