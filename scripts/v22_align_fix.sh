#!/usr/bin/env bash
# v22: structural fix after v21 — mixed opponents + flip-gated capture + fresh train
#
# Usage:
#   bash scripts/v22_align_fix.sh
#   bash scripts/v22_align_fix.sh [num_updates]
set -euo pipefail

# ── Reward: terminal + flip-gated capture + anti_hoard safety net ──
export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_CAPTURE=0.02
export ORBITWARS_SHAPING_CAPTURE_START=0.02
export ORBITWARS_CAPTURE_ANNEAL_UPDATES=999999

export ORBITWARS_SHAPING_ANTI_HOARD=0.02
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.55

# Everything else off (no shaping soup).
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

# Inline eval: skip jax/numpy parity (2/16 edge diffs); still runs full replay vs v20.
export ORBITWARS_SKIP_PARITY=1

NUM_UPDATES="${1:-10000}"

LOGDIR="logs/v22_align_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "═══════════════════════════════════════════════════"
echo " v22 Align-Fix — FRESH START (do NOT resume v21)"
echo "═══════════════════════════════════════════════════"
echo " selfplay : 50% frozen v16a pool + 50% rand"
echo " reward   : terminal ±1 + flip-gated capture=0.02"
echo " policy   : ent_emit=0.004  min_pct_bin=3  allow_hold"
echo " eval     : inline vs-v20 every 100u (SKIP_PARITY=1)"
echo " updates  : $NUM_UPDATES"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v22_align_fix.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
