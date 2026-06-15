#!/usr/bin/env bash
# v24: PPO fine-tune from the v20 BC clone, KL-anchored.
#
# Usage:
#   bash scripts/v24_bc_ft.sh [num_updates]
set -euo pipefail

# Same reward set as v23 Route A: flip-gated capture is precision shaping on
# top of a policy that already knows how to flip (cloned from v20).
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

NUM_UPDATES="${1:-4000}"

LOGDIR="logs/v24_bc_ft_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "═══════════════════════════════════════════════════"
echo " v24 — PPO fine-tune from v20 BC clone"
echo "═══════════════════════════════════════════════════"
echo " init     : ckpt_bc_v20_epw25/ckpt_final.pkl (BC of v20, epw2.5)"
echo " anchor   : KL->BC kl_ref_coef=0.05"
echo " opp      : strong(BC) 40% + frozen 30% + rand 30%"
echo " reward   : flip-gated capture=0.02 fleet_scale=0.01"
echo "            multi_emit=0.02 anti_hoard=0.04"
echo " masks    : faithful-to-BC (no hard stop, min_pct_bin=0)"
echo " updates  : $NUM_UPDATES"
echo " logdir   : $LOGDIR"
echo "═══════════════════════════════════════════════════"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v24_bc_ft.yaml \
  --log-dir "$LOGDIR" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
