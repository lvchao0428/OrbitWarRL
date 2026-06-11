#!/usr/bin/env bash
# v17: hist=50 + ETA-lead dst + Frog curriculum (capture anneal -> sparse)
#
# Usage:
#   bash scripts/v17_frog_hist50.sh                          # latest ckpt, train until 06:00
#   bash scripts/v17_frog_hist50.sh <ckpt.pkl> [num_updates] # explicit resume / length
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-./ckpt_multi_action_v17_frog_hist50}"
CAPTURE_ANNEAL_TOTAL="${ORBITWARS_CAPTURE_ANNEAL_TOTAL:-8000}"
TARGET_HOUR="${TARGET_HOUR:-6}"   # local time, train until this hour (default 06:00)
UPD_PER_SEC="${UPD_PER_SEC:-1.20}" # conservative vs ~1.25 observed on 5090

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

_resolve_resume() {
  if [ -n "${1:-}" ] && [ -f "$1" ]; then
    echo "$1"
    return
  fi
  if [ -n "${1:-}" ] && [ -f "$CKPT_DIR/$1" ]; then
    echo "$CKPT_DIR/$1"
    return
  fi
  local latest
  latest="$(ls -t "$CKPT_DIR"/ckpt_*.pkl 2>/dev/null | head -1 || true)"
  if [ -z "$latest" ]; then
    echo "./ckpt_multi_action_v15_frog/ckpt_009999.pkl"
  else
    echo "$latest"
  fi
}

_ckpt_step() {
  local base step
  base="$(basename "$1" .pkl)"
  step="${base#ckpt_}"
  step="${step#0}"
  step="${step#0}"
  echo "${step:-0}"
}

_updates_until_target_hour() {
  python3 - <<PY
import datetime, math
now = datetime.datetime.now()
target = now.replace(hour=${TARGET_HOUR}, minute=0, second=0, microsecond=0)
if target <= now:
    target += datetime.timedelta(days=1)
left = (target - now).total_seconds()
rate = ${UPD_PER_SEC}
print(max(500, int(math.floor(left * rate))))
PY
}

_set_curriculum_from_prior() {
  local prior="$1"
  local remain cap_start def_start
  if [ "$prior" -ge "$CAPTURE_ANNEAL_TOTAL" ]; then
    export ORBITWARS_SHAPING_CAPTURE=0.0
    export ORBITWARS_SHAPING_CAPTURE_START=0.0
    export ORBITWARS_CAPTURE_ANNEAL_UPDATES=1
    export ORBITWARS_SHAPING_DEFENSE_EMPTY=0.01
    return
  fi
  remain=$((CAPTURE_ANNEAL_TOTAL - prior))
  [ "$remain" -lt 1 ] && remain=1
  cap_start="$(python3 - <<PY
prior = ${prior}
total = ${CAPTURE_ANNEAL_TOTAL}
print(f"{0.03 * (1.0 - prior / total):.6f}")
PY
)"
  def_start="$(python3 - <<PY
prior = ${prior}
total = ${CAPTURE_ANNEAL_TOTAL}
t = min(1.0, prior / total)
print(f"{0.02 * (1.0 - t * 0.5):.6f}")
PY
)"
  export ORBITWARS_SHAPING_CAPTURE="$cap_start"
  export ORBITWARS_SHAPING_CAPTURE_START="$cap_start"
  export ORBITWARS_CAPTURE_ANNEAL_UPDATES="$remain"
  export ORBITWARS_SHAPING_DEFENSE_EMPTY="$def_start"
}

RESUME="$(_resolve_resume "${1:-}")"
CKPT_STEP="$(_ckpt_step "$RESUME")"

if [ -n "${2:-}" ]; then
  NUM_UPDATES="$2"
else
  NUM_UPDATES="$(_updates_until_target_hour)"
fi

_set_curriculum_from_prior "$CKPT_STEP"

LOGDIR="logs/v17_frog_hist50_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "[v17] resume=$RESUME  prior_step≈$CKPT_STEP  num_updates=$NUM_UPDATES"
echo "[v17] capture_start=$ORBITWARS_SHAPING_CAPTURE_START  anneal=$ORBITWARS_CAPTURE_ANNEAL_UPDATES  defense=$ORBITWARS_SHAPING_DEFENSE_EMPTY"
echo "[v17] logdir=$LOGDIR  target~${TARGET_HOUR}:00"

"$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v17_frog_hist50.yaml \
  --log-dir "$LOGDIR" \
  --resume-from "$RESUME" \
  --num-updates "$NUM_UPDATES" \
  2>&1 | tee "$LOGDIR/train.log"
