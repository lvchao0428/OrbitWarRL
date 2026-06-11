#!/usr/bin/env bash
# v14f Phase C long-train (10k updates) — run after B confirm passes
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then
  PY=/home/charlie/anaconda3/bin/python
fi

READY_JSON="logs/v14f_c_longtrain.ready.json"
if [ ! -f "$READY_JSON" ]; then
  echo "[v14f_c_long] ERROR: $READY_JSON not found — Phase B hasn't passed yet."
  exit 1
fi

RESUME_CKPT=$(python3 -c "import json; print(json.load(open('$READY_JSON'))['resume_ckpt'])")
if [ -z "$RESUME_CKPT" ] || [ "$RESUME_CKPT" = "null" ]; then
  echo "[v14f_c_long] ERROR: no resume_ckpt in $READY_JSON"
  exit 1
fi

echo "[v14f_c_long] resume from $RESUME_CKPT"

export ORBITWARS_SHAPING_SCALE=0.0
export ORBITWARS_SHAPING_HOLD_BONUS=0.0
export ORBITWARS_SHAPING_ANTI_HOARD=0.03
export ORBITWARS_SHAPING_ANTI_HOARD_THRESH=0.6
export ORBITWARS_SHAPING_CAPTURE=0.06
export ORBITWARS_SHAPING_RELEASE=0.005
export ORBITWARS_SHAPING_RELEASE_K=15
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.02
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.02
export ORBITWARS_SHAPING_ONE_SHIP_THRESH=3

mkdir -p ckpt_multi_action_v14f_c_long logs

nohup "$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v14e_phase_c_long.yaml \
  --log-dir logs/v14f_c_long \
  --num-updates 10000 \
  --resume-from "$RESUME_CKPT" \
  >> logs/v14f_phase_c_long.log 2>&1 &

echo "pid=$! log=logs/v14f_phase_c_long.log"
