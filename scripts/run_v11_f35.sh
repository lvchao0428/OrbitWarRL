#!/usr/bin/env bash
# v11_f35: f33 arch + 5 src-quality/v20-target-score planet feats (28->33 dim)
#          + anti-1-ship & high-prod-capture reward shaping.
#
# Big combined swing at the verified root cause (DAY8 §11/§12): 55% of launches
# come from <=3-ship planets; the policy never builds a stockpile and never
# beats v20. Three deltas at once (user: "0->1, must SEE it work first").
#
# Usage:
#   bash scripts/run_v11_f35.sh
#   tail -f logs/v11_f35.log
#   bash scripts/check_training_health.sh logs/v11_f35.log
#
# After training:
#   bash scripts/run_f35_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
else
  PY="$PYTHON"
fi

JAX_FRAC="${JAX_FRAC:-0.85}"
CFG="orbit_wars_rl/configs/multi_action_v11_f35.yaml"
LOG="logs/v11_f35.log"
TB="logs/v11_f35"

# f33 base shaping + f35 anti-1-ship / high-prod-capture.
COMMON_SHAPING=(
  "ORBITWARS_SHAPING_EMIT_LOG=0.0"
  "ORBITWARS_SHAPING_EMIT_GATED=0"
  "ORBITWARS_SHAPING_PLANET_SHARE=0.005"
  "ORBITWARS_SHAPING_PROD_SHARE=0.0"
  "ORBITWARS_SHAPING_FLEET_LOG=0.0"
  "ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0"
  "ORBITWARS_SHAPING_RELEASE=0.05"
  "ORBITWARS_SHAPING_RELEASE_K=20.0"
  "ORBITWARS_SHAPING_CAPTURE=0.02"
  "ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.01"
  "ORBITWARS_SHAPING_ONE_SHIP_THRESH=3.0"
  "ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.05"
  "ORBITWARS_SHAPING_HIGH_PROD_THRESH=3.0"
)

mkdir -p logs

echo "[v11_f35] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f35] 33-dim feats (src-quality + v20 target_score) + anti-1-ship reward"
echo "[v11_f35] PPO=f33 recipe, frozen=0.25, 800 upd"

echo "[v11_f35] parity smoke (fresh-init, 33-dim, f33 masks)"
if ! env "${COMMON_SHAPING[@]}" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.inference.test_parity \
    --num-states 16 --emit-hard-stop --flip-hard-mask; then
  echo "[v11_f35] parity FAILED — aborting" >&2
  exit 1
fi

env "${COMMON_SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
echo "monitor: bash scripts/check_training_health.sh $LOG"
