#!/usr/bin/env bash
# v11_f38: curriculum training — shaped reward bootstrap -> anneal -> pure terminal + anchor.
#
# Three stages (each launched separately, resume from previous ckpt):
#   Stage 1 (500 upd): CAPTURE=0.05, PROD_SHARE_DELTA=0.02 — learn to flip planets
#   Stage 2 (500 upd): CAPTURE=0.025, PROD_SHARE_DELTA=0.01 — anneal shaping
#   Stage 3 (2000 upd): all shaping=0 + f29@599 anchor — pure terminal + external anchor
#
# Usage:
#   bash scripts/run_v11_f38.sh [stage]
#   stage = 1 (default), 2, or 3
#
#   tail -f logs/v11_f38.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-1}"

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
LOG="logs/v11_f38.log"
TB="logs/v11_f38"

mkdir -p logs

case "$STAGE" in
  1)
    CFG="orbit_wars_rl/configs/multi_action_v11_f38.yaml"
    RESUME=""
    SHAPING=(
      "ORBITWARS_SHAPING_CAPTURE=0.05"
      "ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.02"
      "ORBITWARS_SHAPING_EMIT_LOG=0.0"
      "ORBITWARS_SHAPING_EMIT_GATED=0"
      "ORBITWARS_SHAPING_PLANET_SHARE=0.0"
      "ORBITWARS_SHAPING_PROD_SHARE=0.0"
      "ORBITWARS_SHAPING_FLEET_LOG=0.0"
      "ORBITWARS_SHAPING_RELEASE=0.0"
      "ORBITWARS_SHAPING_RELEASE_K=20.0"
    )
    echo "[f38-s1] CAPTURE=0.05, PROD_SHARE_DELTA=0.02, 500 upd"
    ;;
  2)
    CFG="orbit_wars_rl/configs/multi_action_v11_f38_s2.yaml"
    RESUME="--resume-from ckpt_multi_action_v11_f38/ckpt_000499.pkl"
    SHAPING=(
      "ORBITWARS_SHAPING_CAPTURE=0.025"
      "ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.01"
      "ORBITWARS_SHAPING_EMIT_LOG=0.0"
      "ORBITWARS_SHAPING_EMIT_GATED=0"
      "ORBITWARS_SHAPING_PLANET_SHARE=0.0"
      "ORBITWARS_SHAPING_PROD_SHARE=0.0"
      "ORBITWARS_SHAPING_FLEET_LOG=0.0"
      "ORBITWARS_SHAPING_RELEASE=0.0"
      "ORBITWARS_SHAPING_RELEASE_K=20.0"
    )
    echo "[f38-s2] anneal: CAPTURE=0.025, PROD_SHARE_DELTA=0.01, resume from s1 @499"
    ;;
  3)
    CFG="orbit_wars_rl/configs/multi_action_v11_f38_s3.yaml"
    RESUME="--resume-from ckpt_multi_action_v11_f38_s2/ckpt_000499.pkl"
    SHAPING=(
      "ORBITWARS_SHAPING_CAPTURE=0.0"
      "ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0"
      "ORBITWARS_SHAPING_EMIT_LOG=0.0"
      "ORBITWARS_SHAPING_EMIT_GATED=0"
      "ORBITWARS_SHAPING_PLANET_SHARE=0.0"
      "ORBITWARS_SHAPING_PROD_SHARE=0.0"
      "ORBITWARS_SHAPING_FLEET_LOG=0.0"
      "ORBITWARS_SHAPING_RELEASE=0.0"
      "ORBITWARS_SHAPING_RELEASE_K=20.0"
    )
    echo "[f38-s3] pure terminal + f29 anchor, resume from s2 @499"
    ;;
  *)
    echo "usage: $0 [1|2|3]" >&2
    exit 1
    ;;
esac

echo "[f38] stage=$STAGE  py=$PY  cfg=$CFG  log=$LOG"

if [ "$STAGE" = "1" ]; then
  echo "[f38] parity smoke"
  if ! JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    "$PY" -m orbit_wars_rl.inference.test_parity \
      --num-states 16 \
      --emit-hard-stop \
      --flip-hard-mask; then
    echo "[f38] parity FAILED — aborting" >&2
    exit 1
  fi
fi

# shellcheck disable=SC2086
env "${SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
    $RESUME \
  >> "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
echo "monitor: bash scripts/check_training_health.sh $LOG"
