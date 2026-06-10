#!/usr/bin/env bash
# v14c — worth_it-gated hold（5090 / 本地 GPU）
#
# Usage:
#   bash scripts/v14c_train.sh
#   nohup bash scripts/v14c_train.sh > logs/v14c.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then
  PY=/home/charlie/anaconda3/bin/python
fi

mkdir -p logs ckpt_multi_action_v14c

export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.05
export ORBITWARS_SHAPING_CAPTURE=0.08
export ORBITWARS_SHAPING_RELEASE=0.01
export ORBITWARS_SHAPING_RELEASE_K=15
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.03
export ORBITWARS_SHAPING_ONE_SHIP_THRESH=3

echo "[v14c] starting from scratch — allow_hold=true force_emit_worth_it=true min_pct_bin=5"
exec "$PY" -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v14c.yaml \
  --log-dir logs/v14c
