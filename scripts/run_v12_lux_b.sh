#!/usr/bin/env bash
# v12_lux_b — 从 v12_lux 最佳 ckpt resume, 修正 clip_frac 过高
#
# 诊断: v12_lux @ upd 429, clip=0.47 kl=0.10 (PPO 更新过猛)
# 修正: lr 3e-4→1e-4, epochs 4→2, minibatches 8→4
#
# Usage:
#   bash scripts/run_v12_lux_b.sh
#   FOREGROUND=1 bash scripts/run_v12_lux_b.sh
#   NUM_UPDATES=5 bash scripts/run_v12_lux_b.sh   # smoke

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else PY=python; fi
else PY="$PYTHON"; fi

CFG="orbit_wars_rl/configs/multi_action_v12_lux_b.yaml"

# 自动选择最新 checkpoint
CKPT_DIR="ckpt_multi_action_v12_lux"
RESUME="${RESUME:-}"
if [ -z "$RESUME" ]; then
  RESUME=$(ls -1 "$CKPT_DIR"/ckpt_*.pkl 2>/dev/null | sort | tail -1 || true)
fi
if [ -z "$RESUME" ] || [ ! -f "$RESUME" ]; then
  echo "[v12_b] ERROR: 找不到 v12_lux checkpoint (尝试: $CKPT_DIR/)"
  echo "  手动指定: RESUME=<path> bash scripts/run_v12_lux_b.sh"
  exit 1
fi

NUM_UPDATES="${NUM_UPDATES:-}"
JAX_FRAC="${JAX_FRAC:-0.90}"
LOG="${LOG:-logs/v12_lux_b.log}"
TB="${TB:-logs/v12_lux_b}"
FOREGROUND="${FOREGROUND:-0}"

mkdir -p logs

ARGS=(--config "$CFG" --log-dir "$TB" --resume-from "$RESUME")
if [ -n "$NUM_UPDATES" ]; then
  ARGS+=(--num-updates "$NUM_UPDATES")
fi

TRAIN_ENV=(
  ORBITWARS_SHAPING_SCALE=0.0
  ORBITWARS_SHAPING_CAPTURE=0.0
  ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE=0.0
  ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0
  ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.0
  ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.0
  ORBITWARS_SHAPING_MULTI_EMIT=0.0
  ORBITWARS_SHAPING_EMIT_LOG=0.0
  ORBITWARS_SHAPING_EMIT_GATED=0
  ORBITWARS_SHAPING_PLANET_SHARE=0.0
  ORBITWARS_SHAPING_PROD_SHARE=0.0
  ORBITWARS_SHAPING_FLEET_LOG=0.0
  ORBITWARS_SHAPING_RELEASE=0.0
  ORBITWARS_SHAPING_KEEP_HOME=0.0
  ORBITWARS_SHAPING_FLEET_SIZE=0.0
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC"
)

echo "[v12_b] 修正 clip_frac: lr 1e-4, epochs=2, minibatches=4"
echo "[v12_b] resume=$RESUME"
echo "[v12_b] log=$LOG"
echo "[v12_b] 目标: clip < 0.15, kl < 0.03"

if [ "$FOREGROUND" = "1" ]; then
  env "${TRAIN_ENV[@]}" \
    "$PY" -m orbit_wars_rl.scripts.train "${ARGS[@]}" \
    2>&1 | tee -a "$LOG"
else
  env "${TRAIN_ENV[@]}" \
    "$PY" -m orbit_wars_rl.scripts.train "${ARGS[@]}" \
    >> "$LOG" 2>&1 &
  echo "pid=$!"
  echo "tail -f $LOG"
  echo "watch: bash scripts/watch_v12_lux.sh $LOG"
fi
