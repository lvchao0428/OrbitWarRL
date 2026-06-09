#!/usr/bin/env bash
# v12_stage2_shaping.sh — 阶段二: 在 sparse 训练基础上加入微量 shaping
#
# 当阶段一训练到 ~3000 updates 后, 如果 WR vs v20 = 0 但 ev/garr 有改善,
# 可以从最佳 checkpoint resume 并加入微量 shaping 加速突破。
#
# 这遵循 Frog Parade 的原则:
#   "I used MATCH_WINNER initially... then switched to FINAL_WINNER"
#   先用 dense 跑稳 → 再切 sparse
#   我们反过来: 先 sparse 跑出基础 → 微量 shaping 精调
#
# Usage:
#   bash scripts/v12_stage2_shaping.sh <resume_ckpt>
#   RESUME=ckpt_multi_action_v12_lux/ckpt_002000.pkl bash scripts/v12_stage2_shaping.sh
#   FOREGROUND=1 bash scripts/v12_stage2_shaping.sh <resume_ckpt>

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else PY=python; fi
else PY="$PYTHON"; fi

RESUME="${1:-${RESUME:-}}"
if [ -z "$RESUME" ]; then
  echo "[stage2] ERROR: 需要指定 resume checkpoint"
  echo "  Usage: bash scripts/v12_stage2_shaping.sh <ckpt_path>"
  exit 1
fi
if [ ! -f "$RESUME" ]; then
  echo "[stage2] ERROR: checkpoint 不存在: $RESUME"
  exit 1
fi

CFG="orbit_wars_rl/configs/multi_action_v12_lux.yaml"
NUM_UPDATES="${NUM_UPDATES:-5000}"
JAX_FRAC="${JAX_FRAC:-0.90}"
LOG="${LOG:-logs/v12_stage2_shaping.log}"
TB="${TB:-logs/v12_stage2_shaping}"
FOREGROUND="${FOREGROUND:-0}"

mkdir -p logs

ARGS=(--config "$CFG" --log-dir "$TB" --resume-from "$RESUME" --num-updates "$NUM_UPDATES")
ARGS+=(--ckpt-dir "ckpt_multi_action_v12_stage2")

# 微量 shaping: 只保留最有效的 CAPTURE (prod-weighted flip bonus)
# 系数极小, 不会主导 ±1 terminal reward, 但提供 early-game gradient
TRAIN_ENV=(
  ORBITWARS_SHAPING_SCALE=0.0
  ORBITWARS_SHAPING_CAPTURE=0.02
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

echo "[stage2] resume=$RESUME  updates=$NUM_UPDATES"
echo "[stage2] shaping: CAPTURE=0.02 (微量, 其余全部=0)"
echo "[stage2] log=$LOG"

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
