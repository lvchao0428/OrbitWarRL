#!/usr/bin/env bash
# run_v13_hold.sh — v13: allow_hold + split PPO + hold shaping + tighter stability
#
# 核心修复 (基于 v12/v13 replay 诊断):
#   1. allow_hold=true: 模型可以选择不发射 (学会积累)
#   2. Split PPO loss: emit head 独立 clip, 权重 60% (不被 src/dst/pct 梯度淹没)
#   3. emit_hard_stop + raised threshold: 需要 30% overkill margin 才触发 emit
#   4. hold_bonus shaping: 不发射且 garrison > 20 时给 +0.01 立即奖励
#   5. release_bonus: 从高产星球释放大舰队给正奖励
#   6. PPO stability: clip_eps=0.10, update_epochs=1, target_kl=0.02
#
# 用法:
#   bash scripts/run_v13_hold.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="orbit_wars_rl/configs/multi_action_v13_hold.yaml"
LOG="logs/v13_hold.log"
mkdir -p logs

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python; fi

# Reward shaping: capture + release (NOT hold_bonus — causes turtle)
export ORBITWARS_SHAPING_HOLD_BONUS="0.0"
export ORBITWARS_SHAPING_CAPTURE="0.05"
export ORBITWARS_SHAPING_PROD_SHARE_DELTA="0.02"
export ORBITWARS_SHAPING_RELEASE="0.005"
export ORBITWARS_SHAPING_RELEASE_K="15.0"

echo "[v13_hold] config=$CONFIG"
echo "[v13_hold] key changes: split_ppo, hold_bonus=0.01, release=0.005, clip=0.10, target_kl=0.02"
echo "[v13_hold] logging to $LOG"
echo ""

"$PY" -m orbit_wars_rl.scripts.train --config "$CONFIG" 2>&1 | tee "$LOG"
