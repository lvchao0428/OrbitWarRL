#!/usr/bin/env bash
# run_v13_hold.sh — v13: allow_hold + low entropy + min_pct_bin
#
# 核心修复 (基于 v12 replay 诊断):
#   1. allow_hold=true: 模型可以选择不发射 (学会积累)
#   2. entropy 降 10x: 加速收敛,停止随机乱射
#   3. min_pct_bin=2: 至少发 30% 兵力 (禁止 1-ship spam)
#   4. emit_hard_stop=true: 没有好目标时不发
#   5. 从 v12_lux_b ckpt_007399 继续训练
#
# 用法:
#   bash scripts/run_v13_hold.sh
#
# 监控:
#   bash scripts/watch_v12_lux.sh logs/v13_hold.log

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="orbit_wars_rl/configs/multi_action_v13_hold.yaml"
LOG="logs/v13_hold.log"
mkdir -p logs

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python; fi

echo "[v13_hold] config=$CONFIG"
echo "[v13_hold] resume from ckpt_multi_action_v12_lux_b/ckpt_007399.pkl"
echo "[v13_hold] key changes: allow_hold=true, ent=0.001, min_pct_bin=2, emit_hard_stop=true"
echo "[v13_hold] logging to $LOG"
echo ""

"$PY" -m orbit_wars_rl.scripts.train --config "$CONFIG" 2>&1 | tee "$LOG"
