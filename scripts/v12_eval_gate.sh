#!/usr/bin/env bash
# v12_eval_gate.sh — 评估 v12 checkpoint 对 v20 的表现
#
# 工作流:
#   1. 从 checkpoint 导出 submission (自动检测模板)
#   2. 运行 replay vs v20 (N 局)
#   3. 输出一行摘要
#
# 用法:
#   bash scripts/v12_eval_gate.sh <ckpt_path> <tag>
#   bash scripts/v12_eval_gate.sh ckpt_multi_action_v12_lux/ckpt_001000.pkl v12_u1000
#   NUM_GAMES=10 bash scripts/v12_eval_gate.sh <ckpt> <tag>
#
# 前置要求:
#   - kaggle_environments 已安装
#   - submission_v20_0513.py 存在

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${1:?Usage: $0 <ckpt> <tag>}"
TAG="${2:?Usage: $0 <ckpt> <tag>}"
NUM_GAMES="${NUM_GAMES:-5}"
PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python; fi

echo "[v12_eval] ckpt=$CKPT  tag=$TAG  games=$NUM_GAMES"

# 使用 quick_replay.sh (已有完整的导出+replay流程)
if [ -f scripts/quick_replay.sh ]; then
  bash scripts/quick_replay.sh "$CKPT" "$TAG"
else
  echo "[v12_eval] ERROR: scripts/quick_replay.sh not found"
  exit 1
fi

# 显示结果
SUMMARY="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
if [ -f "$SUMMARY" ]; then
  echo ""
  echo "━━━ 评估结果 ━━━"
  cat "$SUMMARY"
  echo ""

  # 关键决策指标
  echo "━━━ 决策参考 ━━━"
  "$PY" -c "
import re
with open('$SUMMARY') as f:
    txt = f.read()
m = re.search(r'WLD=(\d+)/(\d+)/(\d+)', txt)
if m:
    w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    total = w + l + d
    wr = w / total if total > 0 else 0
    print(f'  WR vs v20 = {wr:.1%}  ({w}W/{l}L/{d}D)')
    if wr >= 0.5:
        print(f'  → 阶段一目标达成 (WR >= 50%)')
    elif wr >= 0.3:
        print(f'  → 有进步但未达标, 继续训练')
    else:
        print(f'  → 表现不足, 考虑微量 shaping 或回退')
" 2>/dev/null || true
fi
