#!/usr/bin/env bash
# v12_milestone_eval.sh — 批量评估 v12 训练里程碑 checkpoints
#
# 自动评估 ckpt_dir 中的所有 checkpoint, 输出 TSV 摘要表
#
# 用法:
#   bash scripts/v12_milestone_eval.sh                    # 默认 ckpt 目录
#   bash scripts/v12_milestone_eval.sh ckpt_multi_action_v12_lux
#   SKIP_EXISTING=1 bash scripts/v12_milestone_eval.sh    # 跳过已评估的
#
# 输出:
#   logs/v12_milestone_eval.tsv   (追加模式)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT_DIR="${1:-ckpt_multi_action_v12_lux}"
TSV="logs/v12_milestone_eval.tsv"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NUM_GAMES="${NUM_GAMES:-5}"
PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python; fi

if [ ! -d "$CKPT_DIR" ]; then
  echo "[milestone] checkpoint 目录不存在: $CKPT_DIR"
  exit 1
fi

mkdir -p logs

# 写 header
if [ ! -f "$TSV" ]; then
  printf "timestamp\tckpt\tupdate\tWR\twins\tlosses\tdraws\tspf\tgarr\tz0\te2_plus\tflip\n" > "$TSV"
fi

# 遍历 checkpoint
for CKPT in $(ls -1 "$CKPT_DIR"/ckpt_*.pkl 2>/dev/null | sort); do
  UPD=$(basename "$CKPT" .pkl | sed 's/ckpt_0*//')
  TAG="v12_u${UPD}"
  SUMMARY="logs/replay_analyze/${TAG}_vs_v20.summary.txt"

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$SUMMARY" ]; then
    echo "[milestone] 跳过已评估: $CKPT"
    continue
  fi

  echo "[milestone] 评估 $CKPT (tag=$TAG, games=$NUM_GAMES)..."
  NUM_GAMES="$NUM_GAMES" bash scripts/v12_eval_gate.sh "$CKPT" "$TAG" 2>&1 || {
    echo "[milestone] WARN: 评估失败 $CKPT"
    continue
  }

  # 解析结果追加 TSV
  if [ -f "$SUMMARY" ]; then
    "$PY" -c "
import re
with open('$SUMMARY') as f:
    txt = f.read()
m = re.search(r'spf=([\d.]+)\s+garr=([\d.]+)\s+flip=([\d.]+)%\s+z0=([\d.]+)%\s+e8=([\d.]+)%\s+e2\+=([\d.]+)%\s+WLD=(\d+)/(\d+)/(\d+)', txt)
if m:
    spf, garr, flip, z0, e8, e2p = [m.group(i) for i in range(1, 7)]
    w, l, d = int(m.group(7)), int(m.group(8)), int(m.group(9))
    total = w + l + d
    wr = f'{w/total:.2f}' if total > 0 else '0.00'
    import datetime
    ts = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f'{ts}\t$CKPT\t$UPD\t{wr}\t{w}\t{l}\t{d}\t{spf}\t{garr}\t{z0}\t{e2p}\t{flip}')
" >> "$TSV" 2>/dev/null || true
  fi
done

echo ""
echo "━━━ 里程碑评估摘要 ━━━"
if [ -f "$TSV" ]; then
  column -t -s $'\t' "$TSV"
fi
