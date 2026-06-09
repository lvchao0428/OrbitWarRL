#!/usr/bin/env bash
# v12_replay_html.sh — 导出 v12 ckpt 并生成 HTML replay (浏览器可视化)
#
# 用法 (在服务器上):
#   bash scripts/v12_replay_html.sh <ckpt> [seed]
#   bash scripts/v12_replay_html.sh ckpt_multi_action_v12_lux_b/ckpt_000199.pkl
#   bash scripts/v12_replay_html.sh ckpt_multi_action_v12_lux_b/ckpt_000199.pkl 42
#
# 然后在本地:
#   rsync -avz charlie@server:~/project/OrbitWarRL/logs/replay_html/ logs/replay_html/
#   open logs/replay_html/v12_<tag>/replay.html

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${1:?Usage: $0 <ckpt> [seed]}"
SEED="${2:-0}"

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python; fi

# 从 ckpt 路径提取 tag
UPD=$(basename "$CKPT" .pkl | sed 's/ckpt_0*//')
TAG="v12_u${UPD}_s${SEED}"
SUB="submission_rl_${TAG}.py"
OUT_DIR="logs/replay_html/${TAG}"

echo "[replay] ckpt=$CKPT  seed=$SEED  tag=$TAG"

# Step 1: 导出 submission (用 quick_replay 的自动模板检测)
if [ ! -f "$SUB" ]; then
  echo "[replay] 导出 submission..."
  bash scripts/quick_replay.sh "$CKPT" "$TAG" 2>&1 | tail -5 || {
    echo "[replay] WARN: quick_replay 失败, 尝试手动导出"
    # 手动导出 fallback
    TEMPLATE=$(ls -1 submission_rl_v11_f37.py 2>/dev/null | head -1 || echo "")
    if [ -z "$TEMPLATE" ]; then
      echo "[replay] ERROR: 找不到 submission 模板"
      exit 1
    fi
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
      "$PY" -m orbit_wars_rl.scripts.export_submission \
        --ckpt "$CKPT" --template "$TEMPLATE" --out "$SUB" || exit 1
  }
fi

if [ ! -f "$SUB" ]; then
  echo "[replay] ERROR: submission 文件未生成: $SUB"
  exit 1
fi

# Step 2: 生成 HTML replay
echo "[replay] 运行对局 seed=$SEED ..."
V20="submission_v20_0513.py"
if [ ! -f "$V20" ]; then
  echo "[replay] WARN: v20 不存在, 使用 self-play"
  V20="$SUB"
fi

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.scripts.replay_html \
    --agent-a "$SUB" \
    --agent-b "$V20" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR"

echo ""
echo "━━━ 完成 ━━━"
echo "  HTML: $OUT_DIR/replay.html"
echo ""
echo "  同步到本地:"
echo "    rsync -avz charlie@server:~/project/OrbitWarRL/$OUT_DIR/ $OUT_DIR/"
echo "    open $OUT_DIR/replay.html"
