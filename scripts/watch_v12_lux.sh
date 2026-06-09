#!/usr/bin/env bash
# watch_v12_lux.sh — v12_lux 训练过程实时观测仪表板
#
# 用法:
#   bash scripts/watch_v12_lux.sh                    # 本地日志
#   bash scripts/watch_v12_lux.sh logs/v12_lux.log   # 指定日志
#   LOG=remote_v12.log bash scripts/watch_v12_lux.sh # 远程同步日志
#
# 同步远程日志:
#   rsync -avz charlie@server:/home/charlie/project/OrbitWarRL/logs/v12_lux.log logs/

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${1:-${LOG:-$ROOT/logs/v12_lux.log}}"

if [ ! -f "$LOG" ]; then
  echo "[watch] 日志不存在: $LOG"
  echo "  远程同步: rsync -avz charlie@server:/path/to/logs/v12_lux.log logs/"
  exit 1
fi

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python; fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            v12_lux Training Dashboard                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ Log: $LOG"
echo "╚══════════════════════════════════════════════════════════════╝"

# 1. 基本进度
echo ""
echo "━━━ 训练进度 ━━━"
TOTAL=$(grep -c "^upd " "$LOG" 2>/dev/null || echo 0)
echo "  已完成 updates: $TOTAL / 10000"
if [ "$TOTAL" -gt 0 ]; then
  FIRST_SPS=$(grep "^upd " "$LOG" | tail -1 | grep -oE "sps [0-9]+" | awk '{print $2}')
  STEPS=$(grep "^upd " "$LOG" | tail -1 | grep -oE "steps [0-9]+" | awk '{print $2}')
  echo "  当前 SPS: $FIRST_SPS  总步数: $STEPS"
  if [ -n "$FIRST_SPS" ] && [ "$FIRST_SPS" -gt 0 ]; then
    REMAINING=$((10000 - TOTAL))
    # 每 update = 4096 steps, 估算剩余时间
    EST_SEC=$((REMAINING * 4096 / FIRST_SPS))
    EST_HR=$((EST_SEC / 3600))
    EST_MIN=$(( (EST_SEC % 3600) / 60 ))
    echo "  预计剩余: ${EST_HR}h ${EST_MIN}m"
  fi
fi

# 2. 最新训练行
echo ""
echo "━━━ 最新一行 ━━━"
grep "^upd " "$LOG" | tail -1 || echo "(无训练输出)"

# 3. 关键指标趋势 (每 500 update 采样)
echo ""
echo "━━━ 关键指标趋势 (upd / ev / tR / spf / z0 / garr / e2 / emits) ━━━"
grep "^upd " "$LOG" | awk '
  NR==1 || NR%500==0 || NR==ENVIRON["TOTAL"] {
    upd=""; ev=""; tR=""; spf=""; z0=""; garr=""; e2=""; emits=""
    for(i=1;i<=NF;i++) {
      if($i=="upd") upd=$(i+1)
      if($i~/^ev$/) ev=$(i+1)
      if($i=="tR") tR=$(i+1)
      if($i=="spf") spf=$(i+1)
      if($i=="z0") z0=$(i+1)
      if($i=="garr") garr=$(i+1)
      if($i=="e2") e2=$(i+1)
      if($i=="emits") emits=$(i+1)
    }
    printf "  upd %5s  ev %6s  tR %5s  spf %5s  z0 %5s  garr %6s  e2 %5s  emits %s\n", upd, ev, tR, spf, z0, garr, e2, emits
  }
' TOTAL="$TOTAL"

# 4. 最近 10 行滑动窗口
echo ""
echo "━━━ 最近 10 个 update ━━━"
grep "^upd " "$LOG" | tail -10 | awk '{
  upd=""; ev=""; tR=""; spf=""; z0=""; garr=""; e2=""; clip=""; kl=""
  for(i=1;i<=NF;i++) {
    if($i=="upd") upd=$(i+1)
    if($i~/^ev$/) ev=$(i+1)
    if($i=="tR") tR=$(i+1)
    if($i=="spf") spf=$(i+1)
    if($i=="z0") z0=$(i+1)
    if($i=="garr") garr=$(i+1)
    if($i=="e2") e2=$(i+1)
    if($i=="clip") clip=$(i+1)
    if($i=="kl") kl=$(i+1)
  }
  printf "  %5s  ev=%6s  tR=%5s  spf=%5s  z0=%4s  garr=%6s  e2=%4s  clip=%4s  kl=%s\n", upd, ev, tR, spf, z0, garr, e2, clip, kl
}'

# 5. eval vs random
echo ""
echo "━━━ Eval vs Random (WR) ━━━"
grep "WRr" "$LOG" | tail -5 || echo "(无 eval 输出 — 首次在 upd 99)"

# 6. eval vs v20 (inline gate)
echo ""
echo "━━━ Eval vs v20 (inline gate) ━━━"
grep "^\[eval_vs_v20\]" "$LOG" | tail -5 || echo "(无 — 需要 eval_vs_v20=true)"

# 7. Checkpoints
echo ""
echo "━━━ 最近 Checkpoints ━━━"
grep "^\[ckpt\]" "$LOG" | tail -5 || echo "(无 checkpoint 保存)"

# 8. 健康检查
echo ""
echo "━━━ 健康检查 ━━━"
WARNS=0

# clip_frac
LAST_CLIP=$(grep "^upd " "$LOG" | tail -1 | grep -oE "clip [0-9.]+" | awk '{print $2}' || echo "0")
if [ -n "$LAST_CLIP" ]; then
  HIGH=$(awk "BEGIN { print ($LAST_CLIP > 0.20) ? 1 : 0 }")
  if [ "$HIGH" = "1" ]; then
    echo "  ⚠  clip_frac=$LAST_CLIP > 0.20 — 学习率可能过高"
    WARNS=$((WARNS + 1))
  fi
fi

# explained_variance
LAST_EV=$(grep "^upd " "$LOG" | tail -1 | grep -oE "ev [+-]?[0-9.]+" | awk '{print $2}' || echo "0")
if [ -n "$LAST_EV" ]; then
  LOW=$(awk "BEGIN { print ($LAST_EV < 0.3) ? 1 : 0 }")
  if [ "$LOW" = "1" ]; then
    echo "  ⚠  explained_variance=$LAST_EV < 0.3 — value head 学习不足"
    WARNS=$((WARNS + 1))
  fi
fi

# terminal reward (symmetric selfplay: tR 应该在 0 附近, 显著偏离说明有问题)
LAST_TR=$(grep "^upd " "$LOG" | tail -1 | grep -oE "tR [+-]?[0-9.]+" | awk '{print $2}' || echo "0")
if [ -n "$LAST_TR" ]; then
  ABS_TR=$(awk "BEGIN { v=$LAST_TR; if(v<0) v=-v; print v }")
  HIGH_TR=$(awk "BEGIN { print ($ABS_TR > 0.5) ? 1 : 0 }")
  if [ "$HIGH_TR" = "1" ]; then
    echo "  ⚠  |tR|=$ABS_TR > 0.5 — 对称 self-play 中 tR 应接近 0"
    WARNS=$((WARNS + 1))
  fi
fi

if [ "$WARNS" -eq 0 ]; then
  echo "  ✓  全部正常"
fi

# 9. v12 特有指标
echo ""
echo "━━━ v12 特有检查 ━━━"
# 确认 opp=symm (纯对称 self-play)
OPP_TAGS=$(grep "^upd " "$LOG" | awk '{for(i=1;i<=NF;i++) if($i=="opp") print $(i+1)}' | sort -u | tr '\n' ' ')
echo "  对手类型: $OPP_TAGS"
if echo "$OPP_TAGS" | grep -q "symm"; then
  echo "  ✓  symmetric self-play 已激活"
else
  echo "  ⚠  未检测到 opp=symm"
fi

# 确认 shaping=0
if grep -q "SHAPING_SCALE=0.0" "$LOG" 2>/dev/null; then
  echo "  ✓  sparse ±1 reward (shaping=0)"
else
  echo "  ⚠  可能有 reward shaping 残留"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
