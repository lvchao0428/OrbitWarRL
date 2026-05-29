#!/usr/bin/env bash
# check_training_health.sh — Extract training health metrics from any v11 log.
#
# Usage:
#   bash scripts/check_training_health.sh logs/v11_f32.log
#   bash scripts/check_training_health.sh logs/v11_f31.log
#
# Outputs: clip_frac trend, ent trends, loss trend, SPS, update count.
# Warns if clip_frac > 0.20 (top player's warning sign).

set -euo pipefail

LOG="${1:?Usage: $0 <logfile>}"

if [ ! -f "$LOG" ]; then
  echo "[ERROR] Log file not found: $LOG"
  exit 1
fi

echo "=== Training Health Check: $LOG ==="
echo ""

# Total updates
TOTAL=$(grep -c "^upd " "$LOG" 2>/dev/null || echo 0)
echo "Total updates logged: $TOTAL"

if [ "$TOTAL" -eq 0 ]; then
  echo "[WARN] No training updates found in log."
  exit 0
fi

# Latest update
echo ""
echo "--- Latest Update ---"
grep "^upd " "$LOG" | tail -1 || echo "(no update lines found)"

# clip_frac trend (last 10 updates)
echo ""
echo "--- clip_frac Trend (last 10) ---"
grep "^upd " "$LOG" | tail -10 | sed 's/.*clip \([0-9.]*\).*/\1/' | while read -r val; do
  warn=""
  if command -v awk >/dev/null 2>&1; then
    is_warn=$(awk "BEGIN { print ($val > 0.20) ? 1 : 0 }")
    if [ "$is_warn" = "1" ]; then
      warn=" [WARN: >0.20]"
    fi
  fi
  echo "  clip_frac=$val$warn"
done

# Max clip_frac
echo ""
echo "--- clip_frac Stats ---"
MAX_CLIP=$(grep "^upd " "$LOG" | sed 's/.*clip \([0-9.]*\).*/\1/' | sort -rn | head -1)
echo "  max clip_frac=$MAX_CLIP"
if command -v awk >/dev/null 2>&1; then
  is_warn=$(awk "BEGIN { print ($MAX_CLIP > 0.20) ? 1 : 0 }")
  if [ "$is_warn" = "1" ]; then
    echo "  [CRITICAL] clip_frac > 0.20 detected! Cut lr or revert capacity."
  fi
fi

# Entropy trends
echo ""
echo "--- Entropy Trends (last 5) ---"
grep "^upd " "$LOG" | tail -5 | sed 's/.*ent\[s\/d\/p\/e\] \([^ ]*\).*/\1/' | while read -r val; do
  echo "  ent[s/d/p/e]=$val"
done

# Loss trend
echo ""
echo "--- Loss Trend (last 5) ---"
grep "^upd " "$LOG" | tail -5 | sed 's/.*loss \([+-][0-9.]*\).*/\1/' | while read -r val; do
  echo "  loss=$val"
done

# SPS
echo ""
echo "--- SPS ---"
grep "^upd " "$LOG" | tail -1 | sed 's/.*sps \([0-9]*\).*/\1/' | while read -r val; do
  echo "  current SPS=$val"
done

# Key gameplay metrics
echo ""
echo "--- Gameplay Metrics (last update) ---"
LAST=$(grep "^upd " "$LOG" | tail -1 || echo "")
if [ -n "$LAST" ]; then
  echo "$LAST" | awk '{
    for (i=1;i<=NF;i++) {
      if ($i=="emits") emits=$(i+1)
      if ($i=="spf") spf=$(i+1)
      if ($i=="z0") z0=$(i+1)
      if ($i=="garr") garr=$(i+1)
      if ($i=="e2") e2=$(i+1)
      if ($i=="clip") clip=$(i+1)
    }
    printf "  emits=%s  spf=%s  z0=%s  garr=%s  e2=%s  clip=%s\n", emits, spf, z0, garr, e2, clip
  }'
fi

# Self-play inflation detector (spf spike + z0 collapse)
echo ""
echo "--- Self-Play Inflation Check ---"
if command -v awk >/dev/null 2>&1; then
  awk '
    /^upd / {
      upd = $2 + 0
      for (i=1;i<=NF;i++) {
        if ($i=="spf") spf = $(i+1) + 0
        if ($i=="z0") z0 = $(i+1) + 0
        if ($i=="emits") emits = $(i+1) + 0
      }
      if (upd <= 200) { early_n++; early_spf += spf }
      if (upd >= 350 && upd <= 420) { pre_n++; pre_spf += spf; pre_z0 += z0 }
      last_spf = spf; last_z0 = z0; last_emits = emits; last_upd = upd
    }
    END {
      if (early_n > 0) early_avg = early_spf / early_n; else early_avg = 0
      if (pre_n > 0) {
        pre_avg_spf = pre_spf / pre_n
        pre_avg_z0 = pre_z0 / pre_n
        printf "  early_avg_spf (upd<=200) = %.1f\n", early_avg
        printf "  pre_collapse (upd350-420) spf=%.1f z0=%.1f%%\n", pre_avg_spf, pre_avg_z0 * 100
      }
      if (last_upd != "") {
        printf "  latest (upd %d) spf=%.1f z0=%.1f%% emits=%.2f\n", last_upd, last_spf, last_z0 * 100, last_emits
        if (early_avg > 0 && last_spf > early_avg * 4)
          printf "  [WARN] spf > 4x early avg — likely self-play inflation (replay before promote)\n"
        if (last_z0 < 0.02)
          printf "  [WARN] z0 < 2%% — hyper-emit (training metric, check replay vs v20)\n"
      }
    }
  ' "$LOG"
fi

# Warnings
echo ""
echo "--- Warnings ---"
WARN_COUNT=0

# Check clip_frac creep
if command -v awk >/dev/null 2>&1; then
  CLIP_CREEP=$(grep "^upd " "$LOG" | sed 's/.*clip \([0-9.]*\).*/\1/' | awk 'NR<=5{first+=$1} NR>5{last+=$1} END{if(NR>5) print (last/(NR-5)) - (first/5); else print 0}')
  is_creep=$(awk "BEGIN { print ($CLIP_CREEP > 0.05) ? 1 : 0 }")
  if [ "$is_creep" = "1" ]; then
    echo "  [WARN] clip_frac creeping up (+$CLIP_CREEP avg). Monitor closely."
    WARN_COUNT=$((WARN_COUNT + 1))
  fi
fi

# Check KL divergence
LAST_KL=$(grep "^upd " "$LOG" | tail -1 | sed 's/.*kl \([+-][0-9.]*\).*/\1/' 2>/dev/null || echo "0")
if [ -n "$LAST_KL" ] && [ "$LAST_KL" != "0" ]; then
  is_high=$(awk "BEGIN { print ($LAST_KL > 0.05) ? 1 : 0 }" 2>/dev/null || echo "0")
  if [ "$is_high" = "1" ]; then
    echo "  [WARN] KL=$LAST_KL > 0.05. Policy changing too fast."
    WARN_COUNT=$((WARN_COUNT + 1))
  fi
fi

if [ "$WARN_COUNT" -eq 0 ]; then
  echo "  (none)"
fi

echo ""
echo "=== Done ==="
