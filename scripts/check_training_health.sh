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
tail -1 "$LOG" | grep "^upd " || echo "(no update line at end)"

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
LAST=$(tail -1 "$LOG" | grep "^upd " || echo "")
if [ -n "$LAST" ]; then
  echo "$LAST" | sed 's/.*emits \([0-9.]*\)/  emits=\1/' | sed 's/.*spf \([0-9.]*\)/  spf=\1/' | sed 's/.*z0 \([0-9.]*\)/  z0=\1/' | sed 's/.*garr \([0-9.]*\)/  garr=\1/' | sed 's/.*e2 \([0-9.]*\)/  e2=\1/'
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
LAST_KL=$(tail -1 "$LOG" | grep "^upd " | sed 's/.*kl \([+-][0-9.]*\).*/\1/' 2>/dev/null || echo "0")
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
