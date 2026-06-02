#!/usr/bin/env bash
# Post-train f38 eval: replay checkpoints vs v20.
#
# Three ckpt dirs (one per stage):
#   Stage 1: ckpt_multi_action_v11_f38/@199, @499
#   Stage 2: ckpt_multi_action_v11_f38_s2/@199, @499
#   Stage 3: ckpt_multi_action_v11_f38_s3/@199, @499, @999, @1999

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi

echo "=== f38 eval: all stages vs v20 ==="
echo "=== training log health ==="
bash scripts/check_training_health.sh logs/v11_f38.log || true

# Stage 1
for U in 199 499; do
  CKPT="ckpt_multi_action_v11_f38/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then echo "[skip] $CKPT not found"; continue; fi
  bash scripts/quick_replay.sh "$CKPT" "v11_f38_s1_u${U}"
done

# Stage 2
for U in 199 499; do
  CKPT="ckpt_multi_action_v11_f38_s2/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then echo "[skip] $CKPT not found"; continue; fi
  bash scripts/quick_replay.sh "$CKPT" "v11_f38_s2_u${U}"
done

# Stage 3
for U in 199 499 999 1999; do
  CKPT="ckpt_multi_action_v11_f38_s3/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then echo "[skip] $CKPT not found"; continue; fi
  bash scripts/quick_replay.sh "$CKPT" "v11_f38_s3_u${U}"
done

echo ""
echo "=== gate summaries ==="
for TAG in v11_f38_s1_u199 v11_f38_s1_u499 \
           v11_f38_s2_u199 v11_f38_s2_u499 \
           v11_f38_s3_u199 v11_f38_s3_u499 v11_f38_s3_u999 v11_f38_s3_u1999; do
  SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    head -1 "$SUM"
  fi
done
