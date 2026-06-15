#!/usr/bin/env bash
# Start v29 pipeline on remote (called via ssh — avoid pkill in ssh argv).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RPY="${PYTHON:-/home/charlie/anaconda3/bin/python}"
mkdir -p logs/v29_pipeline

if [ -f logs/v29_pipeline/pipeline.pid ]; then
  kill "$(cat logs/v29_pipeline/pipeline.pid)" 2>/dev/null || true
fi
pkill -f 'orbit_wars_rl.scripts.train.*multi_action_v29' 2>/dev/null || true
pkill -f 'orbit_wars_rl.scripts.train.*multi_action_v28' 2>/dev/null || true

EXTRA=("$@")
if [ ${#EXTRA[@]} -gt 0 ]; then
  nohup env PYTHON="$RPY" bash scripts/v29_pipeline.sh "${EXTRA[@]}" \
    >> logs/v29_pipeline/nohup.log 2>&1 &
else
  nohup env PYTHON="$RPY" bash scripts/v29_pipeline.sh --skip-post \
    >> logs/v29_pipeline/nohup.log 2>&1 &
fi
echo $! > logs/v29_pipeline/pipeline.pid
sleep 3
tail -8 logs/v29_pipeline/nohup.log
