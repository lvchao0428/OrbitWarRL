#!/usr/bin/env bash
# Launch v14d binary coordinate search on 5090 / local GPU
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then
  PY=/home/charlie/anaconda3/bin/python
fi

mkdir -p logs ckpt_search logs/search

echo "[v14d_binary] stopping any old v14d train..."
pkill -f "multi_action_v14d|v14d_curriculum" 2>/dev/null || true
sleep 2

echo "[v14d_binary] dry-run trial estimate..."
"$PY" scripts/v14d_curriculum_search.py --space scripts/v14d_search_space.yaml --dry-run

echo "[v14d_binary] starting search..."
nohup "$PY" scripts/v14d_curriculum_search.py \
  --space scripts/v14d_search_space.yaml \
  >> logs/v14d_search.log 2>&1 &
echo "pid=$! log=logs/v14d_search.log state=logs/v14d_search.state.json"
