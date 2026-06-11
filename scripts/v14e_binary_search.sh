#!/usr/bin/env bash
# Launch v14e binary coordinate search — anti-hoard fix
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then
  PY=/home/charlie/anaconda3/bin/python
fi

mkdir -p logs ckpt_search logs/search

echo "[v14e_binary] stopping any old v14d/v14e train..."
pkill -f "v14d_curriculum_search|v14e_curriculum_search|multi_action_v14d|multi_action_v14e" 2>/dev/null || true
sleep 2

echo "[v14e_binary] dry-run trial estimate..."
"$PY" scripts/v14d_curriculum_search.py --space scripts/v14e_search_space.yaml --state logs/v14e_search.state.json --dry-run

echo "[v14e_binary] starting search..."
nohup "$PY" scripts/v14d_curriculum_search.py \
  --space scripts/v14e_search_space.yaml \
  --state logs/v14e_search.state.json \
  >> logs/v14e_search.log 2>&1 &
echo "pid=$! py=$PY log=logs/v14e_search.log state=logs/v14e_search.state.json"
