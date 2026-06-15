#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/v30d_pipeline
nohup bash scripts/v30d_pipeline.sh "$@" > logs/v30d_pipeline/pipeline.log 2>&1 &
echo $! > logs/v30d_pipeline/pipeline.pid
echo "v30d pipeline pid=$(cat logs/v30d_pipeline/pipeline.pid)"
