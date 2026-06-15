#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/v30b_pipeline
nohup bash scripts/v30b_pipeline.sh "$@" > logs/v30b_pipeline/pipeline.log 2>&1 &
echo $! > logs/v30b_pipeline/pipeline.pid
echo "v30b pipeline pid=$(cat logs/v30b_pipeline/pipeline.pid)"
