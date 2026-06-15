#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/v30_pipeline
nohup bash scripts/v30_pipeline.sh "$@" > logs/v30_pipeline/pipeline.log 2>&1 &
echo $! > logs/v30_pipeline/pipeline.pid
echo "v30 pipeline pid=$(cat logs/v30_pipeline/pipeline.pid)"
