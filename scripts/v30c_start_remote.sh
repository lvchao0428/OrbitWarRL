#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/v30c_pipeline
nohup bash scripts/v30c_pipeline.sh "$@" > logs/v30c_pipeline/pipeline.log 2>&1 &
echo $! > logs/v30c_pipeline/pipeline.pid
echo "v30c pipeline pid=$(cat logs/v30c_pipeline/pipeline.pid)"
