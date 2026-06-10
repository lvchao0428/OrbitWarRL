#!/usr/bin/env bash
# Parallel schedule: v11 (2P) + v12 (4P) on one 5090.
#
# GPU is serial — two jobs cannot share VRAM. Modes:
#   pipeline  (default) — v11 3-phase (~12h), then v12 3-phase (~14h)
#   v11_only  — 2P validation only
#   v12_only  — 4P validation only (ORBITWARS_NUM_PLAYERS=4)
#
# Usage:
#   nohup bash scripts/run_parallel_tracks.sh \
#     > logs/parallel_tracks.launcher.log 2>&1 &
#
# If v11 is already running separately, do NOT start pipeline — use v12_only later:
#   nohup env MODE=v12_only bash scripts/run_parallel_tracks.sh \
#     > logs/v12_only.launcher.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-pipeline}"

echo "[parallel] MODE=$MODE  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

case "$MODE" in
  pipeline)
    echo "[parallel] Track A: v11 2P (3 x UPD_PER_PHASE)"
    bash scripts/run_v11_validation.sh
    echo "[parallel] Track B: v12 4P (3 x UPD_PER_PHASE)"
    bash scripts/run_v12_validation.sh
    ;;
  v11_only)
    bash scripts/run_v11_validation.sh
    ;;
  v12_only)
    bash scripts/run_v12_validation.sh
    ;;
  *)
    echo "Unknown MODE=$MODE (use pipeline|v11_only|v12_only)" >&2
    exit 1
    ;;
esac

echo "[parallel] done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
