#!/usr/bin/env bash
# Seed=0 pair_roi calibration loop (feature report + optional ckpt sweep).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPLAY="${REPLAY:-logs/replay_html/v29_u3199_s0/replay.json}"
TURN="${TURN:-}"  # empty => first launch turn in replay
CKPT="${CKPT:-}"  # e.g. ckpt_multi_action_v29_aim/ckpt_003199.pkl

ARGS=(--replay "$REPLAY" --target-id 20 --home-id 12)
if [[ -n "$TURN" ]]; then ARGS+=(--turn "$TURN"); fi
if [[ -n "$CKPT" ]]; then ARGS+=(--ckpt "$CKPT"); fi

python3 -m orbit_wars_rl.scripts.calibrate_pair_roi_seed "${ARGS[@]}"
