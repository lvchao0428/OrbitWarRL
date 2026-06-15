#!/usr/bin/env bash
# v30c smoke: feature tests + 1 PPO update (no BC)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python3}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1
echo "=== Gate0: test_v30_econ (v30c features) ==="
"$PYTHON" -m orbit_wars_rl.features.test_v30_econ
echo "=== Gate1: smoke_v30c ==="
"$PYTHON" -m orbit_wars_rl.scripts.smoke_v30c
echo "[ALL PASS] v30c_smoke"
