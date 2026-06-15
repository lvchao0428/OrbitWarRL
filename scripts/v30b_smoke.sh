#!/usr/bin/env bash
# v30b smoke: warmstart + 1 PPO update with freeze attn dst
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python3}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1
echo "=== Gate0: test_v30_econ ==="
"$PYTHON" -m orbit_wars_rl.features.test_v30_econ
echo "=== Gate1: smoke_v30b ==="
"$PYTHON" -m orbit_wars_rl.scripts.smoke_v30b
echo "[ALL PASS] v30b_smoke"
