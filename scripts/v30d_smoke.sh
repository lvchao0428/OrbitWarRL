#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python3}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1
echo "=== Gate0: test_v30_econ (v30d gate) ==="
"$PYTHON" -m orbit_wars_rl.features.test_v30_econ
echo "[ALL PASS] v30d_smoke"
