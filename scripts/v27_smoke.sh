#!/usr/bin/env bash
# v27 Gate0: unit tests + integration smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export ORBITWARS_SKIP_PARITY=1

PY="${PYTHON:-python3}"
if ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "=== Gate0: test_v27_frontier ==="
"$PY" -m orbit_wars_rl.features.test_v27_frontier

echo ""
echo "=== Gate0: test_capture_roi (regression) ==="
"$PY" -m orbit_wars_rl.features.test_capture_roi

echo "=== Gate0b: test_capture_roi_seed_u3999 (calibration) ==="
"$PY" -m orbit_wars_rl.features.test_capture_roi_seed_u3999

echo ""
echo "=== Gate0: smoke_v27 ==="
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.scripts.smoke_v27

echo ""
echo "[Gate0 PASS] v27 smoke"
