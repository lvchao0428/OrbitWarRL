#!/usr/bin/env bash
# v29 Gate0: regression + pair ROI/dedup + integration smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export ORBITWARS_SKIP_PARITY=1

PY="${PYTHON:-python3}"
if ! "$PY" -c "import jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

echo "=== Gate0: test_v27_frontier (regression) ==="
"$PY" -m orbit_wars_rl.features.test_v27_frontier

echo ""
echo "=== Gate0: test_capture_roi ==="
"$PY" -m orbit_wars_rl.features.test_capture_roi

echo ""
echo "=== Gate0: test_capture_roi_seed_u3999 ==="
"$PY" -m orbit_wars_rl.features.test_capture_roi_seed_u3999

echo ""
echo "=== Gate0: test_v29_pair ==="
"$PY" -m orbit_wars_rl.features.test_v29_pair

echo ""
echo "=== Gate0: smoke_v29 ==="
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.scripts.smoke_v29

echo ""
echo "[Gate0 PASS] v29 smoke"
