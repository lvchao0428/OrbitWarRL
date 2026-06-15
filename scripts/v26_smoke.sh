#!/usr/bin/env bash
# Gate0: unit + integration smoke before v26 pipeline starts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! "$PY" -c "import jax, numpy" 2>/dev/null; then
  PY="/home/charlie/anaconda3/bin/python"
fi
export PYTHON="$PY"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export ORBITWARS_SKIP_PARITY=1

echo "=== Gate0: test_rewards (incl ROI) ==="
"$PY" -m orbit_wars_rl.env.test_rewards

echo ""
echo "=== Gate0: test_capture_roi ==="
"$PY" -m orbit_wars_rl.features.test_capture_roi

echo ""
echo "=== Gate0: smoke_v26 ==="
"$PY" -m orbit_wars_rl.scripts.smoke_v26

echo ""
echo "=== Gate0: preflight ==="
bash scripts/test_v26_preflight.sh

echo ""
echo "Gate0 PASS"
