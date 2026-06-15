#!/usr/bin/env bash
# Parallel v20-vs-v20 BC data collection + merge.
#
# Usage:
#   bash scripts/bc_collect_parallel.sh [n_procs] [games_per_proc] [out_npz]
# Defaults: 8 procs x 50 games -> data/bc_v20_self_400g.npz
#
# Collection is CPU-only (JAX_PLATFORMS=cpu inside collect_data); safe to run
# while nothing else owns the CPU. Each shard uses a disjoint seed range.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N_PROCS="${1:-8}"
GAMES="${2:-50}"
OUT="${3:-data/bc_v20_self_400g.npz}"

PY="${PYTHON:-python3}"
# Need numpy+jax+kaggle_environments; system python3 usually lacks them.
if ! "$PY" -c "import numpy, jax" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi
echo "[bc_collect] python: $PY"

mkdir -p data logs/bc_collect
SHARD_DIR="data/bc_shards_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SHARD_DIR"

echo "[bc_collect] $N_PROCS procs x $GAMES games -> $OUT (shards in $SHARD_DIR)"

pids=()
for i in $(seq 0 $((N_PROCS - 1))); do
  seed=$((1000 * (i + 1)))
  log="logs/bc_collect/shard_${i}.log"
  PYTHONPATH="$ROOT" JAX_PLATFORMS=cpu "$PY" -m orbit_wars_rl.bc.collect_data \
    --num-games "$GAMES" --seed "$seed" \
    --out "$SHARD_DIR/shard_${i}.npz" > "$log" 2>&1 &
  pids+=($!)
  echo "  shard $i: pid ${pids[-1]} seed $seed log $log"
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "[bc_collect] WARNING: at least one shard failed; merging what we have"
fi

"$PY" - "$SHARD_DIR" "$OUT" <<'PY'
import sys, glob
import numpy as np

shard_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{shard_dir}/shard_*.npz"))
if not files:
    raise SystemExit("no shards to merge")
parts = [dict(np.load(f)) for f in files]
keys = parts[0].keys()
merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
n = merged["src"].shape[0]
emits = merged["emit"].sum(axis=1)
print(f"merged {len(files)} shards -> {n} samples")
print(f"  hold rate (0 emits): {(emits == 0).mean() * 100:.1f}%")
print(f"  e2+ rate           : {(emits >= 2).mean() * 100:.1f}%")
print(f"  planet_feats {merged['planet_feats'].shape}  global {merged['global_feats'].shape}")
np.savez_compressed(out, **merged)
import os
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
PY

echo "[bc_collect] done"
