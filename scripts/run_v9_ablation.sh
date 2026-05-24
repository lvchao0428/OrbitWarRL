#!/usr/bin/env bash
# Day 4 v9 ablation launcher. Run on the 5090 (32GB VRAM, 48GB RAM) box.
#
# Usage:
#   bash scripts/run_v9_ablation.sh                    # launch all 4
#   bash scripts/run_v9_ablation.sh v9a v9b            # 1st pair (chained)
#   bash scripts/run_v9_ablation.sh v9c v9d            # 2nd pair (chained)
#   N_CONCURRENT=4 bash scripts/run_v9_ablation.sh     # force all-parallel
#
# Resource budget for 5090 (32GB VRAM, 48GB RAM):
#
#   Per-run estimate (num_envs=128, rollout=256, ep_steps=350, d_model=128):
#     - CPU RAM:  2.5-4 GB (rollout buffer + JAX compile cache)
#     - GPU VRAM: 4-6 GB (model + Adam state + JIT activations)
#
#   This script defaults to N_CONCURRENT=2 (safe on 32GB / 48GB).
#   Override to 4 only if you've verified VRAM headroom with a single run first.
#
# IMPORTANT: JAX defaults to PREALLOCATING 75% of VRAM. We force per-process
#   memory limit via XLA_PYTHON_CLIENT_MEM_FRACTION (set below) so N concurrent
#   jobs don't OOM each other.
#
# Each run writes:
#   logs/multi_action_v9X.log         -- training stdout/stderr
#   logs/multi_action_v9X/            -- tensorboard event files
#   ckpt_multi_action_v9X/            -- pickle checkpoints (every 100 upds)
#
# To watch all 4 in real time:
#   python -m orbit_wars_rl.scripts.monitor_train logs/multi_action_v9*.log
#
# To kill all:
#   pkill -f 'orbit_wars_rl.scripts.train.*multi_action_v9'

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
N_CONCURRENT="${N_CONCURRENT:-2}"

# JAX memory tuning: cap per-process VRAM so concurrent jobs coexist.
# 0.22 * 32GB ~ 7GB per job, leaves ~4GB headroom for system + monitor.
# If only ONE job at a time, you can bump this to 0.50 for faster compile.
case "$N_CONCURRENT" in
  1) JAX_FRAC="${JAX_FRAC:-0.5}"  ;;
  2) JAX_FRAC="${JAX_FRAC:-0.4}"  ;;  # 0.4*32 ~ 12.8GB per job; 2 jobs ~ 25.6GB
  3) JAX_FRAC="${JAX_FRAC:-0.28}" ;;  # 0.28*32 ~ 9GB per job; 3 jobs ~ 27GB
  4) JAX_FRAC="${JAX_FRAC:-0.20}" ;;  # 0.20*32 ~ 6.4GB per job; 4 jobs ~ 25.6GB
  *) JAX_FRAC="${JAX_FRAC:-0.20}" ;;
esac

mkdir -p logs

echo "[config] N_CONCURRENT=$N_CONCURRENT  XLA_PYTHON_CLIENT_MEM_FRACTION=$JAX_FRAC"
echo "[config] each job: ~$(awk "BEGIN { printf \"%.1f\", $JAX_FRAC * 32 }") GB VRAM cap, 2.5-4 GB RAM"
echo ""

# Map: variant -> "ENV_VARS;EXTRA_ARGS"
declare -A LAUNCH=(
  [v9a]=";"
  [v9b]="ORBITWARS_SHAPING_PROD_SHARE=0.01;"
  [v9c]="ORBITWARS_SHAPING_PROD_SHARE=0.01 ORBITWARS_SHAPING_PLANET_SHARE=0.005 ORBITWARS_SHAPING_FLEET_LOG=0.002;"
  [v9d]="ORBITWARS_SHAPING_PROD_SHARE=0.01 ORBITWARS_SHAPING_PLANET_SHARE=0.005;"
)

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(v9a v9b v9c v9d)
fi

for v in "${VARIANTS[@]}"; do
  if [ -z "${LAUNCH[$v]+x}" ]; then
    echo "[ERR] unknown variant '$v' (expected one of: v9a v9b v9c v9d)" >&2
    exit 2
  fi
  envs="${LAUNCH[$v]%;*}"
  cfg="orbit_wars_rl/configs/multi_action_${v}.yaml"
  log="logs/multi_action_${v}.log"
  tb="logs/multi_action_${v}"

  if [ ! -f "$cfg" ]; then
    echo "[ERR] missing config: $cfg" >&2
    exit 2
  fi

  # Skip if already running (idempotent re-launch)
  if pgrep -f "multi_action_${v}.yaml" >/dev/null; then
    echo "[skip] $v already running"
    continue
  fi

  echo "[launch] $v  (shaping: ${envs:-<none>})  log: $log"
  # shellcheck disable=SC2086
  env $envs \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
      nohup "$PY" -m orbit_wars_rl.scripts.train \
      --config "$cfg" \
      --log-dir "$tb" \
      > "$log" 2>&1 &
  echo "  pid=$!"
done

echo ""
echo "All requested v9 runs launched. Quick checks (wait ~60s for compile):"
echo "  watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'"
echo "  grep -h '\\[reward\\]' logs/multi_action_v9*.log"
echo ""
echo "Then start the monitor:"
echo "  python -m orbit_wars_rl.scripts.monitor_train logs/multi_action_v9*.log"
