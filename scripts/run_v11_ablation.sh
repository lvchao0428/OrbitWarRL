#!/usr/bin/env bash
# v11 ABLATION — isolate which v11 ingredient broke vs v20.
#
# Hypothesis from v11 G1 replay vs v20 (spf=1.50, garr=7.86, flip=0.66%):
#   - R4 emit_log + K=16 jointly trained the policy to spam tiny launches
#   - On v20 (early garr ≪ 7) the strategy degenerates to spf=1
#
# We run 3 scratch variants in series, each 800 upd ≈ 4h on 5090.
# After each, snapshot the latest ckpt + record final-50 metrics avg.
#
# Variants (all start from scratch, no curriculum):
#   A) k8_no_emit  : K=8,  R1+R2+R5 only (no R4)         — strict suppression
#   B) k8_full     : K=8,  R1+R2+R4+R5 (gated)            — K vs R4 isolation
#   C) k16_no_emit : K=16, R1+R2+R5 (no R4)               — R4 vs K isolation
#
# Variant A is the most likely winner (strongest suppression of spam).
# B vs A measures R4 effect at K=8;
# C vs A measures K effect at fixed reward set.
#
# Usage (5090, after G2 finishes or in parallel slot):
#   nohup bash scripts/run_v11_ablation.sh \
#     > logs/v11_ablation.launcher.log 2>&1 &
#
# Resume single variant:
#   ONLY=k8_no_emit bash scripts/run_v11_ablation.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
UPD_PER_VARIANT="${UPD_PER_VARIANT:-800}"
JAX_FRAC="${JAX_FRAC:-0.85}"
SUMMARY="logs/v11_ablation_summary.tsv"
ONLY="${ONLY:-}"

mkdir -p logs

# Common shaping (R1 + R2 + R5 + planet_share, level prod / fleet_log OFF).
# R4 gets toggled per variant by appending its env var.
COMMON_SHAPING=(
  "ORBITWARS_SHAPING_PLANET_SHARE=0.005"
  "ORBITWARS_SHAPING_PROD_SHARE=0.0"
  "ORBITWARS_SHAPING_FLEET_LOG=0.0"
  "ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0"
  "ORBITWARS_SHAPING_RELEASE=0.05"
  "ORBITWARS_SHAPING_RELEASE_K=20.0"
  "ORBITWARS_SHAPING_CAPTURE=0.02"
)

_run_variant() {
  local tag="$1"
  local cfg_template="$2"
  local emit_log="$3"   # 0.0 or 0.01
  local emit_gated="$4" # "0" or "1"

  if [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ]; then
    echo "[ablation] skip $tag (ONLY=$ONLY)" >&2
    return 0
  fi

  local log="logs/multi_action_v11_${tag}.log"
  local tb="logs/multi_action_v11_${tag}"
  local ckpt="ckpt_multi_action_v11_${tag}"
  local tmp_cfg
  tmp_cfg="$(mktemp -t "v11_${tag}_XXXX").yaml"

  sed -E \
    -e "s|num_updates: [0-9]+|num_updates: ${UPD_PER_VARIANT}|" \
    -e "s|ckpt_multi_action_v11_(full|k8)|${ckpt}|" \
    "$cfg_template" > "$tmp_cfg"

  echo "" >&2
  echo "================================================================" >&2
  echo "[ablation] tag=${tag}  upd=${UPD_PER_VARIANT}  cfg=${cfg_template}" >&2
  echo "[ablation] R4 EMIT_LOG=${emit_log} GATED=${emit_gated}" >&2
  echo "================================================================" >&2

  local t0
  t0=$(date +%s)

  env "${COMMON_SHAPING[@]}" \
      "ORBITWARS_SHAPING_EMIT_LOG=${emit_log}" \
      "ORBITWARS_SHAPING_EMIT_GATED=${emit_gated}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
      "$PY" -m orbit_wars_rl.scripts.train \
      --config "$tmp_cfg" \
      --log-dir "$tb" \
      > "$log" 2>&1

  local t1 elapsed
  t1=$(date +%s)
  elapsed=$(( t1 - t0 ))

  # average final 50 lines for stable metrics
  snap="$("$PY" - <<PY || true
from pathlib import Path
from orbit_wars_rl.scripts.monitor_train import parse_one_line
log = Path("${log}")
rows = []
for line in log.read_text(errors="replace").splitlines():
    d = parse_one_line(line)
    if d:
        rows.append(d)
if not rows:
    print("? ? ? ? ? ? ?")
else:
    tail = rows[-50:]
    def _avg(k, default=0.0):
        vals = [float(r.get(k, default) or default) for r in tail if r.get(k) not in (None, "?")]
        return sum(vals) / max(len(vals), 1)
    last = rows[-1]
    print(last.get("upd","?"),
          f"{_avg('ev'):.2f}",
          f"{_avg('tG'):.0f}",
          f"{_avg('e2'):.2f}",
          f"{_avg('pkR'):.1f}",
          f"{_avg('z0'):.2f}",
          f"{_avg('spf'):.1f}")
PY
)"
  read -r upd ev tG e2 pkR z0 spf <<< "$snap"

  local latest
  latest="$(ls -1 "${ckpt}"/ckpt_*.pkl 2>/dev/null | sort | tail -1 || true)"

  rm -f "$tmp_cfg"

  if [ ! -f "$SUMMARY" ]; then
    printf "ts\ttag\tupd\tev\ttG\te2\tpkR\tz0\tspf_train\telapsed_s\tckpt\tnotes\n" > "$SUMMARY"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tablation\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tag" "$upd" "$ev" "$tG" "$e2" "$pkR" "$z0" "$spf" \
    "$elapsed" "${latest:-?}" >> "$SUMMARY"

  echo "[ablation] $tag done in ${elapsed}s  ckpt=${latest:-none}" >&2
}

echo "[ablation] start  UPD_PER_VARIANT=$UPD_PER_VARIANT  ONLY=${ONLY:-all}"

# A) K=8, no R4 — most likely winner
_run_variant "k8_no_emit" "orbit_wars_rl/configs/multi_action_v11_k8.yaml" "0.0" "0"

# B) K=8 + R4 — isolates R4 effect at small K
_run_variant "k8_full" "orbit_wars_rl/configs/multi_action_v11_k8.yaml" "0.01" "1"

# C) K=16, no R4 — isolates K effect at no R4
_run_variant "k16_no_emit" "orbit_wars_rl/configs/multi_action_v11_full.yaml" "0.0" "0"

echo ""
echo "================================================================"
echo "[ablation] DONE. Summary:"
column -t -s $'\t' "$SUMMARY" || cat "$SUMMARY"
echo ""
echo "Next: export each ckpt + replay vs v20 (cpu, in parallel):"
echo '  for v in k8_no_emit k8_full k16_no_emit; do'
echo '    bash scripts/quick_replay.sh "ckpt_multi_action_v11_${v}/$(ls -1 ckpt_multi_action_v11_${v}/ckpt_*.pkl | sort | tail -1 | xargs basename)" "v11_${v}"'
echo '  done'
