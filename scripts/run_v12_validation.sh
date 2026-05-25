#!/usr/bin/env bash
# v12 4P FULL STACK — 3-phase serial validation (mirror v11, ORBITWARS_NUM_PLAYERS=4).
#
# Usage (5090, after v11 or standalone):
#   nohup env ORBITWARS_NUM_PLAYERS=4 bash scripts/run_v12_validation.sh \
#     > logs/v12_validation.launcher.log 2>&1 &
#
# Smoke (2 updates):
#   ORBITWARS_NUM_PLAYERS=4 UPD_PER_PHASE=2 bash scripts/run_v12_validation.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export ORBITWARS_NUM_PLAYERS=4

PY="${PYTHON:-python}"
UPD_PER_PHASE="${UPD_PER_PHASE:-800}"
JAX_FRAC="${JAX_FRAC:-0.85}"
CFG_TEMPLATE="orbit_wars_rl/configs/multi_action_v12_4p.yaml"
SUMMARY="logs/v12_validation_summary.tsv"

mkdir -p logs

V12_ENVS=(
  "ORBITWARS_NUM_PLAYERS=4"
  "ORBITWARS_SHAPING_PLANET_SHARE=0.005"
  "ORBITWARS_SHAPING_PROD_SHARE=0.0"
  "ORBITWARS_SHAPING_FLEET_LOG=0.0"
  "ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0"
  "ORBITWARS_SHAPING_RELEASE=0.05"
  "ORBITWARS_SHAPING_RELEASE_K=20.0"
  "ORBITWARS_SHAPING_EMIT_LOG=0.01"
  "ORBITWARS_SHAPING_EMIT_GATED=1"
  "ORBITWARS_SHAPING_CAPTURE=0.02"
)

_run_phase() {
  local phase="$1"
  local tag="$2"
  local resume="${3:-}"
  local strong_ckpt="${4:-}"
  local strong_ratio="${5:-0.0}"

  local log="logs/multi_action_v12_${tag}.log"
  local tb="logs/multi_action_v12_${tag}"
  local ckpt="ckpt_multi_action_v12_${tag}"
  local tmp_cfg
  tmp_cfg="$(mktemp -t "v12_${tag}_XXXX").yaml"

  sed -E \
    -e "s|num_updates: [0-9]+|num_updates: ${UPD_PER_PHASE}|" \
    -e "s|ckpt_multi_action_v12_4p_full|${ckpt}|" \
    -e "s|strong_ckpt_path: \"\"|strong_ckpt_path: \"${strong_ckpt}\"|" \
    -e "s|strong_ratio: 0.0|strong_ratio: ${strong_ratio}|" \
    "$CFG_TEMPLATE" > "$tmp_cfg"

  echo ""
  echo "================================================================"
  echo "[v12/4P] PHASE ${phase}  tag=${tag}  upd=${UPD_PER_PHASE}  log=${log}"
  echo "[v12/4P] resume=${resume:-<scratch>}  strong=${strong_ckpt:-none} ratio=${strong_ratio}"
  echo "================================================================"

  local resume_arg=()
  if [ -n "$resume" ]; then
    resume_arg=(--resume-from "$resume")
  fi

  local t0
  t0=$(date +%s)

  env "${V12_ENVS[@]}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
      "$PY" -m orbit_wars_rl.scripts.train \
      --config "$tmp_cfg" \
      --log-dir "$tb" \
      "${resume_arg[@]}" \
      > "$log" 2>&1

  local t1 elapsed upd ev tG e2 pkR z0
  t1=$(date +%s)
  elapsed=$(( t1 - t0 ))

  snap="$("$PY" - <<PY || true
from pathlib import Path
from orbit_wars_rl.scripts.monitor_train import parse_one_line
log = Path("${log}")
rows = []
for line in log.read_text(errors="replace").splitlines():
    d = parse_one_line(line)
    if d:
        rows.append(d)
if rows:
    m = rows[-1]
    print(m.get("upd", "?"), m.get("ev", "?"), m.get("tG", "?"),
          m.get("e2", "?"), m.get("pkR", "?"), m.get("z0", "?"), m.get("nF", "?"))
PY
)"
  if [ -n "$snap" ]; then
    read -r upd ev tG e2 pkR z0 nF <<< "$snap"
  else
    upd="?" ev="?" tG="?" e2="?" pkR="?" z0="?" nF="?"
  fi

  local latest
  latest="$(ls -1 "${ckpt}"/ckpt_*.pkl 2>/dev/null | sort | tail -1 || true)"

  rm -f "$tmp_cfg"

  if [ ! -f "$SUMMARY" ]; then
    printf "ts\tphase\ttag\tupd\tev\ttG\te2\tpkR\tz0\tnF\telapsed_s\tckpt\tnotes\n" > "$SUMMARY"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t4p-full-stack\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$phase" "$tag" "$upd" "$ev" "$tG" "$e2" "$pkR" "$z0" "$nF" \
    "$elapsed" "${latest:-?}" >> "$SUMMARY"

  echo "[v12/4P] phase ${phase} done in ${elapsed}s  ckpt=${latest:-none}"
  echo "$latest"
}

echo "[v12/4P] FULL STACK validation start  UPD_PER_PHASE=$UPD_PER_PHASE  NUM_PLAYERS=4"
echo "[v12/4P] shaping: ${V12_ENVS[*]}"

CKPT_G1="$(_run_phase 1 g1_scratch "" "" 0.0)"

STRONG="${CKPT_G1%/*}/ckpt_000399.pkl"
if [ ! -f "$STRONG" ]; then
  STRONG="$CKPT_G1"
  echo "[v12/4P] WARN ckpt_000399 missing, using latest for strong opp: $STRONG"
fi

CKPT_G2="$(_run_phase 2 g2_curriculum "$CKPT_G1" "$STRONG" 0.25)"
CKPT_G3="$(_run_phase 3 g3_continue "$CKPT_G2" "$STRONG" 0.25)"

echo ""
echo "================================================================"
echo "[v12/4P] ALL 3 PHASES COMPLETE"
column -t -s $'\t' "$SUMMARY"
echo ""
echo "4P sign-of-life (G3, vs Top10 4P baseline prod_share ~0.25):"
echo "  pdΔ > 0 (baseline 0.25 not 0.5)   tG > 60   e2 > 0.05   ev > 0.5"
