#!/usr/bin/env bash
# v11 FULL STACK — 3-phase serial validation (~4h each, no per-signal ablation).
#
# Top10 all signals in ONE recipe:
#   Reward: R1 delta + R2 release + R4 emit (gated) + R5 capture + planet_share
#   Obs:    A1' threat + global total_garr / fleets_in_flight
#   Scale:  ep500, K=16
#   C1:     phase 2+ mixes 25% vs phase-1 anchor ckpt + 50% frozen self
#
# Usage (5090, background):
#   nohup bash scripts/run_v11_validation.sh \
#     > logs/v11_validation.launcher.log 2>&1 &
#
# Check tomorrow ~10:00:
#   tail -f logs/v11_validation.launcher.log
#   python -m orbit_wars_rl.scripts.monitor_train --once logs/multi_action_v11_g*.log
#   column -t -s $'\t' logs/v11_validation_summary.tsv
#
# Env overrides:
#   UPD_PER_PHASE=800   updates per group (default 800 ≈ 4h)
#   JAX_FRAC=0.85       VRAM cap (serial)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
UPD_PER_PHASE="${UPD_PER_PHASE:-800}"
JAX_FRAC="${JAX_FRAC:-0.85}"
CFG_TEMPLATE="orbit_wars_rl/configs/multi_action_v11_full.yaml"
SUMMARY="logs/v11_validation_summary.tsv"

mkdir -p logs

# Full-stack shaping (all v3 + planet; fleet_log / level prod OFF)
V11_ENVS=(
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

  local log="logs/multi_action_v11_${tag}.log"
  local tb="logs/multi_action_v11_${tag}"
  local ckpt="ckpt_multi_action_v11_${tag}"
  local tmp_cfg
  tmp_cfg="$(mktemp -t "v11_${tag}_XXXX").yaml"

  sed -E \
    -e "s|num_updates: [0-9]+|num_updates: ${UPD_PER_PHASE}|" \
    -e "s|ckpt_multi_action_v11_full|${ckpt}|" \
    -e "s|strong_ckpt_path: \"\"|strong_ckpt_path: \"${strong_ckpt}\"|" \
    -e "s|strong_ratio: 0.0|strong_ratio: ${strong_ratio}|" \
    "$CFG_TEMPLATE" > "$tmp_cfg"

  echo ""
  echo "================================================================"
  echo "[v11] PHASE ${phase}  tag=${tag}  upd=${UPD_PER_PHASE}  log=${log}"
  echo "[v11] resume=${resume:-<scratch>}  strong=${strong_ckpt:-none} ratio=${strong_ratio}"
  echo "================================================================"

  local resume_arg=()
  if [ -n "$resume" ]; then
    resume_arg=(--resume-from "$resume")
  fi

  local t0
  t0=$(date +%s)

  # shellcheck disable=SC2086
  env "${V11_ENVS[@]}" \
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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfull-stack\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$phase" "$tag" "$upd" "$ev" "$tG" "$e2" "$pkR" "$z0" "$nF" \
    "$elapsed" "${latest:-?}" >> "$SUMMARY"

  echo "[v11] phase ${phase} done in ${elapsed}s  ckpt=${latest:-none}"
  echo "$latest"
}

echo "[v11] FULL STACK validation start  UPD_PER_PHASE=$UPD_PER_PHASE"
echo "[v11] shaping: ${V11_ENVS[*]}"

# --- Group 1: scratch foundation (~4h) ---
CKPT_G1="$(_run_phase 1 g1_scratch "" "" 0.0)"

# Anchor for C1: mid-phase-1 snapshot (same arch as v11)
STRONG="${CKPT_G1%/*}/ckpt_000399.pkl"
if [ ! -f "$STRONG" ]; then
  STRONG="$CKPT_G1"
  echo "[v11] WARN ckpt_000399 missing, using latest for strong opp: $STRONG"
fi

# --- Group 2: resume + C1 strong mix (~4h) ---
CKPT_G2="$(_run_phase 2 g2_curriculum "$CKPT_G1" "$STRONG" 0.25)"

# --- Group 3: continue same recipe (~4h) ---
CKPT_G3="$(_run_phase 3 g3_continue "$CKPT_G2" "$STRONG" 0.25)"

echo ""
echo "================================================================"
echo "[v11] ALL 3 PHASES COMPLETE"
column -t -s $'\t' "$SUMMARY"
echo ""
echo "Morning checklist (~10:00):"
echo "  python -m orbit_wars_rl.scripts.monitor_train --once logs/multi_action_v11_g*.log"
echo ""
echo "Sign-of-life (vs Top10 first-80 targets):"
echo "  tG > 60   e2 > 0.05   z0 in 0.15-0.50   pkR > 2.0   ev > 0.5 @ end"
echo ""
echo "If g3 looks healthy, export + h2h:"
echo "  python -m orbit_wars_rl.scripts.export_submission \\"
echo "    --ckpt ${CKPT_G3} --template submission_rl_v4.py \\"
echo "    --out submission_rl_v11_g3.py"
echo "  # then gauntlet vs v20 — see docs/H2H_EVAL_RUNBOOK.md"
