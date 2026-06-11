#!/usr/bin/env python3
"""Gate checks for v14d curriculum phases. Exit 0 = pass, 1 = fail."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import median

_TRAIN = re.compile(
    r"upd\s+\d+\s+.*?"
    r"emits\s+([\d.]+)\s+"
    r"spf\s+([\d.]+)\s+"
    r"z0\s+([\d.]+)\s+"
    r"garr\s+([\d.]+)"
)
_EVAL = re.compile(
    r"spf=([\d.]+)\s+garr=([\d.]+)\s+flip=([\d.]+)%\s+"
    r"z0=([\d.]+)%\s+.*?WLD=([\d/]+)"
)


def _read_train_metrics(log_path: Path, n: int = 30) -> list[dict]:
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = _TRAIN.search(line)
        if m:
            rows.append(
                {
                    "emits": float(m.group(1)),
                    "spf": float(m.group(2)),
                    "z0": float(m.group(3)),
                    "garr": float(m.group(4)),
                }
            )
    return rows[-n:]


def _read_last_eval(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    last = None
    for line in log_path.read_text(errors="replace").splitlines():
        if "[eval_vs_v20]" in line and "spf=" in line:
            m = _EVAL.search(line)
            if m:
                w, l, d = (int(x) for x in m.group(5).split("/"))
                last = {
                    "spf": float(m.group(1)),
                    "garr": float(m.group(2)),
                    "flip": float(m.group(3)),
                    "z0": float(m.group(4)),
                    "wins": w,
                    "losses": l,
                    "draws": d,
                }
    return last


def gate_a(log_path: Path) -> tuple[bool, str]:
    rows = _read_train_metrics(log_path)
    if len(rows) < 10:
        return False, f"too few train rows ({len(rows)})"
    garr = median(r["garr"] for r in rows)
    z0 = median(r["z0"] for r in rows)
    spf = median(r["spf"] for r in rows)
    emits = median(r["emits"] for r in rows)
    ev = _read_last_eval(log_path)
    ok = (
        35.0 <= garr <= 200.0
        and 0.25 <= z0 <= 0.78
        and 15.0 <= spf <= 80.0
        and 0.25 <= emits <= 0.80
    )
    if ev is not None and ev["spf"] < 3.0:
        ok = False
    ev_s = ""
    if ev is not None:
        ev_s = f" v20_spf={ev['spf']:.1f} flip={ev['flip']:.1f}%"
    msg = f"A: garr={garr:.1f} z0={z0:.2f} spf={spf:.1f} emits={emits:.2f}{ev_s}"
    return ok, msg


def gate_b(log_path: Path) -> tuple[bool, str]:
    ev = _read_last_eval(log_path)
    rows = _read_train_metrics(log_path, 20)
    if ev is None:
        return False, "no eval_vs_v20 line"
    ok = (
        ev["flip"] >= 8.0
        and ev["spf"] >= 15.0
        and ev["garr"] >= 18.0
    ) or ev["wins"] >= 1
    spf_self = median(r["spf"] for r in rows) if rows else 0.0
    msg = (
        f"B: v20 spf={ev['spf']:.1f} garr={ev['garr']:.1f} "
        f"flip={ev['flip']:.1f}% z0={ev['z0']:.1f}% WLD={ev['wins']}/{ev['losses']}/{ev['draws']} "
        f"self_spf={spf_self:.1f}"
    )
    return ok, msg


def gate_c(log_path: Path) -> tuple[bool, str]:
    ev = _read_last_eval(log_path)
    if ev is None:
        return False, "no eval_vs_v20 line"
    ok = ev["wins"] >= 1 or (
        ev["flip"] >= 10.0
        and ev["spf"] >= 18.0
        and 20.0 <= ev["z0"] <= 45.0
    )
    msg = (
        f"C: v20 spf={ev['spf']:.1f} garr={ev['garr']:.1f} "
        f"flip={ev['flip']:.1f}% z0={ev['z0']:.1f}% WLD={ev['wins']}/{ev['losses']}/{ev['draws']}"
    )
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("a", "b", "c"))
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    fn = {"a": gate_a, "b": gate_b, "c": gate_c}[args.phase]
    ok, msg = fn(args.log)
    print(f"[gate_{args.phase}] {'PASS' if ok else 'FAIL'} — {msg}")
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"phase": args.phase, "pass": ok, "message": msg}, indent=2)
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
