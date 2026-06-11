#!/usr/bin/env python3
"""Early-abort checks for v14d curriculum / grid search.

Exit 0 = keep training, 2 = abort (pathological), 1 = not enough data yet.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v14d_gate_check import _read_last_eval, _read_train_metrics

_UPD = re.compile(r"upd\s+(\d+)\s+")


def _last_update(log_path: Path) -> int:
    last = 0
    if not log_path.is_file():
        return 0
    for line in log_path.read_text(errors="replace").splitlines():
        m = _UPD.search(line)
        if m:
            last = int(m.group(1))
    return last


def abort_a(log_path: Path, *, min_updates: int = 30) -> tuple[int, str]:
    u = _last_update(log_path)
    if u < min_updates:
        return 1, f"too early (upd={u}<{min_updates})"
    rows = _read_train_metrics(log_path, 15)
    if len(rows) < 8:
        return 1, f"too few metric rows ({len(rows)})"
    garr = median(r["garr"] for r in rows)
    z0 = median(r["z0"] for r in rows)
    spf = median(r["spf"] for r in rows)
    emits = median(r["emits"] for r in rows)
    # v14e: catch hoarding collapse much earlier (v14d collapsed at ~upd 16)
    if z0 >= 0.85 and emits <= 0.12:
        return 2, f"hoard collapse z0={z0:.2f} garr={garr:.0f} emits={emits:.2f}"
    if z0 >= 0.80 and garr >= 120 and emits <= 0.15:
        return 2, f"hoard drift z0={z0:.2f} garr={garr:.0f} emits={emits:.2f}"
    if z0 <= 0.08 and spf <= 8 and emits >= 0.70:
        return 2, f"spam collapse z0={z0:.2f} spf={spf:.1f} emits={emits:.2f}"
    if garr >= 150:
        return 2, f"extreme garr={garr:.0f}"
    ev = _read_last_eval(log_path)
    if ev is not None and u >= 100 and ev["spf"] < 0.5:
        return 2, f"vs v20 zero launch spf={ev['spf']:.1f}"
    return 0, f"ok z0={z0:.2f} garr={garr:.0f} spf={spf:.1f} emits={emits:.2f}"


def abort_b(log_path: Path, *, min_updates: int = 250) -> tuple[int, str]:
    u = _last_update(log_path)
    if u < min_updates:
        return 1, f"too early (upd={u}<{min_updates})"
    rows = _read_train_metrics(log_path, 15)
    if rows and median(r["emits"] for r in rows) >= 0.95 and median(r["spf"] for r in rows) <= 5:
        return 2, "self-play micro-spam"
    ev = _read_last_eval(log_path)
    if ev is None:
        return 1, "no eval yet"
    if u >= 400 and ev["flip"] < 1.0 and ev["spf"] < 5.0:
        return 2, f"no capture signal flip={ev['flip']:.1f}% spf={ev['spf']:.1f}"
    return 0, f"ok v20 flip={ev['flip']:.1f}% spf={ev['spf']:.1f}"


def abort_c(log_path: Path, *, min_updates: int = 350) -> tuple[int, str]:
    u = _last_update(log_path)
    if u < min_updates:
        return 1, f"too early (upd={u}<{min_updates})"
    ev = _read_last_eval(log_path)
    if ev is None:
        return 1, "no eval yet"
    if u >= 600 and ev["wins"] == 0 and ev["flip"] < 5.0:
        return 2, f"stuck flip={ev['flip']:.1f}% WLD=0/{ev['losses']}/{ev['draws']}"
    return 0, f"ok v20 flip={ev['flip']:.1f}% WLD={ev['wins']}/{ev['losses']}/{ev['draws']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("a", "b", "c"))
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--min-updates", type=int, default=None)
    args = ap.parse_args()
    fn = {"a": abort_a, "b": abort_b, "c": abort_c}[args.phase]
    kwargs = {}
    if args.min_updates is not None:
        kwargs["min_updates"] = args.min_updates
    code, msg = fn(args.log, **kwargs)
    label = {0: "CONTINUE", 1: "WAIT", 2: "ABORT"}[code]
    print(f"[early_{args.phase}] {label} — {msg}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
