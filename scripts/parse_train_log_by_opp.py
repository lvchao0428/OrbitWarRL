#!/usr/bin/env python3
"""Parse training logs: behaviour metrics split by opponent tag (P1 diagnostic).

Usage:
  python scripts/parse_train_log_by_opp.py logs/v11_f42.log
  python scripts/parse_train_log_by_opp.py logs/v11_f42.log --align-only
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict


_LINE = re.compile(
    r"^upd\s+(\d+)\s+.*?opp (\w+)\s+.*?spf ([\d.]+)\s+z0 ([\d.]+)\s+"
    r"garr ([\d.]+)\s+.*?e2 ([\d.]+)"
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def parse_log(path: str, align_only: bool = False) -> dict[str, dict[str, float]]:
    by_opp: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"spf": [], "z0": [], "garr": [], "e2": [], "upd": []}
    )
    with open(path) as f:
        for line in f:
            if not line.startswith("upd"):
                continue
            m = _LINE.search(line)
            if not m:
                continue
            opp = m.group(2)
            if align_only and opp not in ("strn", "frzn"):
                continue
            d = by_opp[opp]
            d["upd"].append(float(m.group(1)))
            d["spf"].append(float(m.group(3)))
            d["z0"].append(float(m.group(4)))
            d["garr"].append(float(m.group(5)))
            d["e2"].append(float(m.group(6)))

    out: dict[str, dict[str, float]] = {}
    for opp, d in sorted(by_opp.items()):
        n = len(d["spf"])
        out[opp] = {
            "n": n,
            "spf": _mean(d["spf"]),
            "z0": _mean(d["z0"]),
            "garr": _mean(d["garr"]),
            "e2": _mean(d["e2"]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="training log path")
    ap.add_argument(
        "--align-only",
        action="store_true",
        help="only strn+frzn (replay-aligned proxy)",
    )
    args = ap.parse_args()
    stats = parse_log(args.log, align_only=args.align_only)
    title = "strn+frzn only" if args.align_only else "all opponents"
    print(f"# {args.log}  ({title})")
    print(f"{'opp':>5}  {'n':>4}  {'spf':>7}  {'z0':>6}  {'garr':>7}  {'e2':>6}")
    for opp, s in stats.items():
        print(
            f"{opp:>5}  {int(s['n']):>4}  {s['spf']:>7.1f}  {s['z0']:>6.2f}  "
            f"{s['garr']:>7.1f}  {s['e2']:>6.2f}"
        )
    if not args.align_only and "strn" in stats and "buf" in stats:
        print()
        print("# train/replay gap hint: compare align spf/e2 to replay vs v20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
