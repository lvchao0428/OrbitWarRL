"""Merge multiple state-buffer .npz files for mixed Plan B curriculum.

By default balances sources (subsample to min count) so v20 and top10 contribute
equally despite different collection sizes.

Usage:
    python -m orbit_wars_rl.bc.merge_state_buffers \\
        --inputs data/v20_states_200g.npz data/top10_winner_states.npz \\
        --out data/mixed_v20_top10.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from orbit_wars_rl.bc.state_buffer_util import (
    merge_balanced,
    print_buffer_stats,
    stack_samples,
    subsample_rows,
    write_npz,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge state buffer .npz files.")
    ap.add_argument("--inputs", nargs="+", required=True, help="input .npz paths")
    ap.add_argument("--out", type=str, default="data/mixed_v20_top10.npz")
    ap.add_argument("--balance", action="store_true", default=True,
                    help="subsample each source to min size (default: on)")
    ap.add_argument("--no-balance", action="store_true",
                    help="concat all rows (top10 may dominate)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    for p in args.inputs:
        if not Path(p).is_file():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 1

    if args.no_balance:
        parts = [dict(np.load(p)) for p in args.inputs]
        rows = []
        n_total = sum(int(d["step"].shape[0]) for d in parts)
        print(f"[merge] concatenating {len(parts)} sources → {n_total} states")
        for p, data in zip(args.inputs, parts):
            print_buffer_stats(data, label=Path(p).name)
            n = int(data["step"].shape[0])
            for j in range(n):
                rows.append({k: data[k][j] for k in data})
        stacked = stack_samples(rows)
        write_npz(stacked, out)
        print_buffer_stats(stacked, label=out.name)
    else:
        merge_balanced(args.inputs, out, seed=args.seed)

    size_mb = out.stat().st_size / 1e6
    print(f"\n wrote {out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
