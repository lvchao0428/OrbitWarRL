"""Local head-to-head between two submission files using kaggle_environments.

Usage:
    python -m orbit_wars_rl.scripts.h2h_local \
        --agent-a submission_rl_v1.py \
        --agent-b submission_v20_0513.py \
        --num-games 5 \
        --seeds 0 1 2 3 4            # space-separated
    # or comma-separated also works:
    #   --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from kaggle_environments import make


def _run_one_game(agent_a_path: str, agent_b_path: str, seed: int | None) -> dict:
    cfg = {}
    if seed is not None:
        cfg["seed"] = seed
    env = make("orbit_wars", configuration=cfg, debug=False)
    t0 = time.time()
    env.run([agent_a_path, agent_b_path])
    elapsed = time.time() - t0
    last_step = env.steps[-1]
    rewards = [s["reward"] for s in last_step]
    status = [s["status"] for s in last_step]
    return {
        "rewards": rewards,
        "status": status,
        "elapsed_s": elapsed,
        "num_steps": len(env.steps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-a", type=str, required=True)
    ap.add_argument("--agent-b", type=str, required=True)
    ap.add_argument("--num-games", type=int, default=10)
    ap.add_argument(
        "--seeds",
        type=str,
        nargs="*",
        default=None,
        help=(
            "optional fixed seeds; if shorter than num_games we cycle. "
            "Accepts space-separated (`0 1 2`) and/or comma-separated (`0,1,2`)."
        ),
    )
    args = ap.parse_args()

    # Normalize --seeds: allow comma-separated tokens and mixed forms.
    if args.seeds:
        flat_seeds: list[int] = []
        for tok in args.seeds:
            for part in tok.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    flat_seeds.append(int(part))
                except ValueError:
                    print(f"--seeds: cannot parse '{part}' as int", file=sys.stderr)
                    return 2
        args.seeds = flat_seeds

    for f in (args.agent_a, args.agent_b):
        if not Path(f).exists():
            print(f"agent not found: {f}", file=sys.stderr)
            return 2

    a_wins = 0
    b_wins = 0
    draws = 0
    errors = 0

    for i in range(args.num_games):
        seed = None
        if args.seeds:
            seed = args.seeds[i % len(args.seeds)]
        try:
            r = _run_one_game(args.agent_a, args.agent_b, seed=seed)
        except Exception as exc:
            print(f"  game {i}: error {exc}")
            errors += 1
            continue

        ra, rb = r["rewards"]
        if ra is None or rb is None:
            outcome = "ERR"
            errors += 1
        elif ra > rb:
            outcome = "A "; a_wins += 1
        elif rb > ra:
            outcome = "B "; b_wins += 1
        else:
            outcome = "T "; draws += 1
        print(f"  game {i:2d}  seed={seed!s:>4}  steps={r['num_steps']:3d}  "
              f"a={ra}  b={rb}  status={r['status']}  {outcome}  ({r['elapsed_s']:.1f}s)")

    n = args.num_games
    print()
    print(f"agent A: {args.agent_a}")
    print(f"agent B: {args.agent_b}")
    print(f"  A wins: {a_wins}/{n}  ({a_wins/n:.2f})")
    print(f"  B wins: {b_wins}/{n}  ({b_wins/n:.2f})")
    print(f"  draws : {draws}/{n}")
    print(f"  errors: {errors}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
