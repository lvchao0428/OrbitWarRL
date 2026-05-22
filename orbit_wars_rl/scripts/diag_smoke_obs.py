"""Diagnose smoke obs: import submission, manually call greedy_multi_action,
print (src, dst, pct) before they get filtered by decode_multi_to_kaggle_moves.

Usage:
    python -m orbit_wars_rl.scripts.diag_smoke_obs --sub submission_rl_v6p2_ckpt_000999.py
    python -m orbit_wars_rl.scripts.diag_smoke_obs --sub submission_rl_v6p2_u299.py
"""
from __future__ import annotations

import argparse
import importlib.util as iu


def make_smoke_obs(player: int = 0, av: float = 0.04) -> dict:
    """4-planet smoke obs identical to the one in export_submission.py."""
    return {
        "player": player,
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 30, 3],
            [1, -1, 80.0, 20.0, 1.0, 15, 2],
            [2, -1, 20.0, 80.0, 1.0, 15, 2],
            [3, 1, 80.0, 80.0, 1.0, 30, 3],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 30, 3],
            [1, -1, 80.0, 20.0, 1.0, 15, 2],
            [2, -1, 20.0, 80.0, 1.0, 15, 2],
            [3, 1, 80.0, 80.0, 1.0, 30, 3],
        ],
        "angular_velocity": av,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="path to submission_rl_*.py")
    ap.add_argument("--player", type=int, default=0)
    args = ap.parse_args()

    spec = iu.spec_from_file_location("_diag_sub", args.sub)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {args.sub}")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    obs = make_smoke_obs(player=args.player)
    print(f"=== smoke obs for {args.sub} (player={args.player}) ===")
    print(f"  planets: {obs['planets']}")
    print(f"  fleets:  {obs['fleets']}")

    enc = mod.encode_obs(obs, step=0, episode_steps=500, player=args.player)
    W = mod._load_weights()
    src_list, dst_list, pct_list = mod.greedy_multi_action(W, enc)
    print(f"  raw greedy_multi_action:")
    print(f"    src_list = {src_list}")
    print(f"    dst_list = {dst_list}")
    print(f"    pct_list = {pct_list}")
    if src_list:
        for i, (s, d, p) in enumerate(zip(src_list, dst_list, pct_list)):
            marker = " <-- DROPPED (src==dst)" if s == d else ""
            print(f"    fleet {i}: src={s} dst={d} pct_bin={p}{marker}")

    moves = mod.agent(obs, {"episodeSteps": 500})
    print(f"  agent() moves: {moves}")
    print(f"  -> {'EMPTY (filtered)' if not moves else f'{len(moves)} moves'}")

    # Also try player=1 perspective to detect the asA=N asB=0 issue.
    if args.player == 0:
        obs1 = make_smoke_obs(player=1)
        enc1 = mod.encode_obs(obs1, step=0, episode_steps=500, player=1)
        src_list1, dst_list1, pct_list1 = mod.greedy_multi_action(W, enc1)
        moves1 = mod.agent(obs1, {"episodeSteps": 500})
        print(f"\n=== as player=1 (symmetric obs) ===")
        print(f"    src_list = {src_list1}")
        print(f"    dst_list = {dst_list1}")
        print(f"    moves = {moves1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
