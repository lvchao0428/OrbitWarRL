"""Diagnose a submission against a *real* kaggle game obs (not a synthetic
4-planet smoke obs). Prints src/dst/pct lists for several turns and shows
ships before/after to identify policy bugs like "always emit max 8 fleets"
or "src commit lock".

Usage:
    python -m orbit_wars_rl.scripts.diag_real_game_obs \
        --sub submission_rl_v6p3_ckpt_004999.py \
        --opponent submission_v20_0513.py \
        --player 0 \
        --seed 0 \
        --turns 0,5,10,15,20,30,50

Prints per-turn:
    turn N  P=npl sP=ships_on_planet F=nflt sF=ships_in_flight
            src_list, dst_list, pct_list (raw greedy_multi_action output)
            agent() final moves
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import sys
from pathlib import Path
from typing import Dict, List, Any

from kaggle_environments import make


def _load_mod(path: str):
    # Use the file stem as the module name so the loaded module can refer to
    # itself via ``sys.modules[__name__]`` -- required for things like
    # @dataclass which look up ``cls.__module__`` during decoration.
    name = Path(path).stem
    spec = iu.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = iu.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def _summary(obs: Dict[str, Any], player: int) -> str:
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    n_p = sum(1 for p in planets if int(p[1]) == player)
    sp = sum(int(p[5]) for p in planets if int(p[1]) == player)
    n_f = sum(1 for f in fleets if int(f[1]) == player)
    sf = sum(int(f[6]) for f in fleets if int(f[1]) == player)
    return f"P={n_p:>2} sP={sp:>4} F={n_f:>2} sF={sf:>4} S={sp+sf:>4}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--player", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--turns",
        type=str,
        default="0,5,10,15,20,30,50",
        help="comma-separated turn indices to dump",
    )
    args = ap.parse_args()

    target_turns = sorted({int(x) for x in args.turns.split(",")})
    print(f"target turns: {target_turns}")
    print(f"sub          : {args.sub}")
    print(f"opponent     : {args.opponent}")
    print(f"player       : {args.player}")
    print(f"seed         : {args.seed}")

    sub_mod = _load_mod(args.sub)

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)

    # Build a 2-agent setup but we'll roll the game ourselves so we can
    # inject diag calls between steps.
    env.reset(num_agents=2)

    # State of the game
    # Get the most recent step entries
    step_entries = env.steps[0]
    obs_for_player = step_entries[args.player].get("observation", {})

    # We need an opponent agent to step the env. Load the opponent mod.
    opp_mod = _load_mod(args.opponent)

    print()
    print(f"{'turn':>5} | {'sub_summary':<26} | {'opp_summary':<26} | "
          f"diag")
    print("-" * 110)

    max_turn = max(target_turns)
    for turn in range(max_turn + 1):
        cur_step = env.steps[-1]
        obs_p = cur_step[args.player].get("observation", {})
        obs_o = cur_step[1 - args.player].get("observation", {})
        cfg_p = cur_step[args.player].get("configuration", {}) or {"episodeSteps": 500}
        cfg_o = cur_step[1 - args.player].get("configuration", {}) or {"episodeSteps": 500}

        # diag sub on this obs (deterministic argmax)
        if turn in target_turns:
            try:
                enc = sub_mod.encode_obs(
                    obs_p, step=turn, episode_steps=int(cfg_p.get("episodeSteps", 500)),
                    player=args.player,
                )
                W = sub_mod._load_weights()
                src_list, dst_list, pct_list = sub_mod.greedy_multi_action(W, enc)
                moves = sub_mod.agent(obs_p, cfg_p)
            except Exception as exc:
                src_list, dst_list, pct_list = [], [], []
                moves = []
                exc_str = f"diag-ERR: {exc}"
            else:
                exc_str = ""

            sub_summary = _summary(obs_p, args.player)
            opp_summary = _summary(obs_o, 1 - args.player)
            print(
                f"{turn:>5} | {sub_summary:<26} | {opp_summary:<26} | "
                f"src={src_list} dst={dst_list} pct={pct_list}"
            )
            if moves:
                # Show pct values for moves: ships per move
                ship_list = [int(m[2]) for m in moves]
                print(f"      | -> agent() moves ships: {ship_list}  (count={len(moves)})")
            else:
                print(f"      | -> agent() moves: [] (filtered or no ships)")
            if exc_str:
                print(f"      | {exc_str}")

        # If we've already hit the last turn we wanted, exit early.
        if turn >= max_turn:
            break

        # Step env: call both agents and feed actions
        try:
            sub_action = sub_mod.agent(obs_p, cfg_p)
        except Exception as exc:
            sub_action = []
            print(f"  WARN: sub agent at turn {turn} raised: {exc}")
        try:
            opp_action = opp_mod.agent(obs_o, cfg_o)
        except Exception as exc:
            opp_action = []
            print(f"  WARN: opp agent at turn {turn} raised: {exc}")

        if args.player == 0:
            actions = [sub_action, opp_action]
        else:
            actions = [opp_action, sub_action]

        env.step(actions)

        if env.steps[-1][0].get("status") == "DONE":
            print(f"... env DONE at turn {turn+1}")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
