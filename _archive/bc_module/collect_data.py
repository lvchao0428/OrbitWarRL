"""Collect (obs, action) tuples by running v20 against itself in kaggle env.

Each tuple holds:
  - encoded observation tensors (planet_feats, fleet_feats, global_feats, masks)
    -- already JAX-numpy ready for train_bc
  - planet_ships_raw / planet_x_raw / planet_y_raw / home_idx_raw -- raw
    turn-start fields needed by current K-step pair features inside evaluate()
  - my_planet_mask (P,) -- separate bool array for use in PPO-style heads
  - K-step targets from action_inverse: src/dst/pct/emit/loss_mask/emit_free

Stored as a single .npz with stacked arrays so train_bc can iterate via
slice indexing without dict gather costs.

Usage:
    python -m orbit_wars_rl.bc.collect_data \
        --num-games 200 --opponent submission_v20_0513.py \
        --agent submission_v20_0513.py --out data/bc_v20_self.npz

Smoke run:
    python -m orbit_wars_rl.bc.collect_data --num-games 2 --out /tmp/bc_smoke.npz

Notes
-----
* kaggle_environments is slow (~80-150ms per turn with v20 thinking); 200
  games × ~120 turns × 2 players ≈ 48k samples in ~30-50 minutes on a
  single thread.
* We save both players' perspectives every turn. This doubles the dataset
  and exposes the network to both color positions.
* We do NOT include episodes where v20 errored (kaggle marks them as
  ``status=ERROR``); those moves are missing/empty.
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, List

import jax.numpy as jnp
import numpy as np

# Stop kaggle envs from auto-installing scipy / other heavy deps when running
# locally. We only need orbit_wars.
import os
os.environ.setdefault("KAGGLE_DISABLE_COLAB_FALLBACK", "1")

from orbit_wars_rl.bc.action_inverse import kaggle_moves_to_targets
from orbit_wars_rl.env import constants
from orbit_wars_rl.features.encode import encode
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate


_K = constants.MAX_FLEETS_PER_TURN
_P = constants.MAX_PLANETS
_F = constants.MAX_FLEETS


def _load_agent(path: str):
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
    if not hasattr(mod, "agent"):
        raise SystemExit(f"{path} has no agent() function")
    return mod.agent


def _encode_one(state, player: int, episode_steps: int) -> dict[str, np.ndarray]:
    """Run our encoder once on a kaggle-bridged state, return numpy arrays."""
    enc = encode(state, jnp.int32(player), jnp.int32(episode_steps))
    return {
        "planet_feats": np.asarray(enc.planet_feats),
        "planet_mask": np.asarray(enc.planet_mask),
        "fleet_feats": np.asarray(enc.fleet_feats),
        "fleet_mask": np.asarray(enc.fleet_mask),
        "global_feats": np.asarray(enc.global_feats),
        "my_planet_mask": np.asarray(enc.my_planet_mask),
        "enemy_planet_mask": np.asarray(enc.enemy_planet_mask),
        "neutral_planet_mask": np.asarray(enc.neutral_planet_mask),
        "planet_ships_raw": np.asarray(state.planet_ships, dtype=np.int32),
        "planet_x_raw": np.asarray(state.planet_x, dtype=np.float32),
        "planet_y_raw": np.asarray(state.planet_y, dtype=np.float32),
        "home_idx_raw": np.asarray(state.home_planet_idx[player], dtype=np.int32),
    }


def _state_for_player(env, player: int):
    """Build our EnvState from the kaggle obs of one player. We use whichever
    side is active *at this turn* (kaggle stores both perspectives).

    Returns ``None`` if the obs has more planets than ``MAX_PLANETS`` -- the
    bridge would raise; we silently skip those samples instead of aborting
    the whole game. Caller treats ``None`` as a "skip this sample" signal.
    """
    obs = env.state[player]["observation"]
    n_planets = len(obs.get("planets") or [])
    if n_planets > _P:
        return None
    return kaggle_obs_to_envstate(obs)


def _collect_game(
    env, agent_a, agent_b, episode_steps: int, debug: bool = False
) -> tuple[List[dict[str, np.ndarray]], int]:
    """Play one v20-vs-v20 game; return (sample list, num_emits_total)."""
    env.reset()
    samples = []
    n_emits = 0
    turn = 0

    while True:
        state_p0 = env.state[0]
        state_p1 = env.state[1]

        if state_p0["status"] != "ACTIVE" and state_p1["status"] != "ACTIVE":
            break

        # Both agents see *their* observation (kaggle hides info per player,
        # but in orbit_wars both players see the full board -- still, we
        # call obs through both perspectives so the bridge gets the
        # ``player`` index right).
        obs_p0 = state_p0["observation"]
        obs_p1 = state_p1["observation"]
        # Mark perspective: kaggle obs has obs["player"] set to 0 or 1.
        # If missing, we infer from the state index.
        obs_p0.setdefault("player", 0)
        obs_p1.setdefault("player", 1)

        try:
            move_p0 = agent_a(obs_p0, env.configuration) if state_p0["status"] == "ACTIVE" else []
        except Exception:
            move_p0 = []
        try:
            move_p1 = agent_b(obs_p1, env.configuration) if state_p1["status"] == "ACTIVE" else []
        except Exception:
            move_p1 = []

        # Build samples for both perspectives BEFORE stepping (so encoded
        # state matches the obs v20 saw when it decided).
        for player, moves in [(0, move_p0), (1, move_p1)]:
            if env.state[player]["status"] != "ACTIVE":
                continue
            bridged = _state_for_player(env, player)
            if bridged is None:
                # too many planets -- skip this sample but keep the game going
                continue
            enc = _encode_one(bridged, player, episode_steps)
            targets = kaggle_moves_to_targets(bridged, player, moves)

            sample = {**enc, **targets}
            samples.append(sample)
            n_emits += int(np.sum(targets["emit"]))

            if debug and turn < 3:
                k = int(np.sum(targets["emit"]))
                ck = ["emit" if e else "-" for e in targets["emit"]]
                print(
                    f"  [turn {turn} p{player}] moves={len(moves)} "
                    f"keep={k}  src={list(targets['src'][:k])} "
                    f"dst={list(targets['dst'][:k])} pct={list(targets['pct'][:k])}"
                )

        env.step([move_p0, move_p1])
        turn += 1
        # Belt-and-suspenders: bail if env doesn't update for safety
        if turn > episode_steps + 5:
            print(f"  warn: turn limit hit ({turn}), bailing")
            break

    return samples, n_emits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=2,
                        help="number of self-play games to collect")
    parser.add_argument("--agent", type=str, default="submission_v20_0513.py")
    parser.add_argument("--opponent", type=str, default="submission_v20_0513.py",
                        help="opponent in self-play; default same as agent (BC dataset)")
    parser.add_argument("--out", type=str, default="data/bc_v20_self.npz")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading agent: {args.agent}")
    agent_a = _load_agent(args.agent)
    print(f"loading opponent: {args.opponent}")
    agent_b = _load_agent(args.opponent)

    # kaggle_environments must be imported AFTER we set env vars.
    from kaggle_environments import make

    all_samples: List[dict[str, np.ndarray]] = []
    n_games_done = 0
    n_total_emits = 0
    move_counts = Counter()
    t0 = time.time()

    for g in range(args.num_games):
        env = make("orbit_wars", debug=False)
        env.configuration.seed = args.seed + g
        # Smoothly capture the games
        try:
            samples, n_emits = _collect_game(env, agent_a, agent_b,
                                              args.episode_steps,
                                              debug=args.debug)
        except Exception as e:
            print(f"  game {g} errored: {e!r}", flush=True)
            continue
        all_samples.extend(samples)
        n_games_done += 1
        n_total_emits += n_emits
        for s in samples:
            move_counts[int(np.sum(s["emit"]))] += 1

        if g % 5 == 0 or g == args.num_games - 1:
            dt = time.time() - t0
            print(
                f"  game {g+1}/{args.num_games}: {len(samples)} samples "
                f"emits/game={n_emits} | total={len(all_samples)} ({dt:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"  game {g+1}: +{len(samples)} samples ({time.time()-t0:.1f}s)",
                flush=True,
            )

    if not all_samples:
        print("ERROR: no samples collected")
        return 1

    # Stack everything; consistent leading axis = sample index.
    print(f"\nstacking {len(all_samples)} samples...")
    keys = list(all_samples[0].keys())
    stacked = {k: np.stack([s[k] for s in all_samples], axis=0) for k in keys}

    # Report dataset statistics.
    n = stacked["src"].shape[0]
    emit_total = int(stacked["emit"].sum())
    emit_per_sample = stacked["emit"].sum(axis=-1)
    pct_hist = Counter()
    for s in all_samples:
        for k in range(_K):
            if s["emit"][k]:
                pct_hist[int(s["pct"][k])] += 1
    print(f"  total samples: {n}")
    print(f"  total emits  : {emit_total} (avg {emit_total/n:.2f} per sample)")
    print(f"  emits-per-sample histogram:")
    for nk in sorted(move_counts.keys()):
        print(f"    {nk} emits: {move_counts[nk]} samples ({100*move_counts[nk]/n:.1f}%)")
    print(f"  pct_bin distribution:")
    for b in range(constants.NUM_PCT_BINS):
        v = constants.PCT_BIN_VALUES[b]
        print(f"    bin {b} ({v}): {pct_hist.get(b,0)} ({100*pct_hist.get(b,0)/max(emit_total,1):.1f}%)")

    print(f"\nwriting {out}...")
    np.savez_compressed(out, **stacked)
    print(f"  size: {out.stat().st_size / 1e6:.1f} MB")
    print(f"  done. {n_games_done} games in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    # Force stdout line-buffering so background-launched jobs show progress.
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
