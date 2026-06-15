"""Collect (obs, action) tuples by running v20 against itself in kaggle env.

Each tuple holds:
  - encoded observation tensors (planet_feats, fleet_feats, global_feats, masks)
    -- already JAX-numpy ready for train_bc. Encoded with the CURRENT feature
    stack (v21+: planet 63 = base 43 + planet_hist 5x4, global 427 = base 27
    + global_hist 50x8), with the history ring buffers maintained across
    turns exactly like env.step / submission_rl_v21 do.
  - planet_ships_raw / planet_x_raw / planet_y_raw / home_idx_raw -- raw
    turn-start fields needed by current K-step pair features inside evaluate()
  - planet_orbit_phase_raw / planet_orbit_radius_raw / planet_is_orbiting_raw
    / angular_velocity_raw -- orbital fields for the ETA-lead pair features
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
  games x ~120 turns x 2 players ~ 48k samples in ~30-50 minutes on a
  single thread. Shard with --seed and merge for parallel collection.
* We save both players' perspectives every turn. This doubles the dataset
  and exposes the network to both color positions.
* History semantics mirror env.step: hist is empty (zeros) when encoding
  turn 0; from turn 1 on, the current state's slice is rolled in BEFORE
  encoding, so global_hist[-1] == slice(current state). This is exactly
  what submission_rl_v21.py does at inference.
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
# Collection is CPU-bound on the kaggle env; don't grab the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from orbit_wars_rl.bc.action_inverse import kaggle_moves_to_targets
from orbit_wars_rl.env import constants
from orbit_wars_rl.features.encode import encode
from orbit_wars_rl.features.history import (
    empty_global_hist,
    empty_planet_hist,
    update_global_hist,
    update_planet_hist,
)
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
        # v23: orbital fields for ETA-lead pair features inside evaluate().
        "planet_orbit_phase_raw": np.asarray(state.planet_orbit_phase, dtype=np.float32),
        "planet_orbit_radius_raw": np.asarray(state.planet_orbit_radius, dtype=np.float32),
        "planet_is_orbiting_raw": np.asarray(state.planet_is_orbiting, dtype=np.bool_),
        "angular_velocity_raw": np.asarray(state.angular_velocity, dtype=np.float32),
    }


def _resolve_home_indices(obs: dict[str, Any]) -> list[int]:
    """Resolve both players' home slots from a TURN-0 observation.

    At turn 0 each player owns exactly one planet (their home), so the
    owner field of ``planets`` is authoritative. ``initial_planets`` shows
    homes as owner=-1 (pre-assignment) and cannot be used directly.
    """
    home = [0, 0]
    for p in obs.get("planets") or []:
        o = int(p[1])
        if o in (0, 1):
            home[o] = int(p[0])
    return home


def _collect_game(
    env, agent_a, agent_b, episode_steps: int, debug: bool = False
) -> tuple[List[dict[str, np.ndarray]], int]:
    """Play one v20-vs-v20 game; return (sample list, num_emits_total)."""
    env.reset()
    samples = []
    n_emits = 0
    turn = 0

    # History ring buffers, maintained across the whole game (both players'
    # POV slices live in the same [2, ...] arrays, same as EnvState).
    hist_g = empty_global_hist()
    hist_p = empty_planet_hist()
    home_idx: list[int] | None = None

    while True:
        state_p0 = env.state[0]
        state_p1 = env.state[1]

        if state_p0["status"] != "ACTIVE" and state_p1["status"] != "ACTIVE":
            break

        obs_p0 = state_p0["observation"]
        obs_p1 = state_p1["observation"]
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

        # Bridge ONCE per turn (board state is identical for both players;
        # only the POV index differs at encode time). Inject the carried
        # history + home indices, then roll in this turn's slice (t > 0).
        obs_any = obs_p0 if state_p0["status"] == "ACTIVE" else obs_p1
        n_planets = len(obs_any.get("planets") or [])
        if n_planets > _P:
            # Cannot bridge / keep hist consistent -- drop the rest of the
            # game (over-MAX_PLANETS maps are not in our training domain).
            print(f"  warn: {n_planets} planets > MAX_PLANETS={_P}, dropping game tail")
            break
        bridged = kaggle_obs_to_envstate(obs_any)
        if home_idx is None:
            home_idx = _resolve_home_indices(obs_any)
        bridged = bridged.replace(
            global_hist=hist_g,
            planet_hist=hist_p,
            home_planet_idx=jnp.asarray(home_idx, dtype=jnp.int32),
        )
        if turn > 0:
            # Same ordering as env.step / submission inference: hist last
            # frame == slice(current state) for every turn after the first.
            bridged = update_global_hist(bridged, episode_steps)
            bridged = update_planet_hist(bridged, episode_steps)
        hist_g = bridged.global_hist
        hist_p = bridged.planet_hist

        # Build samples for both perspectives BEFORE stepping (so encoded
        # state matches the obs v20 saw when it decided).
        for player, moves in [(0, move_p0), (1, move_p1)]:
            if env.state[player]["status"] != "ACTIVE":
                continue
            enc = _encode_one(bridged, player, episode_steps)
            targets = kaggle_moves_to_targets(bridged, player, moves)

            sample = {**enc, **targets}
            samples.append(sample)
            n_emits += int(np.sum(targets["emit"]))

            if debug and turn < 3:
                k = int(np.sum(targets["emit"]))
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

        dt = time.time() - t0
        print(
            f"  game {g+1}/{args.num_games}: {len(samples)} samples "
            f"emits/game={n_emits} | total={len(all_samples)} ({dt:.1f}s)",
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
    pct_hist = Counter()
    for s in all_samples:
        for k in range(_K):
            if s["emit"][k]:
                pct_hist[int(s["pct"][k])] += 1
    print(f"  total samples: {n}")
    print(f"  total emits  : {emit_total} (avg {emit_total/n:.2f} per sample)")
    print(f"  planet_feats : {stacked['planet_feats'].shape}")
    print(f"  global_feats : {stacked['global_feats'].shape}")
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
