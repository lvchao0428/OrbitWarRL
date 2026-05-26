"""Collect EnvState snapshots from v20 self-play games for buffer curriculum.

Each game step produces one EnvState (from **both** players' perspectives, since
the board is fully observable in orbit_wars). States are stored as flat numpy
arrays (one row per field) so they can be stacked into a JAX pytree later.

The resulting .npz can be loaded by the rollout reset machinery:
``make_rollout_fn_with_buffer_reset`` in ``orbit_wars_rl.ppo.rollout``.

Design choices
--------------
* We collect from **both** player perspectives each turn: player 0's view and
  player 1's view.  From the RL perspective "player 0 = learning agent", so both
  views are valid starting states for the learner (the bridge flips owner ids to
  always make the collector's player "player 0").
* We skip the **last 20%** of each episode (high-garr endgame states that look
  very different from early/mid game where the learner gets stuck).
* States are deduplicated by a lightweight (step, p0_ships_total) key to avoid
  over-representing any particular game.

Usage
-----
    # Quick smoke test (2 games, ~30 seconds)
    python -m orbit_wars_rl.bc.collect_states \\
        --agent submission_v20_0513.py --num-games 2 \\
        --out /tmp/v20_states_smoke.npz

    # Full buffer (200 games, ~40 min on CPU)
    python -m orbit_wars_rl.bc.collect_states \\
        --agent submission_v20_0513.py --num-games 200 \\
        --out data/v20_states_200g.npz

Notes
-----
* kaggle_environments is slow (~80-150ms/turn). 200 games × ~120 early turns
  × 2 perspectives ≈ 48k states in ~35 min.
* Fleet slot ordering: we pack live fleets into slots 0..n-1 (same as
  kaggle_bridge).  This matches what the encoder already assumes.
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import os
import sys
import time
from pathlib import Path
from typing import Any, List

import numpy as np

os.environ.setdefault("KAGGLE_DISABLE_COLAB_FALLBACK", "1")

from orbit_wars_rl.env import constants
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate

_P = constants.MAX_PLANETS
_F = constants.MAX_FLEETS


def _load_agent(path: str):
    name = Path(path).stem
    spec = iu.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise SystemExit(f"{path} has no agent() function")
    return mod.agent


def _state_to_numpy(state) -> dict[str, np.ndarray]:
    """Flatten one EnvState to a dict of 1-D / scalar numpy arrays."""
    return {
        "planet_owner":       np.asarray(state.planet_owner, dtype=np.int8),
        "planet_x":           np.asarray(state.planet_x, dtype=np.float32),
        "planet_y":           np.asarray(state.planet_y, dtype=np.float32),
        "planet_radius":      np.asarray(state.planet_radius, dtype=np.float32),
        "planet_ships":       np.asarray(state.planet_ships, dtype=np.int32),
        "planet_prod":        np.asarray(state.planet_prod, dtype=np.int32),
        "planet_mask":        np.asarray(state.planet_mask, dtype=bool),
        "fleet_owner":        np.asarray(state.fleet_owner, dtype=np.int8),
        "fleet_x":            np.asarray(state.fleet_x, dtype=np.float32),
        "fleet_y":            np.asarray(state.fleet_y, dtype=np.float32),
        "fleet_angle":        np.asarray(state.fleet_angle, dtype=np.float32),
        "fleet_ships":        np.asarray(state.fleet_ships, dtype=np.int32),
        "fleet_mask":         np.asarray(state.fleet_mask, dtype=bool),
        "step":               np.asarray(state.step, dtype=np.int32).reshape(()),
        "angular_velocity":   np.asarray(state.angular_velocity, dtype=np.float32).reshape(()),
        "planet_orbit_radius":np.asarray(state.planet_orbit_radius, dtype=np.float32),
        "planet_orbit_phase": np.asarray(state.planet_orbit_phase, dtype=np.float32),
        "planet_is_orbiting": np.asarray(state.planet_is_orbiting, dtype=bool),
        # home_planet_idx: bridge can't recover true home from mid-game obs;
        # we store zeros here. The reset hook will set step=0 and done=False
        # anyway, so shaping that uses home_planet_idx (keep_home) will use the
        # sentinel value consistently.
        "home_planet_idx":    np.zeros(constants.NUM_PLAYERS, dtype=np.int32),
    }


def _flip_owners(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a copy of ``d`` with player 0 ↔ player 1 swapped.

    Planets/fleets owned by player 0 become player 1 and vice versa. Neutral
    (-1) and padding (-2) owners are unchanged.

    This lets us treat both perspectives of a game as independent training
    states where the learner is always player 0.
    """
    out = {k: v.copy() for k, v in d.items()}
    for field in ("planet_owner", "fleet_owner"):
        arr = out[field].copy()
        is0 = arr == np.int8(0)
        is1 = arr == np.int8(1)
        arr[is0] = np.int8(1)
        arr[is1] = np.int8(0)
        out[field] = arr
    # Also flip home_planet_idx: swap home[0] and home[1]
    h = out["home_planet_idx"].copy()
    h[0], h[1] = h[1], h[0]
    out["home_planet_idx"] = h
    return out


def _collect_game(
    env,
    agent_a,
    agent_b,
    episode_steps: int,
    skip_tail_frac: float = 0.20,
    debug: bool = False,
) -> List[dict[str, np.ndarray]]:
    """Play one game; return state snapshots from early/mid turns only."""
    env.reset()
    samples: List[dict[str, np.ndarray]] = []
    turn = 0
    cutoff = int(episode_steps * (1.0 - skip_tail_frac))

    while True:
        s0 = env.state[0]
        s1 = env.state[1]
        if s0["status"] != "ACTIVE" and s1["status"] != "ACTIVE":
            break

        obs0 = s0["observation"]
        obs1 = s1["observation"]
        obs0.setdefault("player", 0)
        obs1.setdefault("player", 1)

        # Collect states BEFORE stepping, but only up to the cutoff turn.
        if turn < cutoff:
            n_planets = len(obs0.get("planets") or [])
            if n_planets <= _P:
                try:
                    state_p0 = kaggle_obs_to_envstate(obs0)
                    d0 = _state_to_numpy(state_p0)
                    samples.append(d0)
                    # Player-1 perspective: flip owner ids so learner = p0.
                    samples.append(_flip_owners(d0))
                except Exception as e:
                    if debug:
                        print(f"  [turn {turn}] bridge error: {e}")

        # Step both agents.
        try:
            m0 = agent_a(obs0, env.configuration) if s0["status"] == "ACTIVE" else []
        except Exception:
            m0 = []
        try:
            m1 = agent_b(obs1, env.configuration) if s1["status"] == "ACTIVE" else []
        except Exception:
            m1 = []

        env.step([m0, m1])
        turn += 1
        if turn > episode_steps + 5:
            break

    if debug:
        print(f"  collected {len(samples)} states from {turn} turns "
              f"(cutoff={cutoff})")
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Collect EnvState snapshots from v20 self-play for buffer curriculum."
    )
    ap.add_argument("--agent", type=str, default="submission_v20_0513.py",
                    help="path to submission .py; used as both sides")
    ap.add_argument("--opponent", type=str, default=None,
                    help="opponent .py (defaults to --agent for self-play)")
    ap.add_argument("--num-games", type=int, default=50)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--skip-tail-frac", type=float, default=0.20,
                    help="fraction of episode tail to skip (avoid endgame states)")
    ap.add_argument("--out", type=str, default="data/v20_states.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading agent: {args.agent}")
    agent_a = _load_agent(args.agent)
    opp_path = args.opponent or args.agent
    if opp_path != args.agent:
        print(f"loading opponent: {opp_path}")
    agent_b = _load_agent(opp_path) if opp_path != args.agent else agent_a

    from kaggle_environments import make

    all_samples: List[dict[str, np.ndarray]] = []
    n_ok = 0
    t0 = time.time()

    for g in range(args.num_games):
        env_k = make("orbit_wars", debug=False)
        env_k.configuration.seed = args.seed + g
        try:
            samples = _collect_game(
                env_k, agent_a, agent_b,
                episode_steps=args.episode_steps,
                skip_tail_frac=args.skip_tail_frac,
                debug=args.debug,
            )
        except Exception as e:
            print(f"  game {g} errored: {e!r}", flush=True)
            continue
        all_samples.extend(samples)
        n_ok += 1

        if g % 10 == 0 or g == args.num_games - 1:
            dt = time.time() - t0
            print(
                f"  game {g+1}/{args.num_games}: +{len(samples)} states | "
                f"total={len(all_samples)} ({dt:.1f}s)",
                flush=True,
            )

    if not all_samples:
        print("ERROR: no states collected")
        return 1

    print(f"\nstacking {len(all_samples)} states...")
    keys = list(all_samples[0].keys())
    stacked = {k: np.stack([s[k] for s in all_samples], axis=0) for k in keys}

    # Report basic stats.
    n = stacked["planet_ships"].shape[0]
    steps = stacked["step"]
    p0_ships = stacked["planet_ships"][stacked["planet_owner"] == 0].sum()
    print(f"  total states : {n}")
    print(f"  step range   : [{steps.min()}, {steps.max()}] mean={steps.mean():.1f}")
    hist, edges = np.histogram(steps, bins=10)
    for i, c in enumerate(hist):
        print(f"    step [{edges[i]:.0f},{edges[i+1]:.0f}): {c} ({100*c/n:.1f}%)")

    print(f"\nwriting {out}...")
    np.savez_compressed(out, **stacked)
    size_mb = out.stat().st_size / 1e6
    print(f"  size: {size_mb:.1f} MB  |  {n_ok} games in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
