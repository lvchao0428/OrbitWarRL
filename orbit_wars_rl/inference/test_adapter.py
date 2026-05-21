"""Verify ``kaggle_adapter.encode_kaggle_obs`` agrees with the training encoder.

We build a fresh ``EnvState`` from the JAX env, encode it through both:

  1. ``orbit_wars_rl.features.encode`` (training-time, jax)
  2. ``orbit_wars_rl.inference.kaggle_adapter.encode_kaggle_obs`` (numpy, fed a
     "Kaggle-style" obs dict reconstructed from the state)

and assert every per-planet / per-fleet / global feature matches to ~1e-5.

Usage:
    python -m orbit_wars_rl.inference.test_adapter
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from orbit_wars_rl.env import OrbitWarsEnv
from orbit_wars_rl.features import encode as jax_encode
from orbit_wars_rl.inference.kaggle_adapter import encode_kaggle_obs


def _state_to_kaggle_obs(state, player: int) -> dict:
    """Reconstruct a Kaggle-style obs dict from an MVP EnvState.

    Slot index == planet id for the MVP env, which matches what Kaggle would
    feed in for static planets, so this is a faithful proxy.
    """
    planets = []
    p_owner = np.asarray(state.planet_owner)
    p_x = np.asarray(state.planet_x)
    p_y = np.asarray(state.planet_y)
    p_r = np.asarray(state.planet_radius)
    p_s = np.asarray(state.planet_ships)
    p_p = np.asarray(state.planet_prod)
    p_m = np.asarray(state.planet_mask)
    for i in range(p_owner.shape[0]):
        if not bool(p_m[i]):
            continue
        planets.append([int(i), int(p_owner[i]), float(p_x[i]), float(p_y[i]),
                        float(p_r[i]), int(p_s[i]), int(p_p[i])])

    fleets = []
    f_o = np.asarray(state.fleet_owner)
    f_x = np.asarray(state.fleet_x)
    f_y = np.asarray(state.fleet_y)
    f_a = np.asarray(state.fleet_angle)
    f_s = np.asarray(state.fleet_ships)
    f_m = np.asarray(state.fleet_mask)
    for i in range(f_o.shape[0]):
        if not bool(f_m[i]):
            continue
        fleets.append([int(i), int(f_o[i]), float(f_x[i]), float(f_y[i]),
                       float(f_a[i]), -1, int(f_s[i])])

    return {
        "player": player,
        "planets": planets,
        "fleets": fleets,
        "step": int(state.step),
    }


def run(seed: int = 11) -> int:
    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    state = env.reset(jax.random.PRNGKey(seed))
    player = 0

    jax_obs = jax_encode(state, player, episode_steps=60)
    kaggle_obs = _state_to_kaggle_obs(state, player)
    np_obs = encode_kaggle_obs(kaggle_obs, player=player, step=int(state.step), episode_steps=60)

    failed = False

    def _check(name: str, a: np.ndarray, b: np.ndarray, tol: float = 1e-5) -> None:
        nonlocal failed
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        if a_np.dtype == bool or b_np.dtype == bool:
            diff = (a_np.astype(np.int8) ^ b_np.astype(np.int8)).astype(np.float32)
        else:
            diff = np.abs(a_np.astype(np.float32) - b_np.astype(np.float32))
        mx = float(diff.max() if diff.size else 0.0)
        ok = mx < tol
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}] {name:<22}  max={mx:.3e}")
        if not ok:
            failed = True

    _check("planet_feats", jax_obs.planet_feats, np_obs.planet_feats)
    _check("planet_mask", jax_obs.planet_mask, np_obs.planet_mask, tol=0.5)
    _check("fleet_feats", jax_obs.fleet_feats, np_obs.fleet_feats)
    _check("fleet_mask", jax_obs.fleet_mask, np_obs.fleet_mask, tol=0.5)
    _check("global_feats", jax_obs.global_feats, np_obs.global_feats)
    _check("my_planet_mask", jax_obs.my_planet_mask, np_obs.my_planet_mask, tol=0.5)
    _check("enemy_planet_mask", jax_obs.enemy_planet_mask, np_obs.enemy_planet_mask, tol=0.5)
    _check("neutral_planet_mask", jax_obs.neutral_planet_mask, np_obs.neutral_planet_mask, tol=0.5)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
