"""EnvState -> network-ready features. All jit-pure.

Each per-entity row is built so that **padding rows are exactly zero**, with
``mask`` carrying the validity flag. This keeps attention key-padding clean and
makes the global aggregates safe to compute via sum / mean.

Feature layouts (kept in lock-step with the heads):

  planet (12 dims):
    [0..2]  owner one-hot from the *current player*'s POV: [is_mine, is_enemy, is_neutral]
    [3]     x_norm  in [-1, 1]
    [4]     y_norm  in [-1, 1]
    [5]     radius_norm
    [6]     log1p(ships) / 8
    [7]     prod / 5
    [8]     dist_to_sun / BOARD
    [9]     fraction inbound friendly ships / max
    [10]    fraction inbound enemy ships  / max
    [11]    is_padding (1 if pad, else 0) -- redundant w/ mask but harmless

  fleet (8 dims):
    [0..2]  owner one-hot from POV: [mine, enemy, neutral=always 0 but kept for symmetry]
    [3]     x_norm
    [4]     y_norm
    [5]     sin(angle), cos(angle) stacked into [5], [6]
    [6]     cos(angle)
    [7]     log1p(ships) / 8

  global (10 dims):
    [0]   step / episode_steps
    [1]   my_ships_frac
    [2]   opp_ships_frac
    [3]   my_planet_frac
    [4]   opp_planet_frac
    [5]   my_prod_frac
    [6]   opp_prod_frac
    [7..9] phase one-hot [early, mid, late]
"""

from __future__ import annotations

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.dynamics import fleet_speed
from orbit_wars_rl.env.state import EnvState


PLANET_FEAT_DIM = 12
FLEET_FEAT_DIM = 8
GLOBAL_FEAT_DIM = 10

_BOARD = jnp.float32(constants.BOARD)
_BOARD_HALF = jnp.float32(constants.BOARD * 0.5)
_SUN = jnp.array([constants.SUN_X, constants.SUN_Y], dtype=jnp.float32)


@chex.dataclass(frozen=True)
class EncodedObs:
    """Player-conditioned observation tensors."""

    planet_feats: chex.Array
    planet_mask: chex.Array
    fleet_feats: chex.Array
    fleet_mask: chex.Array
    global_feats: chex.Array

    my_planet_mask: chex.Array
    enemy_planet_mask: chex.Array
    neutral_planet_mask: chex.Array


def _inbound_ships(
    state: EnvState,
    owner_keep: int,
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
) -> jnp.ndarray:
    """Approximate the total in-flight ships heading toward each planet.

    We use a coarse heuristic (closest planet to the fleet's *velocity ray*),
    sufficient for features. Each fleet is attributed to exactly one planet.
    """
    speed = fleet_speed(state.fleet_ships)
    eps = jnp.float32(1e-6)
    spd_safe = jnp.maximum(speed, eps)
    dxn = jnp.cos(state.fleet_angle)
    dyn = jnp.sin(state.fleet_angle)

    rel_x = planet_x[None, :] - state.fleet_x[:, None]
    rel_y = planet_y[None, :] - state.fleet_y[:, None]
    proj = rel_x * dxn[:, None] + rel_y * dyn[:, None]
    perp_x = rel_x - proj * dxn[:, None]
    perp_y = rel_y - proj * dyn[:, None]
    perp_d2 = perp_x * perp_x + perp_y * perp_y

    in_front = proj > 0
    cost = jnp.where(in_front, perp_d2, jnp.float32(1e9))
    cost = jnp.where(state.planet_mask[None, :], cost, jnp.float32(1e9))
    target_idx = jnp.argmin(cost, axis=1)

    is_owner_keep = state.fleet_owner == owner_keep
    contributes = state.fleet_mask & is_owner_keep
    ships_keep = jnp.where(contributes, state.fleet_ships, jnp.int32(0))

    inbound = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    inbound = inbound.at[target_idx].add(ships_keep.astype(jnp.float32))
    return inbound


def _encode_planets(state: EnvState, player: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    is_mine = (state.planet_owner == player) & state.planet_mask
    is_enemy = (state.planet_owner >= 0) & (state.planet_owner != player) & state.planet_mask
    is_neutral = (state.planet_owner == constants.NEUTRAL_OWNER) & state.planet_mask

    x_norm = (state.planet_x - _BOARD_HALF) / _BOARD_HALF
    y_norm = (state.planet_y - _BOARD_HALF) / _BOARD_HALF
    radius_norm = state.planet_radius / 5.0
    log_ships = jnp.log1p(jnp.maximum(state.planet_ships, 0).astype(jnp.float32)) / 8.0
    prod_norm = state.planet_prod.astype(jnp.float32) / 5.0
    dist_sun = jnp.sqrt(
        (state.planet_x - _SUN[0]) ** 2 + (state.planet_y - _SUN[1]) ** 2
    ) / _BOARD

    opp = 1 - player
    in_friend = _inbound_ships(state, player, state.planet_x, state.planet_y)
    in_foe = _inbound_ships(state, opp, state.planet_x, state.planet_y)
    in_friend_norm = jnp.log1p(in_friend) / 8.0
    in_foe_norm = jnp.log1p(in_foe) / 8.0

    is_padding = jnp.logical_not(state.planet_mask).astype(jnp.float32)

    feats = jnp.stack(
        [
            is_mine.astype(jnp.float32),
            is_enemy.astype(jnp.float32),
            is_neutral.astype(jnp.float32),
            x_norm,
            y_norm,
            radius_norm,
            log_ships,
            prod_norm,
            dist_sun,
            in_friend_norm,
            in_foe_norm,
            is_padding,
        ],
        axis=-1,
    )
    feats = feats * state.planet_mask[:, None].astype(jnp.float32)
    return feats, is_mine, is_enemy, is_neutral


def _encode_fleets(state: EnvState, player: int) -> jnp.ndarray:
    is_mine = (state.fleet_owner == player) & state.fleet_mask
    is_enemy = (state.fleet_owner >= 0) & (state.fleet_owner != player) & state.fleet_mask

    x_norm = (state.fleet_x - _BOARD_HALF) / _BOARD_HALF
    y_norm = (state.fleet_y - _BOARD_HALF) / _BOARD_HALF
    sin_a = jnp.sin(state.fleet_angle)
    cos_a = jnp.cos(state.fleet_angle)
    log_ships = jnp.log1p(jnp.maximum(state.fleet_ships, 0).astype(jnp.float32)) / 8.0
    zero = jnp.zeros_like(x_norm)

    feats = jnp.stack(
        [
            is_mine.astype(jnp.float32),
            is_enemy.astype(jnp.float32),
            zero,
            x_norm,
            y_norm,
            sin_a,
            cos_a,
            log_ships,
        ],
        axis=-1,
    )
    feats = feats * state.fleet_mask[:, None].astype(jnp.float32)
    return feats


def _encode_global(state: EnvState, player: int, episode_steps: int) -> jnp.ndarray:
    opp = 1 - player

    def player_ships(p: int) -> jnp.ndarray:
        ps_planet = jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_ships, 0
        ).sum()
        ps_fleet = jnp.where(
            (state.fleet_owner == p) & state.fleet_mask, state.fleet_ships, 0
        ).sum()
        return (ps_planet + ps_fleet).astype(jnp.float32)

    def player_planets(p: int) -> jnp.ndarray:
        return ((state.planet_owner == p) & state.planet_mask).sum().astype(jnp.float32)

    def player_prod(p: int) -> jnp.ndarray:
        return jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_prod, 0
        ).sum().astype(jnp.float32)

    total_ships = jnp.maximum(player_ships(player) + player_ships(opp), jnp.float32(1.0))
    total_planets = jnp.maximum(player_planets(player) + player_planets(opp), jnp.float32(1.0))
    total_prod = jnp.maximum(player_prod(player) + player_prod(opp), jnp.float32(1.0))

    step_norm = state.step.astype(jnp.float32) / jnp.float32(episode_steps)
    is_early = (step_norm < 0.18).astype(jnp.float32)
    is_mid = ((step_norm >= 0.18) & (step_norm < 0.64)).astype(jnp.float32)
    is_late = (step_norm >= 0.64).astype(jnp.float32)

    return jnp.stack(
        [
            step_norm,
            player_ships(player) / total_ships,
            player_ships(opp) / total_ships,
            player_planets(player) / total_planets,
            player_planets(opp) / total_planets,
            player_prod(player) / total_prod,
            player_prod(opp) / total_prod,
            is_early,
            is_mid,
            is_late,
        ],
    )


def encode(state: EnvState, player: int, episode_steps: int = constants.DEFAULT_EPISODE_STEPS) -> EncodedObs:
    planet_feats, is_mine, is_enemy, is_neutral = _encode_planets(state, player)
    fleet_feats = _encode_fleets(state, player)
    global_feats = _encode_global(state, player, episode_steps)

    return EncodedObs(
        planet_feats=planet_feats,
        planet_mask=state.planet_mask,
        fleet_feats=fleet_feats,
        fleet_mask=state.fleet_mask,
        global_feats=global_feats,
        my_planet_mask=is_mine,
        enemy_planet_mask=is_enemy,
        neutral_planet_mask=is_neutral,
    )
