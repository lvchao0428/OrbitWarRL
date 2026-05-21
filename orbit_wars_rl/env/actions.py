"""Discrete action conversion: (src_idx, dst_idx, pct_bin) -> fleet launch params.

MVP rules:
- Each player emits one action per turn (no multi-action yet).
- ``src_idx`` indexes a planet slot the player owns. If the chosen slot is
  invalid (not owned, padded, zero ships), the launch is silently dropped.
- ``angle`` is derived from straight src -> dst geometry (no lead intercept).
- ``num_ships`` = floor(src.ships * pct_bin), at least 1 if any are present.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


@chex.dataclass(frozen=True)
class PlayerAction:
    """Discrete action triple for a single player."""

    src_idx: chex.Array
    dst_idx: chex.Array
    pct_bin: chex.Array


_PCT_BIN_TABLE = jnp.array(constants.PCT_BIN_VALUES, dtype=jnp.float32)


def noop_action() -> PlayerAction:
    return PlayerAction(
        src_idx=jnp.int32(0),
        dst_idx=jnp.int32(0),
        pct_bin=jnp.int32(0),
    )


def decode_action(
    state: EnvState,
    action: PlayerAction,
    player: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (valid, src_idx_safe, ships_to_send, angle, dst_idx_safe).

    ``valid`` is False if the source is invalid for any reason; callers should
    skip the launch in that case (typically by masking).
    """
    src_idx = jnp.clip(action.src_idx, 0, constants.MAX_PLANETS - 1)
    dst_idx = jnp.clip(action.dst_idx, 0, constants.MAX_PLANETS - 1)
    pct_idx = jnp.clip(action.pct_bin, 0, constants.NUM_PCT_BINS - 1)
    pct = _PCT_BIN_TABLE[pct_idx]

    src_owner = state.planet_owner[src_idx]
    src_alive = state.planet_mask[src_idx]
    dst_alive = state.planet_mask[dst_idx]
    src_ships = state.planet_ships[src_idx]

    owns_src = src_owner == player
    has_ships = src_ships > 0
    different_target = src_idx != dst_idx
    valid = owns_src & src_alive & dst_alive & has_ships & different_target

    ships_to_send = jnp.maximum(
        jnp.int32(1),
        jnp.floor(src_ships.astype(jnp.float32) * pct).astype(jnp.int32),
    )
    ships_to_send = jnp.minimum(ships_to_send, src_ships)
    ships_to_send = jnp.where(valid, ships_to_send, jnp.int32(0))

    sx = state.planet_x[src_idx]
    sy = state.planet_y[src_idx]
    dx = state.planet_x[dst_idx]
    dy = state.planet_y[dst_idx]
    angle = jnp.arctan2(dy - sy, dx - sx)

    return valid, src_idx, ships_to_send, angle, dst_idx
