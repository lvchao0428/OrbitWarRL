"""Discrete action conversion: (src_idx, dst_idx, pct_bin) -> fleet launch params.

Two action shapes are supported:

* ``PlayerAction`` -- legacy single-fleet-per-turn action used by old random
  opponents, eval harnesses, and the v1/v2 ckpts.
* ``MultiPlayerAction`` -- up to K fleets per turn (autoregressive policy),
  each step carries an ``emit_mask`` flag so the network can choose to stop
  early. ``K = constants.MAX_FLEETS_PER_TURN``.

``decode_action`` operates on a single ``PlayerAction``; the multi-fleet path
in ``dynamics.launch_fleets`` reuses the same single-action decoder K times
with a running ``reserved_ships`` buffer to avoid double-spending a source.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


@chex.dataclass(frozen=True)
class PlayerAction:
    """Discrete action triple for a single fleet launch."""

    src_idx: chex.Array
    dst_idx: chex.Array
    pct_bin: chex.Array


@chex.dataclass(frozen=True)
class MultiPlayerAction:
    """Up to ``K = MAX_FLEETS_PER_TURN`` fleet launches per turn.

    Shapes are all ``[K]`` (no batch dim, vmap to add one):

    * ``src_idx``    int32 -- planet slot to launch from (0 if not emitting)
    * ``dst_idx``    int32 -- target planet slot
    * ``pct_bin``    int32 -- pct table index
    * ``emit_mask``  bool  -- True means actually launch this step

    Invariants the trainer/runner must uphold:
    * Once ``emit_mask[t] == False`` for some t, all later steps should also
      be False (downstream code will treat any True after a False as a no-op
      anyway because dynamics consumes them in order with reserved_ships).
    """

    src_idx: chex.Array      # int32 [K]
    dst_idx: chex.Array      # int32 [K]
    pct_bin: chex.Array      # int32 [K]
    emit_mask: chex.Array    # bool  [K]


_PCT_BIN_TABLE = jnp.array(constants.PCT_BIN_VALUES, dtype=jnp.float32)


def noop_action() -> PlayerAction:
    return PlayerAction(
        src_idx=jnp.int32(0),
        dst_idx=jnp.int32(0),
        pct_bin=jnp.int32(0),
    )


def noop_multi_action(k: int = constants.MAX_FLEETS_PER_TURN) -> MultiPlayerAction:
    return MultiPlayerAction(
        src_idx=jnp.zeros((k,), dtype=jnp.int32),
        dst_idx=jnp.zeros((k,), dtype=jnp.int32),
        pct_bin=jnp.zeros((k,), dtype=jnp.int32),
        emit_mask=jnp.zeros((k,), dtype=jnp.bool_),
    )


def single_to_multi(action: PlayerAction, k: int = constants.MAX_FLEETS_PER_TURN) -> MultiPlayerAction:
    """Wrap a legacy single PlayerAction as a MultiPlayerAction with K=1 emit.

    The K-1 trailing slots get emit_mask=False and arbitrary indices.
    Used so legacy random_opponent / eval code can drive the new env without
    changes. JIT-pure (no Python control flow on traced shapes).
    """
    zeros_k = jnp.zeros((k,), dtype=jnp.int32)
    emit_mask = jnp.zeros((k,), dtype=jnp.bool_).at[0].set(True)
    return MultiPlayerAction(
        src_idx=zeros_k.at[0].set(action.src_idx.astype(jnp.int32)),
        dst_idx=zeros_k.at[0].set(action.dst_idx.astype(jnp.int32)),
        pct_bin=zeros_k.at[0].set(action.pct_bin.astype(jnp.int32)),
        emit_mask=emit_mask,
    )


def decode_action(
    state: EnvState,
    action: PlayerAction,
    player: int,
    reserved_ships: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (valid, src_idx_safe, ships_to_send, angle, dst_idx_safe).

    ``valid`` is False if the source is invalid for any reason; callers should
    skip the launch in that case (typically by masking).

    ``reserved_ships`` (optional int32 [MAX_PLANETS]) is the running total of
    ships already promised to other launches in this same turn. The available
    ships at ``src`` are computed as ``state.planet_ships[src] - reserved[src]``
    so multi-fleet turns don't oversubscribe a single planet.
    """
    src_idx = jnp.clip(action.src_idx, 0, constants.MAX_PLANETS - 1)
    dst_idx = jnp.clip(action.dst_idx, 0, constants.MAX_PLANETS - 1)
    pct_idx = jnp.clip(action.pct_bin, 0, constants.NUM_PCT_BINS - 1)
    pct = _PCT_BIN_TABLE[pct_idx]

    src_owner = state.planet_owner[src_idx]
    src_alive = state.planet_mask[src_idx]
    dst_alive = state.planet_mask[dst_idx]

    raw_src_ships = state.planet_ships[src_idx]
    if reserved_ships is None:
        src_ships = raw_src_ships
    else:
        src_ships = jnp.maximum(raw_src_ships - reserved_ships[src_idx], jnp.int32(0))

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
