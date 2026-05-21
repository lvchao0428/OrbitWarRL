"""Env state as flat jnp arrays. All shapes are static across an entire run."""

from __future__ import annotations

import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class EnvState:
    """Single-env state. Add a leading batch dim via vmap.

    Planets and fleets live in fixed-size slots; ``mask`` tells which are real.
    Owners use ``-2`` for padding, ``-1`` for neutral, ``0..NUM_PLAYERS-1`` for players.
    """

    planet_owner: chex.Array
    planet_x: chex.Array
    planet_y: chex.Array
    planet_radius: chex.Array
    planet_ships: chex.Array
    planet_prod: chex.Array
    planet_mask: chex.Array

    fleet_owner: chex.Array
    fleet_x: chex.Array
    fleet_y: chex.Array
    fleet_angle: chex.Array
    fleet_ships: chex.Array
    fleet_mask: chex.Array

    step: chex.Array
    done: chex.Array
    rng: chex.Array


def empty_state(max_planets: int, max_fleets: int) -> EnvState:
    """All-zero placeholder; useful for shape probing or unit tests."""
    return EnvState(
        planet_owner=jnp.full((max_planets,), -2, dtype=jnp.int8),
        planet_x=jnp.zeros((max_planets,), dtype=jnp.float32),
        planet_y=jnp.zeros((max_planets,), dtype=jnp.float32),
        planet_radius=jnp.zeros((max_planets,), dtype=jnp.float32),
        planet_ships=jnp.zeros((max_planets,), dtype=jnp.int32),
        planet_prod=jnp.zeros((max_planets,), dtype=jnp.int32),
        planet_mask=jnp.zeros((max_planets,), dtype=jnp.bool_),
        fleet_owner=jnp.full((max_fleets,), -2, dtype=jnp.int8),
        fleet_x=jnp.zeros((max_fleets,), dtype=jnp.float32),
        fleet_y=jnp.zeros((max_fleets,), dtype=jnp.float32),
        fleet_angle=jnp.zeros((max_fleets,), dtype=jnp.float32),
        fleet_ships=jnp.zeros((max_fleets,), dtype=jnp.int32),
        fleet_mask=jnp.zeros((max_fleets,), dtype=jnp.bool_),
        step=jnp.int32(0),
        done=jnp.bool_(False),
        rng=jnp.zeros((2,), dtype=jnp.uint32),
    )
