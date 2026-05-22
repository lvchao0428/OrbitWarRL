"""Env state as flat jnp arrays. All shapes are static across an entire run."""

from __future__ import annotations

import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class EnvState:
    """Single-env state. Add a leading batch dim via vmap.

    Planets and fleets live in fixed-size slots; ``mask`` tells which are real.
    Owners use ``-2`` for padding, ``-1`` for neutral, ``0..NUM_PLAYERS-1`` for players.

    Orbital fields (added 2026-05-22 to match Kaggle env):
      * ``angular_velocity``: scalar rad/turn for the whole episode, sampled at
        reset uniformly from [0.025, 0.05] -- matches Kaggle's range.
      * ``planet_orbit_radius``: distance from each planet to the sun centre
        (50, 50). For static planets this just records their current distance.
      * ``planet_orbit_phase``: current angle (rad) of each planet around the
        sun, measured from +x axis. ``planet_x = 50 + r*cos(phase)`` and same
        for y. Updated every step on orbiting planets only.
      * ``planet_is_orbiting``: bool[MAX_PLANETS]. True iff the planet rotates.
        Static planets keep ``phase`` fixed, so ``planet_x/y`` stays put.

    ``planet_x`` and ``planet_y`` remain the source of truth for collision
    detection and policy features. They are recomputed each step from
    ``(orbit_radius, phase)`` so downstream code doesn't need to change.
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

    angular_velocity: chex.Array
    planet_orbit_radius: chex.Array
    planet_orbit_phase: chex.Array
    planet_is_orbiting: chex.Array


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
        angular_velocity=jnp.float32(0.0),
        planet_orbit_radius=jnp.zeros((max_planets,), dtype=jnp.float32),
        planet_orbit_phase=jnp.zeros((max_planets,), dtype=jnp.float32),
        planet_is_orbiting=jnp.zeros((max_planets,), dtype=jnp.bool_),
    )
