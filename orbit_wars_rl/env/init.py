"""Map generation. MVP: static planets, 4-fold symmetric, 2P opposing.

Returns a fresh ``EnvState`` with planets filling the first ``4*num_groups``
slots and the rest padded out.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


def _symmetric_group_offsets(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Given an anchor (x, y) in quadrant 1, build 4 mirrored positions.

    Mirror order matters and matches Kaggle: anchor goes to slot 0 (quadrant
    1: high x, high y), slot 1 mirrors across the vertical line x=50, slot 2
    mirrors across y=50, slot 3 is the diagonal opposite (used as home for
    player 1 in 2P games).
    """
    cx = jnp.float32(constants.BOARD * 0.5)
    cy = jnp.float32(constants.BOARD * 0.5)
    return jnp.stack(
        [
            jnp.stack([x, y]),
            jnp.stack([2 * cx - x, y]),
            jnp.stack([x, 2 * cy - y]),
            jnp.stack([2 * cx - x, 2 * cy - y]),
        ],
        axis=0,
    )


# Kaggle env has the home group be one of the symmetric groups picked at
# random; that means a home planet *can* be orbiting (confirmed via seeds 1,
# 500, 1000, 3000). Our implementation respects this by deriving
# planet_is_orbiting purely from (distance_to_sun + radius < 50), not from
# home-group status.
#
# Kaggle's anchor range is wider than ours: empirically we've seen home
# orbital radii up to ~61 (seed 100), implying anchors can sit near the
# board corners. Our current sampler bounds anchors to [12, 47] which keeps
# orb_r in roughly [4.2, 53.7]. Closing that gap is a separate concern --
# logging it here so it's not forgotten.


def reset(rng: jnp.ndarray, num_groups: int = 5, shuffle_slots: bool | None = None) -> EnvState:
    """Generate a 2P map with ``num_groups`` symmetric planet groups.

    ``num_groups`` is a Python int (static); the *positions* within each group
    are randomized via ``rng``. Slots beyond ``num_groups*4`` are masked off.

    ``shuffle_slots`` permutes the assignment of (owner, position, ships, prod)
    across the first 4*num_groups slot ids. Without it, player 0 home is
    always at slot 0 and player 1 home is always at slot 3 -- which lets the
    transformer-policy memorize "slot 0 = my home" instead of using features.
    Real Kaggle env has no such slot-id stability; shuffling closes that gap.

    If ``shuffle_slots`` is None (default), reads env var ORBITWARS_SHUFFLE_SLOTS
    (1/true/yes => on, 0/false/no => off). If unset, defaults to ON. This lets
    legacy ckpts trained on fixed slots be evaluated fairly by setting
    ORBITWARS_SHUFFLE_SLOTS=0 at the command line.
    """
    if shuffle_slots is None:
        env_val = os.environ.get("ORBITWARS_SHUFFLE_SLOTS", "1").strip().lower()
        shuffle_slots = env_val not in ("0", "false", "no", "off", "")

    assert constants.MIN_PLANET_GROUPS <= num_groups <= constants.MAX_PLANET_GROUPS
    assert num_groups * constants.PLANETS_PER_GROUP <= constants.MAX_PLANETS

    rng_pos, rng_prod, rng_ships, rng_perm, rng_omega, rng_state = jax.random.split(rng, 6)

    half = constants.BOARD * 0.5
    margin = 12.0
    anchors_x = jax.random.uniform(
        rng_pos,
        (num_groups,),
        minval=jnp.float32(margin),
        maxval=jnp.float32(half - 3.0),
    )
    anchors_y = jax.random.uniform(
        jax.random.fold_in(rng_pos, 1),
        (num_groups,),
        minval=jnp.float32(margin),
        maxval=jnp.float32(half - 3.0),
    )

    group_xy = jax.vmap(_symmetric_group_offsets)(anchors_x, anchors_y)
    xy = group_xy.reshape(num_groups * 4, 2)

    productions = jax.random.randint(
        rng_prod,
        (num_groups,),
        minval=constants.PROD_MIN,
        maxval=constants.PROD_MAX + 1,
    )
    productions = jnp.repeat(productions, 4)
    radius = 1.0 + jnp.log(productions.astype(jnp.float32))

    is_home_group_planet = jnp.zeros((num_groups * 4,), dtype=jnp.bool_).at[0].set(True).at[3].set(True)
    # Pre-permutation home slot ids: player 0 = slot 0, player 1 = slot 3.
    # After the optional permutation we update these via the permutation
    # inverse so they keep pointing at the actual home planets.
    home_idx_pre = jnp.array([0, 3], dtype=jnp.int32)

    raw_ships = jax.random.randint(
        rng_ships,
        (num_groups,),
        minval=constants.NEUTRAL_SHIPS_MIN,
        maxval=constants.NEUTRAL_SHIPS_MAX + 1,
    )
    raw_ships = jnp.repeat(raw_ships, 4)
    ships = jnp.where(is_home_group_planet, jnp.int32(constants.HOME_PLANET_SHIPS), raw_ships)

    owners = jnp.full((num_groups * 4,), constants.NEUTRAL_OWNER, dtype=jnp.int8)
    owners = owners.at[0].set(jnp.int8(0)).at[3].set(jnp.int8(1))

    # Orbit fields. We derive these BEFORE the slot permutation so the four
    # planets in each group rotate together (preserving the 4-fold symmetry).
    sun_x = jnp.float32(constants.SUN_X)
    sun_y = jnp.float32(constants.SUN_Y)
    dx = xy[:, 0] - sun_x
    dy = xy[:, 1] - sun_y
    orbit_radius_live = jnp.sqrt(dx * dx + dy * dy)
    orbit_phase_live = jnp.arctan2(dy, dx)
    is_orbiting_live = (orbit_radius_live + radius) < jnp.float32(constants.ORBIT_RADIUS_LIMIT)

    # Sample one angular velocity for the whole episode.
    angular_velocity = jax.random.uniform(
        rng_omega, (),
        minval=jnp.float32(constants.ORBIT_OMEGA_MIN),
        maxval=jnp.float32(constants.ORBIT_OMEGA_MAX),
    )

    # Permute slot ids so player 0 home isn't always at slot 0 / player 1 at slot 3.
    # This is the cheapest way to close the env-vs-Kaggle gap: real Kaggle places
    # planet ids in arbitrary order each match, so the policy must use features
    # not slot indices. We only permute the live slots, not the padding.
    # Orbit fields are permuted alongside their planet so the (radius, phase,
    # is_orbiting) triple stays tied to its planet.
    if shuffle_slots:
        perm = jax.random.permutation(rng_perm, num_groups * 4)
        xy = xy[perm]
        productions = productions[perm]
        radius = radius[perm]
        ships = ships[perm]
        owners = owners[perm]
        orbit_radius_live = orbit_radius_live[perm]
        orbit_phase_live = orbit_phase_live[perm]
        is_orbiting_live = is_orbiting_live[perm]
        # Map old-slot home idxs to their new positions. ``perm[i] = j`` means
        # the planet that was at slot j now lives at slot i; we want the
        # inverse: where did old slot 0 / slot 3 end up?
        inv_perm = jnp.argsort(perm)
        home_idx_pre = inv_perm[home_idx_pre]

    pad = constants.MAX_PLANETS - num_groups * 4
    planet_x = jnp.concatenate([xy[:, 0], jnp.zeros((pad,), dtype=jnp.float32)])
    planet_y = jnp.concatenate([xy[:, 1], jnp.zeros((pad,), dtype=jnp.float32)])
    planet_radius = jnp.concatenate([radius, jnp.zeros((pad,), dtype=jnp.float32)])
    planet_ships = jnp.concatenate([ships, jnp.zeros((pad,), dtype=jnp.int32)])
    planet_prod = jnp.concatenate([productions, jnp.zeros((pad,), dtype=jnp.int32)])
    planet_owner = jnp.concatenate([owners, jnp.full((pad,), -2, dtype=jnp.int8)])
    planet_mask = jnp.concatenate([
        jnp.ones((num_groups * 4,), dtype=jnp.bool_),
        jnp.zeros((pad,), dtype=jnp.bool_),
    ])
    planet_orbit_radius = jnp.concatenate([
        orbit_radius_live, jnp.zeros((pad,), dtype=jnp.float32),
    ])
    planet_orbit_phase = jnp.concatenate([
        orbit_phase_live, jnp.zeros((pad,), dtype=jnp.float32),
    ])
    planet_is_orbiting = jnp.concatenate([
        is_orbiting_live, jnp.zeros((pad,), dtype=jnp.bool_),
    ])

    return EnvState(
        planet_owner=planet_owner,
        planet_x=planet_x,
        planet_y=planet_y,
        planet_radius=planet_radius,
        planet_ships=planet_ships,
        planet_prod=planet_prod,
        planet_mask=planet_mask,
        fleet_owner=jnp.full((constants.MAX_FLEETS,), -2, dtype=jnp.int8),
        fleet_x=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.float32),
        fleet_y=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.float32),
        fleet_angle=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.float32),
        fleet_ships=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.int32),
        fleet_mask=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.bool_),
        step=jnp.int32(0),
        done=jnp.bool_(False),
        rng=rng_state.astype(jnp.uint32),
        angular_velocity=angular_velocity,
        planet_orbit_radius=planet_orbit_radius,
        planet_orbit_phase=planet_orbit_phase,
        planet_is_orbiting=planet_is_orbiting,
        home_planet_idx=home_idx_pre,
    )
