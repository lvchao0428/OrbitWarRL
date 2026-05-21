"""Map generation. MVP: static planets, 4-fold symmetric, 2P opposing.

Returns a fresh ``EnvState`` with planets filling the first ``4*num_groups``
slots and the rest padded out.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


def _symmetric_group_offsets(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Given an anchor (x, y) in quadrant 1, build 4 mirrored positions."""
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


def reset(rng: jnp.ndarray, num_groups: int = 5) -> EnvState:
    """Generate a 2P map with ``num_groups`` symmetric planet groups.

    ``num_groups`` is a Python int (static); the *positions* within each group
    are randomized via ``rng``. Slots beyond ``num_groups*4`` are masked off.
    """
    assert constants.MIN_PLANET_GROUPS <= num_groups <= constants.MAX_PLANET_GROUPS
    assert num_groups * constants.PLANETS_PER_GROUP <= constants.MAX_PLANETS

    rng_pos, rng_prod, rng_ships, rng_state = jax.random.split(rng, 4)

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
    )
