"""Hand-crafted round-trip tests for action_inverse.

These are the *foundational* tests for BC: if action_inverse is buggy then
all downstream BC training is poisoned. Test cases derived from a synthetic
4-planet setup we control fully.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.bc.action_inverse import kaggle_moves_to_targets
from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


def _make_state(
    planets: list[tuple[int, float, float, int]],
) -> EnvState:
    """Build a minimal EnvState from a small list of planets.

    planets : [(owner, x, y, ships), ...]
    """
    P = constants.MAX_PLANETS
    F = constants.MAX_FLEETS
    owner = np.full((P,), constants.PADDING_OWNER, dtype=np.int8)
    x = np.zeros((P,), dtype=np.float32)
    y = np.zeros((P,), dtype=np.float32)
    rad = np.zeros((P,), dtype=np.float32)
    ships = np.zeros((P,), dtype=np.int32)
    prod = np.zeros((P,), dtype=np.int32)
    mask = np.zeros((P,), dtype=np.bool_)

    for i, (o, xi, yi, sh) in enumerate(planets):
        owner[i] = o
        x[i] = xi
        y[i] = yi
        rad[i] = 1.0
        ships[i] = sh
        prod[i] = 1
        mask[i] = True

    return EnvState(
        planet_owner=jnp.asarray(owner),
        planet_x=jnp.asarray(x),
        planet_y=jnp.asarray(y),
        planet_radius=jnp.asarray(rad),
        planet_ships=jnp.asarray(ships),
        planet_prod=jnp.asarray(prod),
        planet_mask=jnp.asarray(mask),
        fleet_owner=jnp.full((F,), constants.PADDING_OWNER, dtype=jnp.int8),
        fleet_x=jnp.zeros((F,), dtype=jnp.float32),
        fleet_y=jnp.zeros((F,), dtype=jnp.float32),
        fleet_angle=jnp.zeros((F,), dtype=jnp.float32),
        fleet_ships=jnp.zeros((F,), dtype=jnp.int32),
        fleet_mask=jnp.zeros((F,), dtype=jnp.bool_),
        step=jnp.int32(0),
        done=jnp.bool_(False),
        rng=jnp.zeros((2,), dtype=jnp.uint32),
        angular_velocity=jnp.float32(0.0),
        planet_orbit_radius=jnp.zeros((P,), dtype=jnp.float32),
        planet_orbit_phase=jnp.zeros((P,), dtype=jnp.float32),
        planet_is_orbiting=jnp.zeros((P,), dtype=jnp.bool_),
    )


def test_single_move_roundtrip():
    """v20 emits one move 0 -> 1, half ships; verify src/dst/pct/emit/loss."""
    state = _make_state([
        (0, 10.0, 10.0, 20),   # planet 0: my, 20 ships
        (1, 80.0, 50.0, 30),   # planet 1: enemy, east of 0
        (1, 10.0, 90.0, 15),   # planet 2: enemy, north of 0
        (-1, 50.0, 50.0, 5),   # planet 3: neutral, NE of 0
    ])
    # angle from p0 to p1: atan2(50-10, 80-10) = atan2(40, 70)
    angle_01 = math.atan2(40.0, 70.0)
    moves = [[0, angle_01, 10]]  # send 10 ships from 0 to 1 (50%)

    t = kaggle_moves_to_targets(state, player=0, moves=moves)

    assert t["src"][0] == 0
    assert t["dst"][0] == 1, f"expected dst=1, got {t['dst'][0]}"
    # pct: 10/20 = 0.5, closest of (.10,.20,.30,.40,.55,.70,.85,1.00) is .55 (idx 4)
    assert t["pct"][0] == 4, f"expected pct_bin=4, got {t['pct'][0]}"
    assert bool(t["emit"][0]) is True
    assert bool(t["loss_mask"][0]) is True
    # step 1: stop supervision
    assert bool(t["loss_mask"][1]) is True
    assert bool(t["emit"][1]) is False
    # step 2..7: no supervision
    for s in range(2, 8):
        assert bool(t["loss_mask"][s]) is False, f"step {s} should not have loss"


def test_two_moves_same_src_reserved():
    """v20 emits two moves from the same planet; second pct must account for reserved."""
    state = _make_state([
        (0, 10.0, 10.0, 100),  # planet 0: my, 100 ships
        (1, 80.0, 10.0, 50),   # planet 1: east enemy
        (1, 10.0, 80.0, 50),   # planet 2: north enemy
    ])
    # send 30 to p1, then 30 to p2 (both from p0)
    # first move: 30/100 = 0.30 -> bin 2 -> decoded floor(100*.30)=30. reserved=30.
    # second move: 30/(100-30)=30/70 = 0.4286 -> closest bin .40 (idx 3) -> floor(70*.40)=28.
    # action_inverse currently snaps to 0.40, that's the *desired* pct target.
    a01 = math.atan2(0.0, 70.0)
    a02 = math.atan2(70.0, 0.0)
    moves = [
        [0, a01, 30],
        [0, a02, 30],
    ]
    t = kaggle_moves_to_targets(state, player=0, moves=moves)

    assert t["src"][0] == 0 and t["dst"][0] == 1
    assert t["pct"][0] == 2, f"expected pct[0]=2 (.30), got {t['pct'][0]}"
    assert t["src"][1] == 0 and t["dst"][1] == 2
    assert t["pct"][1] == 3, f"expected pct[1]=3 (.40), got {t['pct'][1]}"
    assert bool(t["emit"][0]) and bool(t["emit"][1])
    assert bool(t["loss_mask"][2]) is True
    assert bool(t["emit"][2]) is False


def test_empty_moves():
    """v20 returns []; verify all emit=False, loss only at step 0."""
    state = _make_state([
        (0, 10.0, 10.0, 5),
        (1, 80.0, 50.0, 5),
    ])
    t = kaggle_moves_to_targets(state, player=0, moves=[])
    assert not bool(t["emit"][0])
    assert bool(t["loss_mask"][0]) is True
    for s in range(1, 8):
        assert bool(t["loss_mask"][s]) is False


def test_invalid_src_skipped():
    """Move with src belonging to enemy is treated as no-emit."""
    state = _make_state([
        (0, 10.0, 10.0, 20),
        (1, 80.0, 50.0, 30),
    ])
    a = math.atan2(0.0, 1.0)
    # src=1 belongs to enemy -- should be silently dropped
    moves = [[1, a, 10]]
    t = kaggle_moves_to_targets(state, player=0, moves=moves)
    # the move was dropped, so loss_mask[0]=True with emit=False (stop)
    assert not bool(t["emit"][0])


def test_overflow_capped_at_K():
    """v20 emits more than K moves; we keep first K only and stop supervision."""
    state = _make_state([(0, 50.0, 50.0, 1000)] + [
        (1, 10.0 * (i + 1), 90.0, 5) for i in range(9)
    ])
    angles = [math.atan2(40.0, 10.0 * (i + 1) - 50.0) for i in range(9)]
    moves = [[0, angles[i], 5] for i in range(9)]  # 9 moves, K=8
    t = kaggle_moves_to_targets(state, player=0, moves=moves)
    assert sum(int(b) for b in t["emit"]) == 8
    # no stop step because we filled all K
    assert sum(int(b) for b in t["loss_mask"]) == 8


if __name__ == "__main__":
    test_single_move_roundtrip()
    test_two_moves_same_src_reserved()
    test_empty_moves()
    test_invalid_src_skipped()
    test_overflow_capped_at_K()
    print("[OK] all action_inverse tests pass")
