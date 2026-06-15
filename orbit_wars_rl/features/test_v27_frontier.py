"""Unit tests for v27 frontier / anti-shuffle features (dims 26, 30, 41-42).

Run: ``python -m orbit_wars_rl.features.test_v27_frontier``
"""

from __future__ import annotations

import jax.numpy as jnp

from orbit_wars_rl.env import constants, state
from orbit_wars_rl.features.encode import encode
from orbit_wars_rl.features.frontier_util import (
    capture_need_exact_norm,
    frontier_score_per_planet,
    interior_planet_bin,
    shuffle_dst_risk_norm,
)


def _two_planet_state(*, x0: float, y0: float, x1: float, y1: float, owner1: int):
    s = state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)
    s = s.replace(
        planet_mask=s.planet_mask.at[0].set(True).at[1].set(True),
        planet_owner=s.planet_owner.at[0].set(0).at[1].set(owner1),
        planet_ships=s.planet_ships.at[0].set(80).at[1].set(10),
        planet_prod=s.planet_prod.at[0].set(5).at[1].set(4),
        planet_x=s.planet_x.at[0].set(x0).at[1].set(x1),
        planet_y=s.planet_y.at[0].set(y0).at[1].set(y1),
        planet_radius=s.planet_radius.at[0].set(5).at[1].set(5),
    )
    return s


def test_frontier_adjacent_neutral() -> None:
    s = _two_planet_state(x0=100.0, y0=100.0, x1=150.0, y1=100.0, owner1=-1)
    frontier = frontier_score_per_planet(s, player=0)
    interior = interior_planet_bin(s, 0, frontier)
    assert float(frontier[0]) > 0.4, f"owned border planet should be frontier, got {frontier[0]}"
    assert float(interior[0]) < 0.5
    assert float(frontier[1]) > 0.0
    print(f"[check] frontier owned={float(frontier[0]):.3f} neutral={float(frontier[1]):.3f}")


def test_shuffle_risk_on_safe_owned() -> None:
    s = _two_planet_state(x0=100.0, y0=100.0, x1=500.0, y1=500.0, owner1=0)
    s = s.replace(
        fleet_mask=s.fleet_mask.at[0].set(True),
        fleet_owner=s.fleet_owner.at[0].set(0),
        fleet_ships=s.fleet_ships.at[0].set(20),
        fleet_x=s.fleet_x.at[0].set(200.0),
        fleet_y=s.fleet_y.at[0].set(500.0),
        fleet_angle=s.fleet_angle.at[0].set(0.0),
    )
    threat = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    friendly_w2 = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    friendly_w2 = friendly_w2.at[1].set(15.0)
    risk = shuffle_dst_risk_norm(s, player=0, threat_ratio=threat, friendly_inbound=friendly_w2)
    assert float(risk[1]) > 0.5, "interior owned with inbound should flag shuffle risk"
    print(f"[check] shuffle_dst_risk interior={float(risk[1]):.3f}")


def test_capture_need_exact() -> None:
    s = _two_planet_state(x0=100.0, y0=100.0, x1=150.0, y1=100.0, owner1=-1)
    need = jnp.full((constants.MAX_PLANETS,), 12.0, dtype=jnp.float32)
    inbound = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    inbound = inbound.at[1].set(8.0)
    exact = capture_need_exact_norm(s, player=0, need=need, friendly_inbound=inbound)
    assert 0.2 < float(exact[1]) < 0.5, f"remaining need fraction expected ~0.33, got {exact[1]}"
    print(f"[check] capture_need_exact neutral={float(exact[1]):.3f}")


def test_encode_dim_slots() -> None:
    s = _two_planet_state(x0=100.0, y0=100.0, x1=150.0, y1=100.0, owner1=-1)
    obs = encode(s, player=0, episode_steps=500)
    pf = obs.planet_feats
    assert pf.shape == (constants.MAX_PLANETS, constants.PLANET_FEAT_DIM)
    assert float(pf[0, 26]) > 0.0, "dim26 frontier_score on border owned"
    assert float(pf[1, 30]) >= 0.0, "dim30 capture_need_exact on neutral"
    assert obs.global_feats.shape[0] >= 427
    assert 0.0 <= float(obs.global_feats[17]) <= 1.0, "global dim17 frontier_owned_norm"
    print("[check] encode v27 dim slots OK")


def main() -> int:
    test_frontier_adjacent_neutral()
    test_shuffle_risk_on_safe_owned()
    test_capture_need_exact()
    test_encode_dim_slots()
    print("\n[ALL PASS] v27 frontier feature tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
