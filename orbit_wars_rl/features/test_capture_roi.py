"""Unit tests for capture_roi_norm (planet dim 32).

Run: ``python -m orbit_wars_rl.features.test_capture_roi``
"""

from __future__ import annotations

import jax.numpy as jnp

from orbit_wars_rl.env import constants, state
from orbit_wars_rl.features.encode import encode


def _factory_vs_weak_fixture():
    """Nearby high-prod factory beats nearby weak mop (opposite of old v26)."""
    s = state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)
    s = s.replace(
        planet_mask=s.planet_mask.at[0].set(True).at[1].set(True).at[2].set(True),
        planet_owner=s.planet_owner.at[0].set(0).at[1].set(constants.NEUTRAL_OWNER).at[2].set(
            constants.NEUTRAL_OWNER
        ),
        planet_x=s.planet_x.at[0].set(0.0).at[1].set(5.0).at[2].set(6.0),
        planet_y=s.planet_y.at[0].set(0.0).at[1].set(0.0).at[2].set(0.0),
        planet_ships=s.planet_ships.at[0].set(100).at[1].set(2).at[2].set(18),
        planet_prod=s.planet_prod.at[0].set(5).at[1].set(1).at[2].set(4),
        planet_radius=s.planet_radius.at[0].set(3.0).at[1].set(3.0).at[2].set(3.0),
    )
    return s


def main() -> int:
    s = _factory_vs_weak_fixture()
    obs = encode(s, player=0, episode_steps=500)

    assert obs.planet_feats.shape == (constants.MAX_PLANETS, constants.PLANET_FEAT_DIM)
    assert obs.global_feats.shape[0] >= 27

    roi_weak = float(obs.planet_feats[1, 32])
    roi_factory = float(obs.planet_feats[2, 32])
    print(f"[check] capture_roi_norm weak={roi_weak:.4f} factory={roi_factory:.4f}")
    if roi_factory <= roi_weak:
        raise AssertionError(
            f"nearby factory should beat weak mop: {roi_factory} <= {roi_weak}"
        )
    print("[OK ] nearby prod4/g18 beats weak prod1/g2 on dim32")

    local_roi = float(obs.global_feats[15])
    print(f"[check] local_roi_targets_norm (dim15) = {local_roi:.4f}")
    if local_roi <= 0.0:
        raise AssertionError(f"expected positive local_roi_targets_norm, got {local_roi}")
    print("[OK ] global dim15 counts high-ROI capturable targets")

    print("\n[ALL PASS] capture_roi feature tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
