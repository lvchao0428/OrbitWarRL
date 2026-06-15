"""v29 pair ROI + same-turn dst dedup tests.

Run: ``python -m orbit_wars_rl.features.test_v29_pair``
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.features.pair import (
    DST_PAIR_DIM,
    dst_pair_features,
    mark_dst_used,
    mask_used_dst_logits,
)
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate

_REPLAY = Path("logs/replay_html/v27_u3999_s0/replay.json")
_HOME_ID = 12
_FACTORY_ID = 20
_FAR_ID = 22
_WEAK_ID = 16


def _planet_slot(obs: dict, planet_id: int) -> int:
    for i, p in enumerate(obs["planets"]):
        if int(p[0]) == planet_id:
            return i
    raise KeyError(planet_id)


def _check_pair_roi_ranking() -> None:
    if not _REPLAY.is_file():
        print(f"[SKIP] pair ROI ranking — missing {_REPLAY}")
        return

    with open(_REPLAY) as f:
        replay = json.load(f)

    for t in (0, 14):
        obs = dict(replay["steps"][t][0]["observation"])
        obs["player"] = 0
        state = kaggle_obs_to_envstate(obs)
        src_idx = _planet_slot(obs, _HOME_ID)
        P = int(state.planet_mask.shape[0])
        target = state.planet_mask & jnp.logical_not(state.planet_owner == 0)
        remaining = state.planet_ships.astype(jnp.int32)

        pair_feats, _ = dst_pair_features(
            state.planet_x,
            state.planet_y,
            state.planet_ships,
            state.planet_mask,
            target,
            remaining,
            jnp.int32(src_idx),
            planet_orbit_phase=state.planet_orbit_phase,
            planet_orbit_radius=state.planet_orbit_radius,
            planet_is_orbiting=state.planet_is_orbiting,
            angular_velocity=state.angular_velocity,
            planet_prod=state.planet_prod.astype(jnp.float32),
        )
        assert pair_feats.shape == (P, DST_PAIR_DIM), pair_feats.shape

        i20 = _planet_slot(obs, _FACTORY_ID)
        i22 = _planet_slot(obs, _FAR_ID)
        i16 = _planet_slot(obs, _WEAK_ID)
        roi20 = float(pair_feats[i20, 5])
        roi22 = float(pair_feats[i22, 5])
        roi16 = float(pair_feats[i16, 5])
        print(
            f"[t={t:2d}] pair_roi home->{_FACTORY_ID}={roi20:.4f} "
            f"->{_FAR_ID}={roi22:.4f} ->{_WEAK_ID}={roi16:.4f}"
        )
        if roi20 <= roi22:
            raise AssertionError(f"t={t}: pair_roi id20 must beat id22: {roi20} <= {roi22}")
        if roi20 <= roi16:
            raise AssertionError(f"t={t}: pair_roi id20 must beat id16: {roi20} <= {roi16}")

    print("[OK ] pair_roi_norm ranks id=20 above id=22/id=16 from home")


def _check_dedup_mask() -> None:
    P = 8
    logits = jnp.zeros((P,), dtype=jnp.float32)
    logits = logits.at[5].set(10.0)
    used = jnp.zeros((P,), dtype=jnp.bool_)
    used = mark_dst_used(used, jnp.int32(5), jnp.bool_(True))
    pmask = jnp.ones((P,), dtype=jnp.bool_)
    masked = mask_used_dst_logits(logits, used, pmask, jnp.int32(2))
    assert float(masked[5]) < -1e8, f"dst=5 should be blocked, got {masked[5]}"
    assert int(jnp.argmax(masked)) != 5
    print("[OK ] used_dst blocks previously chosen dst in logits")


def main() -> int:
    _check_pair_roi_ranking()
    _check_dedup_mask()
    print("\n[ALL PASS] test_v29_pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
