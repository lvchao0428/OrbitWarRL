"""v30 economics head + unified ROI + flip max-pct tests.

Run: ``python -m orbit_wars_rl.features.test_v30_econ``
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.features.capture_roi_util import ROI_EXEMPT_THRESHOLD, ROI_NORM
from orbit_wars_rl.features.pair import (
    DST_PAIR_DIM,
    ECON_DST_DIM,
    EMIT_PAIR_DIM,
    dst_flip_block_mask,
    dst_pair_features,
    econ_dst_features,
    emit_pair_globals,
    roi_teacher_dst,
)
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate

_REPLAY = Path("logs/replay_html/v27_u3999_s0/replay.json")
_HOME_ID = 12
_FACTORY_ID = 20
_FAR_ID = 22
_WEAK_ID = 16


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
        src_idx = _HOME_ID
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
            planet_prod=state.planet_prod.astype(jnp.float32),
        )
        i20, i22, i16 = _FACTORY_ID, _FAR_ID, _WEAK_ID
        roi20 = float(pair_feats[i20, 5])
        roi22 = float(pair_feats[i22, 5])
        roi16 = float(pair_feats[i16, 5])
        print(
            f"[t={t:2d}] pair_roi home->{_FACTORY_ID}={roi20:.4f} "
            f"->{_FAR_ID}={roi22:.4f} ->{_WEAK_ID}={roi16:.4f} norm={ROI_NORM}"
        )
        if roi20 <= roi22:
            raise AssertionError(f"t={t}: pair_roi id20 must beat id22: {roi20} <= {roi22}")
        if roi20 <= roi16:
            raise AssertionError(f"t={t}: pair_roi id20 must beat id16: {roi20} <= {roi16}")

    print("[OK ] pair_roi ranks id=20 above id=22/id=16")


def _check_flip_exempt_id20() -> None:
    """At opening, id=20 must not be flip-blocked (high ROI exempt)."""
    if not _REPLAY.is_file():
        print(f"[SKIP] flip exempt — missing {_REPLAY}")
        return

    with open(_REPLAY) as f:
        replay = json.load(f)

    obs = dict(replay["steps"][0][0]["observation"])
    obs["player"] = 0
    state = kaggle_obs_to_envstate(obs)
    src_idx = _HOME_ID
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
        planet_prod=state.planet_prod.astype(jnp.float32),
    )
    block = dst_flip_block_mask(
        state.planet_ships,
        state.planet_mask,
        target,
        remaining,
        jnp.int32(src_idx),
        pair_roi_norm=pair_feats[..., 5],
    )
    roi20 = float(pair_feats[_FACTORY_ID, 5])
    blocked20 = bool(block[_FACTORY_ID])
    print(f"[flip] id20 roi={roi20:.4f} exempt_thr={ROI_EXEMPT_THRESHOLD} blocked={blocked20}")
    if blocked20:
        raise AssertionError("id=20 must not be flip-blocked at opening (ROI exempt)")
    teacher = int(roi_teacher_dst(pair_feats, state.planet_mask, target, jnp.int32(src_idx)))
    print(f"[flip] roi_teacher dst={teacher} (expect {_FACTORY_ID})")
    if teacher != _FACTORY_ID:
        raise AssertionError(f"roi teacher should be id={_FACTORY_ID}, got {teacher}")
    print("[OK ] flip exempt + roi teacher id=20 @ t=0")


def _check_emit_gate_t0_t12() -> None:
    """v30d: no emit_worth at t=0; fires at t=12 when home>=need(20)."""
    if not _REPLAY.is_file():
        print(f"[SKIP] emit gate — missing {_REPLAY}")
        return

    with open(_REPLAY) as f:
        replay = json.load(f)

    for t, expect in ((0, 0.0), (12, 1.0)):
        obs = dict(replay["steps"][t][0]["observation"])
        obs["player"] = 0
        state = kaggle_obs_to_envstate(obs)
        tgt = state.planet_mask & jnp.logical_not(state.planet_owner == 0)
        rem = state.planet_ships.astype(jnp.int32)
        my = state.planet_owner == 0
        prod = state.planet_prod.astype(jnp.float32)
        hi = jnp.int32(_HOME_ID)
        emit_g = emit_pair_globals(
            state.planet_x, state.planet_y, state.planet_ships,
            state.planet_mask, my, tgt, rem,
            hi, jnp.float32(rem[_HOME_ID]), jnp.float32((rem * my).sum()),
            planet_prod=prod,
        )
        worth = float(emit_g[0])
        urgency = float(emit_g[11])
        home_g = int(rem[_HOME_ID])
        print(f"[t={t:2d}] home_g={home_g} emit_worth_it={worth:.1f} urgency={urgency:.3f}")
        if expect < 0.5 and worth > 0.5:
            raise AssertionError(f"t={t}: emit_worth_it should be 0, got {worth}")
        if expect > 0.5 and worth < 0.5:
            raise AssertionError(f"t={t}: emit_worth_it should be 1, got {worth}")
    print("[OK ] v30d opening gate t=0 off, t=12 on")


def _check_capture_ready_t12() -> None:
    """At t=12 home margin=0: capture_ready spikes and emit_worth_it fires."""
    if not _REPLAY.is_file():
        print(f"[SKIP] capture_ready t=12 — missing {_REPLAY}")
        return

    with open(_REPLAY) as f:
        replay = json.load(f)

    obs = dict(replay["steps"][12][0]["observation"])
    obs["player"] = 0
    state = kaggle_obs_to_envstate(obs)
    home_idx = _HOME_ID
    target = state.planet_mask & jnp.logical_not(state.planet_owner == 0)
    remaining = state.planet_ships.astype(jnp.int32)
    prod = state.planet_prod.astype(jnp.float32)

    pair_feats, _ = dst_pair_features(
        state.planet_x, state.planet_y, state.planet_ships,
        state.planet_mask, target, remaining, jnp.int32(home_idx),
        planet_prod=prod,
    )
    ready20 = float(pair_feats[_FACTORY_ID, 6])
    print(f"[t=12] capture_ready home->{_FACTORY_ID}={ready20:.2f} (expect 1.0)")
    if ready20 < 0.9:
        raise AssertionError(f"t=12: capture_ready id20 should be ~1.0, got {ready20}")

    my_mask = state.planet_owner == 0
    home_init = float(remaining[home_idx])
    total_init = float((remaining * my_mask).sum())
    emit_g = emit_pair_globals(
        state.planet_x, state.planet_y, state.planet_ships,
        state.planet_mask, my_mask, target, remaining,
        jnp.int32(home_idx), jnp.float32(home_init), jnp.float32(total_init),
        planet_prod=prod,
    )
    worth = float(emit_g[0])
    urgency = float(emit_g[11])
    print(f"[t=12] emit_worth_it={worth:.1f} emit_urgency={urgency:.3f}")
    if worth < 0.5:
        raise AssertionError(f"t=12: emit_worth_it should be 1.0, got {worth}")
    if urgency < 0.3:
        raise AssertionError(f"t=12: emit_urgency too low: {urgency}")
    print("[OK ] capture_ready + emit urgency @ t=12")


def _check_econ_dim() -> None:
    assert DST_PAIR_DIM == 7
    assert EMIT_PAIR_DIM == 12
    assert ECON_DST_DIM == 8
    x = jnp.zeros((40, 7))
    prod = jnp.ones((40,))
    tgt = jnp.ones((40,), dtype=jnp.bool_)
    out = econ_dst_features(x, prod, jnp.ones((40,), dtype=jnp.bool_), tgt)
    assert out.shape == (40, ECON_DST_DIM), out.shape
    print(f"[OK ] ECON_DST_DIM={ECON_DST_DIM}")


def main() -> int:
    _check_econ_dim()
    _check_pair_roi_ranking()
    _check_flip_exempt_id20()
    _check_emit_gate_t0_t12()
    _check_capture_ready_t12()
    print("\n[ALL PASS] test_v30_econ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
