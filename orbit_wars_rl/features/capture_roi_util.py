"""Shared capture ROI economics (planet dim32 + pair + flip + reward).

Calibration target: v27 u3999 seed=0 — nearby high-prod neutral id=20
(prod=3, g=20) must rank above distant id=22 and weak id=16.

v30: single payback+eta formula; pair decisions use src→dst ETA; flip
feasibility uses max pct bin (1.0), not fixed 0.7.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.env import constants

# Max launch fraction (highest pct bin) for flip feasibility estimates.
MAX_LAUNCH_PCT: float = float(max(constants.PCT_BIN_VALUES))
NEED_PAD_NEUTRAL: float = 2.0
NEED_PAD_ENEMY: float = 8.0
ROI_EXEMPT_THRESHOLD: float = 0.12  # high-ROI dst exempt from flip hard-block
ROI_NORM: float = 0.35  # raw roi scale before clip (id=20 ~0.34 @ seed=0)
ROI_READY_BETA: float = 3.0  # boost roi_teacher when capture window opens
EMIT_URGENCY_THRESH: float = 0.35  # emit_worth_it via home capture urgency
EMIT_URGENCY_FORCE: float = 0.45  # legacy; v30d gates force via emit_worth_it only
OPENING_FACTORY_IDX: int = 20  # seed=0 opening factory slot (prod=3, g=20)


def need_est_from_garr(
    garr: jnp.ndarray | np.ndarray,
    *,
    is_neutral: jnp.ndarray | np.ndarray | None = None,
    pad_neutral: float = NEED_PAD_NEUTRAL,
    pad_enemy: float = NEED_PAD_ENEMY,
) -> jnp.ndarray:
    """Estimated ships to capture: garrison + padding."""
    g = garr.astype(jnp.float32)
    if is_neutral is None:
        pad = jnp.float32(pad_neutral)
    else:
        pad = jnp.where(
            is_neutral.astype(jnp.bool_),
            jnp.float32(pad_neutral),
            jnp.float32(pad_enemy),
        )
    return jnp.maximum(g + pad, jnp.float32(1.0))


def min_dist_to_owned(
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
    is_mine: jnp.ndarray,
) -> jnp.ndarray:
    """Per-planet distance to nearest owned planet (inf if none)."""
    my_x = jnp.where(is_mine, planet_x, jnp.float32(1e6))
    my_y = jnp.where(is_mine, planet_y, jnp.float32(1e6))
    dx = planet_x[:, None] - my_x[None, :]
    dy = planet_y[:, None] - my_y[None, :]
    return jnp.sqrt(dx * dx + dy * dy).min(axis=-1)


def capture_roi_raw(
    prod_f: jnp.ndarray,
    need_est: jnp.ndarray,
    dist_or_eta: jnp.ndarray,
    *,
    fleet_speed: float | jnp.ndarray | None = None,
    norm: float | jnp.ndarray = ROI_NORM,
    use_eta_directly: bool = False,
) -> jnp.ndarray:
    """prod / (payback + eta); higher = better capture.

    ``dist_or_eta``: src→dst distance, or precomputed eta if
    ``use_eta_directly=True``.
    """
    spd = jnp.float32(fleet_speed or constants.DEFAULT_MAX_SHIP_SPEED)
    eta = dist_or_eta if use_eta_directly else dist_or_eta / spd
    payback = need_est / jnp.maximum(prod_f, jnp.float32(1.0))
    roi = prod_f / jnp.maximum(payback + eta, jnp.float32(1.0))
    return roi / jnp.float32(norm)


def capture_roi_from_src(
    prod_f: jnp.ndarray,
    need_est: jnp.ndarray,
    src_dst_dist: jnp.ndarray,
    *,
    fleet_speed: float | jnp.ndarray | None = None,
    norm: float | jnp.ndarray = ROI_NORM,
) -> jnp.ndarray:
    """Pair-level ROI: eta = src→dst distance / speed."""
    return capture_roi_raw(
        prod_f, need_est, src_dst_dist, fleet_speed=fleet_speed, norm=norm,
    )


def capture_roi_planet_level(
    prod_f: jnp.ndarray,
    need_est: jnp.ndarray,
    min_dist_to_mine: jnp.ndarray,
    *,
    fleet_speed: float | jnp.ndarray | None = None,
    norm: float | jnp.ndarray = ROI_NORM,
) -> jnp.ndarray:
    """Planet dim32: eta = distance to nearest owned planet."""
    return capture_roi_raw(
        prod_f,
        need_est,
        min_dist_to_mine,
        fleet_speed=fleet_speed,
        norm=norm,
    )


def ships_at_max_pct(remaining: jnp.ndarray) -> jnp.ndarray:
    """Ships sendable at max pct bin from ``remaining`` garrison."""
    rem = jnp.maximum(remaining.astype(jnp.float32), jnp.float32(1.0))
    return jnp.floor(rem * jnp.float32(MAX_LAUNCH_PCT))


def flip_margin_at_max_pct(
    remaining: jnp.ndarray,
    garr_dst: jnp.ndarray,
) -> jnp.ndarray:
    """Flip margin using max pct: floor(rem*1.0) - garr (broadcasts)."""
    ships = ships_at_max_pct(remaining)
    garr = garr_dst.astype(jnp.float32)
    if ships.ndim < garr.ndim:
        ships = ships[..., None]
    return ships - garr


def flip_ok_at_max_pct(
    remaining: jnp.ndarray,
    garr_dst: jnp.ndarray,
) -> jnp.ndarray:
    """True iff max-pct send strictly exceeds dst garrison."""
    return flip_margin_at_max_pct(remaining, garr_dst) > jnp.float32(0.0)


def capture_ready_from_slack(slack: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Spike when max-pct send is within capture window of dst garrison.

    slack = floor(rem*max_pct) - garr; [0,2]->1.0, [3,5]->0.5, else 0.
    """
    s = slack.astype(jnp.float32) if hasattr(slack, "astype") else slack
    return jnp.where(
        s < jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.where(
            s <= jnp.float32(2.0),
            jnp.float32(1.0),
            jnp.where(s <= jnp.float32(5.0), jnp.float32(0.5), jnp.float32(0.0)),
        ),
    )


def pair_roi_effective(
    roi: jnp.ndarray,
    capture_ready: jnp.ndarray,
    *,
    beta: float | jnp.ndarray = ROI_READY_BETA,
) -> jnp.ndarray:
    """ROI boosted when capture window is open (teacher + emit urgency)."""
    return roi * (jnp.float32(1.0) + jnp.float32(beta) * capture_ready)
