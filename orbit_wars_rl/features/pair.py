"""K-step pair / per-step feature helpers (f26 / f29).

These are computed *inside* the autoregressive head loop, not at turn start,
because they depend on:

  * the running ``reserved`` book-keeping buffer (which grows each step)
  * the current ``src_t`` / ``dst_t`` (for per-pair signals)

They feed the heads as extra scalar/vector kwargs and are intentionally kept
out of ``features/encode.py`` (whose output is per-turn-static).

Provided:

  ``dst_pair_features(planet_x, planet_y, planet_ships, planet_mask,
                       is_target_mask, remaining, src_idx)``
      -> (pair_feats: (P, 5), sun_block_mask: (P,) bool)  [f29: +pair_margin_norm]
      Per-candidate-dst features given current source. ``sun_block_mask`` is
      True for dst whose src->dst segment crosses the sun. Used both as a
      head input (sun_risk float) and as a hard logit mask in DstHead.

  ``emit_pair_globals(planet_x, planet_y, planet_ships, planet_mask,
                       my_mask, target_mask, remaining,
                       home_remaining, home_init,
                       total_remaining, total_init)``
      -> (g: (4,)) per-step global scalar that summarises feasibility / budget.

  ``dst_flip_block_mask(planet_ships, planet_mask, is_target_mask,
                       remaining, src_idx)``
      -> (..., P) bool -- True iff dst cannot be flipped with floor(rem*0.7)
      at ``src_idx`` (enemy/neutral only; f31 hard mask).

  ``pct_pair_features(garr_dst, remaining_src)``
      -> (2,) [min_bin_norm, pair_flip_bin5] given chosen (src, dst).
      ``min_bin_norm`` is the smallest pct-bin index that sends enough ships
      to flip ``garr_dst``, divided by (NUM_PCT_BINS-1).  Replaces the f26
      ``(garr+1)/remaining`` ratio which correlated with bin0 when remaining
      was large (home planet spam).

All functions are jit-pure (no python branching on traced values).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.features.encode import _predict_planet_pos


_BOARD = jnp.float32(constants.BOARD)
_SUN = jnp.array([constants.SUN_X, constants.SUN_Y], dtype=jnp.float32)
_SUN_R = jnp.float32(constants.SUN_RADIUS)
# Margin: treat fleet path as "hits sun" if min-dist < SUN_R * SUN_PATH_MARGIN.
# constants.SUN_PATH_MARGIN = 1.25 -> ~12.5 unit forbidden corridor radius.
_SUN_GUARD = _SUN_R * jnp.float32(constants.SUN_PATH_MARGIN)
# Hard mask threshold (>= 0.9 means src->dst path nearly tangent to sun guard).
_SUN_BLOCK_THRESH = jnp.float32(0.9)
_PCT_BIN_TABLE = jnp.array(constants.PCT_BIN_VALUES, dtype=jnp.float32)
_NUM_PCT_BINS = constants.NUM_PCT_BINS


def _segment_min_dist_to_point(
    ax: jnp.ndarray, ay: jnp.ndarray,
    bx: jnp.ndarray, by: jnp.ndarray,
    cx: jnp.ndarray, cy: jnp.ndarray,
) -> jnp.ndarray:
    """Minimum distance from point C to line segment AB.

    All inputs broadcast (used with B-shape == (P,), A scalar)."""
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    ab2 = abx * abx + aby * aby + jnp.float32(1e-6)
    t = (acx * abx + acy * aby) / ab2
    t = jnp.clip(t, 0.0, 1.0)
    px = ax + t * abx
    py = ay + t * aby
    dx = cx - px
    dy = cy - py
    return jnp.sqrt(dx * dx + dy * dy + jnp.float32(1e-6))


def dst_pair_features(
    planet_x: jnp.ndarray,        # (P,) float
    planet_y: jnp.ndarray,        # (P,) float
    planet_ships: jnp.ndarray,    # (P,) int -- raw garrison (per dst)
    planet_mask: jnp.ndarray,     # (P,) bool
    is_target_mask: jnp.ndarray,  # (P,) bool -- enemy | neutral
    remaining: jnp.ndarray,       # (P,) int  -- remaining at every src (only src_idx row used)
    src_idx: jnp.ndarray,         # () int
    *,
    planet_orbit_phase: jnp.ndarray | None = None,
    planet_orbit_radius: jnp.ndarray | None = None,
    planet_is_orbiting: jnp.ndarray | None = None,
    angular_velocity: jnp.ndarray | None = None,
    fleet_speed: jnp.ndarray | float | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-candidate-dst features given a chosen src.

    Returns:
      pair_feats: (P, 5) float32 in [0, 1]:
        [0] dist_src_dst / BOARD
        [1] sun_risk = clip(1 - min_dist / SUN_GUARD, 0, 1)  (1 = path through sun)
        [2] ships_needed_norm = (garr[dst]+1) / remaining[src]  clip [0,1]
        [3] pair_flip_bin5 = float( floor(remaining[src]*0.7) > garr[dst] ) AND target
        [4] pair_margin_norm = clip((floor(rem*0.7)-garr)/rem, 0, 1) AND target
      sun_block_mask: (P,) bool -- True iff sun_risk >= 0.9 (hard mask candidate)
    """
    P = planet_x.shape[0]
    src_x = planet_x[src_idx]
    src_y = planet_y[src_idx]
    rem_src = jnp.maximum(remaining[src_idx], jnp.int32(1)).astype(jnp.float32)

    dx = planet_x - src_x
    dy = planet_y - src_y
    dist = jnp.sqrt(dx * dx + dy * dy + jnp.float32(1e-6))

    # v17: ETA-based lead targeting for orbiting destinations.
    if (
        planet_orbit_phase is not None
        and planet_orbit_radius is not None
        and planet_is_orbiting is not None
        and angular_velocity is not None
    ):
        spd = jnp.float32(fleet_speed if fleet_speed is not None else constants.DEFAULT_MAX_SHIP_SPEED)
        eta = dist / jnp.maximum(spd, jnp.float32(0.5))
        rotate_mask_f = planet_is_orbiting.astype(jnp.float32)
        omega_b = angular_velocity
        if omega_b is not None and omega_b.ndim < planet_x.ndim:
            omega_b = omega_b[..., None]
        lead_x, lead_y = _predict_planet_pos(
            planet_orbit_phase,
            planet_orbit_radius,
            omega_b,
            eta,
            planet_x,
            planet_y,
            rotate_mask_f,
        )
        dx_lead = lead_x - src_x
        dy_lead = lead_y - src_y
        dist_lead = jnp.sqrt(dx_lead * dx_lead + dy_lead * dy_lead + jnp.float32(1e-6))
        use_lead = rotate_mask_f > jnp.float32(0.0)
        dist = jnp.where(use_lead, dist_lead, dist)
        planet_x_eff = jnp.where(use_lead, lead_x, planet_x)
        planet_y_eff = jnp.where(use_lead, lead_y, planet_y)
    else:
        planet_x_eff = planet_x
        planet_y_eff = planet_y

    dist_norm = dist / _BOARD

    min_to_sun = _segment_min_dist_to_point(
        src_x, src_y, planet_x_eff, planet_y_eff, _SUN[0], _SUN[1]
    )
    # Self-pair (src==dst) -> degenerate segment; force sun_risk=0 to avoid NaN-like behaviour.
    self_pair = jnp.arange(P) == src_idx
    sun_risk = jnp.clip(1.0 - min_to_sun / _SUN_GUARD, 0.0, 1.0)
    sun_risk = jnp.where(self_pair, jnp.float32(0.0), sun_risk)

    garr_dst = planet_ships.astype(jnp.float32)
    ships_needed = (garr_dst + 1.0) / rem_src
    ships_needed_norm = jnp.clip(ships_needed, 0.0, 1.0)

    ships_at_bin5 = jnp.floor(rem_src * jnp.float32(0.7))
    pair_flip_bin5 = (ships_at_bin5 > garr_dst).astype(jnp.float32) * is_target_mask.astype(
        jnp.float32
    )
    margin = (ships_at_bin5 - garr_dst) / rem_src
    pair_margin_norm = jnp.clip(margin, 0.0, 1.0) * is_target_mask.astype(jnp.float32)

    pair_feats = jnp.stack(
        [dist_norm, sun_risk, ships_needed_norm, pair_flip_bin5, pair_margin_norm], axis=-1
    )
    # Zero rows for padding planets so they cannot leak through the MLP.
    pair_feats = pair_feats * planet_mask[:, None].astype(jnp.float32)

    sun_block_mask = (sun_risk >= _SUN_BLOCK_THRESH) & planet_mask & (~self_pair)
    return pair_feats, sun_block_mask


def dst_pair_features_batched(
    planet_x: jnp.ndarray,        # (..., P)
    planet_y: jnp.ndarray,        # (..., P)
    planet_ships: jnp.ndarray,    # (..., P)
    planet_mask: jnp.ndarray,     # (..., P)
    is_target_mask: jnp.ndarray,  # (..., P)
    remaining: jnp.ndarray,       # (..., P)
    src_idx: jnp.ndarray,         # (...,)
    *,
    planet_orbit_phase: jnp.ndarray | None = None,
    planet_orbit_radius: jnp.ndarray | None = None,
    planet_is_orbiting: jnp.ndarray | None = None,
    angular_velocity: jnp.ndarray | None = None,
    fleet_speed: jnp.ndarray | float | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Same as ``dst_pair_features`` with arbitrary leading batch dims.

    Implements via take_along_axis to gather src coords/ships.
    """
    if planet_x.ndim == 1:
        return dst_pair_features(
            planet_x, planet_y, planet_ships, planet_mask,
            is_target_mask, remaining, src_idx,
            planet_orbit_phase=planet_orbit_phase,
            planet_orbit_radius=planet_orbit_radius,
            planet_is_orbiting=planet_is_orbiting,
            angular_velocity=angular_velocity,
            fleet_speed=fleet_speed,
        )
    P = planet_x.shape[-1]
    src_idx_e = src_idx[..., None]
    src_x = jnp.take_along_axis(planet_x, src_idx_e, axis=-1)
    src_y = jnp.take_along_axis(planet_y, src_idx_e, axis=-1)
    rem_at_src = jnp.take_along_axis(remaining, src_idx_e, axis=-1).astype(jnp.float32)
    rem_at_src = jnp.maximum(rem_at_src, jnp.float32(1.0))

    dx = planet_x - src_x
    dy = planet_y - src_y
    dist = jnp.sqrt(dx * dx + dy * dy + jnp.float32(1e-6))

    if (
        planet_orbit_phase is not None
        and planet_orbit_radius is not None
        and planet_is_orbiting is not None
        and angular_velocity is not None
    ):
        spd = jnp.float32(fleet_speed if fleet_speed is not None else constants.DEFAULT_MAX_SHIP_SPEED)
        eta = dist / jnp.maximum(spd, jnp.float32(0.5))
        rotate_mask_f = planet_is_orbiting.astype(jnp.float32)
        omega_b = angular_velocity
        if omega_b is not None and omega_b.ndim < planet_x.ndim:
            omega_b = omega_b[..., None]
        lead_x, lead_y = _predict_planet_pos(
            planet_orbit_phase,
            planet_orbit_radius,
            omega_b,
            eta,
            planet_x,
            planet_y,
            rotate_mask_f,
        )
        dx_lead = lead_x - src_x
        dy_lead = lead_y - src_y
        dist_lead = jnp.sqrt(dx_lead * dx_lead + dy_lead * dy_lead + jnp.float32(1e-6))
        use_lead = rotate_mask_f > jnp.float32(0.0)
        dist = jnp.where(use_lead, dist_lead, dist)
        planet_x_eff = jnp.where(use_lead, lead_x, planet_x)
        planet_y_eff = jnp.where(use_lead, lead_y, planet_y)
    else:
        planet_x_eff = planet_x
        planet_y_eff = planet_y

    dist_norm = dist / _BOARD

    min_to_sun = _segment_min_dist_to_point(
        src_x, src_y, planet_x_eff, planet_y_eff, _SUN[0], _SUN[1]
    )
    arange = jnp.arange(P)
    # Broadcast self_pair = arange == src_idx along leading dims.
    self_pair = arange[(None,) * (planet_x.ndim - 1) + (slice(None),)] == src_idx[..., None]
    sun_risk = jnp.clip(1.0 - min_to_sun / _SUN_GUARD, 0.0, 1.0)
    sun_risk = jnp.where(self_pair, jnp.float32(0.0), sun_risk)

    garr_dst = planet_ships.astype(jnp.float32)
    ships_needed = (garr_dst + 1.0) / rem_at_src
    ships_needed_norm = jnp.clip(ships_needed, 0.0, 1.0)

    ships_at_bin5 = jnp.floor(rem_at_src * jnp.float32(0.7))
    pair_flip_bin5 = (ships_at_bin5 > garr_dst).astype(jnp.float32) * is_target_mask.astype(
        jnp.float32
    )
    margin = (ships_at_bin5 - garr_dst) / rem_at_src
    pair_margin_norm = jnp.clip(margin, 0.0, 1.0) * is_target_mask.astype(jnp.float32)

    pair_feats = jnp.stack(
        [dist_norm, sun_risk, ships_needed_norm, pair_flip_bin5, pair_margin_norm], axis=-1
    )
    pair_feats = pair_feats * planet_mask[..., None].astype(jnp.float32)

    sun_block_mask = (sun_risk >= _SUN_BLOCK_THRESH) & planet_mask & jnp.logical_not(self_pair)
    return pair_feats, sun_block_mask


def dst_flip_block_mask(
    planet_ships: jnp.ndarray,    # (..., P) int -- garrison at each dst
    planet_mask: jnp.ndarray,     # (..., P) bool
    is_target_mask: jnp.ndarray,  # (..., P) bool -- enemy | neutral
    remaining: jnp.ndarray,       # (..., P) int -- remaining at each src
    src_idx: jnp.ndarray,         # (...,) int
) -> jnp.ndarray:
    """Hard mask: block dst that cannot be flipped with 70% of remaining at src.

    Returns (..., P) bool where True means **forbidden** (same convention as
    ``sun_block_mask``). Owned planets are never blocked (``is_target_mask``
    is false for mine).
    """
    rem_at_src = jnp.take_along_axis(
        remaining.astype(jnp.float32), src_idx[..., None], axis=-1
    )
    rem_at_src = jnp.maximum(rem_at_src, jnp.float32(1.0))
    ships_at_bin5 = jnp.floor(rem_at_src * jnp.float32(0.7))[..., 0]
    garr_dst = planet_ships.astype(jnp.float32)
    flip_ok = ships_at_bin5[..., None] > garr_dst
    block = jnp.logical_not(flip_ok) & is_target_mask & planet_mask
    return block


def emit_pair_globals(
    planet_x: jnp.ndarray,       # (..., P)
    planet_y: jnp.ndarray,       # (..., P)
    planet_ships: jnp.ndarray,   # (..., P) int  -- per-planet garrison (dst side)
    planet_mask: jnp.ndarray,    # (..., P) bool
    my_mask: jnp.ndarray,        # (..., P) bool
    target_mask: jnp.ndarray,    # (..., P) bool (enemy | neutral)
    remaining: jnp.ndarray,      # (..., P) int  -- remaining at each of MY planets (others irrelevant)
    home_idx: jnp.ndarray,       # (...,) int  -- this player's home slot
    home_init: jnp.ndarray,      # (...,) float -- starting garrison at home (HOME_PLANET_SHIPS or first-turn)
    total_init: jnp.ndarray,     # (...,) float -- total my garrison at turn start (sum of ships_raw on mine)
) -> jnp.ndarray:
    """6 scalar features describing remaining attack budget & emit stop signal.

    Returns (..., 6):
      [0] emit_worth_it           -- 1.0 iff any (src,dst) pair has margin>0 with current remaining
      [1] best_pair_margin_norm   -- log1p(max over feasible pairs of margin) / log1p(MAX_PLANET_SHIPS)
      [2] home_remain_ratio       -- remaining[home] / max(home_init, 1)  clip [0,1]
      [3] total_remain_ratio      -- (sum remaining over mine) / max(total_init, 1) clip [0,1]
      [4] feasible_target_count_norm -- how many distinct dst can be flipped / MAX_PLANETS
      [5] surplus_ratio           -- (total_remaining - sum_min_needs) / total_remaining, clip [0,1]
    """
    P = planet_x.shape[-1]

    rem_my = jnp.where(my_mask, remaining, jnp.int32(0)).astype(jnp.float32)
    ships_at_bin5_per_src = jnp.floor(rem_my * jnp.float32(0.7))  # (..., P)
    garr_dst = planet_ships.astype(jnp.float32)                   # (..., P)

    # Pair feasibility: shape (..., P, P)
    margin = ships_at_bin5_per_src[..., :, None] - garr_dst[..., None, :]
    # Restrict to src in mine, dst in target, both masked valid.
    pair_valid = (
        my_mask[..., :, None]
        & target_mask[..., None, :]
        & planet_mask[..., :, None]
        & planet_mask[..., None, :]
    )
    # Require 10% overkill margin for emit_worth_it.
    # Not too high (causes turtle) nor zero (causes spam).
    min_overkill = jnp.float32(0.1) * jnp.maximum(garr_dst[..., None, :], jnp.float32(1.0))
    feasible = pair_valid & (margin > min_overkill)
    emit_worth_it = feasible.any(axis=(-2, -1)).astype(jnp.float32)

    # Best margin (set non-feasible to 0).
    margin_masked = jnp.where(feasible, margin, jnp.float32(0.0))
    best_margin = margin_masked.reshape(margin_masked.shape[:-2] + (P * P,)).max(axis=-1)
    best_margin = jnp.maximum(best_margin, jnp.float32(0.0))
    best_margin_norm = jnp.log1p(best_margin) / jnp.float32(8.0)
    best_margin_norm = jnp.clip(best_margin_norm, 0.0, 1.0)

    # Home / total ratio.
    home_idx_e = home_idx[..., None]
    rem_at_home = jnp.take_along_axis(rem_my, home_idx_e, axis=-1)[..., 0]
    home_remain_ratio = jnp.clip(
        rem_at_home / jnp.maximum(home_init, jnp.float32(1.0)), 0.0, 1.0
    )
    total_remaining = rem_my.sum(axis=-1)
    total_remain_ratio = jnp.clip(
        total_remaining / jnp.maximum(total_init, jnp.float32(1.0)), 0.0, 1.0
    )

    # [4] feasible_target_count: how many distinct dst planets can be flipped
    # by at least one of my src planets. Tells emit head "there are N soft
    # targets right now" -- direct multi-route signal.
    dst_flippable = feasible.any(axis=-2)  # (..., P) -- any src can flip this dst
    feasible_count = dst_flippable.astype(jnp.float32).sum(axis=-1)  # (...,)
    feasible_target_count_norm = jnp.clip(
        feasible_count / jnp.float32(constants.MAX_PLANETS), 0.0, 1.0
    )

    # [5] surplus_ratio: after flipping all feasible targets with the cheapest
    # src for each, how much of my total remaining is left over?
    # For each feasible dst, min_need = garr_dst + 1 (cheapest flip cost).
    target_need = (garr_dst + jnp.float32(1.0)) * target_mask.astype(jnp.float32)
    # Only count dst that are actually flippable by at least one src.
    flippable_need = target_need * dst_flippable.astype(jnp.float32)
    sum_min_needs = flippable_need.sum(axis=-1)  # (...,)
    surplus_ratio = jnp.clip(
        (total_remaining - sum_min_needs) / jnp.maximum(total_remaining, jnp.float32(1.0)),
        0.0,
        1.0,
    )

    return jnp.stack(
        [emit_worth_it, best_margin_norm, home_remain_ratio, total_remain_ratio,
         feasible_target_count_norm, surplus_ratio],
        axis=-1,
    )


def pct_min_bin_index(
    garr_dst: jnp.ndarray,       # (...,) float
    remaining_src: jnp.ndarray,  # (...,) float
) -> jnp.ndarray:
    """(...,) int32 smallest pct bin index whose ship count exceeds garr_dst."""
    rem = jnp.maximum(remaining_src, jnp.float32(1.0))
    ships_at_bins = jnp.floor(rem[..., None] * _PCT_BIN_TABLE)
    flip_at_bin = ships_at_bins > garr_dst[..., None]
    bin_indices = jnp.arange(_NUM_PCT_BINS, dtype=jnp.int32)
    masked_idx = jnp.where(
        flip_at_bin,
        bin_indices,
        jnp.int32(_NUM_PCT_BINS - 1),
    )
    return masked_idx.min(axis=-1)


def pct_low_bin_mask(
    min_bin: jnp.ndarray,        # (...,) int32
    num_bins: int = _NUM_PCT_BINS,
) -> jnp.ndarray:
    """(..., num_bins) bool -- True iff bin index is allowed (>= min_bin)."""
    arange = jnp.arange(num_bins, dtype=jnp.int32)
    if min_bin.ndim == 0:
        return arange >= min_bin
    return arange[(None,) * min_bin.ndim + (slice(None),)] >= min_bin[..., None]


def pct_pair_features(
    garr_dst: jnp.ndarray,       # (...,) float -- garrison at chosen dst
    remaining_src: jnp.ndarray,  # (...,) float -- remaining at chosen src
    *,
    enemy_inbound_norm: jnp.ndarray | None = None,   # (...,) float -- in_foe_norm at dst
    net_garrison_t15_dst: jnp.ndarray | None = None,  # (...,) float -- predicted garrison balance at dst
    src_prod_ratio: jnp.ndarray | None = None,         # (...,) float -- src prod / total my prod
    fleet_count_norm: jnp.ndarray | None = None,       # (...,) float -- t / MAX_FLEETS_PER_TURN
) -> jnp.ndarray:
    """(..., PCT_PAIR_DIM) pct decision features given chosen (src, dst).

    Base features (always present):
      [0] min_bin_norm    -- smallest bin that flips garr_dst, / (NUM_PCT_BINS-1)
      [1] pair_flip_bin5  -- 1 if floor(rem*0.7) > garr_dst

    Extended features (v14, when provided):
      [2] enemy_inbound_norm   -- foe ships heading to dst (log-normalised)
      [3] net_garrison_t15_dst -- predicted garrison balance at dst 15 steps out
      [4] src_prod_ratio       -- src production as fraction of total own production
      [5] fleet_count_norm     -- which autoregressive step we're on (0..1)
    """
    min_bin = pct_min_bin_index(garr_dst, remaining_src)
    min_bin_norm = min_bin.astype(jnp.float32) / jnp.float32(_NUM_PCT_BINS - 1)
    rem = jnp.maximum(remaining_src, jnp.float32(1.0))

    ships_at_bin5 = jnp.floor(rem * jnp.float32(0.7))
    pair_flip_bin5 = (ships_at_bin5 > garr_dst).astype(jnp.float32)

    parts = [min_bin_norm, pair_flip_bin5]
    if enemy_inbound_norm is not None:
        parts.append(enemy_inbound_norm)
    if net_garrison_t15_dst is not None:
        parts.append(net_garrison_t15_dst)
    if src_prod_ratio is not None:
        parts.append(src_prod_ratio)
    if fleet_count_norm is not None:
        parts.append(fleet_count_norm)
    return jnp.stack(parts, axis=-1)


# --------------------- public constants for sanity checks ---------------------
DST_PAIR_DIM = 5
EMIT_PAIR_DIM = 6
PCT_PAIR_DIM = 6
SUN_BLOCK_THRESH = float(_SUN_BLOCK_THRESH)
