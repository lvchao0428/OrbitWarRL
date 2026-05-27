"""K-step pair / per-step feature helpers (f26).

These are computed *inside* the autoregressive head loop, not at turn start,
because they depend on:

  * the running ``reserved`` book-keeping buffer (which grows each step)
  * the current ``src_t`` / ``dst_t`` (for per-pair signals)

They feed the heads as extra scalar/vector kwargs and are intentionally kept
out of ``features/encode.py`` (whose output is per-turn-static).

Provided:

  ``dst_pair_features(planet_x, planet_y, planet_ships, planet_mask,
                       is_target_mask, remaining, src_idx)``
      -> (pair_feats: (P, 4), sun_block_mask: (P,) bool)
      Per-candidate-dst features given current source. ``sun_block_mask`` is
      True for dst whose src->dst segment crosses the sun. Used both as a
      head input (sun_risk float) and as a hard logit mask in DstHead.

  ``emit_pair_globals(planet_x, planet_y, planet_ships, planet_mask,
                       my_mask, target_mask, remaining,
                       home_remaining, home_init,
                       total_remaining, total_init)``
      -> (g: (4,)) per-step global scalar that summarises feasibility / budget.

  ``pct_pair_features(garr_dst, remaining_src)``
      -> (2,) [pair_needed_pct, pair_flip_bin5] given chosen (src, dst).

All functions are jit-pure (no python branching on traced values).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants


_BOARD = jnp.float32(constants.BOARD)
_SUN = jnp.array([constants.SUN_X, constants.SUN_Y], dtype=jnp.float32)
_SUN_R = jnp.float32(constants.SUN_RADIUS)
# Margin: treat fleet path as "hits sun" if min-dist < SUN_R * SUN_PATH_MARGIN.
# constants.SUN_PATH_MARGIN = 1.25 -> ~12.5 unit forbidden corridor radius.
_SUN_GUARD = _SUN_R * jnp.float32(constants.SUN_PATH_MARGIN)
# Hard mask threshold (>= 0.9 means src->dst path nearly tangent to sun guard).
_SUN_BLOCK_THRESH = jnp.float32(0.9)


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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-candidate-dst (4,) features given a chosen src.

    Returns:
      pair_feats: (P, 4) float32 in [0, 1]:
        [0] dist_src_dst / BOARD
        [1] sun_risk = clip(1 - min_dist / SUN_GUARD, 0, 1)  (1 = path through sun)
        [2] ships_needed_norm = (garr[dst]+1) / remaining[src]  clip [0,1]
        [3] pair_flip_bin5 = float( floor(remaining[src]*0.7) > garr[dst] ) AND target
      sun_block_mask: (P,) bool -- True iff sun_risk >= 0.9 (hard mask candidate)
    """
    P = planet_x.shape[0]
    src_x = planet_x[src_idx]
    src_y = planet_y[src_idx]
    rem_src = jnp.maximum(remaining[src_idx], jnp.int32(1)).astype(jnp.float32)

    dx = planet_x - src_x
    dy = planet_y - src_y
    dist = jnp.sqrt(dx * dx + dy * dy + jnp.float32(1e-6))
    dist_norm = dist / _BOARD

    min_to_sun = _segment_min_dist_to_point(
        src_x, src_y, planet_x, planet_y, _SUN[0], _SUN[1]
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

    pair_feats = jnp.stack(
        [dist_norm, sun_risk, ships_needed_norm, pair_flip_bin5], axis=-1
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Same as ``dst_pair_features`` with arbitrary leading batch dims.

    Implements via take_along_axis to gather src coords/ships.
    """
    if planet_x.ndim == 1:
        return dst_pair_features(
            planet_x, planet_y, planet_ships, planet_mask,
            is_target_mask, remaining, src_idx,
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
    dist_norm = dist / _BOARD

    min_to_sun = _segment_min_dist_to_point(
        src_x, src_y, planet_x, planet_y, _SUN[0], _SUN[1]
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

    pair_feats = jnp.stack(
        [dist_norm, sun_risk, ships_needed_norm, pair_flip_bin5], axis=-1
    )
    pair_feats = pair_feats * planet_mask[..., None].astype(jnp.float32)

    sun_block_mask = (sun_risk >= _SUN_BLOCK_THRESH) & planet_mask & jnp.logical_not(self_pair)
    return pair_feats, sun_block_mask


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
    """4 scalar features describing remaining attack budget & feasible targets.

    Returns (..., 4):
      [0] n_feasible_pairs_norm   -- count(src in mine, dst in target, flip_bin5(src,dst)) / MAX_PLANETS
      [1] best_pair_margin_norm   -- log1p(max over feasible pairs of margin) / log1p(MAX_PLANET_SHIPS)
      [2] home_remain_ratio       -- remaining[home] / max(home_init, 1)  clip [0,1]
      [3] total_remain_ratio      -- (sum remaining over mine) / max(total_init, 1) clip [0,1]
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
    feasible = pair_valid & (margin > 0)
    feasible_f = feasible.astype(jnp.float32)
    n_feasible = feasible_f.sum(axis=(-2, -1))
    n_feasible_norm = jnp.clip(
        n_feasible / jnp.float32(constants.MAX_PLANETS), 0.0, 1.0
    )

    # Best margin (set non-feasible to 0).
    margin_masked = jnp.where(feasible, margin, jnp.float32(0.0))
    best_margin = margin_masked.reshape(margin_masked.shape[:-2] + (P * P,)).max(axis=-1)
    best_margin = jnp.maximum(best_margin, jnp.float32(0.0))
    # log1p / log1p(1000): same scale family as planet log_ships.
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

    return jnp.stack(
        [n_feasible_norm, best_margin_norm, home_remain_ratio, total_remain_ratio],
        axis=-1,
    )


def pct_pair_features(
    garr_dst: jnp.ndarray,       # (...,) float -- garrison at chosen dst
    remaining_src: jnp.ndarray,  # (...,) float -- remaining at chosen src
) -> jnp.ndarray:
    """(...,2) [pair_needed_pct, pair_flip_bin5_pair]."""
    rem = jnp.maximum(remaining_src, jnp.float32(1.0))
    pair_needed_pct = jnp.clip((garr_dst + 1.0) / rem, 0.0, 1.0)
    ships_at_bin5 = jnp.floor(rem * jnp.float32(0.7))
    pair_flip_bin5 = (ships_at_bin5 > garr_dst).astype(jnp.float32)
    return jnp.stack([pair_needed_pct, pair_flip_bin5], axis=-1)


# --------------------- public constants for sanity checks ---------------------
DST_PAIR_DIM = 4
EMIT_PAIR_DIM = 4
PCT_PAIR_DIM = 2
SUN_BLOCK_THRESH = float(_SUN_BLOCK_THRESH)
