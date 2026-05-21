"""Map a Kaggle Orbit Wars observation -> numpy features the model expects.

The training side uses an MVP env where ``planet_id`` happens to equal its
slot index (because static planets are written into the first ``4*num_groups``
slots). At Kaggle inference time we receive the real obs:

    planets[i] = [id, owner, x, y, radius, ships, production]
    fleets[i]  = [id, owner, x, y, angle, from_planet_id, ships]
    player     = int            # which player we control
    comets, comet_planet_ids, ...

We replicate the feature encoder using **planet_id as the slot index** to
preserve training-time correspondence; planets with id >= MAX_PLANETS are
skipped. The action goes back as ``[planet_id, angle, num_ships]`` triples.

This module is **numpy-only**; it never imports jax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np

# Match orbit_wars_rl/env/constants.py. Kept in-line so this module is
# importable from a single-file Kaggle submission with no package imports.
MAX_PLANETS = 40
MAX_FLEETS = 128
BOARD = 100.0
BOARD_HALF = 50.0
SUN_X = 50.0
SUN_Y = 50.0
NUM_PCT_BINS = 4
PCT_BIN_VALUES = (0.25, 0.5, 0.75, 1.0)
NEUTRAL_OWNER = -1
PADDING_OWNER = -2

PLANET_FEAT_DIM = 12
FLEET_FEAT_DIM = 8
GLOBAL_FEAT_DIM = 10

DEFAULT_EPISODE_STEPS = 500
_LOG_1000 = float(np.log(1000.0))


@dataclass
class EncodedObsNp:
    planet_feats: np.ndarray       # (MAX_PLANETS, 12)
    planet_mask: np.ndarray        # (MAX_PLANETS,) bool
    fleet_feats: np.ndarray        # (MAX_FLEETS, 8)
    fleet_mask: np.ndarray         # (MAX_FLEETS,) bool
    global_feats: np.ndarray       # (10,)
    my_planet_mask: np.ndarray     # (MAX_PLANETS,) bool
    enemy_planet_mask: np.ndarray  # (MAX_PLANETS,) bool
    neutral_planet_mask: np.ndarray  # (MAX_PLANETS,) bool


def _fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    s = max(float(ships), 1.0)
    rel = (math.log(s) / _LOG_1000) ** 1.5
    return min(1.0 + (max_speed - 1.0) * rel, max_speed)


def encode_kaggle_obs(
    obs: Dict[str, Any],
    player: Optional[int] = None,
    step: Optional[int] = None,
    episode_steps: int = DEFAULT_EPISODE_STEPS,
) -> EncodedObsNp:
    """Return MVP-style features for one Kaggle observation.

    ``player`` defaults to ``obs['player']``; ``step`` defaults to ``obs.get('step', 0)``
    (Kaggle places the step under different keys depending on context; we read
    the most common one and fall back to 0).
    """
    if player is None:
        player = int(obs.get("player", 0))
    opp = 1 - player  # MVP is 2P only; for 4P we'd take the "strongest opp"

    raw_planets: List[Any] = list(obs.get("planets") or [])
    raw_fleets: List[Any] = list(obs.get("fleets") or [])

    if step is None:
        step = int(obs.get("step") or obs.get("stepCount") or 0)

    planet_owner = np.full((MAX_PLANETS,), PADDING_OWNER, dtype=np.int32)
    planet_x = np.zeros((MAX_PLANETS,), dtype=np.float32)
    planet_y = np.zeros((MAX_PLANETS,), dtype=np.float32)
    planet_radius = np.zeros((MAX_PLANETS,), dtype=np.float32)
    planet_ships = np.zeros((MAX_PLANETS,), dtype=np.int32)
    planet_prod = np.zeros((MAX_PLANETS,), dtype=np.int32)
    planet_mask = np.zeros((MAX_PLANETS,), dtype=bool)

    for p in raw_planets:
        pid = int(p[0])
        if pid < 0 or pid >= MAX_PLANETS:
            continue
        planet_owner[pid] = int(p[1])
        planet_x[pid] = float(p[2])
        planet_y[pid] = float(p[3])
        planet_radius[pid] = float(p[4])
        planet_ships[pid] = int(p[5])
        planet_prod[pid] = int(p[6])
        planet_mask[pid] = True

    fleet_owner = np.full((MAX_FLEETS,), PADDING_OWNER, dtype=np.int32)
    fleet_x = np.zeros((MAX_FLEETS,), dtype=np.float32)
    fleet_y = np.zeros((MAX_FLEETS,), dtype=np.float32)
    fleet_angle = np.zeros((MAX_FLEETS,), dtype=np.float32)
    fleet_ships = np.zeros((MAX_FLEETS,), dtype=np.int32)
    fleet_mask = np.zeros((MAX_FLEETS,), dtype=bool)

    # Kaggle fleets carry their own ids; we just pack them sequentially into
    # the first MAX_FLEETS slots. The model never refers to fleets by id, only
    # by their feature vectors, so packing order doesn't matter as long as the
    # mask aligns.
    for i, f in enumerate(raw_fleets[:MAX_FLEETS]):
        fleet_owner[i] = int(f[1])
        fleet_x[i] = float(f[2])
        fleet_y[i] = float(f[3])
        fleet_angle[i] = float(f[4])
        fleet_ships[i] = int(f[6])
        fleet_mask[i] = True

    is_mine = (planet_owner == player) & planet_mask
    is_enemy = (planet_owner >= 0) & (planet_owner != player) & planet_mask
    is_neutral = (planet_owner == NEUTRAL_OWNER) & planet_mask

    x_norm = (planet_x - BOARD_HALF) / BOARD_HALF
    y_norm = (planet_y - BOARD_HALF) / BOARD_HALF
    radius_norm = planet_radius / 5.0
    log_ships = np.log1p(np.maximum(planet_ships, 0).astype(np.float32)) / 8.0
    prod_norm = planet_prod.astype(np.float32) / 5.0
    dist_sun = np.sqrt((planet_x - SUN_X) ** 2 + (planet_y - SUN_Y) ** 2) / BOARD

    in_friend = _inbound_ships_np(
        fleet_owner, fleet_x, fleet_y, fleet_angle, fleet_ships, fleet_mask,
        planet_x, planet_y, planet_mask, owner_keep=player,
    )
    in_foe = _inbound_ships_np(
        fleet_owner, fleet_x, fleet_y, fleet_angle, fleet_ships, fleet_mask,
        planet_x, planet_y, planet_mask, owner_keep=opp,
    )
    in_friend_norm = np.log1p(in_friend) / 8.0
    in_foe_norm = np.log1p(in_foe) / 8.0
    is_padding = (~planet_mask).astype(np.float32)

    planet_feats = np.stack(
        [
            is_mine.astype(np.float32),
            is_enemy.astype(np.float32),
            is_neutral.astype(np.float32),
            x_norm,
            y_norm,
            radius_norm,
            log_ships,
            prod_norm,
            dist_sun,
            in_friend_norm,
            in_foe_norm,
            is_padding,
        ],
        axis=-1,
    ).astype(np.float32)
    planet_feats *= planet_mask[:, None].astype(np.float32)

    f_is_mine = (fleet_owner == player) & fleet_mask
    f_is_enemy = (fleet_owner >= 0) & (fleet_owner != player) & fleet_mask
    fx_norm = (fleet_x - BOARD_HALF) / BOARD_HALF
    fy_norm = (fleet_y - BOARD_HALF) / BOARD_HALF
    sin_a = np.sin(fleet_angle)
    cos_a = np.cos(fleet_angle)
    f_log_ships = np.log1p(np.maximum(fleet_ships, 0).astype(np.float32)) / 8.0
    zero = np.zeros_like(fx_norm, dtype=np.float32)

    fleet_feats = np.stack(
        [
            f_is_mine.astype(np.float32),
            f_is_enemy.astype(np.float32),
            zero,
            fx_norm,
            fy_norm,
            sin_a,
            cos_a,
            f_log_ships,
        ],
        axis=-1,
    ).astype(np.float32)
    fleet_feats *= fleet_mask[:, None].astype(np.float32)

    def _player_ships(p_id: int) -> float:
        p_sum = float(planet_ships[(planet_owner == p_id) & planet_mask].sum())
        f_sum = float(fleet_ships[(fleet_owner == p_id) & fleet_mask].sum())
        return p_sum + f_sum

    def _player_planets(p_id: int) -> float:
        return float(((planet_owner == p_id) & planet_mask).sum())

    def _player_prod(p_id: int) -> float:
        return float(planet_prod[(planet_owner == p_id) & planet_mask].sum())

    total_ships = max(_player_ships(player) + _player_ships(opp), 1.0)
    total_planets = max(_player_planets(player) + _player_planets(opp), 1.0)
    total_prod = max(_player_prod(player) + _player_prod(opp), 1.0)

    step_norm = float(step) / float(episode_steps)
    is_early = float(step_norm < 0.18)
    is_mid = float(0.18 <= step_norm < 0.64)
    is_late = float(step_norm >= 0.64)

    global_feats = np.array(
        [
            step_norm,
            _player_ships(player) / total_ships,
            _player_ships(opp) / total_ships,
            _player_planets(player) / total_planets,
            _player_planets(opp) / total_planets,
            _player_prod(player) / total_prod,
            _player_prod(opp) / total_prod,
            is_early,
            is_mid,
            is_late,
        ],
        dtype=np.float32,
    )

    return EncodedObsNp(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        fleet_feats=fleet_feats,
        fleet_mask=fleet_mask,
        global_feats=global_feats,
        my_planet_mask=is_mine,
        enemy_planet_mask=is_enemy,
        neutral_planet_mask=is_neutral,
    )


def _inbound_ships_np(
    fleet_owner: np.ndarray,
    fleet_x: np.ndarray, fleet_y: np.ndarray,
    fleet_angle: np.ndarray, fleet_ships: np.ndarray, fleet_mask: np.ndarray,
    planet_x: np.ndarray, planet_y: np.ndarray, planet_mask: np.ndarray,
    owner_keep: int,
) -> np.ndarray:
    """Mirror of orbit_wars_rl.features.encode._inbound_ships, in numpy."""
    dxn = np.cos(fleet_angle)
    dyn = np.sin(fleet_angle)
    rel_x = planet_x[None, :] - fleet_x[:, None]
    rel_y = planet_y[None, :] - fleet_y[:, None]
    proj = rel_x * dxn[:, None] + rel_y * dyn[:, None]
    perp_x = rel_x - proj * dxn[:, None]
    perp_y = rel_y - proj * dyn[:, None]
    perp_d2 = perp_x * perp_x + perp_y * perp_y
    in_front = proj > 0
    cost = np.where(in_front, perp_d2, np.float32(1e9))
    cost = np.where(planet_mask[None, :], cost, np.float32(1e9))
    target_idx = np.argmin(cost, axis=1)

    is_owner_keep = fleet_owner == owner_keep
    contributes = fleet_mask & is_owner_keep
    ships_keep = np.where(contributes, fleet_ships, 0)

    inbound = np.zeros((MAX_PLANETS,), dtype=np.float32)
    for fi in range(fleet_owner.shape[0]):
        if contributes[fi]:
            inbound[target_idx[fi]] += float(ships_keep[fi])
    return inbound


def decode_to_kaggle_move(
    obs: Dict[str, Any],
    src_idx: int,
    dst_idx: int,
    pct_bin: int,
    player: Optional[int] = None,
) -> List[List[float]]:
    """Translate ``(src, dst, pct_bin)`` into Kaggle's ``[[planet_id, angle, ships]]``.

    Drops the launch (returns ``[]``) if any of the standard sanity checks fail
    (src not owned, no ships, src==dst), mirroring ``actions.decode_action`` so
    the model never confidently emits an illegal action even on edge cases.
    """
    if player is None:
        player = int(obs.get("player", 0))

    planets_by_id: Dict[int, Any] = {}
    for p in obs.get("planets") or []:
        planets_by_id[int(p[0])] = p

    if src_idx not in planets_by_id or dst_idx not in planets_by_id:
        return []
    if src_idx == dst_idx:
        return []

    src = planets_by_id[src_idx]
    dst = planets_by_id[dst_idx]
    src_owner, src_ships = int(src[1]), int(src[5])
    if src_owner != player or src_ships <= 0:
        return []

    pct_bin = max(0, min(NUM_PCT_BINS - 1, int(pct_bin)))
    pct = PCT_BIN_VALUES[pct_bin]
    ships_to_send = max(1, int(math.floor(src_ships * pct)))
    ships_to_send = min(ships_to_send, src_ships)

    sx, sy = float(src[2]), float(src[3])
    dx, dy = float(dst[2]), float(dst[3])
    angle = math.atan2(dy - sy, dx - sx)

    return [[int(src_idx), float(angle), int(ships_to_send)]]
