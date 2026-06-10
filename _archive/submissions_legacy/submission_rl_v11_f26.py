"""Orbit Wars RL v11 f26 — f25 features + K-loop pair head augmentations (Day7).

Must match ``orbit_wars_rl/features/encode.py`` + ``features/pair.py`` (f26):
  - PLANET_FEAT_DIM = 28  (same as f25)
  - FLEET_FEAT_DIM  = 10  (same as f25)
  - GLOBAL_FEAT_DIM = 17  (same as f25)
  - MAX_FLEETS_PER_TURN = 8

Pair head augmentations (K-loop, per step):
  DstHead   += 4 scalars per candidate dst (dist, sun_risk, ships_needed, flip_bin5)
              + hard ``sun_block_mask`` on src->dst paths crossing the sun.
  EmitHead  += 4 scalar globals (n_feasible_pairs, best_margin, home_ratio, total_ratio).
  PctHead   += 2 scalars for chosen (src,dst) [pair_needed_pct, pair_flip_bin5].

These do NOT change PLANET/FLEET/GLOBAL feat dims -- they only widen the
inputs of dst_fc1 / emit_fc1 / pct_fc1, so the same ckpt header still detects
planet=28/global=17/K=8 but pair head shapes differ from f25's.

Weights are injected into the WEIGHTS_B64 line near the bottom of this file
by ``orbit_wars_rl/scripts/export_submission.py``.

NOTE: do not write the literal assignment ``WEIGHTS_B64 = <quoted-string>``
form anywhere in this docstring -- the injector uses a start-of-line
anchored regex, but extra matches in early drafts caused us to lose a
training run (see docs/DAY2_PROGRESS.md ${section}9.10).
"""

from __future__ import annotations

import base64
import io
import math
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --- env constants (mirror orbit_wars_rl.env.constants) -----------------------
MAX_PLANETS = 40
MAX_FLEETS = 128
BOARD = 100.0
BOARD_HALF = 50.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
SUN_PATH_MARGIN = 1.25
SUN_GUARD = SUN_RADIUS * SUN_PATH_MARGIN
SUN_BLOCK_THRESH = 0.9
NUM_PCT_BINS = 8
PCT_BIN_VALUES = (0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00)
NEUTRAL_OWNER = -1
PADDING_OWNER = -2

PLANET_FEAT_DIM = 28
FLEET_FEAT_DIM = 10
GLOBAL_FEAT_DIM = 17

# Orbital constants (must mirror env/constants.py).
ROTATION_RADIUS_LIMIT = 50.0
ORBIT_OMEGA_MAX = 0.05

# Lead-target prediction times (mirror features/encode.LEAD_TIMES).
LEAD_TIMES = (15.0, 30.0)

DEFAULT_EPISODE_STEPS = 500
N_LAYERS = 2
MAX_FLEETS_PER_TURN = 8

_LOG_1000 = float(np.log(1000.0))
_PCT_BIN_TABLE_NP = np.array(PCT_BIN_VALUES, dtype=np.float32)


# --- feature encoding (mirror features.encode) --------------------------------

def _inbound_ships_np(
    fleet_owner, fleet_x, fleet_y, fleet_angle, fleet_ships, fleet_mask,
    planet_x, planet_y, planet_mask, owner_keep: int,
) -> np.ndarray:
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

    contributes = fleet_mask & (fleet_owner == owner_keep)
    ships_keep = np.where(contributes, fleet_ships, 0)

    inbound = np.zeros((MAX_PLANETS,), dtype=np.float32)
    for fi in range(fleet_owner.shape[0]):
        if contributes[fi]:
            inbound[target_idx[fi]] += float(ships_keep[fi])
    return inbound


def _fleet_speed_np(ships: np.ndarray, max_speed: float = 6.0) -> np.ndarray:
    s = np.maximum(ships.astype(np.float32), 1.0)
    rel = (np.log(s) / _LOG_1000) ** 1.5
    return np.minimum(1.0 + (max_speed - 1.0) * rel, max_speed)


def _soft_inbound_and_eta_np(
    fleet_owner, fleet_x, fleet_y, fleet_angle, fleet_ships, fleet_mask,
    planet_x, planet_y, planet_mask, owner_keep: int, episode_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-square soft inbound + min ETA (mirror features.encode A1')."""
    eps = np.float32(1e-4)
    rel_x = planet_x[None, :] - fleet_x[:, None]
    rel_y = planet_y[None, :] - fleet_y[:, None]
    dist2 = rel_x * rel_x + rel_y * rel_y + eps
    dist = np.sqrt(dist2)
    speed = _fleet_speed_np(fleet_ships)
    eta = dist / np.maximum(speed[:, None], eps)

    cos_a = np.cos(fleet_angle)
    sin_a = np.sin(fleet_angle)
    proj = rel_x * cos_a[:, None] + rel_y * sin_a[:, None]
    toward = (proj > 0).astype(np.float32)

    is_owner = ((fleet_owner == owner_keep) & fleet_mask).astype(np.float32)
    weight = toward * is_owner[:, None] / dist2
    ships = fleet_ships.astype(np.float32)
    inbound_soft = (weight * ships[:, None]).sum(axis=0)

    big_eta = np.float32(1e6)
    eta_masked = np.where(weight > 0, eta, big_eta)
    eta_min = eta_masked.min(axis=0)
    eta_min = np.where(eta_min >= big_eta, np.float32(0.0), eta_min)
    eta_norm = eta_min / np.float32(max(episode_steps, 1))
    inbound_soft = np.where(planet_mask, inbound_soft, 0.0)
    eta_norm = np.where(planet_mask, eta_norm, 0.0)
    return inbound_soft.astype(np.float32), eta_norm.astype(np.float32)


def encode_obs(obs: Dict[str, Any], player: int, step: int, episode_steps: int) -> dict:
    opp = 1 - player

    raw_planets = list(obs.get("planets") or [])
    raw_fleets = list(obs.get("fleets") or [])

    planet_owner = np.full((MAX_PLANETS,), PADDING_OWNER, dtype=np.int32)
    planet_x = np.zeros((MAX_PLANETS,), dtype=np.float32)
    planet_y = np.zeros((MAX_PLANETS,), dtype=np.float32)
    planet_radius = np.zeros((MAX_PLANETS,), dtype=np.float32)
    planet_ships = np.zeros((MAX_PLANETS,), dtype=np.int32)
    planet_prod = np.zeros((MAX_PLANETS,), dtype=np.int32)
    planet_mask = np.zeros((MAX_PLANETS,), dtype=bool)
    for p in raw_planets:
        pid = int(p[0])
        if 0 <= pid < MAX_PLANETS:
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

    foe_soft, eta_foe_min = _soft_inbound_and_eta_np(
        fleet_owner, fleet_x, fleet_y, fleet_angle, fleet_ships, fleet_mask,
        planet_x, planet_y, planet_mask, owner_keep=opp, episode_steps=episode_steps,
    )
    friend_soft, _ = _soft_inbound_and_eta_np(
        fleet_owner, fleet_x, fleet_y, fleet_angle, fleet_ships, fleet_mask,
        planet_x, planet_y, planet_mask, owner_keep=player, episode_steps=episode_steps,
    )
    garr = np.maximum(planet_ships.astype(np.float32), 1.0)
    threat_ratio = np.clip(foe_soft / garr / 8.0, 0.0, 1.0)
    net_inbound = np.clip((foe_soft - friend_soft) / garr / 8.0, -1.0, 1.0)

    my_ships_raw = np.where(is_mine, planet_ships.astype(np.float32), 0.0)
    my_total_garrison = my_ships_raw.sum()
    my_n_planets = is_mine.astype(np.float32).sum()
    my_avg_garrison = my_total_garrison / max(my_n_planets, 1.0)

    target_garr = np.where(is_mine, 0.0, planet_ships.astype(np.float32))
    flip_cost_norm = np.clip(
        target_garr / max(my_avg_garrison, 1.0) / 3.0, 0.0, 1.0
    )

    friendly_surplus = (in_friend - planet_ships.astype(np.float32)) / np.maximum(
        planet_ships.astype(np.float32), 1.0
    )
    friendly_surplus = np.clip(friendly_surplus, -1.0, 1.0)

    ships_at_bin3 = np.floor(my_avg_garrison * 0.4)
    capturable_bin3 = np.where(
        is_mine,
        0.0,
        (ships_at_bin3 > planet_ships.astype(np.float32)).astype(np.float32),
    )

    my_max_garrison = my_ships_raw.max()
    needed_pct_norm = np.where(
        is_mine,
        0.0,
        np.clip(target_garr / max(my_max_garrison, 1.0), 0.0, 1.0),
    )

    ships_at_bin5 = np.floor(my_max_garrison * 0.7)
    capturable_bin5 = np.where(
        is_mine,
        0.0,
        (ships_at_bin5 > planet_ships.astype(np.float32)).astype(np.float32),
    )

    is_capturable_target = (is_enemy | is_neutral).astype(np.float32)
    inv_strength = np.clip(
        1.0 - np.log1p(np.maximum(planet_ships, 0).astype(np.float32)) / 8.0,
        0.0,
        1.0,
    )
    weak_target_score = inv_strength * is_capturable_target

    is_padding = (~planet_mask).astype(np.float32)

    dx_sun = planet_x - SUN_X
    dy_sun = planet_y - SUN_Y
    orbit_radius_np = np.sqrt(dx_sun * dx_sun + dy_sun * dy_sun)
    orbit_phase_np = np.arctan2(dy_sun, dx_sun)
    is_orbiting_np = ((orbit_radius_np + planet_radius) < ROTATION_RADIUS_LIMIT) & planet_mask
    is_orbiting_f = is_orbiting_np.astype(np.float32)
    orbit_phase_norm = orbit_phase_np.astype(np.float32) / np.float32(np.pi)
    orbit_radius_norm = orbit_radius_np.astype(np.float32) / BOARD_HALF

    av_raw = float(obs.get("angular_velocity") or 0.0)
    rotate_mask_f = is_orbiting_f
    def _lead(t: float):
        new_phase = orbit_phase_np + av_raw * t
        rot_x = SUN_X + orbit_radius_np * np.cos(new_phase)
        rot_y = SUN_Y + orbit_radius_np * np.sin(new_phase)
        out_x = rotate_mask_f * rot_x + (1.0 - rotate_mask_f) * planet_x
        out_y = rotate_mask_f * rot_y + (1.0 - rotate_mask_f) * planet_y
        return out_x.astype(np.float32), out_y.astype(np.float32)
    lead_x_15, lead_y_15 = _lead(LEAD_TIMES[0])
    lead_x_30, lead_y_30 = _lead(LEAD_TIMES[1])
    lead_x_15_norm = (lead_x_15 - BOARD_HALF) / BOARD_HALF
    lead_y_15_norm = (lead_y_15 - BOARD_HALF) / BOARD_HALF
    lead_x_30_norm = (lead_x_30 - BOARD_HALF) / BOARD_HALF
    lead_y_30_norm = (lead_y_30 - BOARD_HALF) / BOARD_HALF

    planet_feats = np.stack(
        [
            is_mine.astype(np.float32), is_enemy.astype(np.float32), is_neutral.astype(np.float32),
            x_norm, y_norm, radius_norm, log_ships, prod_norm, dist_sun,
            in_friend_norm, in_foe_norm, is_padding,
            is_orbiting_f, orbit_phase_norm, orbit_radius_norm,
            lead_x_15_norm, lead_y_15_norm, lead_x_30_norm, lead_y_30_norm,
            threat_ratio, net_inbound, eta_foe_min,
            flip_cost_norm, friendly_surplus, capturable_bin3,
            needed_pct_norm, capturable_bin5, weak_target_score,
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

    dxn = np.cos(fleet_angle)
    dyn = np.sin(fleet_angle)
    rel_x = planet_x[None, :] - fleet_x[:, None]
    rel_y = planet_y[None, :] - fleet_y[:, None]
    proj = rel_x * dxn[:, None] + rel_y * dyn[:, None]
    perp_x = rel_x - proj * dxn[:, None]
    perp_y = rel_y - proj * dyn[:, None]
    perp_d2 = perp_x * perp_x + perp_y * perp_y
    cost = np.where(proj > 0, perp_d2, np.float32(1e9))
    cost = np.where(planet_mask[None, :], cost, np.float32(1e9))
    target_idx = np.argmin(cost, axis=1)

    t_dx = planet_x[target_idx] - fleet_x
    t_dy = planet_y[target_idx] - fleet_y
    target_dist_norm = np.sqrt(t_dx * t_dx + t_dy * t_dy + 1e-6) / BOARD
    target_garrison_norm = np.log1p(
        np.maximum(planet_ships[target_idx].astype(np.float32), 0.0)
    ) / 8.0

    fleet_feats = np.stack(
        [
            f_is_mine.astype(np.float32), f_is_enemy.astype(np.float32), zero,
            fx_norm, fy_norm, sin_a, cos_a, f_log_ships,
            target_dist_norm, target_garrison_norm,
        ],
        axis=-1,
    ).astype(np.float32)
    fleet_feats *= fleet_mask[:, None].astype(np.float32)

    def _player_ships(p_id: int) -> float:
        return float(planet_ships[(planet_owner == p_id) & planet_mask].sum()) + \
               float(fleet_ships[(fleet_owner == p_id) & fleet_mask].sum())

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
    av_norm = av_raw / ORBIT_OMEGA_MAX
    my_garr_total = float(planet_ships[is_mine].sum())
    n_fleets_mine = float(f_is_mine.sum())
    n_fleets_enemy = float(f_is_enemy.sum())
    max_garr_norm = np.log1p(my_max_garrison) / 10.0
    weak_mask = (weak_target_score > 0.3).astype(np.float32)
    n_weak_targets_norm = weak_mask.sum() / MAX_PLANETS
    total_weak_garr = (weak_mask * target_garr).sum()
    ships_to_capture_all_weak_norm = np.log1p(total_weak_garr) / 10.0
    global_feats = np.array(
        [
            step_norm,
            _player_ships(player) / total_ships, _player_ships(opp) / total_ships,
            _player_planets(player) / total_planets, _player_planets(opp) / total_planets,
            _player_prod(player) / total_prod, _player_prod(opp) / total_prod,
            is_early, is_mid, is_late,
            av_norm,
            np.log1p(my_garr_total) / 10.0,
            n_fleets_mine / MAX_FLEETS,
            n_fleets_enemy / MAX_FLEETS,
            max_garr_norm,
            n_weak_targets_norm,
            ships_to_capture_all_weak_norm,
        ],
        dtype=np.float32,
    )

    # f26: expose planet_x/y so greedy_multi_action can compute pair feats.
    return dict(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        fleet_feats=fleet_feats,
        fleet_mask=fleet_mask,
        global_feats=global_feats,
        my_planet_mask=is_mine,
        planet_ships=planet_ships,
        planet_x=planet_x,
        planet_y=planet_y,
    )


# --- f26 pair feature helpers (mirror features/pair.py) ----------------------

def _segment_min_dist_to_point(ax, ay, bx, by, cx, cy):
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    ab2 = abx * abx + aby * aby + np.float32(1e-6)
    t = (acx * abx + acy * aby) / ab2
    t = np.clip(t, 0.0, 1.0)
    px = ax + t * abx
    py = ay + t * aby
    dx = cx - px
    dy = cy - py
    return np.sqrt(dx * dx + dy * dy + np.float32(1e-6))


def _dst_pair_features(planet_x, planet_y, planet_ships, planet_mask,
                       is_target_mask, remaining, src_idx):
    P = planet_x.shape[0]
    src_x = float(planet_x[src_idx])
    src_y = float(planet_y[src_idx])
    rem_src = float(max(int(remaining[src_idx]), 1))

    dx = planet_x - src_x
    dy = planet_y - src_y
    dist = np.sqrt(dx * dx + dy * dy + 1e-6)
    dist_norm = (dist / BOARD).astype(np.float32)

    min_to_sun = _segment_min_dist_to_point(src_x, src_y, planet_x, planet_y,
                                            SUN_X, SUN_Y)
    self_pair = np.arange(P) == src_idx
    sun_risk = np.clip(1.0 - min_to_sun / SUN_GUARD, 0.0, 1.0).astype(np.float32)
    sun_risk = np.where(self_pair, np.float32(0.0), sun_risk)

    garr_dst = planet_ships.astype(np.float32)
    ships_needed_norm = np.clip((garr_dst + 1.0) / rem_src, 0.0, 1.0).astype(np.float32)
    ships_at_bin5 = np.floor(rem_src * np.float32(0.7))
    pair_flip_bin5 = ((ships_at_bin5 > garr_dst).astype(np.float32)
                      * is_target_mask.astype(np.float32))

    pair_feats = np.stack(
        [dist_norm, sun_risk, ships_needed_norm, pair_flip_bin5], axis=-1
    ).astype(np.float32)
    pair_feats *= planet_mask.astype(np.float32)[:, None]

    sun_block_mask = (sun_risk >= SUN_BLOCK_THRESH) & planet_mask & np.logical_not(self_pair)
    return pair_feats, sun_block_mask


def _emit_pair_globals(planet_x, planet_y, planet_ships, planet_mask, my_mask,
                       target_mask, remaining, home_idx, home_init, total_init):
    P = planet_x.shape[0]
    rem_my = np.where(my_mask, remaining, 0).astype(np.float32)
    ships_at_bin5_src = np.floor(rem_my * np.float32(0.7))
    garr_dst = planet_ships.astype(np.float32)

    margin = ships_at_bin5_src[:, None] - garr_dst[None, :]
    pair_valid = (my_mask[:, None] & target_mask[None, :]
                  & planet_mask[:, None] & planet_mask[None, :])
    feasible = pair_valid & (margin > 0)
    n_feasible = float(feasible.sum())
    n_feasible_norm = float(min(n_feasible / MAX_PLANETS, 1.0))

    margin_masked = np.where(feasible, margin, 0.0)
    best_margin = float(margin_masked.max(initial=0.0))
    best_margin = max(best_margin, 0.0)
    best_margin_norm = float(min(np.log1p(best_margin) / 8.0, 1.0))

    rem_at_home = float(rem_my[home_idx]) if 0 <= home_idx < P else 0.0
    home_remain_ratio = float(min(rem_at_home / max(home_init, 1.0), 1.0))
    total_remaining = float(rem_my.sum())
    total_remain_ratio = float(min(total_remaining / max(total_init, 1.0), 1.0))

    return np.array(
        [n_feasible_norm, best_margin_norm, home_remain_ratio, total_remain_ratio],
        dtype=np.float32,
    )


def _pct_pair_features(garr_dst, remaining_src):
    rem = max(float(remaining_src), 1.0)
    pair_needed_pct = min(max((float(garr_dst) + 1.0) / rem, 0.0), 1.0)
    ships_at_bin5 = float(np.floor(rem * 0.7))
    pair_flip_bin5 = 1.0 if ships_at_bin5 > float(garr_dst) else 0.0
    return np.array([pair_needed_pct, pair_flip_bin5], dtype=np.float32)


# --- numpy forward (mirror inference.numpy_forward) ---------------------------

def _layer_norm(x, scale, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _gelu(x):
    c = np.float32(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * (x ** 3))))


def _dense(x, W, b):
    return x @ W + b


def _project_qkv(x, kernel, bias):
    return np.einsum("td,dhe->the", x, kernel) + bias


def _attention(q, k, v, key_mask):
    depth = q.shape[-1]
    q = q / np.sqrt(depth, dtype=q.dtype)
    w = np.einsum("qhd,khd->hqk", q, k)
    neg = np.finfo(q.dtype).min
    w = np.where(key_mask[None, None, :], w, neg)
    w = w - w.max(axis=-1, keepdims=True)
    w = np.exp(w)
    w = w / w.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", w, v)


def _attention_out(attn, kernel, bias):
    return np.einsum("thd,hde->te", attn, kernel) + bias


def _self_attn_block(x, key_mask, W, prefix):
    h = _layer_norm(x, W[f"{prefix}/ln1/scale"], W[f"{prefix}/ln1/bias"])
    q = _project_qkv(h, W[f"{prefix}/attn/query/kernel"], W[f"{prefix}/attn/query/bias"])
    k = _project_qkv(h, W[f"{prefix}/attn/key/kernel"], W[f"{prefix}/attn/key/bias"])
    v = _project_qkv(h, W[f"{prefix}/attn/value/kernel"], W[f"{prefix}/attn/value/bias"])
    attn = _attention(q, k, v, key_mask)
    out = _attention_out(attn, W[f"{prefix}/attn/out/kernel"], W[f"{prefix}/attn/out/bias"])
    x = x + out

    h = _layer_norm(x, W[f"{prefix}/ln2/scale"], W[f"{prefix}/ln2/bias"])
    h = _dense(h, W[f"{prefix}/mlp/fc1/kernel"], W[f"{prefix}/mlp/fc1/bias"])
    h = _gelu(h)
    h = _dense(h, W[f"{prefix}/mlp/fc2/kernel"], W[f"{prefix}/mlp/fc2/bias"])
    return x + h


def _encode_tokens(W, enc) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pf = enc["planet_feats"]
    pm = enc["planet_mask"]
    ff = enc["fleet_feats"]
    fm = enc["fleet_mask"]
    gf = enc["global_feats"]

    planet_tok = _dense(pf, W["encoder/planet_proj/kernel"], W["encoder/planet_proj/bias"])
    fleet_tok = _dense(ff, W["encoder/fleet_proj/kernel"], W["encoder/fleet_proj/bias"])
    global_tok = _dense(gf, W["encoder/global_proj/kernel"], W["encoder/global_proj/bias"])[None, :]

    te = W["encoder/type_embed"]
    global_tok = global_tok + te[0:1]
    planet_tok = planet_tok + te[1:2]
    fleet_tok = fleet_tok + te[2:3]

    tokens = np.concatenate([global_tok, planet_tok, fleet_tok], axis=0)
    key_mask = np.concatenate([np.ones((1,), dtype=bool), pm.astype(bool), fm.astype(bool)])

    for i in range(N_LAYERS):
        tokens = _self_attn_block(tokens, key_mask, W, prefix=f"encoder/block{i}")
    tokens = _layer_norm(tokens, W["encoder/ln_out/scale"], W["encoder/ln_out/bias"])

    P = pf.shape[0]
    global_emb = tokens[0]
    planet_emb = tokens[1:1 + P]
    fleet_emb = tokens[1 + P:]
    p_mask_f = pm.astype(np.float32)[:, None]
    planet_pool = (planet_emb * p_mask_f).sum(axis=0) / max(float(p_mask_f.sum()), 1.0)
    f_mask_f = fm.astype(np.float32)[:, None]
    fleet_pool = (fleet_emb * f_mask_f).sum(axis=0) / max(float(f_mask_f.sum()), 1.0)
    return global_emb, planet_emb, fleet_emb, planet_pool, fleet_pool


_NEG = -1e9


def _mask_logits(logits, mask):
    return np.where(mask, logits, _NEG).astype(np.float32)


def _src_head(W, planet_emb, my_mask, remaining_norm=None):
    if remaining_norm is not None:
        x = np.concatenate(
            [planet_emb, remaining_norm.astype(np.float32)[..., None]], axis=-1,
        )
    else:
        x = planet_emb
    x = _dense(x, W["src_head/fc1/kernel"], W["src_head/fc1/bias"])
    x = _gelu(x)
    logits = _dense(x, W["src_head/src_score/kernel"], W["src_head/src_score/bias"])[..., 0]
    return _mask_logits(logits, my_mask.astype(bool))


def _dst_head(W, planet_emb, src_emb, planet_mask, my_mask, reserved_norm=None,
              pair_feats=None, sun_block_mask=None):
    del my_mask  # owned planets allowed as dst (reinforcement); only mask padding
    q = _project_qkv(src_emb[None, :], W["dst_head/cross_attn/query/kernel"], W["dst_head/cross_attn/query/bias"])
    k = _project_qkv(planet_emb, W["dst_head/cross_attn/key/kernel"], W["dst_head/cross_attn/key/bias"])
    v = _project_qkv(planet_emb, W["dst_head/cross_attn/value/kernel"], W["dst_head/cross_attn/value/bias"])
    attended = _attention(q, k, v, planet_mask.astype(bool))
    cond = _attention_out(attended, W["dst_head/cross_attn/out/kernel"], W["dst_head/cross_attn/out/bias"])[0]
    extras = []
    if reserved_norm is not None:
        extras.append(reserved_norm.astype(np.float32)[..., None])
    if pair_feats is not None:
        extras.append(pair_feats.astype(np.float32))
    if extras:
        planet_rows = np.concatenate([planet_emb] + extras, axis=-1)
    else:
        planet_rows = planet_emb
    joined = np.concatenate(
        [planet_rows, np.broadcast_to(cond, (planet_rows.shape[0], cond.shape[0]))], axis=-1
    )
    x = _dense(joined, W["dst_head/dst_fc1/kernel"], W["dst_head/dst_fc1/bias"])
    x = _gelu(x)
    logits = _dense(x, W["dst_head/dst_score/kernel"], W["dst_head/dst_score/bias"])[..., 0]
    eff_mask = planet_mask.astype(bool)
    if sun_block_mask is not None:
        allowed = eff_mask & np.logical_not(sun_block_mask.astype(bool))
        if bool(allowed.any()):
            eff_mask = allowed
    return _mask_logits(logits, eff_mask)


def _pct_head(W, src_emb, dst_emb, global_emb, src_remaining_norm=None,
              pair_feats=None):
    feats = [src_emb, dst_emb, global_emb]
    if src_remaining_norm is not None:
        s = np.asarray(src_remaining_norm, dtype=np.float32).reshape(())
        feats.append(np.array([s], dtype=np.float32))
    if pair_feats is not None:
        feats.append(np.asarray(pair_feats, dtype=np.float32))
    px = np.concatenate(feats, axis=-1)
    px = _dense(px, W["pct_head/fc1/kernel"], W["pct_head/fc1/bias"])
    px = _gelu(px)
    return _dense(px, W["pct_head/logits/kernel"], W["pct_head/logits/bias"])


def _emit_head(W, global_emb, planet_pool, step_idx, max_steps=MAX_FLEETS_PER_TURN,
               total_remaining_norm=None, pair_feats_g=None):
    step_oh = np.zeros((max_steps,), dtype=np.float32)
    if 0 <= step_idx < max_steps:
        step_oh[step_idx] = 1.0
    feats = [global_emb, planet_pool, step_oh]
    if total_remaining_norm is not None:
        t = np.asarray(total_remaining_norm, dtype=np.float32).reshape(())
        feats.append(np.array([t], dtype=np.float32))
    if pair_feats_g is not None:
        feats.append(np.asarray(pair_feats_g, dtype=np.float32))
    x = np.concatenate(feats, axis=-1)
    x = _dense(x, W["emit_head/fc1/kernel"], W["emit_head/fc1/bias"])
    x = _gelu(x)
    return _dense(x, W["emit_head/logits/kernel"], W["emit_head/logits/bias"])


def greedy_multi_action(W, enc, home_idx: int = 0) -> Tuple[List[int], List[int], List[int]]:
    """Run the K-step greedy autoregressive policy with f26 pair features.

    Returns (src_list, dst_list, pct_list) of equal length, containing only
    the emitted launches.
    """
    global_emb, planet_emb, _f, planet_pool, _fp = _encode_tokens(W, enc)
    my_mask = enc["my_planet_mask"].astype(bool)
    planet_mask = enc["planet_mask"].astype(bool)
    target_mask = planet_mask & np.logical_not(my_mask)
    ships = enc["planet_ships"].astype(np.int32).copy()
    planet_x = enc["planet_x"].astype(np.float32)
    planet_y = enc["planet_y"].astype(np.float32)
    P = planet_emb.shape[0]
    reserved = np.zeros((P,), dtype=np.int32)
    still_emit = True

    ships_my_f = np.where(my_mask, ships, 0).astype(np.float32)
    total_init = float(ships_my_f.sum())
    home_init = float(ships_my_f[home_idx]) if 0 <= home_idx < P else 0.0

    src_out: List[int] = []
    dst_out: List[int] = []
    pct_out: List[int] = []

    for t in range(MAX_FLEETS_PER_TURN):
        remaining = ships - reserved
        avail_mask = my_mask & (remaining > 0)
        any_avail = bool(avail_mask.any())
        no_options = not any_avail
        eff_mask = avail_mask if any_avail else my_mask

        # v7: reserved-aware features (must match training: log1p(.)/8)
        remaining_clip = np.maximum(remaining, 0).astype(np.float32)
        reserved_f = np.maximum(reserved, 0).astype(np.float32)
        remaining_norm = np.log1p(remaining_clip) / 8.0
        reserved_norm = np.log1p(reserved_f) / 8.0
        total_remaining = float((remaining_clip * my_mask.astype(np.float32)).sum())
        total_remaining_norm = np.float32(np.log1p(total_remaining) / 8.0)

        # f26: emit pair globals (depend on reserved, not on chosen src/dst).
        emit_pair_g = _emit_pair_globals(
            planet_x, planet_y, ships, planet_mask, my_mask, target_mask,
            remaining, home_idx, home_init, total_init,
        )
        e_logits = _emit_head(
            W, global_emb, planet_pool, t,
            total_remaining_norm=total_remaining_norm,
            pair_feats_g=emit_pair_g,
        )
        if t == 0:
            decision = (not no_options)
        else:
            emit_pred = int(np.argmax(e_logits))
            decision = (emit_pred == 1) and (not no_options)
        emit_t = decision and still_emit

        s_logits = _src_head(W, planet_emb, eff_mask, remaining_norm)
        src_t = int(np.argmax(s_logits))
        src_emb = planet_emb[src_t]
        # f26: dst pair features depend on chosen src.
        dst_pair, sun_block = _dst_pair_features(
            planet_x, planet_y, ships, planet_mask, target_mask,
            remaining, src_t,
        )
        d_logits = _dst_head(
            W, planet_emb, src_emb, planet_mask, my_mask,
            reserved_norm=reserved_norm,
            pair_feats=dst_pair, sun_block_mask=sun_block,
        )
        d_logits[src_t] = _NEG
        dst_t = int(np.argmax(d_logits))
        dst_emb = planet_emb[dst_t]
        src_remaining_norm = float(remaining_norm[src_t])
        # f26: pct pair feats use chosen (src, dst).
        pct_pair = _pct_pair_features(float(ships[dst_t]), float(remaining[src_t]))
        p_logits = _pct_head(
            W, src_emb, dst_emb, global_emb,
            src_remaining_norm=src_remaining_norm,
            pair_feats=pct_pair,
        )
        pct_t = int(np.argmax(p_logits))

        if emit_t:
            avail_at_src = max(int(ships[src_t]) - int(reserved[src_t]), 0)
            mult = np.float32(avail_at_src) * _PCT_BIN_TABLE_NP[pct_t]
            ships_t = max(1, int(np.floor(mult)))
            ships_t = min(ships_t, avail_at_src)
            reserved[src_t] += ships_t
            src_out.append(src_t)
            dst_out.append(dst_t)
            pct_out.append(pct_t)
        else:
            still_emit = False

    return src_out, dst_out, pct_out


# --- action decoding to Kaggle format -----------------------------------------

def decode_multi_to_kaggle_moves(
    obs: Dict[str, Any],
    src_list: List[int],
    dst_list: List[int],
    pct_list: List[int],
    player: int,
) -> List[List[float]]:
    planets_by_id: Dict[int, Any] = {int(p[0]): p for p in (obs.get("planets") or [])}
    reserved: Dict[int, int] = {}
    moves: List[List[float]] = []
    for src_idx, dst_idx, pct_bin in zip(src_list, dst_list, pct_list):
        if src_idx not in planets_by_id or dst_idx not in planets_by_id or src_idx == dst_idx:
            continue
        src = planets_by_id[src_idx]
        dst = planets_by_id[dst_idx]
        if int(src[1]) != player:
            continue
        avail = int(src[5]) - reserved.get(src_idx, 0)
        if avail <= 0:
            continue
        pct_bin = max(0, min(NUM_PCT_BINS - 1, int(pct_bin)))
        pct = PCT_BIN_VALUES[pct_bin]
        ships_to_send = max(1, int(math.floor(avail * pct)))
        ships_to_send = min(ships_to_send, avail)
        angle = math.atan2(float(dst[3]) - float(src[3]), float(dst[2]) - float(src[2]))
        moves.append([int(src_idx), float(angle), int(ships_to_send)])
        reserved[src_idx] = reserved.get(src_idx, 0) + ships_to_send
    return moves


# --- weights (filled in by orbit_wars_rl/scripts/export_submission.py) --------

WEIGHTS_B64 = "__WEIGHTS_B64__"
_PARAMS_CACHE: Optional[Dict[str, np.ndarray]] = None


def _load_weights() -> Dict[str, np.ndarray]:
    global _PARAMS_CACHE
    if _PARAMS_CACHE is not None:
        return _PARAMS_CACHE
    if WEIGHTS_B64 == "__WEIGHTS_B64__":
        raise RuntimeError(
            "submission_rl_v11.py: WEIGHTS_B64 placeholder is not filled in. "
            "Run: python -m orbit_wars_rl.scripts.export_submission "
            "--template submission_rl_v11.py --ckpt <path>"
        )
    raw = zlib.decompress(base64.b64decode(WEIGHTS_B64))
    with np.load(io.BytesIO(raw)) as data:
        _PARAMS_CACHE = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    return _PARAMS_CACHE


# --- Kaggle entry -------------------------------------------------------------

_LAST_INITIAL_PLANET_KEY: Optional[Tuple] = None
_STEP_COUNTER: int = 0
_HOME_IDX: int = 0


def _detect_new_episode(obs: Dict[str, Any]) -> bool:
    global _LAST_INITIAL_PLANET_KEY
    ip = obs.get("initial_planets") or []
    key = tuple((int(p[0]), float(p[2]), float(p[3])) for p in ip[:8])
    if key != _LAST_INITIAL_PLANET_KEY:
        _LAST_INITIAL_PLANET_KEY = key
        return True
    return False


def _resolve_home_idx(obs: Dict[str, Any], player: int) -> int:
    """Locate this player's home planet slot. Prefer ``initial_planets`` (only
    populated on turn 0 but persistent) else fall back to whichever owned
    planet has the highest production at turn-start. Default 0 on failure.
    """
    for p in (obs.get("initial_planets") or []):
        if int(p[1]) == player:
            pid = int(p[0])
            if 0 <= pid < MAX_PLANETS:
                return pid
    # Fallback: highest-prod owned planet.
    best_pid = 0
    best_prod = -1
    for p in (obs.get("planets") or []):
        if int(p[1]) == player:
            pid = int(p[0])
            prod = int(p[6])
            if prod > best_prod and 0 <= pid < MAX_PLANETS:
                best_prod = prod
                best_pid = pid
    return best_pid


def agent(obs, config=None):
    """Kaggle-required entry. Returns list of [src_id, angle, ships] moves."""
    global _STEP_COUNTER, _HOME_IDX
    try:
        if _detect_new_episode(obs):
            _STEP_COUNTER = 0
            _HOME_IDX = _resolve_home_idx(obs, int(obs.get("player", 0)))
        else:
            _STEP_COUNTER += 1

        player = int(obs.get("player", 0))
        episode_steps = int((config or {}).get("episodeSteps", DEFAULT_EPISODE_STEPS))

        planets = list(obs.get("planets") or [])
        if not any(int(p[1]) == player for p in planets):
            return []

        W = _load_weights()
        enc = encode_obs(obs, player=player, step=_STEP_COUNTER, episode_steps=episode_steps)
        if not bool(enc["my_planet_mask"].any()):
            return []
        src_list, dst_list, pct_list = greedy_multi_action(W, enc, home_idx=_HOME_IDX)
        return decode_multi_to_kaggle_moves(obs, src_list, dst_list, pct_list, player=player)
    except Exception:
        return []
