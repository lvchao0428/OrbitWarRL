"""Numpy-only forward pass mirroring ``ActorCritic`` for a single observation.

This module imports **only numpy**, so the entire inference path can be inlined
into a Kaggle submission file without dragging jax/flax/optax along.

Conventions
-----------
- Single-observation mode (no batch dim). Shapes:
    planet_feats   (P, Fp)
    planet_mask    (P,)
    fleet_feats    (F, Ff)
    fleet_mask     (F,)
    global_feats   (Fg,)
    my_planet_mask (P,)
- All weights live in a flat dict ``W: Dict[str, np.ndarray]``. Keys follow the
  flax layout, see ``inference/weights.py``.
- Numerical layout matches the training-time ActorCritic:
    1. encoder: project + add type_embed; N pre-norm self-attn blocks; final ln.
    2. SrcHead: dense(1) over planet embeddings, masked by ``my_planet_mask``.
    3. DstHead: cross-attn with src embedding as query; gated by
       ``planet_mask & ~my_planet_mask``.
    4. PctHead: MLP over [src_emb || dst_emb || global_emb].
    5. ValueHead: MLP over [global_emb || planet_pool || fleet_pool].
- Tested against the flax model in ``test_parity.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from orbit_wars_rl.env import constants as _const


_LOGIT_NEG_INF = -1e9

# Pair-feature constants (mirror features/pair.py).
_SUN_X = float(_const.SUN_X)
_SUN_Y = float(_const.SUN_Y)
_SUN_R = float(_const.SUN_RADIUS)
_SUN_GUARD = _SUN_R * float(_const.SUN_PATH_MARGIN)
_SUN_BLOCK_THRESH = 0.9
_BOARD_F = float(_const.BOARD)
_MAX_PLANETS = int(_const.MAX_PLANETS)
_PCT_BIN_TABLE_NP = np.array(_const.PCT_BIN_VALUES, dtype=np.float32)


# --------- f26 pair feature helpers (numpy mirror of features/pair.py) -------


def _segment_min_dist_to_point_np(
    ax: float, ay: float,
    bx: np.ndarray, by: np.ndarray,
    cx: float, cy: float,
) -> np.ndarray:
    """Min distance from C=(cx,cy) to segment A=(ax,ay)->B=(bx,by). B is (P,)."""
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


def _dst_pair_features_np(
    planet_x: np.ndarray,        # (P,)
    planet_y: np.ndarray,        # (P,)
    planet_ships: np.ndarray,    # (P,) int
    planet_mask: np.ndarray,     # (P,) bool
    is_target_mask: np.ndarray,  # (P,) bool
    remaining: np.ndarray,       # (P,) int
    src_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    P = planet_x.shape[0]
    src_x = float(planet_x[src_idx])
    src_y = float(planet_y[src_idx])
    rem_src = float(max(int(remaining[src_idx]), 1))

    dx = planet_x - src_x
    dy = planet_y - src_y
    dist = np.sqrt(dx * dx + dy * dy + 1e-6)
    dist_norm = (dist / _BOARD_F).astype(np.float32)

    min_to_sun = _segment_min_dist_to_point_np(
        src_x, src_y, planet_x, planet_y, _SUN_X, _SUN_Y
    )
    self_pair = np.arange(P) == src_idx
    sun_risk = np.clip(1.0 - min_to_sun / _SUN_GUARD, 0.0, 1.0).astype(np.float32)
    sun_risk = np.where(self_pair, np.float32(0.0), sun_risk)

    garr_dst = planet_ships.astype(np.float32)
    ships_needed_norm = np.clip((garr_dst + 1.0) / rem_src, 0.0, 1.0).astype(np.float32)
    ships_at_bin5 = np.floor(rem_src * np.float32(0.7))
    pair_flip_bin5 = (
        (ships_at_bin5 > garr_dst).astype(np.float32) * is_target_mask.astype(np.float32)
    )
    margin = (ships_at_bin5 - garr_dst) / rem_src
    pair_margin_norm = np.clip(margin, 0.0, 1.0).astype(np.float32) * is_target_mask.astype(np.float32)

    pair_feats = np.stack(
        [dist_norm, sun_risk, ships_needed_norm, pair_flip_bin5, pair_margin_norm], axis=-1
    ).astype(np.float32)
    pair_feats *= planet_mask.astype(np.float32)[:, None]

    sun_block_mask = (sun_risk >= _SUN_BLOCK_THRESH) & planet_mask & np.logical_not(self_pair)
    return pair_feats, sun_block_mask


def _dst_flip_block_mask_np(
    planet_ships: np.ndarray,    # (P,) int
    planet_mask: np.ndarray,     # (P,) bool
    is_target_mask: np.ndarray,  # (P,) bool
    remaining: np.ndarray,       # (P,) int
    src_idx: int,
) -> np.ndarray:
    rem_src = float(max(int(remaining[src_idx]), 1))
    ships_at_bin5 = float(np.floor(rem_src * np.float32(0.7)))
    garr_dst = planet_ships.astype(np.float32)
    flip_ok = ships_at_bin5 > garr_dst
    block = np.logical_not(flip_ok) & is_target_mask.astype(bool) & planet_mask.astype(bool)
    return block


def _emit_pair_globals_np(
    planet_x: np.ndarray,
    planet_y: np.ndarray,
    planet_ships: np.ndarray,
    planet_mask: np.ndarray,
    my_mask: np.ndarray,
    target_mask: np.ndarray,
    remaining: np.ndarray,
    home_idx: int,
    home_init: float,
    total_init: float,
) -> np.ndarray:
    P = planet_x.shape[0]
    rem_my = np.where(my_mask, remaining, 0).astype(np.float32)
    ships_at_bin5_src = np.floor(rem_my * np.float32(0.7))     # (P,)
    garr_dst = planet_ships.astype(np.float32)                 # (P,)

    margin = ships_at_bin5_src[:, None] - garr_dst[None, :]    # (P, P)
    pair_valid = (
        my_mask[:, None] & target_mask[None, :]
        & planet_mask[:, None] & planet_mask[None, :]
    )
    feasible = pair_valid & (margin > 0)
    emit_worth_it = 1.0 if bool(feasible.any()) else 0.0

    margin_masked = np.where(feasible, margin, 0.0)
    best_margin = float(margin_masked.max(initial=0.0))
    best_margin = max(best_margin, 0.0)
    best_margin_norm = float(min(np.log1p(best_margin) / 8.0, 1.0))

    rem_at_home = float(rem_my[home_idx])
    home_remain_ratio = float(min(rem_at_home / max(home_init, 1.0), 1.0))
    total_remaining = float(rem_my.sum())
    total_remain_ratio = float(min(total_remaining / max(total_init, 1.0), 1.0))

    # [4] feasible_target_count: how many distinct dst planets can be flipped
    dst_flippable = feasible.any(axis=0)  # (P,) any src can flip this dst
    feasible_count = float(dst_flippable.astype(np.float32).sum())
    feasible_target_count_norm = float(min(feasible_count / _MAX_PLANETS, 1.0))

    # [5] surplus_ratio: after flipping all feasible targets cheaply, leftover
    target_need = (garr_dst + np.float32(1.0)) * target_mask.astype(np.float32)
    flippable_need = target_need * dst_flippable.astype(np.float32)
    sum_min_needs = float(flippable_need.sum())
    surplus_ratio = float(
        min(max((total_remaining - sum_min_needs) / max(total_remaining, 1.0), 0.0), 1.0)
    )

    return np.array(
        [emit_worth_it, best_margin_norm, home_remain_ratio, total_remain_ratio,
         feasible_target_count_norm, surplus_ratio],
        dtype=np.float32,
    )


def _pct_min_bin_index_np(garr_dst: float, remaining_src: float) -> int:
    rem = max(remaining_src, 1.0)
    ships_at_bins = np.floor(rem * _PCT_BIN_TABLE_NP)
    flip_at_bin = ships_at_bins > garr_dst
    if bool(flip_at_bin.any()):
        return int(np.argmax(flip_at_bin))
    return len(_PCT_BIN_TABLE_NP) - 1


def _pct_low_bin_mask_np(min_bin: int, num_bins: int = len(_PCT_BIN_TABLE_NP)) -> np.ndarray:
    return np.arange(num_bins, dtype=np.int32) >= min_bin


def _pct_pair_features_np(
    garr_dst: float,
    remaining_src: float,
    *,
    enemy_inbound_norm: float = 0.0,
    net_garrison_t15_dst: float = 0.0,
    src_prod_ratio: float = 0.0,
    fleet_count_norm: float = 0.0,
    extended: bool = False,
) -> np.ndarray:
    min_bin = _pct_min_bin_index_np(garr_dst, remaining_src)
    min_bin_norm = min_bin / (len(_PCT_BIN_TABLE_NP) - 1)
    rem = max(remaining_src, 1.0)
    ships_at_bin5 = float(np.floor(rem * 0.7))
    pair_flip_bin5 = 1.0 if ships_at_bin5 > garr_dst else 0.0
    if extended:
        return np.array([
            min_bin_norm, pair_flip_bin5,
            enemy_inbound_norm, net_garrison_t15_dst,
            src_prod_ratio, fleet_count_norm,
        ], dtype=np.float32)
    return np.array([min_bin_norm, pair_flip_bin5], dtype=np.float32)


def _layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """flax default LayerNorm: (x-mean)/sqrt(var+eps) * scale + bias along last dim."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    """flax default ``nn.gelu`` uses ``approximate=True`` (tanh formulation)."""
    c = np.float32(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * (x ** 3))))


def _dense(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b


def _mask_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, logits, _LOGIT_NEG_INF).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _project_qkv(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x: (T, D); kernel: (D, H, Dh); bias: (H, Dh). returns (T, H, Dh)."""
    return np.einsum("td,dhe->the", x, kernel) + bias  # bias broadcasts (1,H,Dh)


def _attention(
    q: np.ndarray,      # (Tq, H, Dh)
    k: np.ndarray,      # (Tk, H, Dh)
    v: np.ndarray,      # (Tk, H, Dh)
    key_mask: np.ndarray,  # (Tk,) bool, True = valid
) -> np.ndarray:
    """Dot-product attention. Returns (Tq, H, Dh)."""
    depth = q.shape[-1]
    q = q / np.sqrt(depth, dtype=q.dtype)
    # (Tq, H, Dh), (Tk, H, Dh) -> (H, Tq, Tk)
    w = np.einsum("qhd,khd->hqk", q, k)
    # mask shape (H, Tq, Tk); broadcast from (1, 1, Tk)
    neg = np.finfo(q.dtype).min
    w = np.where(key_mask[None, None, :], w, neg)
    w = _softmax(w, axis=-1)
    # (H, Tq, Tk), (Tk, H, Dh) -> (Tq, H, Dh)
    out = np.einsum("hqk,khd->qhd", w, v)
    return out


def _attention_out(attn: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """attn: (T, H, Dh); kernel: (H, Dh, D); bias: (D,) -> (T, D)."""
    return np.einsum("thd,hde->te", attn, kernel) + bias


def _self_attention_block(x: np.ndarray, key_mask: np.ndarray, W: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """One pre-norm self-attn block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""
    h = _layer_norm(x, W[f"{prefix}/ln1/scale"], W[f"{prefix}/ln1/bias"])
    q = _project_qkv(h, W[f"{prefix}/attn/query/kernel"], W[f"{prefix}/attn/query/bias"])
    k = _project_qkv(h, W[f"{prefix}/attn/key/kernel"], W[f"{prefix}/attn/key/bias"])
    v = _project_qkv(h, W[f"{prefix}/attn/value/kernel"], W[f"{prefix}/attn/value/bias"])
    attn = _attention(q, k, v, key_mask)
    attn_out = _attention_out(attn, W[f"{prefix}/attn/out/kernel"], W[f"{prefix}/attn/out/bias"])
    x = x + attn_out

    h = _layer_norm(x, W[f"{prefix}/ln2/scale"], W[f"{prefix}/ln2/bias"])
    h = _dense(h, W[f"{prefix}/mlp/fc1/kernel"], W[f"{prefix}/mlp/fc1/bias"])
    h = _gelu(h)
    h = _dense(h, W[f"{prefix}/mlp/fc2/kernel"], W[f"{prefix}/mlp/fc2/bias"])
    return x + h


def encode_tokens(
    W: Dict[str, np.ndarray],
    planet_feats: np.ndarray,
    planet_mask: np.ndarray,
    fleet_feats: np.ndarray,
    fleet_mask: np.ndarray,
    global_feats: np.ndarray,
    n_layers: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (global_emb, planet_emb, fleet_emb, planet_pool, fleet_pool)."""
    planet_tok = _dense(planet_feats, W["encoder/planet_proj/kernel"], W["encoder/planet_proj/bias"])
    fleet_tok = _dense(fleet_feats, W["encoder/fleet_proj/kernel"], W["encoder/fleet_proj/bias"])
    global_tok = _dense(global_feats, W["encoder/global_proj/kernel"], W["encoder/global_proj/bias"])[None, :]

    type_emb = W["encoder/type_embed"]  # (3, D)
    global_tok = global_tok + type_emb[0:1]
    planet_tok = planet_tok + type_emb[1:2]
    fleet_tok = fleet_tok + type_emb[2:3]

    P = planet_tok.shape[0]
    F = fleet_tok.shape[0]

    tokens = np.concatenate([global_tok, planet_tok, fleet_tok], axis=0)  # (1+P+F, D)
    key_mask = np.concatenate([np.ones((1,), dtype=bool), planet_mask.astype(bool), fleet_mask.astype(bool)])

    for i in range(n_layers):
        tokens = _self_attention_block(tokens, key_mask, W, prefix=f"encoder/block{i}")

    tokens = _layer_norm(tokens, W["encoder/ln_out/scale"], W["encoder/ln_out/bias"])

    global_emb = tokens[0]
    planet_emb = tokens[1:1 + P]
    fleet_emb = tokens[1 + P:]

    p_mask_f = planet_mask.astype(np.float32)[:, None]
    planet_pool = (planet_emb * p_mask_f).sum(axis=0) / max(float(p_mask_f.sum()), 1.0)
    f_mask_f = fleet_mask.astype(np.float32)[:, None]
    fleet_pool = (fleet_emb * f_mask_f).sum(axis=0) / max(float(f_mask_f.sum()), 1.0)

    return global_emb, planet_emb, fleet_emb, planet_pool, fleet_pool


def src_head(
    W: Dict[str, np.ndarray],
    planet_emb: np.ndarray,
    my_planet_mask: np.ndarray,
    remaining_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Returns src_logits of shape (P,) masked by my_planet_mask.

    v7: ``remaining_norm`` (P,) appended as a per-planet scalar feature.
    """
    if remaining_norm is not None:
        x = np.concatenate(
            [planet_emb, remaining_norm.astype(np.float32)[..., None]], axis=-1
        )
    else:
        x = planet_emb
    x = _dense(x, W["src_head/fc1/kernel"], W["src_head/fc1/bias"])
    x = _gelu(x)
    logits = _dense(x, W["src_head/src_score/kernel"], W["src_head/src_score/bias"])[..., 0]
    return _mask_logits(logits, my_planet_mask.astype(bool))


def dst_head(
    W: Dict[str, np.ndarray],
    planet_emb: np.ndarray,
    src_emb: np.ndarray,
    planet_mask: np.ndarray,
    my_planet_mask: np.ndarray,
    src_idx: int | None = None,
    reserved_norm: np.ndarray | None = None,
    pair_feats: np.ndarray | None = None,
    sun_block_mask: np.ndarray | None = None,
    flip_block_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-attention conditioned on src_emb. Returns dst_logits of shape (P,).

    v7: ``reserved_norm`` (P,) appended.
    f26: ``pair_feats`` (P, 4) appended; ``sun_block_mask`` (P,) hard-masks
    sun-crossing dst (falls back to standard mask when all candidates are
    blocked).
    """
    del my_planet_mask
    q = _project_qkv(src_emb[None, :], W["dst_head/cross_attn/query/kernel"], W["dst_head/cross_attn/query/bias"])
    k = _project_qkv(planet_emb, W["dst_head/cross_attn/key/kernel"], W["dst_head/cross_attn/key/bias"])
    v = _project_qkv(planet_emb, W["dst_head/cross_attn/value/kernel"], W["dst_head/cross_attn/value/bias"])
    attended = _attention(q, k, v, planet_mask.astype(bool))  # (1, H, Dh)
    cond = _attention_out(attended, W["dst_head/cross_attn/out/kernel"], W["dst_head/cross_attn/out/bias"])[0]

    P = planet_emb.shape[0]
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
        [planet_rows, np.broadcast_to(cond, (P, cond.shape[0]))], axis=-1
    )
    x = _dense(joined, W["dst_head/dst_fc1/kernel"], W["dst_head/dst_fc1/bias"])
    x = _gelu(x)
    logits = _dense(x, W["dst_head/dst_score/kernel"], W["dst_head/dst_score/bias"])[..., 0]

    eff_mask = planet_mask.astype(bool)
    if src_idx is not None:
        eff_mask = eff_mask.copy()
        eff_mask[src_idx] = False
    if sun_block_mask is not None:
        allowed = eff_mask & np.logical_not(sun_block_mask.astype(bool))
        if bool(allowed.any()):
            eff_mask = allowed
    if flip_block_mask is not None:
        allowed = eff_mask & np.logical_not(flip_block_mask.astype(bool))
        if bool(allowed.any()):
            eff_mask = allowed
    return _mask_logits(logits, eff_mask)


def pct_head(
    W: Dict[str, np.ndarray],
    src_emb: np.ndarray,
    dst_emb: np.ndarray,
    global_emb: np.ndarray,
    src_remaining_norm: float | np.ndarray | None = None,
    pair_feats: np.ndarray | None = None,
    pct_low_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    feats = [src_emb, dst_emb, global_emb]
    if src_remaining_norm is not None:
        s = np.asarray(src_remaining_norm, dtype=np.float32).reshape(())
        feats.append(np.array([s], dtype=np.float32))
    if pair_feats is not None:
        feats.append(np.asarray(pair_feats, dtype=np.float32))
    x = np.concatenate(feats, axis=-1)
    x = _dense(x, W["pct_head/fc1/kernel"], W["pct_head/fc1/bias"])
    x = _gelu(x)
    logits = _dense(x, W["pct_head/logits/kernel"], W["pct_head/logits/bias"])
    if pct_low_bin_mask is not None:
        allowed = pct_low_bin_mask.astype(bool)
        if bool(allowed.any()):
            logits = _mask_logits(logits, allowed)
    return logits


def emit_head(
    W: Dict[str, np.ndarray],
    global_emb: np.ndarray,
    planet_pool: np.ndarray,
    step_idx: int,
    max_steps: int = 8,
    total_remaining_norm: float | np.ndarray | None = None,
    pair_feats_g: np.ndarray | None = None,
    emit_force_stop: bool = False,
) -> np.ndarray:
    """Returns emit_logits of shape (2,). [stop, continue]."""
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
    logits = _dense(x, W["emit_head/logits/kernel"], W["emit_head/logits/bias"])
    if emit_force_stop:
        logits = logits.copy()
        logits[1] = _LOGIT_NEG_INF
    return logits


def value_head(
    W: Dict[str, np.ndarray],
    global_emb: np.ndarray,
    planet_emb: np.ndarray,
    planet_mask: np.ndarray,
    fleet_emb: np.ndarray,
    fleet_mask: np.ndarray,
) -> float:
    """Multi-query cross-attention ValueHead (v6+).

    Mirrors heads.py::ValueHead exactly. N learned query tokens (conditioned
    on global state) cross-attend over per-entity planet+fleet embeddings,
    then concat all attended queries + global_emb -> MLP -> scalar.
    """
    queries = W["value_head/queries"]  # (Q, D)
    n_queries = queries.shape[0]
    g_proj = _dense(global_emb, W["value_head/q_cond/kernel"], W["value_head/q_cond/bias"])  # (D,)
    q_tokens = queries + g_proj[None, :]  # (Q, D)

    kv = np.concatenate([planet_emb, fleet_emb], axis=0)  # (P+F, D)
    kv_mask = np.concatenate([planet_mask.astype(bool), fleet_mask.astype(bool)], axis=0)

    q = _project_qkv(q_tokens, W["value_head/value_attn/query/kernel"], W["value_head/value_attn/query/bias"])
    k = _project_qkv(kv, W["value_head/value_attn/key/kernel"], W["value_head/value_attn/key/bias"])
    v = _project_qkv(kv, W["value_head/value_attn/value/kernel"], W["value_head/value_attn/value/bias"])
    attended = _attention(q, k, v, kv_mask)  # (Q, H, Dh)
    pooled = _attention_out(attended, W["value_head/value_attn/out/kernel"], W["value_head/value_attn/out/bias"])  # (Q, D)
    pooled = pooled.reshape(-1)  # (Q*D,)

    joined = np.concatenate([pooled, global_emb], axis=-1)
    x = _dense(joined, W["value_head/fc1/kernel"], W["value_head/fc1/bias"])
    x = _gelu(x)
    out = _dense(x, W["value_head/value/kernel"], W["value_head/value/bias"])
    return float(out[..., 0])


def forward(
    W: Dict[str, np.ndarray],
    planet_feats: np.ndarray,
    planet_mask: np.ndarray,
    fleet_feats: np.ndarray,
    fleet_mask: np.ndarray,
    global_feats: np.ndarray,
    my_planet_mask: np.ndarray,
    n_layers: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns (src_logits, dst_logits_given_argmax_src, pct_logits_given_argmax, value).

    Legacy single-action forward (no pair feats). Used only by old tests
    that don't have geometry handy. Real inference goes through
    ``greedy_multi_action``.
    """
    global_emb, planet_emb, fleet_emb, planet_pool, fleet_pool = encode_tokens(
        W, planet_feats, planet_mask, fleet_feats, fleet_mask, global_feats, n_layers=n_layers,
    )
    s_logits = src_head(W, planet_emb, my_planet_mask)
    src_idx = int(np.argmax(s_logits))
    src_emb = planet_emb[src_idx]
    d_logits = dst_head(W, planet_emb, src_emb, planet_mask, my_planet_mask)
    d_logits = d_logits.copy()
    d_logits[src_idx] = -1e9
    dst_idx = int(np.argmax(d_logits))
    dst_emb = planet_emb[dst_idx]
    p_logits = pct_head(W, src_emb, dst_emb, global_emb)
    v = value_head(W, global_emb, planet_emb, planet_mask, fleet_emb, fleet_mask)
    return s_logits, d_logits, p_logits, v


def greedy_action(
    W: Dict[str, np.ndarray],
    planet_feats: np.ndarray,
    planet_mask: np.ndarray,
    fleet_feats: np.ndarray,
    fleet_mask: np.ndarray,
    global_feats: np.ndarray,
    my_planet_mask: np.ndarray,
    n_layers: int = 2,
) -> Tuple[int, int, int, float]:
    """Legacy single-action greedy. Argmax src/dst/pct + value.

    v7: forwards zero remaining/reserved features (since this single-action
    path has no K-step state). Matches training's t=0 with fresh reserved=0.
    """
    global_emb, planet_emb, fleet_emb, planet_pool, fleet_pool = encode_tokens(
        W, planet_feats, planet_mask, fleet_feats, fleet_mask, global_feats, n_layers=n_layers,
    )
    P = planet_emb.shape[0]
    # No ship state available here, so emulate "start of turn" (all-zero
    # reserved). For src/dst we pass log1p(planet_ships)/8 if we have it;
    # without ships info we pass zero-like masks. v7 networks pass these
    # explicitly elsewhere; legacy code paths only used for tests.
    zero_per_planet = np.zeros((P,), dtype=np.float32)
    s_logits = src_head(W, planet_emb, my_planet_mask, zero_per_planet)
    src_idx = int(np.argmax(s_logits))
    src_emb = planet_emb[src_idx]
    d_logits = dst_head(
        W, planet_emb, src_emb, planet_mask, my_planet_mask,
        reserved_norm=zero_per_planet,
    )
    d_logits = d_logits.copy()
    d_logits[src_idx] = -1e9
    dst_idx = int(np.argmax(d_logits))
    dst_emb = planet_emb[dst_idx]
    p_logits = pct_head(W, src_emb, dst_emb, global_emb, src_remaining_norm=0.0)
    pct_idx = int(np.argmax(p_logits))
    v = value_head(W, global_emb, planet_emb, planet_mask, fleet_emb, fleet_mask)
    return src_idx, dst_idx, pct_idx, v


def greedy_multi_action(
    W: Dict[str, np.ndarray],
    planet_feats: np.ndarray,
    planet_mask: np.ndarray,
    fleet_feats: np.ndarray,
    fleet_mask: np.ndarray,
    global_feats: np.ndarray,
    my_planet_mask: np.ndarray,
    planet_ships: np.ndarray,    # (P,) int -- raw garrison
    n_layers: int = 2,
    max_fleets_per_turn: int = 8,
    planet_x: np.ndarray | None = None,   # (P,) float -- f26 pair feats
    planet_y: np.ndarray | None = None,
    home_idx: int = 0,
    emit_hard_stop: bool = False,
    emit_hard_stop_min_step: int = 1,
    flip_hard_mask: bool = False,
    allow_hold: bool = False,
    force_emit_worth_it: bool = False,
    min_pct_bin: int = 0,
    pct_pair_dim: int = 2,
    planet_prod: np.ndarray | None = None,   # (P,) float -- production rate
    in_foe_norm: np.ndarray | None = None,   # (P,) float -- enemy inbound (log-normalised)
    net_garrison_t15: np.ndarray | None = None,  # (P,) float -- predicted garrison balance t+15
) -> Tuple[list, list, list, list, float]:
    """Greedy multi-fleet action mirroring ActorCritic.__call__(deterministic=True).

    f26: when ``planet_x``/``planet_y`` are provided, pair features (dst geometry,
    emit budget, pct pair) are passed to the heads and sun-mask is enforced.
    If they are ``None``, the legacy path runs without pair signals (used only
    by pre-f26 ckpts; PPO/replay must supply geometry).
    """
    global_emb, planet_emb, fleet_emb, planet_pool, fleet_pool = encode_tokens(
        W, planet_feats, planet_mask, fleet_feats, fleet_mask, global_feats, n_layers=n_layers,
    )
    P = planet_emb.shape[0]
    ships = planet_ships.astype(np.int32).copy()
    reserved = np.zeros((P,), dtype=np.int32)
    still_emit = True

    src_out, dst_out, pct_out, emit_out = [], [], [], []
    my_mask_b = my_planet_mask.astype(bool)
    planet_mask_b = planet_mask.astype(bool)
    target_mask_b = planet_mask_b & np.logical_not(my_mask_b)
    use_pair = (planet_x is not None) and (planet_y is not None)
    if use_pair:
        ships_my_f = np.where(my_mask_b, ships, 0).astype(np.float32)
        total_init = float(ships_my_f.sum())
        home_init = float(ships_my_f[home_idx]) if 0 <= home_idx < P else 0.0
    else:
        ships_my_f = None
        total_init = 0.0
        home_init = 0.0

    for t in range(max_fleets_per_turn):
        remaining = ships - reserved
        avail_mask = my_mask_b & (remaining > 0)
        any_avail = bool(avail_mask.any())
        no_options = not any_avail
        eff_mask = avail_mask if any_avail else my_mask_b

        # v7: compute reserved-aware features for this step.
        remaining_clip = np.maximum(remaining, 0).astype(np.float32)
        reserved_f = np.maximum(reserved, 0).astype(np.float32)
        remaining_norm = np.log1p(remaining_clip) / 8.0
        reserved_norm = np.log1p(reserved_f) / 8.0
        total_remaining = float((remaining_clip * my_mask_b.astype(np.float32)).sum())
        total_remaining_norm = np.float32(np.log1p(total_remaining) / 8.0)

        emit_pair_g = None
        if use_pair:
            emit_pair_g = _emit_pair_globals_np(
                planet_x, planet_y, ships, planet_mask_b,
                my_mask_b, target_mask_b, remaining,
                home_idx, home_init, total_init,
            )
        emit_worth_it = (emit_pair_g is not None) and (float(emit_pair_g[0]) > 0.0)
        emit_force_stop = (
            emit_hard_stop
            and (t >= emit_hard_stop_min_step)
            and (not emit_worth_it)
        )
        e_logits = emit_head(
            W, global_emb, planet_pool, t,
            max_steps=max_fleets_per_turn,
            total_remaining_norm=total_remaining_norm,
            pair_feats_g=emit_pair_g,
            emit_force_stop=emit_force_stop,
        )
        force_first = (
            t == 0
            and (not no_options)
            and (
                (not allow_hold)
                or (force_emit_worth_it and emit_worth_it)
            )
        )
        if force_first:
            decision = True
        else:
            emit_pred = int(np.argmax(e_logits))
            decision = (emit_pred == 1) and (not no_options)
        emit_t = decision and still_emit

        s_logits = src_head(W, planet_emb, eff_mask, remaining_norm)
        src_t = int(np.argmax(s_logits))
        src_emb = planet_emb[src_t]

        dst_pair = None
        sun_block = None
        flip_block = None
        if use_pair:
            dst_pair, sun_block = _dst_pair_features_np(
                planet_x, planet_y, ships, planet_mask_b,
                target_mask_b, remaining, src_t,
            )
            if flip_hard_mask:
                flip_block = _dst_flip_block_mask_np(
                    ships, planet_mask_b, target_mask_b, remaining, src_t,
                )
        d_logits = dst_head(
            W, planet_emb, src_emb, planet_mask, my_planet_mask,
            src_idx=src_t,
            reserved_norm=reserved_norm,
            pair_feats=dst_pair, sun_block_mask=sun_block,
            flip_block_mask=flip_block,
        )
        dst_t = int(np.argmax(d_logits))
        dst_emb = planet_emb[dst_t]
        src_remaining_norm = float(remaining_norm[src_t])

        pct_pair = None
        pct_mask = None
        if use_pair:
            garr = float(ships[dst_t])
            rem = float(remaining[src_t])
            ext_kwargs: dict = {}
            if pct_pair_dim >= 6:
                _ein = float(in_foe_norm[dst_t]) if in_foe_norm is not None else 0.0
                _ngt = float(net_garrison_t15[dst_t]) if net_garrison_t15 is not None else 0.0
                _sp = 0.0
                if planet_prod is not None:
                    my_prod_total = float((planet_prod * my_mask_b.astype(np.float32)).sum())
                    _sp = float(planet_prod[src_t]) / max(my_prod_total, 1e-6)
                _fc = float(t) / max(max_fleets_per_turn - 1, 1)
                ext_kwargs = dict(
                    enemy_inbound_norm=_ein,
                    net_garrison_t15_dst=_ngt,
                    src_prod_ratio=_sp,
                    fleet_count_norm=_fc,
                    extended=True,
                )
            pct_pair = _pct_pair_features_np(garr, rem, **ext_kwargs)
            min_bin = _pct_min_bin_index_np(garr, rem)
            if min_pct_bin > 0:
                min_bin = max(min_bin, int(min_pct_bin))
            pct_mask = _pct_low_bin_mask_np(min_bin)
        p_logits = pct_head(
            W, src_emb, dst_emb, global_emb,
            src_remaining_norm=src_remaining_norm,
            pair_feats=pct_pair,
            pct_low_bin_mask=pct_mask,
        )
        pct_t = int(np.argmax(p_logits))

        if emit_t:
            avail_at_src = max(int(ships[src_t]) - int(reserved[src_t]), 0)
            # IMPORTANT: use float32 multiplication to match JAX's _ships_to_send
            # logic exactly.
            mult = np.float32(avail_at_src) * _PCT_BIN_TABLE_NP[pct_t]
            ships_t = max(1, int(np.floor(mult)))
            ships_t = min(ships_t, avail_at_src)
            reserved[src_t] += ships_t
            src_out.append(src_t)
            dst_out.append(dst_t)
            pct_out.append(pct_t)
            emit_out.append(True)
        else:
            emit_out.append(False)
            still_emit = False

    v = value_head(W, global_emb, planet_emb, planet_mask, fleet_emb, fleet_mask)
    return src_out, dst_out, pct_out, emit_out, v
