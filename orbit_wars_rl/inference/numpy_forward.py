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


_LOGIT_NEG_INF = -1e9


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


def src_head(W: Dict[str, np.ndarray], planet_emb: np.ndarray, my_planet_mask: np.ndarray) -> np.ndarray:
    """Returns src_logits of shape (P,) masked by my_planet_mask.

    Mirrors the two-layer MLP SrcHead (v6+): fc1 -> gelu -> src_score.
    """
    x = _dense(planet_emb, W["src_head/fc1/kernel"], W["src_head/fc1/bias"])
    x = _gelu(x)
    logits = _dense(x, W["src_head/src_score/kernel"], W["src_head/src_score/bias"])[..., 0]
    return _mask_logits(logits, my_planet_mask.astype(bool))


def dst_head(
    W: Dict[str, np.ndarray],
    planet_emb: np.ndarray,
    src_emb: np.ndarray,
    planet_mask: np.ndarray,
    my_planet_mask: np.ndarray,
) -> np.ndarray:
    """Cross-attention conditioned on src_emb. Returns dst_logits of shape (P,).

    Any valid planet is a legal target (including own planets -- the game
    treats same-owner arrivals as reinforcements). ``my_planet_mask`` is
    accepted for signature compatibility but ignored.
    """
    del my_planet_mask
    q = _project_qkv(src_emb[None, :], W["dst_head/cross_attn/query/kernel"], W["dst_head/cross_attn/query/bias"])
    k = _project_qkv(planet_emb, W["dst_head/cross_attn/key/kernel"], W["dst_head/cross_attn/key/bias"])
    v = _project_qkv(planet_emb, W["dst_head/cross_attn/value/kernel"], W["dst_head/cross_attn/value/bias"])
    attended = _attention(q, k, v, planet_mask.astype(bool))  # (1, H, Dh)
    cond = _attention_out(attended, W["dst_head/cross_attn/out/kernel"], W["dst_head/cross_attn/out/bias"])[0]

    P = planet_emb.shape[0]
    joined = np.concatenate([planet_emb, np.broadcast_to(cond, (P, cond.shape[0]))], axis=-1)
    x = _dense(joined, W["dst_head/dst_fc1/kernel"], W["dst_head/dst_fc1/bias"])
    x = _gelu(x)
    logits = _dense(x, W["dst_head/dst_score/kernel"], W["dst_head/dst_score/bias"])[..., 0]

    return _mask_logits(logits, planet_mask.astype(bool))


def pct_head(
    W: Dict[str, np.ndarray],
    src_emb: np.ndarray,
    dst_emb: np.ndarray,
    global_emb: np.ndarray,
) -> np.ndarray:
    x = np.concatenate([src_emb, dst_emb, global_emb], axis=-1)
    x = _dense(x, W["pct_head/fc1/kernel"], W["pct_head/fc1/bias"])
    x = _gelu(x)
    return _dense(x, W["pct_head/logits/kernel"], W["pct_head/logits/bias"])


def emit_head(
    W: Dict[str, np.ndarray],
    global_emb: np.ndarray,
    planet_pool: np.ndarray,
    step_idx: int,
    max_steps: int = 8,
) -> np.ndarray:
    """Returns emit_logits of shape (2,). [stop, continue]."""
    step_oh = np.zeros((max_steps,), dtype=np.float32)
    if 0 <= step_idx < max_steps:
        step_oh[step_idx] = 1.0
    x = np.concatenate([global_emb, planet_pool, step_oh], axis=-1)
    x = _dense(x, W["emit_head/fc1/kernel"], W["emit_head/fc1/bias"])
    x = _gelu(x)
    return _dense(x, W["emit_head/logits/kernel"], W["emit_head/logits/bias"])


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

    DstHead/PctHead depend on the chosen src; this convenience returns the
    *argmax-conditioned* logits for deterministic Kaggle inference. For
    parity tests we expose ``encode_tokens``/``src_head``/``dst_head``/``pct_head``
    individually so callers can pass arbitrary src/dst.
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
    """Legacy single-action greedy. Argmax src/dst/pct + value."""
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
    pct_idx = int(np.argmax(p_logits))
    v = value_head(W, global_emb, planet_emb, planet_mask, fleet_emb, fleet_mask)
    return src_idx, dst_idx, pct_idx, v


_PCT_BIN_TABLE_NP = np.array(
    [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00], dtype=np.float32
)


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
) -> Tuple[list, list, list, list, float]:
    """Greedy multi-fleet action mirroring ActorCritic.__call__(deterministic=True).

    Returns:
      src_list, dst_list, pct_list -- ints (only emit==True steps included)
      emit_list -- bool list per step (length K, but trimmed list above only
                   holds the emitted ones)
      value
    """
    global_emb, planet_emb, fleet_emb, planet_pool, fleet_pool = encode_tokens(
        W, planet_feats, planet_mask, fleet_feats, fleet_mask, global_feats, n_layers=n_layers,
    )
    P = planet_emb.shape[0]
    ships = planet_ships.astype(np.int32).copy()
    reserved = np.zeros((P,), dtype=np.int32)
    still_emit = True

    src_out, dst_out, pct_out, emit_out = [], [], [], []
    for t in range(max_fleets_per_turn):
        remaining = ships - reserved
        avail_mask = my_planet_mask.astype(bool) & (remaining > 0)
        any_avail = bool(avail_mask.any())
        no_options = not any_avail
        eff_mask = avail_mask if any_avail else my_planet_mask.astype(bool)

        e_logits = emit_head(W, global_emb, planet_pool, t, max_steps=max_fleets_per_turn)
        if t == 0:
            decision = (not no_options)
        else:
            emit_pred = int(np.argmax(e_logits))
            decision = (emit_pred == 1) and (not no_options)
        emit_t = decision and still_emit

        s_logits = src_head(W, planet_emb, eff_mask)
        src_t = int(np.argmax(s_logits))
        src_emb = planet_emb[src_t]
        d_logits = dst_head(W, planet_emb, src_emb, planet_mask, my_planet_mask)
        d_logits = d_logits.copy()
        d_logits[src_t] = -1e9
        dst_t = int(np.argmax(d_logits))
        dst_emb = planet_emb[dst_t]
        p_logits = pct_head(W, src_emb, dst_emb, global_emb)
        pct_t = int(np.argmax(p_logits))

        if emit_t:
            avail_at_src = max(int(ships[src_t]) - int(reserved[src_t]), 0)
            pct = float(_PCT_BIN_TABLE_NP[pct_t])
            ships_t = max(1, int(np.floor(avail_at_src * pct)))
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
