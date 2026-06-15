"""Load a flax pickle checkpoint and flatten the param tree into ``Dict[str, ndarray]``.

The training side uses ``runner.save_checkpoint`` which pickles a dict with
``params``, ``opt_state``, ``step``. Only ``params`` is used here. The param
tree mirrors the flax module layout (see ``ActorCritic.setup``):

  encoder/{type_embed, planet_proj, fleet_proj, global_proj,
           block0..block{N-1}/{ln1, attn/{query,key,value,out}, ln2, mlp/{fc1,fc2}},
           ln_out}
  src_head/src_score
  dst_head/{cross_attn/{query,key,value,out}, dst_score}
  pct_head/{fc1, logits}
  value_head/{fc1, value}

Keys are flattened with ``/`` separators so they round-trip through numpy ``.npz``.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Iterable

import numpy as np


def _flatten(tree: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            child_prefix = f"{prefix}{k}/" if isinstance(v, dict) else f"{prefix}{k}"
            out.update(_flatten(v, child_prefix))
        return out
    arr = np.asarray(tree)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    out[prefix.rstrip("/")] = arr
    return out


def flatten_params(params: Any) -> Dict[str, np.ndarray]:
    """Take a flax-style param pytree (dict[str, ...]) and return ``{path: ndarray}``.

    Strips the outer ``params`` key if present (``model.init`` wraps in it).
    """
    if isinstance(params, dict) and set(params.keys()) == {"params"}:
        params = params["params"]
    return _flatten(params)


def load_flat_params(path: str) -> Dict[str, np.ndarray]:
    """Load ``ckpt_XXXXXX.pkl`` produced by ``runner.save_checkpoint`` and flatten it."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"{path}: expected dict with 'params' key, got {type(payload)}")
    return flatten_params(payload["params"])


def save_npz(path: str, flat: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **flat)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def infer_arch_from_flat(flat: Dict[str, np.ndarray]) -> dict[str, int]:
    """Infer model dims from a flattened ckpt (for parity / export).

    Returns d_model, n_layers, n_heads, ff_dim, planet_feat_dim, global_feat_dim,
    max_fleets_per_turn.
    """
    proj_key = "encoder/planet_proj/kernel"
    if proj_key not in flat:
        raise ValueError(f"missing {proj_key} in ckpt; cannot infer arch")
    planet_feat_dim = int(flat[proj_key].shape[0])
    d_model = int(flat[proj_key].shape[-1])

    glob_key = "encoder/global_proj/kernel"
    if glob_key not in flat:
        raise ValueError(f"missing {glob_key} in ckpt")
    global_feat_dim = int(flat[glob_key].shape[0])

    n_layers = 0
    while f"encoder/block{n_layers}/ln1/scale" in flat:
        n_layers += 1
    if n_layers == 0:
        raise ValueError("no encoder/block* found in ckpt")

    qk = "encoder/block0/attn/query/kernel"
    if qk in flat and flat[qk].ndim == 3:
        n_heads = int(flat[qk].shape[1])
    else:
        n_heads = 4

    fc1_key = "encoder/block0/mlp/fc1/kernel"
    ff_dim = int(flat[fc1_key].shape[-1]) if fc1_key in flat else 128

    emit_key = "emit_head/fc1/kernel"
    if emit_key not in flat:
        raise ValueError(f"missing {emit_key} in ckpt")
    emit_in = int(flat[emit_key].shape[0])
    # emit input layouts:
    #   legacy (pre-f26):  2*d_model + K + 1 (total_remaining_norm)
    #   f26-f36:           2*d_model + K + 1 + 4 (pair_feats_g, 4-dim)
    #   f37+:              2*d_model + K + 1 + 6 (pair_feats_g, 6-dim)
    #   v21:               2*d_model + K + 1 + 11 (pair_feats_g, 11-dim)
    # Try each interpretation; pick the first that yields K in [1, 16].
    k_legacy = emit_in - 2 * d_model - 1
    k_f26 = emit_in - 2 * d_model - 5
    k_f37 = emit_in - 2 * d_model - 7
    k_v30c = emit_in - 2 * d_model - 13
    k_v21 = emit_in - 2 * d_model - 12
    has_emit_pair = False
    emit_pair_dim = 0
    if 1 <= k_v30c <= 16:
        max_fleets_per_turn = k_v30c
        has_emit_pair = True
        emit_pair_dim = 12
    elif 1 <= k_v21 <= 16:
        max_fleets_per_turn = k_v21
        has_emit_pair = True
        emit_pair_dim = 11
    elif 1 <= k_f37 <= 16:
        max_fleets_per_turn = k_f37
        has_emit_pair = True
        emit_pair_dim = 6
    elif 1 <= k_f26 <= 16:
        max_fleets_per_turn = k_f26
        has_emit_pair = True
        emit_pair_dim = 4
    elif 1 <= k_legacy <= 32:
        max_fleets_per_turn = k_legacy
    else:
        raise ValueError(
            f"cannot infer max_fleets_per_turn from {emit_key} shape "
            f"{flat[emit_key].shape} d_model={d_model} "
            f"(legacy={k_legacy}, f26={k_f26}, f37={k_f37}, v21={k_v21})"
        )

    # Detect dst pair feats and pct pair feats via fc1 input shape.
    # dst_fc1 input layout:
    #   legacy:  (d_model + 1[reserved]) + d_model  -- two parts concatenated
    #   f26:     (d_model + 1[reserved] + 4[pair]) + d_model
    has_dst_pair = False
    dst_pair_dim = 0
    has_f29_dst = False
    dst_key = "dst_head/dst_fc1/kernel"
    if dst_key in flat:
        dst_in = int(flat[dst_key].shape[0])
        if dst_in == 2 * d_model + 8:
            has_dst_pair = True
            has_f29_dst = True
            dst_pair_dim = 7
        elif dst_in == 2 * d_model + 7:
            has_dst_pair = True
            has_f29_dst = True
            dst_pair_dim = 6
        elif dst_in == 2 * d_model + 6:
            has_dst_pair = True
            has_f29_dst = True
            dst_pair_dim = 5
        elif dst_in == 2 * d_model + 5:
            has_dst_pair = True
            dst_pair_dim = 4
        elif dst_in == 2 * d_model + 1:
            has_dst_pair = False
            dst_pair_dim = 0
        else:
            raise ValueError(
                f"unexpected dst_fc1 input dim={dst_in}; "
                f"d_model={d_model}, expected 2*d_model+1, +5 (f26), +6 (f29), +7 (v29), or +8 (v30c)"
            )

    # pct_head/fc1 input layout:
    #   legacy: 3*d_model + 1
    #   f26:    3*d_model + 1 + 2  (pct_pair_dim=2)
    #   v14:    3*d_model + 1 + 6  (pct_pair_dim=6, extended pair feats)
    #   v21:    3*d_model + 1 + 7  (pct_pair_dim=7, +lead_dist_norm)
    has_pct_pair = False
    pct_pair_dim = 0
    pct_key = "pct_head/fc1/kernel"
    if pct_key in flat:
        pct_in = int(flat[pct_key].shape[0])
        if pct_in == 3 * d_model + 8:
            has_pct_pair = True
            pct_pair_dim = 7
        elif pct_in == 3 * d_model + 7:
            has_pct_pair = True
            pct_pair_dim = 6
        elif pct_in == 3 * d_model + 3:
            has_pct_pair = True
            pct_pair_dim = 2
        elif pct_in == 3 * d_model + 1:
            has_pct_pair = False
            pct_pair_dim = 0
        else:
            raise ValueError(
                f"unexpected pct_head/fc1 input dim={pct_in}; "
                f"d_model={d_model}, expected 3*d_model+1, +3, +7, or +8"
            )

    has_pair = has_emit_pair and has_dst_pair and has_pct_pair

    # Infer NUM_PCT_BINS from pct_head/logits output dim.
    pct_logits_key = "pct_head/logits/kernel"
    num_pct_bins = int(flat[pct_logits_key].shape[-1]) if pct_logits_key in flat else 8

    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ff_dim": ff_dim,
        "planet_feat_dim": planet_feat_dim,
        "global_feat_dim": global_feat_dim,
        "max_fleets_per_turn": max_fleets_per_turn,
        "has_pair": has_pair,
        "has_emit_pair": has_emit_pair,
        "has_dst_pair": has_dst_pair,
        "has_f29_dst": has_f29_dst,
        "dst_pair_dim": dst_pair_dim,
        "emit_pair_dim": emit_pair_dim,
        "has_pct_pair": has_pct_pair,
        "pct_pair_dim": pct_pair_dim,
        "num_pct_bins": num_pct_bins,
        "has_econ_dst": "dst_economics_head/fc1/kernel" in flat,
        "econ_dst_dim": (
            int(flat["dst_economics_head/fc1/kernel"].shape[0])
            if "dst_economics_head/fc1/kernel" in flat
            else 0
        ),
    }


def assert_expected_keys(flat: Dict[str, np.ndarray], n_layers: int = 2) -> None:
    """Sanity check that ``flat`` contains every key the numpy forward needs.

    Schema (v6+):
      * encoder: planet/fleet/global proj + type_embed + N pre-norm blocks + ln_out
      * src_head: two-layer MLP (fc1 -> src_score)
      * dst_head: cross_attn(qkv,out) + two-layer MLP (dst_fc1 -> dst_score)
      * pct_head: fc1 -> logits
      * emit_head: fc1 -> logits
      * value_head: multi-query attention -- queries param + q_cond + value_attn(qkv,out)
        + fc1 + value
    """
    expected: set[str] = set()
    expected.add("encoder/type_embed")
    for proj in ("planet_proj", "fleet_proj", "global_proj"):
        expected.add(f"encoder/{proj}/kernel")
        expected.add(f"encoder/{proj}/bias")
    for i in range(n_layers):
        for ln in ("ln1", "ln2"):
            expected.add(f"encoder/block{i}/{ln}/scale")
            expected.add(f"encoder/block{i}/{ln}/bias")
        for qkv in ("query", "key", "value", "out"):
            expected.add(f"encoder/block{i}/attn/{qkv}/kernel")
            expected.add(f"encoder/block{i}/attn/{qkv}/bias")
        for fc in ("fc1", "fc2"):
            expected.add(f"encoder/block{i}/mlp/{fc}/kernel")
            expected.add(f"encoder/block{i}/mlp/{fc}/bias")
    expected.add("encoder/ln_out/scale")
    expected.add("encoder/ln_out/bias")

    # SrcHead: fc1 -> src_score
    for fc in ("fc1", "src_score"):
        expected.add(f"src_head/{fc}/kernel")
        expected.add(f"src_head/{fc}/bias")

    # DstHead: cross_attn + dst_fc1 -> dst_score
    for qkv in ("query", "key", "value", "out"):
        expected.add(f"dst_head/cross_attn/{qkv}/kernel")
        expected.add(f"dst_head/cross_attn/{qkv}/bias")
    for fc in ("dst_fc1", "dst_score"):
        expected.add(f"dst_head/{fc}/kernel")
        expected.add(f"dst_head/{fc}/bias")

    # v30 DstEconomicsHead (optional)
    if "dst_economics_head/fc1/kernel" in flat:
        for fc in ("fc1", "econ_score"):
            expected.add(f"dst_economics_head/{fc}/kernel")
            expected.add(f"dst_economics_head/{fc}/bias")

    # PctHead + EmitHead: fc1 -> logits
    for h in ("pct_head", "emit_head"):
        for fc in ("fc1", "logits"):
            expected.add(f"{h}/{fc}/kernel")
            expected.add(f"{h}/{fc}/bias")

    # ValueHead (multi-query attention):
    #   queries (param), q_cond/{k,b}, value_attn/{q,k,v,out}/{k,b}, fc1, value
    expected.add("value_head/queries")
    expected.add("value_head/q_cond/kernel")
    expected.add("value_head/q_cond/bias")
    for qkv in ("query", "key", "value", "out"):
        expected.add(f"value_head/value_attn/{qkv}/kernel")
        expected.add(f"value_head/value_attn/{qkv}/bias")
    for fc in ("fc1", "value"):
        expected.add(f"value_head/{fc}/kernel")
        expected.add(f"value_head/{fc}/bias")

    missing = expected - set(flat.keys())
    if missing:
        raise KeyError(f"weights missing keys: {sorted(missing)[:6]}{'...' if len(missing) > 6 else ''}")
