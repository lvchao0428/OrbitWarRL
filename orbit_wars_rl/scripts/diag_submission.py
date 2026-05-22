"""Diagnose a (silent) submission failure: imports an exported submission,
runs ``greedy_multi_action`` on a battery of obs probes, and prints every
step's intermediate decision (src, dst, pct, emit gate). Use when
``[export] smoke moves: []`` shows up.

Usage:
    python -m orbit_wars_rl.scripts.diag_submission --submission submission_rl_v4p2.py
"""

from __future__ import annotations

import argparse
import importlib.util as iu
import sys
from pathlib import Path

import numpy as np


def _load_submission(path: str):
    spec = iu.spec_from_file_location("_diag_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBES = [
    (
        "v3_legacy_omega0",
        {
            "player": 0,
            "planets": [
                [0, 0, 20.0, 20.0, 1.0, 30, 3],
                [1, -1, 80.0, 20.0, 1.0, 15, 2],
                [2, -1, 20.0, 80.0, 1.0, 15, 2],
                [3, 1, 80.0, 80.0, 1.0, 30, 3],
            ],
            "fleets": [],
            "initial_planets": [
                [0, 0, 20.0, 20.0, 1.0, 30, 3],
                [1, -1, 80.0, 20.0, 1.0, 15, 2],
                [2, -1, 20.0, 80.0, 1.0, 15, 2],
                [3, 1, 80.0, 80.0, 1.0, 30, 3],
            ],
        },
    ),
    (
        "v3_legacy_omega004",
        {
            "player": 0,
            "planets": [
                [0, 0, 20.0, 20.0, 1.0, 30, 3],
                [1, -1, 80.0, 20.0, 1.0, 15, 2],
                [2, -1, 20.0, 80.0, 1.0, 15, 2],
                [3, 1, 80.0, 80.0, 1.0, 30, 3],
            ],
            "fleets": [],
            "initial_planets": [
                [0, 0, 20.0, 20.0, 1.0, 30, 3],
                [1, -1, 80.0, 20.0, 1.0, 15, 2],
                [2, -1, 20.0, 80.0, 1.0, 15, 2],
                [3, 1, 80.0, 80.0, 1.0, 30, 3],
            ],
            "angular_velocity": 0.04,
        },
    ),
    (
        "orbital_5planet_mid",
        {
            "player": 0,
            "planets": [
                # 4 orbiting + 1 home (close to sun)
                [0, 0, 30.0, 50.0, 1.0, 50, 3],
                [1, 1, 70.0, 50.0, 1.0, 50, 3],
                [2, -1, 50.0, 30.0, 1.0, 25, 2],
                [3, -1, 50.0, 70.0, 1.0, 25, 2],
                [4, -1, 15.0, 50.0, 1.0, 10, 1],
            ],
            "fleets": [],
            "initial_planets": [
                [0, 0, 30.0, 50.0, 1.0, 50, 3],
                [1, 1, 70.0, 50.0, 1.0, 50, 3],
                [2, -1, 50.0, 30.0, 1.0, 25, 2],
                [3, -1, 50.0, 70.0, 1.0, 25, 2],
                [4, -1, 15.0, 50.0, 1.0, 10, 1],
            ],
            "angular_velocity": 0.035,
        },
    ),
]


def diagnose(mod, probe_name: str, obs: dict) -> None:
    print(f"\n=== probe: {probe_name} ===")
    player = int(obs["player"])

    # Step 1: encode_obs.
    enc = mod.encode_obs(obs, player, step=0, episode_steps=500)
    print(f"  encoded planet_mask sum = {int(enc['planet_mask'].sum())}")
    print(f"  my_planet_mask sum      = {int(enc['my_planet_mask'].sum())}")
    print(f"  planet_ships first 6    = {enc['planet_ships'][:6].tolist()}")
    print(f"  global_feats            = {np.round(enc['global_feats'], 3).tolist()}")
    print(f"  planet_feats shape      = {enc['planet_feats'].shape}")

    # Step 2: weights present?
    if mod.WEIGHTS_B64 == "__WEIGHTS_B64__":
        print("  WEIGHTS_B64 is placeholder -- cannot run forward.")
        return

    # Step 3: run greedy_multi_action with verbose patches.
    W = mod._load_weights()
    print(f"  weights loaded: {len(W)} tensors")
    # Manually re-run greedy_multi_action but with logging.
    _encode_tokens = mod._encode_tokens
    _emit_head = mod._emit_head
    _src_head = mod._src_head
    _dst_head = mod._dst_head
    _pct_head = mod._pct_head

    global_emb, planet_emb, _f, planet_pool, _fp = _encode_tokens(W, enc)
    my_mask = enc["my_planet_mask"].astype(bool)
    planet_mask = enc["planet_mask"].astype(bool)
    ships = enc["planet_ships"].astype(np.int32).copy()
    P = planet_emb.shape[0]
    reserved = np.zeros((P,), dtype=np.int32)
    still_emit = True

    src_out, dst_out, pct_out = [], [], []

    K = getattr(mod, "MAX_FLEETS_PER_TURN", 8)
    print(f"  greedy autoreg with K={K}:")
    for t in range(K):
        remaining = ships - reserved
        avail_mask = my_mask & (remaining > 0)
        any_avail = bool(avail_mask.any())
        no_options = not any_avail
        eff_mask = avail_mask if any_avail else my_mask

        e_logits = _emit_head(W, global_emb, planet_pool, t)
        if t == 0:
            decision = (not no_options)
        else:
            emit_pred = int(np.argmax(e_logits))
            decision = (emit_pred == 1) and (not no_options)
        emit_t = decision and still_emit

        s_logits = _src_head(W, planet_emb, eff_mask)
        src_t = int(np.argmax(s_logits))
        src_emb = planet_emb[src_t]
        d_logits = _dst_head(W, planet_emb, src_emb, planet_mask, my_mask)
        dst_t = int(np.argmax(d_logits))
        dst_emb = planet_emb[dst_t]
        p_logits = _pct_head(W, src_emb, dst_emb, global_emb)
        pct_t = int(np.argmax(p_logits))

        # Show full logits (truncated to first 6) for visibility.
        e_soft = np.exp(e_logits - e_logits.max()); e_soft /= e_soft.sum()
        s_logits_show = s_logits[:6].tolist() if len(s_logits) >= 6 else s_logits.tolist()
        d_logits_show = d_logits[:6].tolist() if len(d_logits) >= 6 else d_logits.tolist()

        print(
            f"  t={t}  no_opts={no_options}  emit?={emit_t}  "
            f"emit_prob(stop,go)={tuple(round(float(x),3) for x in e_soft)}  "
            f"src={src_t}(top6 logits {[round(x,2) for x in s_logits_show]})  "
            f"dst={dst_t}(top6 logits {[round(x,2) for x in d_logits_show]})  "
            f"pct_bin={pct_t}"
        )

        if emit_t:
            avail_at_src = max(int(ships[src_t]) - int(reserved[src_t]), 0)
            pct = float(mod._PCT_BIN_TABLE_NP[pct_t])
            import math
            ships_t = max(1, int(math.floor(avail_at_src * pct)))
            ships_t = min(ships_t, avail_at_src)
            reserved[src_t] += ships_t
            src_out.append(src_t)
            dst_out.append(dst_t)
            pct_out.append(pct_t)
        else:
            still_emit = False
            if t > 0:
                break  # mirror greedy halt

    print(f"  emitted: src={src_out} dst={dst_out} pct={pct_out}")

    moves = mod.decode_multi_to_kaggle_moves(obs, src_out, dst_out, pct_out, player)
    print(f"  decoded moves -> kaggle: {moves}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, required=True,
                    help="path to a submission_rl_v*.py file (with real weights)")
    args = ap.parse_args()

    if not Path(args.submission).exists():
        print(f"not found: {args.submission}", file=sys.stderr)
        return 2

    mod = _load_submission(args.submission)
    for name, obs in PROBES:
        try:
            diagnose(mod, name, obs)
        except Exception as exc:  # noqa: BLE001
            print(f"probe {name} raised {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
