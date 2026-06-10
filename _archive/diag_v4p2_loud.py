"""Loud diagnostic for submission_rl_v4p2.py: bypass the silent
``except Exception: return []`` swallower in agent() and re-raise every
error verbatim. Also prints intermediate forward-pass shapes so we can
catch any shape mismatch.

Usage on 5090:
    python -m orbit_wars_rl.scripts.diag_v4p2_loud --submission submission_rl_v4p2.py
"""

from __future__ import annotations

import argparse
import importlib.util as iu
import sys
import traceback
from pathlib import Path

import numpy as np


def _load_submission(path: str):
    spec = iu.spec_from_file_location("_diag_v4p2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBES = [
    (
        "fake_obs_no_omega",
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
        "fake_obs_with_omega",
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
]


def diagnose(mod, name: str, obs: dict) -> None:
    print(f"\n=== {name} ===")
    player = int(obs.get("player", 0))

    # Step 1: encode_obs (raw, no swallowing).
    print("[step 1] encode_obs ...")
    enc = mod.encode_obs(obs, player=player, step=0, episode_steps=500)
    print(f"  keys: {list(enc.keys())}")
    for k, v in enc.items():
        if hasattr(v, "shape"):
            print(f"    {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"    {k}: {v}")
    print(f"  planet_feats[0] = {np.round(enc['planet_feats'][0], 3).tolist()}")
    print(f"  planet_feats[1] = {np.round(enc['planet_feats'][1], 3).tolist()}")
    print(f"  global_feats    = {np.round(enc['global_feats'], 3).tolist()}")

    # Step 2: load weights (raw, no swallowing).
    print("\n[step 2] _load_weights ...")
    if mod.WEIGHTS_B64 == "__WEIGHTS_B64__":
        print("  WEIGHTS_B64 is placeholder; cannot run forward.")
        return
    W = mod._load_weights()
    print(f"  loaded {len(W)} tensors")
    for k in sorted(W.keys())[:5]:
        print(f"    {k}: shape={W[k].shape}")
    print(f"    ... ({len(W) - 5} more)")

    # Step 3: encode_tokens forward.
    print("\n[step 3] _encode_tokens ...")
    try:
        global_emb, planet_emb, _f, planet_pool, _fp = mod._encode_tokens(W, enc)
        print(f"  global_emb shape   = {global_emb.shape}")
        print(f"  planet_emb shape   = {planet_emb.shape}")
        print(f"  planet_pool shape  = {planet_pool.shape}")
    except Exception:
        traceback.print_exc()
        return

    # Step 4: greedy_multi_action with full per-step trace.
    print("\n[step 4] greedy_multi_action (verbose) ...")
    my_mask = enc["my_planet_mask"].astype(bool)
    planet_mask = enc["planet_mask"].astype(bool)
    ships = enc["planet_ships"].astype(np.int32).copy()
    P = planet_emb.shape[0]
    reserved = np.zeros((P,), dtype=np.int32)
    still_emit = True

    K = getattr(mod, "MAX_FLEETS_PER_TURN", 8)
    src_out, dst_out, pct_out = [], [], []

    for t in range(K):
        remaining = ships - reserved
        avail_mask = my_mask & (remaining > 0)
        any_avail = bool(avail_mask.any())
        no_options = not any_avail
        eff_mask = avail_mask if any_avail else my_mask

        try:
            e_logits = mod._emit_head(W, global_emb, planet_pool, t)
            s_logits = mod._src_head(W, planet_emb, eff_mask)
            src_t = int(np.argmax(s_logits))
            src_emb = planet_emb[src_t]
            d_logits = mod._dst_head(W, planet_emb, src_emb, planet_mask, my_mask)
            dst_t = int(np.argmax(d_logits))
            dst_emb = planet_emb[dst_t]
            p_logits = mod._pct_head(W, src_emb, dst_emb, global_emb)
            pct_t = int(np.argmax(p_logits))
        except Exception:
            print(f"  t={t}: FORWARD CRASHED:")
            traceback.print_exc()
            break

        if t == 0:
            decision = (not no_options)
        else:
            emit_pred = int(np.argmax(e_logits))
            decision = (emit_pred == 1) and (not no_options)
        emit_t = decision and still_emit

        e_soft = np.exp(e_logits - e_logits.max()); e_soft /= e_soft.sum()
        print(
            f"  t={t}  no_opts={no_options}  forced_emit={t==0 and not no_options}  "
            f"emit_argmax={int(np.argmax(e_logits))}  "
            f"emit_p(stop,go)=({round(float(e_soft[0]),3)},{round(float(e_soft[1]),3)})  "
            f"still_emit={still_emit}  -> emit_t={emit_t}"
        )
        print(
            f"          src={src_t} (logit={round(float(s_logits[src_t]),3)})  "
            f"dst={dst_t} (logit={round(float(d_logits[dst_t]),3)}) "
            f"pct_bin={pct_t}  same_planet={src_t==dst_t}"
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
                break

    print(f"\n  emitted lists: src={src_out} dst={dst_out} pct={pct_out}")

    # Step 5: decode.
    print("\n[step 5] decode_multi_to_kaggle_moves ...")
    moves = mod.decode_multi_to_kaggle_moves(obs, src_out, dst_out, pct_out, player)
    print(f"  decoded moves: {moves}")
    if not moves and src_out:
        print("  -> decode swallowed all emits. Inspect the loop in")
        print("     decode_multi_to_kaggle_moves: src_idx not in planets_by_id,")
        print("     dst_idx not in planets_by_id, src_idx == dst_idx, owner check.")

    # Step 6: agent() (the canonical call -- this is what kaggle invokes).
    print("\n[step 6] agent() (canonical entry, silent except handler) ...")
    moves2 = mod.agent(obs, {"episodeSteps": 500})
    print(f"  agent() returns: {moves2}")
    if moves2 != moves:
        print("  !! agent() result differs from manual trace. Check silent except,")
        print("     _STEP_COUNTER / _detect_new_episode globals, or differing")
        print("     encode_obs args.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, required=True)
    args = ap.parse_args()

    if not Path(args.submission).exists():
        print(f"not found: {args.submission}", file=sys.stderr)
        return 2

    mod = _load_submission(args.submission)
    for name, obs in PROBES:
        try:
            diagnose(mod, name, obs)
        except Exception:
            print(f"\n!!! probe {name} crashed at TOP LEVEL:")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
