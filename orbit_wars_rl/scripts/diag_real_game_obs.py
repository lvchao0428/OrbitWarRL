"""Diagnose: take a REAL game step-1 obs from kaggle env (no fake corner
planets!) and run the submission's numpy forward on it, dumping every
intermediate decision -- which planet is selected as src? what does the
DST head argmax to? what's the emit gate's stop/go softmax? what does
the pct head pick?

This is the diagnostic that should have run before any of the H2H —
``replay_dump`` showed v4p2 outputting ``(12, X, 1)`` for ~15 turns no
matter the game state, but ``diag_v4p2_loud`` (with fake 4-planet obs)
showed ``(0, 0.0, 22)``. So the bug is hidden behind real-game-shaped
obs, not in the forward / weights themselves.

Usage on 5090:
    python -m orbit_wars_rl.scripts.diag_real_game_obs \
        --submission submission_rl_v4p2.py \
        --seed 0 \
        --num-steps 3
"""

from __future__ import annotations

import argparse
import importlib.util as iu
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from kaggle_environments import make


def _load_submission(path: str):
    spec = iu.spec_from_file_location("_diag_real_obs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_step1_obs(seed: int) -> Dict[str, Any]:
    """Reset kaggle env once and return the player-0 obs at step 0."""
    cfg = {"seed": seed}
    env = make("orbit_wars", configuration=cfg, debug=False)
    # Step the env once with no actions to populate state.
    # Actually, env.state already contains the initial observation.
    obs = env.state[0]["observation"]
    return obs


def diagnose_obs(mod, obs: Dict[str, Any], label: str) -> None:
    print(f"\n========== {label} ==========")
    player = int(obs.get("player", 0))
    print(f"player={player}")
    raw_planets = obs.get("planets") or []
    print(f"n_planets in obs: {len(raw_planets)}")
    own_ids = sorted([int(p[0]) for p in raw_planets if int(p[1]) == player])
    enemy_ids = sorted([int(p[0]) for p in raw_planets if int(p[1]) == (1 - player)])
    neutral_ids = sorted([int(p[0]) for p in raw_planets if int(p[1]) == -1])
    print(f"  own planets (id): {own_ids}")
    print(f"  enemy planets (id): {enemy_ids}")
    print(f"  neutral planets (id): {neutral_ids[:10]}{'...' if len(neutral_ids)>10 else ''}")
    print(f"  angular_velocity: {obs.get('angular_velocity', 'MISSING')}")

    # Print own home detail
    for p in raw_planets:
        if int(p[1]) == player:
            print(f"  own home raw row: id={int(p[0])} owner={int(p[1])} "
                  f"xy=({float(p[2]):.2f},{float(p[3]):.2f}) r={float(p[4]):.2f} "
                  f"ships={int(p[5])} prod={int(p[6])}")

    enc = mod.encode_obs(obs, player=player, step=0, episode_steps=500)
    print(f"\n[encode_obs out]")
    pm = enc["planet_mask"].astype(bool)
    mm = enc["my_planet_mask"].astype(bool)
    print(f"  planet_mask.sum() = {int(pm.sum())}")
    print(f"  my_planet_mask.sum() = {int(mm.sum())}")
    mine_idx = [i for i, b in enumerate(mm) if b]
    print(f"  my_planet_mask indices: {mine_idx}")
    print(f"  planet_ships (mask=True only): "
          f"{[(i, int(enc['planet_ships'][i])) for i in range(40) if pm[i]][:10]}")

    if mod.WEIGHTS_B64 == "__WEIGHTS_B64__":
        print("  WEIGHTS_B64 placeholder -- cannot run forward.")
        return

    W = mod._load_weights()
    global_emb, planet_emb, _f, planet_pool, _fp = mod._encode_tokens(W, enc)

    # --- Step 0 of multi-action ---
    print(f"\n[autoregressive step 0]")
    t = 0
    e_logits = mod._emit_head(W, global_emb, planet_pool, t)
    e_soft = np.exp(e_logits - e_logits.max()); e_soft /= e_soft.sum()
    print(f"  emit_head logits = ({float(e_logits[0]):.3f}, {float(e_logits[1]):.3f}) "
          f"softmax = ({float(e_soft[0]):.3f} stop, {float(e_soft[1]):.3f} go)  "
          f"argmax = {int(np.argmax(e_logits))}")

    # src head: only own planets are valid
    eff_mask = mm  # at t=0, all own planets are eligible (no reservations yet)
    s_logits = mod._src_head(W, planet_emb, eff_mask)
    src_t = int(np.argmax(s_logits))
    # Show top-3 valid src
    valid_pairs = sorted(
        [(i, float(s_logits[i])) for i in range(40) if eff_mask[i]],
        key=lambda x: -x[1],
    )
    print(f"  src head argmax = {src_t}, logit = {float(s_logits[src_t]):.3f}")
    print(f"  valid src logits (top): {valid_pairs[:5]}")

    # dst head conditioned on src_t
    src_emb = planet_emb[src_t]
    d_logits = mod._dst_head(W, planet_emb, src_emb, pm, mm)
    dst_t = int(np.argmax(d_logits))
    # Show top-5 dst options
    valid_dst = sorted(
        [(i, float(d_logits[i])) for i in range(40) if pm[i]],
        key=lambda x: -x[1],
    )
    d_soft = np.exp(d_logits - d_logits.max()); d_soft /= d_soft.sum()
    # entropy over valid mass
    valid_soft = d_soft * pm.astype(np.float32)
    valid_soft /= max(float(valid_soft.sum()), 1e-9)
    valid_only = valid_soft[pm]
    ent = float(-(valid_only * np.log(valid_only + 1e-12)).sum())
    print(f"  dst head argmax = {dst_t}, logit = {float(d_logits[dst_t]):.3f}")
    print(f"  dst entropy over valid = {ent:.3f} (max = ln({int(pm.sum())}) = {np.log(int(pm.sum())):.3f})")
    print(f"  valid dst logits (top 5): {valid_dst[:5]}")

    dst_emb = planet_emb[dst_t]
    p_logits = mod._pct_head(W, src_emb, dst_emb, global_emb)
    pct_t = int(np.argmax(p_logits))
    p_soft = np.exp(p_logits - p_logits.max()); p_soft /= p_soft.sum()
    print(f"  pct head argmax = {pct_t}  bin_value = {mod._PCT_BIN_TABLE_NP[pct_t]:.2f}  "
          f"softmax = {[round(float(x),3) for x in p_soft]}")

    # Decode the single emit to kaggle format
    moves = mod.decode_multi_to_kaggle_moves(obs, [src_t], [dst_t], [pct_t], player)
    print(f"\n  decoded move: {moves}")

    # What does agent() canonical return on this obs?
    moves2 = mod.agent(obs, {"episodeSteps": 500})
    print(f"  agent() canonical: {moves2}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=2,
                    help="number of distinct seeds to test")
    args = ap.parse_args()

    if not Path(args.submission).exists():
        print(f"not found: {args.submission}", file=sys.stderr)
        return 2
    mod = _load_submission(args.submission)

    for s_off in range(args.num_seeds):
        seed = args.seed + s_off
        try:
            obs = _make_step1_obs(seed)
            diagnose_obs(mod, obs, label=f"step 0 obs from kaggle env seed={seed}")
        except Exception as exc:  # noqa: BLE001
            print(f"seed {seed}: failed {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
