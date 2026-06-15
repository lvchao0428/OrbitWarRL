"""Seed-calibrated pair_roi tuning for opening aim (id=20 @ seed=0).

Finds how strongly ``pair_roi`` must be amplified (feature scale or logit
bias) so greedy dst argmax hits the target planet on a fixed replay turn.

Usage (feature-only, no ckpt):
    python -m orbit_wars_rl.scripts.calibrate_pair_roi_seed \\
        --replay logs/replay_html/v29_u3199_s0/replay.json --turn 11

With checkpoint (full dst logits + sweep):
    python -m orbit_wars_rl.scripts.calibrate_pair_roi_seed \\
        --replay logs/replay_html/v29_u3199_s0/replay.json --turn 11 \\
        --ckpt ckpt_multi_action_v29_aim/ckpt_003199.pkl

Run: ``bash scripts/calibrate_pair_roi_seed.sh``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.features.pair import DST_PAIR_DIM, dst_flip_block_mask, dst_pair_features
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPLAY = _ROOT / "logs/replay_html/v27_u3999_s0/replay.json"
_HOME_ID = 12
_TARGET_ID = 20


def _home_ships(obs: dict, home_id: int) -> int:
    return int(next(p for p in obs["planets"] if int(p[0]) == home_id)[5])


def _first_launch_turn(replay: dict, player: int = 0) -> int | None:
    for t, step in enumerate(replay["steps"]):
        act = step[player].get("action") or []
        if act:
            return t
    return None


def _pre_action_home_ships(replay: dict, turn: int, home_id: int = _HOME_ID) -> int:
    if turn <= 0:
        obs = replay["steps"][0][0]["observation"]
        return int(next(p for p in obs["planets"] if int(p[0]) == home_id)[5])
    prev = replay["steps"][turn - 1][0]["observation"]
    return int(next(p for p in prev["planets"] if int(p[0]) == home_id)[5]) + 1


def _load_submission():
    path = _ROOT / "submission_rl_v21.py"
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pair_table(
    obs: dict,
    *,
    src_idx: int,
    roi_norm: float = 1.0,
    roi_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = kaggle_obs_to_envstate(obs)
    P = int(state.planet_mask.shape[0])
    target = state.planet_mask & jnp.logical_not(state.planet_owner == 0)
    remaining = state.planet_ships.astype(jnp.int32)
    pair_feats, sun_block = dst_pair_features(
        state.planet_x,
        state.planet_y,
        state.planet_ships,
        state.planet_mask,
        target,
        remaining,
        jnp.int32(src_idx),
        planet_orbit_phase=state.planet_orbit_phase,
        planet_orbit_radius=state.planet_orbit_radius,
        planet_is_orbiting=state.planet_is_orbiting,
        angular_velocity=state.angular_velocity,
        planet_prod=state.planet_prod.astype(jnp.float32),
    )
    pf = np.array(pair_feats, dtype=np.float32)
    if roi_norm != 1.0 or roi_scale != 1.0:
        from orbit_wars_rl.features.capture_roi_util import capture_roi_from_src

        rem = float(max(int(remaining[src_idx]), 1))
        garr = state.planet_ships.astype(jnp.float32)
        prod_f = state.planet_prod.astype(jnp.float32)
        src_x = float(state.planet_x[src_idx])
        src_y = float(state.planet_y[src_idx])
        dx = state.planet_x - src_x
        dy = state.planet_y - src_y
        dist = jnp.sqrt(dx * dx + dy * dy + jnp.float32(1e-6))
        need_est = garr + jnp.float32(2.0)
        roi_raw = capture_roi_from_src(
            prod_f, need_est, dist, norm=jnp.float32(roi_norm),
        )
        roi = np.clip(np.array(roi_raw, dtype=np.float32), 0.0, 1.0)
        roi = np.clip(roi * float(roi_scale), 0.0, 1.0) * np.array(target, dtype=np.float32)
        pf[:, 5] = roi

    flip_block = np.array(
        dst_flip_block_mask(
            state.planet_ships,
            state.planet_mask,
            target,
            remaining,
            jnp.int32(src_idx),
        ),
        dtype=bool,
    )
    return pf, np.array(sun_block, dtype=bool), flip_block


def _print_feature_report(
    obs: dict,
    *,
    turn: int,
    src_idx: int,
    target_id: int,
    actual_dst_id: int | None,
) -> np.ndarray:
    pf, _sun, flip_block = _pair_table(obs, src_idx=src_idx)
    rows: list[tuple[float, int, int, int, float, float, bool]] = []
    for i, p in enumerate(obs["planets"]):
        pid, owner = int(p[0]), int(p[1])
        if owner != -1:
            continue
        rows.append(
            (
                float(pf[i, 5]),
                pid,
                int(p[5]),
                int(p[6]),
                float(pf[i, 3]),
                float(pf[i, 4]),
                bool(flip_block[i]),
            )
        )
    rows.sort(reverse=True)
    tgt_idx = target_id
    roi_tgt = float(pf[tgt_idx, 5])
    rank_tgt = next(i for i, r in enumerate(rows) if r[1] == target_id) + 1
    n_flip_ok = sum(1 for r in rows if not r[6])
    n_blocked = sum(1 for r in rows if r[6])

    print(f"\n[features] turn={turn} src=home(id={_HOME_ID}) rem={_home_ships(obs, _HOME_ID)}")
    print(f"  pair_roi rank id={target_id}: #{rank_tgt} roi={roi_tgt:.4f}")
    print(f"  flip_block: {n_blocked} neutrals blocked, {n_flip_ok} pass (fallback if all blocked)")
    print("  top neutrals by pair_roi:")
    for roi, pid, g, prod, flip, margin, blocked in rows[:6]:
        tag = "BLOCK" if blocked else "ok"
        print(
            f"    id={pid:2d} g={g:3d} prod={prod} roi={roi:.4f} "
            f"flip={flip:.0f} margin={margin:.3f} [{tag}]"
        )
    if actual_dst_id is not None:
        act_idx = actual_dst_id
        print(
            f"  actual dst id={actual_dst_id}: roi={pf[act_idx, 5]:.4f} "
            f"flip_block={bool(flip_block[act_idx])}"
        )
    return pf


def _dst_logits_once(
    sub_mod: Any,
    W: dict[str, np.ndarray],
    obs: dict,
    *,
    turn: int,
    src_idx: int,
    roi_scale: float = 1.0,
    logit_alpha: float = 0.0,
    apply_flip_block: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    enc = sub_mod.encode_obs(
        obs, player=0, step=turn, episode_steps=500,
    )
    pf, sun_block, flip_block = _pair_table(obs, src_idx=src_idx, roi_scale=roi_scale)

    global_emb, planet_emb, _, _, _ = sub_mod._encode_tokens(W, enc)
    my_mask = enc["my_planet_mask"].astype(bool)
    planet_mask = enc["planet_mask"].astype(bool)
    target_mask = planet_mask & np.logical_not(my_mask)
    ships = enc["planet_ships"].astype(np.int32)
    remaining = ships.copy()
    remaining_norm = np.log1p(np.maximum(remaining, 0).astype(np.float32)) / 8.0
    reserved_norm = np.zeros_like(remaining_norm)
    src_emb = planet_emb[src_idx]

    flip_m = flip_block if apply_flip_block else None
    d_logits = sub_mod._dst_head(
        W,
        planet_emb,
        src_emb,
        planet_mask,
        my_mask,
        src_idx=src_idx,
        reserved_norm=reserved_norm,
        pair_feats=pf,
        sun_block_mask=sun_block,
        flip_block_mask=flip_m,
    )
    if logit_alpha != 0.0:
        d_logits = d_logits + np.float32(logit_alpha) * pf[:, 5]
    return d_logits.astype(np.float32), pf


def _top_logits(
    obs: dict,
    logits: np.ndarray,
    *,
    k: int = 8,
) -> list[tuple[float, int, int, int]]:
    rows: list[tuple[float, int, int, int]] = []
    for i, p in enumerate(obs["planets"]):
        pid, owner = int(p[0]), int(p[1])
        if owner != -1 or pid >= logits.shape[0]:
            continue
        rows.append((float(logits[pid]), pid, int(p[5]), int(p[6])))
    rows.sort(reverse=True)
    return rows[:k]


def _sweep(
    sub_mod: Any,
    W: dict[str, np.ndarray],
    obs: dict,
    *,
    turn: int,
    src_idx: int,
    target_id: int,
) -> None:
    base_logits, base_pf = _dst_logits_once(
        sub_mod, W, obs, turn=turn, src_idx=src_idx,
    )
    tgt_idx = target_id
    base_pick = int(np.argmax(base_logits))
    base_pid = int(obs["planets"][base_pick][0])

    print(f"\n[ckpt] base argmax: slot={base_pick} id={base_pid} "
          f"logit={base_logits[base_pick]:.3f}")
    print(f"       target id={target_id} logit={base_logits[tgt_idx]:.3f} "
          f"gap={base_logits[base_pick]-base_logits[tgt_idx]:+.3f}")
    print("  top dst logits:")
    for logit, pid, g, prod in _top_logits(obs, base_logits):
        mark = " <-- target" if pid == target_id else (" <-- base" if pid == base_pid else "")
        print(f"    id={pid:2d} g={g:3d} prod={prod} logit={logit:.3f}{mark}")

    # --- sweep logit_alpha (inference-only bias; easiest to tune) ---
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    min_alpha: float | None = None
    for alpha in alphas:
        logits, _ = _dst_logits_once(
            sub_mod, W, obs, turn=turn, src_idx=src_idx, logit_alpha=alpha,
        )
        pick = int(np.argmax(logits))
        pid = int(obs["planets"][pick][0])
        ok = pid == target_id
        print(f"  alpha={alpha:5.1f} -> id={pid:2d} logit_tgt={logits[tgt_idx]:.3f} {'OK' if ok else ''}")
        if ok and min_alpha is None:
            min_alpha = alpha

    if min_alpha is not None:
        print(f"\n[sweep] min logit_alpha for id={target_id}: {min_alpha}")
    else:
        print(f"\n[sweep] logit_alpha up to {alphas[-1]} still misses id={target_id}")

    # --- sweep roi_scale (feature path; needs train/export parity) ---
    scales = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    min_scale: float | None = None
    for scale in scales:
        logits, _ = _dst_logits_once(
            sub_mod, W, obs, turn=turn, src_idx=src_idx, roi_scale=scale,
        )
        pick = int(np.argmax(logits))
        pid = int(obs["planets"][pick][0])
        ok = pid == target_id
        print(f"  roi_scale={scale:4.1f} -> id={pid:2d} {'OK' if ok else ''}")
        if ok and min_scale is None:
            min_scale = scale

    if min_scale is not None:
        print(f"[sweep] min pair_roi feature scale for id={target_id}: {min_scale}")
    else:
        print(f"[sweep] roi_scale up to {scales[-1]} still misses id={target_id} (MLP may ignore dim6)")

    # --- roi_norm (raw economics before clip) ---
    print("\n[hint] If feature scale fails but logit_alpha works, patch training init:")
    d_model = int(W["encoder/planet_proj/kernel"].shape[1])
    col = d_model + 1 + 5  # reserved + pair dims [0..5]
    w = W.get("dst_head/dst_fc1/kernel")
    if w is not None and w.shape[0] > col:
        col_w = w[col, :]
        print(f"  dst_fc1 pair_roi column norm={float(np.linalg.norm(col_w)):.6f} "
              f"(zero-pad from v28 => ~0)")
        if min_alpha is not None:
            print(
                f"  suggest init: scale dst_fc1 row {col} by ~{max(min_alpha, 1.0):.0f} "
                f"or add pair_roi_logit_coef={min_alpha} at inference during continued train"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate pair_roi on seed=0 opening.")
    ap.add_argument("--replay", type=Path, default=_DEFAULT_REPLAY)
    ap.add_argument("--turn", type=int, default=-1, help="decision turn (default: first launch)")
    ap.add_argument("--home-id", type=int, default=_HOME_ID)
    ap.add_argument("--target-id", type=int, default=_TARGET_ID)
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()

    replay_path = args.replay if args.replay.is_absolute() else _ROOT / args.replay
    if not replay_path.is_file():
        raise SystemExit(f"missing replay: {replay_path}")

    with open(replay_path) as f:
        replay = json.load(f)

    turn = args.turn
    if turn < 0:
        found = _first_launch_turn(replay)
        if found is None:
            raise SystemExit("no launch in replay; set --turn explicitly")
        turn = found

    # Pre-action obs: replay stores post-action state; use previous turn + prod.
    if turn > 0:
        obs = dict(replay["steps"][turn - 1][0]["observation"])
    else:
        obs = dict(replay["steps"][0][0]["observation"])
    obs["player"] = 0

    src_idx = args.home_id
    pre_home = _pre_action_home_ships(replay, turn, args.home_id)

    act = replay["steps"][turn][0].get("action") or []
    actual_dst_id: int | None = None
    if act:
        src, ang, ships = int(act[0][0]), float(act[0][1]), int(act[0][2])
        hx = float(next(p for p in obs["planets"] if int(p[0]) == src)[2])
        hy = float(next(p for p in obs["planets"] if int(p[0]) == src)[3])
        best: tuple[float, int] | None = None
        for p in obs["planets"]:
            pid = int(p[0])
            px, py = float(p[2]), float(p[3])
            ang_p = math.atan2(py - hy, px - hx)
            d = abs((ang - ang_p + math.pi) % (2 * math.pi) - math.pi)
            if best is None or d < best[0]:
                best = (d, pid)
        actual_dst_id = best[1] if best else None

    print(f"=== calibrate_pair_roi_seed ===")
    print(f"replay={replay_path.name} decision_turn={turn} pre_home_ships={pre_home}")
    print(f"target=id={args.target_id} actual_first_dst={actual_dst_id}")

    _print_feature_report(
        obs,
        turn=turn,
        src_idx=src_idx,
        target_id=args.target_id,
        actual_dst_id=actual_dst_id,
    )

    ckpt = args.ckpt
    if ckpt is not None:
        ckpt_path = ckpt if ckpt.is_absolute() else _ROOT / ckpt
        if not ckpt_path.is_file():
            raise SystemExit(f"missing ckpt: {ckpt_path}")
        from orbit_wars_rl.inference.weights import infer_arch_from_flat, load_flat_params

        flat = load_flat_params(str(ckpt_path))
        arch = infer_arch_from_flat(flat)
        print(f"\n[ckpt] {ckpt_path.name} dst_pair_dim={arch.get('dst_pair_dim')} "
              f"d_model={arch.get('d_model')}")
        sub_mod = _load_submission()
        _sweep(sub_mod, flat, obs, turn=turn, src_idx=src_idx, target_id=args.target_id)
    else:
        print("\n[skip] no --ckpt: feature ranking only. Pass ckpt to sweep logit_alpha / roi_scale.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
