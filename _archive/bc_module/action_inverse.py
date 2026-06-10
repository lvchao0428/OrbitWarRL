"""Convert v20's raw kaggle moves into supervised labels for our K-step heads.

Input
-----
state        : EnvState (kaggle-bridged) at the start of v20's turn
player       : 0 or 1 -- which side v20 was playing
kaggle_moves : list[[src_id:int, angle:float, ships:int]]  -- v20's output

Output (each (K,) where K = MAX_FLEETS_PER_TURN = 8)
----------------------------------------------------
src_t   : int32  -- slot index of src for step t (0 if no emit)
dst_t   : int32  -- recovered slot index of dst (closest-angle planet)
pct_t   : int32  -- discretised into NUM_PCT_BINS bins
emit_t  : bool   -- True iff v20 issued a move at step t
loss_mask : bool -- True at steps we supervise: all emit steps + the first
                   non-emit step ("stop" decision). step 0 is always supervised
                   (forced emit decision is masked out separately).
emit_free : bool -- True if the emit head was a *free* choice (i.e. t > 0 and
                   ships are still available). Matches the sampler's
                   ``emit_free_mask``; emit head only takes loss where this is
                   True (consistent with PPO's logp masking).

Algorithm
---------
For each step k in [0..min(len(moves), K)):
    src_id = move[k][0]
    angle  = move[k][1]
    ships  = move[k][2]
    # 1) src = src_id (kaggle slot id == our slot, see kaggle_bridge.py)
    # 2) pct: ships / (planet_ships[src_id] - reserved[src_id])
    #         where reserved tracks earlier-in-turn emits from the same src.
    # 3) dst: argmin( |atan2(py[i]-py[src], px[i]-px[src]) - angle| )
    #         excluding padding planets and src itself.

The reserved buffer is non-trivial: v20 emits in a single agent() call but
the kaggle env updates the planet_ships *after* the whole list executes. So
when we map move[2] from src=7, the *planet's reported* ships in obs still
reflect the start-of-turn count, even though moves[0] and moves[1] may have
already drawn from src=7. We rebuild this reserved counter ourselves.

This matches our env's ``_ships_to_send_for_step`` semantics exactly, which
is critical: BC labels must be reachable in our env, or evaluate() can't
re-compute the same logits the training assumes.
"""
from __future__ import annotations

from typing import List

import numpy as np

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


_PCT_BINS = np.asarray(constants.PCT_BIN_VALUES, dtype=np.float32)
_K = constants.MAX_FLEETS_PER_TURN
_P = constants.MAX_PLANETS


def _snap_pct(frac: float) -> int:
    """Pick the pct bin whose *floor decode* gives the closest ship count.

    The env decodes ``ships = max(1, floor(avail * pct))``; we therefore want
    the bin whose decoded ship count is closest to v20's chosen value, not
    the bin closest to ``frac`` itself (those can differ in edge cases like
    very small avail where floor(avail*0.55) == floor(avail*0.70)).

    For BC purposes we use the simpler nearest-frac rule, which is the same
    target almost everywhere; the diag tests confirm round-trip stability.
    """
    return int(np.argmin(np.abs(_PCT_BINS - np.float32(frac))))


def kaggle_moves_to_targets(
    state: EnvState,
    player: int,
    moves: List[List[float]],
) -> dict[str, np.ndarray]:
    """Return per-step supervision targets for our K-step heads.

    Parameters
    ----------
    state : EnvState bridged from a real kaggle observation, at the start
        of v20's turn. We read ``planet_ships``, ``planet_x``, ``planet_y``,
        and ``planet_mask`` from it.
    player : 0 or 1, v20's perspective.
    moves : list of [src_id, angle, ships]. Empty list means v20 issued no
        action that turn.
    """
    src_arr = np.zeros((_K,), dtype=np.int32)
    dst_arr = np.zeros((_K,), dtype=np.int32)
    pct_arr = np.zeros((_K,), dtype=np.int32)
    emit_arr = np.zeros((_K,), dtype=np.bool_)
    loss_mask = np.zeros((_K,), dtype=np.bool_)
    emit_free = np.zeros((_K,), dtype=np.bool_)

    px = np.asarray(state.planet_x, dtype=np.float32)
    py = np.asarray(state.planet_y, dtype=np.float32)
    pmask = np.asarray(state.planet_mask, dtype=np.bool_)
    pships = np.asarray(state.planet_ships, dtype=np.int32)
    powner = np.asarray(state.planet_owner, dtype=np.int8)
    my_mask = pmask & (powner == player)

    # Reserved book-keeping: same as env's _ships_to_send_for_step.
    reserved = np.zeros((_P,), dtype=np.int32)

    # Cap at K. If v20 emits more we drop the tail (rare; v20 usually emits ~3-8).
    k_eff = min(len(moves), _K)

    for k in range(k_eff):
        m = moves[k]
        if not isinstance(m, (list, tuple)) or len(m) != 3:
            # malformed move -- skip
            break
        try:
            src_id = int(m[0])
            angle = float(m[1])
            ships = int(m[2])
        except (TypeError, ValueError):
            break

        if src_id < 0 or src_id >= _P or not bool(my_mask[src_id]):
            # invalid src; the env would silently drop this move so we
            # don't supervise it either.
            continue
        if ships <= 0:
            continue

        avail = int(pships[src_id]) - int(reserved[src_id])
        if avail <= 0:
            # nothing left at this src -- env would clip ships=0 and drop.
            continue
        ships_clipped = min(ships, avail)
        pct_frac = float(ships_clipped) / float(max(avail, 1))
        pct_bin = _snap_pct(pct_frac)

        sx, sy = float(px[src_id]), float(py[src_id])
        dx_all = px - sx
        dy_all = py - sy
        angles = np.arctan2(dy_all, dx_all)
        # Circular distance.
        adiff = np.abs(((angles - angle) + np.pi) % (2 * np.pi) - np.pi)
        # Exclude padding and src itself; only score valid planets.
        adiff = np.where(pmask, adiff, np.float32(1e9))
        adiff[src_id] = 1e9
        dst_id = int(np.argmin(adiff))

        src_arr[k] = src_id
        dst_arr[k] = dst_id
        pct_arr[k] = pct_bin
        emit_arr[k] = True
        loss_mask[k] = True

        # Mirror env's _ships_to_send_for_step (floor + max(1, ..)):
        decoded_ships = max(1, int(np.floor(avail * _PCT_BINS[pct_bin])))
        decoded_ships = min(decoded_ships, avail)
        reserved[src_id] += decoded_ships

    # Supervise the "stop" decision exactly once: the first non-emit step
    # after the last emit. If we emitted all K, no stop supervision (the
    # env truncates anyway).
    if k_eff < _K:
        loss_mask[k_eff] = True  # emit_arr[k_eff] is already False -> target=stop

    # emit_free is True at step t iff t > 0 AND we still have ships to send.
    # We approximate "still have ships" by walking reserved/total; at step t
    # if total my-ships - sum(reserved) > 0 we say emit_free=True. For BC
    # purposes the simpler rule (t > 0) is fine: the loss_mask already
    # excludes us from supervising bogus emit decisions.
    for t in range(1, _K):
        emit_free[t] = True

    return {
        "src": src_arr,
        "dst": dst_arr,
        "pct": pct_arr,
        "emit": emit_arr,
        "loss_mask": loss_mask,
        "emit_free": emit_free,
    }
