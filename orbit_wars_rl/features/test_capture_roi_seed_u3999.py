"""Seed-calibrated ROI test: v27 u3999 replay opening must prefer id=20.

Run: ``python -m orbit_wars_rl.features.test_capture_roi_seed_u3999``

Pass criteria (t=0 and t=14, player 0):
  * neutral id=20 has highest dim32 among capturable neutrals, OR is tied #1
    with other high-prod nearby targets but strictly beats id=22 and id=16.
  * roi(id=20) > roi(id=22) and roi(id=20) > roi(id=16).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_wars_rl.features.encode import encode
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate

_REPLAY = Path("logs/replay_html/v27_u3999_s0/replay.json")
_HOME_ID = 12
_FACTORY_ID = 20  # nearby high-prod (prod=3, g=20)
_FAR_ID = 22  # same stats, dist≈56 — agent wrongly picked this at t=14
_WEAK_ID = 16  # old v26 formula ranked this above id=20


def _roi_table(obs: dict, t: int) -> list[tuple[float, int, int, int]]:
    state = kaggle_obs_to_envstate(obs)
    enc = encode(state, player=0, episode_steps=500)
    pf = np.array(enc.planet_feats)
    rows: list[tuple[float, int, int, int]] = []
    for p in obs["planets"]:
        pid, owner = int(p[0]), int(p[1])
        if owner != -1:
            continue
        idx = next(j for j, pp in enumerate(obs["planets"]) if int(pp[0]) == pid)
        rows.append((float(pf[idx, 32]), pid, int(p[5]), int(p[6])))
    rows.sort(reverse=True)
    return rows


def _check_turn(replay: dict, t: int) -> None:
    obs = dict(replay["steps"][t][0]["observation"])
    obs["player"] = 0
    rows = _roi_table(obs, t)
    by_id = {pid: roi for roi, pid, _, _ in rows}

    roi20 = by_id[_FACTORY_ID]
    roi22 = by_id[_FAR_ID]
    roi16 = by_id[_WEAK_ID]
    top_pid = rows[0][1]
    rank20 = next(i for i, r in enumerate(rows) if r[1] == _FACTORY_ID) + 1

    print(
        f"[t={t:2d}] top={top_pid} roi20={roi20:.4f} roi22={roi22:.4f} "
        f"roi16={roi16:.4f} rank20={rank20} top3={[(r[1], round(r[0], 3)) for r in rows[:3]]}"
    )

    if roi20 <= roi22:
        raise AssertionError(f"t={t}: id20 roi must exceed id22: {roi20} <= {roi22}")
    if roi20 <= roi16:
        raise AssertionError(f"t={t}: id20 roi must exceed id16: {roi20} <= {roi16}")
    if rank20 > 2:
        raise AssertionError(
            f"t={t}: id20 should rank #1-2 among neutrals for opening capture, got #{rank20}"
        )


def main() -> int:
    if not _REPLAY.is_file():
        raise SystemExit(f"missing replay fixture: {_REPLAY}")

    with open(_REPLAY) as f:
        replay = json.load(f)

    owned_t0 = [int(p[0]) for p in replay["steps"][0][0]["observation"]["planets"] if int(p[1]) == 0]
    print(f"[info] home={_HOME_ID} owned@t0={owned_t0} replay={_REPLAY}")

    for t in (0, 14):
        _check_turn(replay, t)

    print("\n[ALL PASS] seed u3999 ROI calibration (id=20 beats id=22/id=16, rank top-2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
