"""Collect EnvState snapshots from Kaggle replay JSON for Plan B buffer curriculum.

Reads ``top10_episodes_*/episodes/episodes/*.json`` (or any orbit_wars replay JSON).
Uses observation only — **no action labels** (same as ``collect_states.py``).

4P replays are remapped to 2P board ids (winner perspective → player 0, all other
human players → player 1) so they load into our NUM_PLAYERS=2 trainer.

Usage
-----
    # Smoke (10 files)
    python -m orbit_wars_rl.bc.collect_states_from_json \\
        --replay-dir top10_episodes_2026-05-04/episodes/episodes \\
        --max-files 10 --out /tmp/top10_smoke.npz

    # Full top10 winners (5090, ~30-60 min)
    python -m orbit_wars_rl.bc.collect_states_from_json \\
        --replay-dir top10_episodes_2026-05-04/episodes/episodes \\
        --winners-only --out data/top10_winner_states.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from orbit_wars_rl.bc.state_buffer_util import (
    flip_owners,
    print_buffer_stats,
    remap_4p_to_2p,
    stack_samples,
    state_to_numpy,
    write_npz,
)
from orbit_wars_rl.env import constants
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate

_P = constants.MAX_PLANETS


def _winner_players(rewards: List[float]) -> List[int]:
    if not rewards:
        return []
    best = max(rewards)
    if best <= 0:
        return []
    return [i for i, r in enumerate(rewards) if r == best]


def _collect_episode(
    path: Path,
    *,
    skip_tail_frac: float,
    winners_only: bool,
    remap_4p: bool,
    flip_augment: bool,
    max_turn: Optional[int],
) -> List[dict[str, np.ndarray]]:
    with open(path) as f:
        replay: dict[str, Any] = json.load(f)

    cfg = replay.get("configuration") or {}
    episode_steps = int(cfg.get("episodeSteps", 500))
    rewards = list(replay.get("rewards") or [])
    steps = replay.get("steps") or []
    num_players = len(rewards) if rewards else len(steps[0]) if steps else 0
    if num_players <= 0:
        return []

    players = _winner_players(rewards) if winners_only else list(range(num_players))
    if not players:
        return []

    cutoff = int(episode_steps * (1.0 - skip_tail_frac))
    if max_turn is not None:
        cutoff = min(cutoff, max_turn)

    samples: List[dict[str, np.ndarray]] = []
    for turn, step in enumerate(steps):
        if turn >= cutoff:
            break
        if not step or len(step) < num_players:
            break

        for p in players:
            cell = step[p] or {}
            obs = dict(cell.get("observation") or step[0].get("observation") or {})
            if not obs.get("planets"):
                continue
            if len(obs["planets"]) > _P:
                continue
            obs["player"] = p
            try:
                state = kaggle_obs_to_envstate(obs)
            except Exception:
                continue
            d = state_to_numpy(state)
            if remap_4p and num_players > 2:
                d = remap_4p_to_2p(d, p)
            samples.append(d)
            if flip_augment:
                samples.append(flip_owners(d))

    return samples


def _iter_replay_paths(replay_dir: Optional[str], replay_glob: Optional[str]) -> List[Path]:
    if replay_glob:
        return sorted(Path(p) for p in __import__("glob").glob(replay_glob))
    if replay_dir:
        return sorted(Path(replay_dir).glob("*.json"))
    raise ValueError("need --replay-dir or --replay-glob")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Collect EnvState snapshots from replay JSON for buffer curriculum."
    )
    ap.add_argument("--replay-dir", type=str, default=None,
                    help="directory containing *.json episode files")
    ap.add_argument("--replay-glob", type=str, default=None,
                    help="glob pattern, e.g. 'top10_episodes_*/episodes/episodes/*.json'")
    ap.add_argument("--max-files", type=int, default=0,
                    help="limit number of JSON files (0 = all)")
    ap.add_argument("--winners-only", dest="winners_only", action="store_true", default=True)
    ap.add_argument("--all-players", dest="winners_only", action="store_false")
    ap.add_argument("--skip-tail-frac", type=float, default=0.20,
                    help="skip last fraction of each episode (avoid endgame hoarding)")
    ap.add_argument("--max-turn", type=int, default=None,
                    help="optional hard cap on turn index (in addition to skip-tail)")
    ap.add_argument("--no-remap-4p", action="store_true",
                    help="keep raw 4P owner ids (not for 2P training)")
    ap.add_argument("--no-flip", action="store_true",
                    help="disable player-0/1 flip augmentation")
    ap.add_argument("--out", type=str, default="data/top10_winner_states.npz")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    winners_only = args.winners_only
    paths = _iter_replay_paths(args.replay_dir, args.replay_glob)
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        print("ERROR: no replay JSON files found", file=sys.stderr)
        return 1

    out = Path(args.out)
    all_samples: List[dict[str, np.ndarray]] = []
    n_ok = 0
    t0 = time.time()

    print(f"collecting from {len(paths)} replay JSON files → {out}")
    for i, path in enumerate(paths):
        try:
            samples = _collect_episode(
                path,
                skip_tail_frac=args.skip_tail_frac,
                winners_only=winners_only,
                remap_4p=not args.no_remap_4p,
                flip_augment=not args.no_flip,
                max_turn=args.max_turn,
            )
        except Exception as e:
            if args.debug:
                print(f"  {path.name} error: {e!r}")
            continue
        if samples:
            all_samples.extend(samples)
            n_ok += 1
        if i % 50 == 0 or i == len(paths) - 1:
            dt = time.time() - t0
            print(
                f"  file {i+1}/{len(paths)} ({path.name}): +{len(samples)} | "
                f"total={len(all_samples)} ({dt:.1f}s)",
                flush=True,
            )

    if not all_samples:
        print("ERROR: no states collected")
        return 1

    print(f"\nstacking {len(all_samples)} states...")
    stacked = stack_samples(all_samples)
    print_buffer_stats(stacked)
    print(f"\nwriting {out}...")
    write_npz(stacked, out)
    size_mb = out.stat().st_size / 1e6
    print(f"  size: {size_mb:.1f} MB  |  {n_ok}/{len(paths)} files in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
