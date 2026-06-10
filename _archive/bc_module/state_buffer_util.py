"""Shared helpers for Plan B state-buffer .npz files (collect_states / JSON)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

from orbit_wars_rl.env import constants


def state_to_numpy(state) -> dict[str, np.ndarray]:
    """Flatten one EnvState to a dict of numpy arrays (one sample row)."""
    return {
        "planet_owner": np.asarray(state.planet_owner, dtype=np.int8),
        "planet_x": np.asarray(state.planet_x, dtype=np.float32),
        "planet_y": np.asarray(state.planet_y, dtype=np.float32),
        "planet_radius": np.asarray(state.planet_radius, dtype=np.float32),
        "planet_ships": np.asarray(state.planet_ships, dtype=np.int32),
        "planet_prod": np.asarray(state.planet_prod, dtype=np.int32),
        "planet_mask": np.asarray(state.planet_mask, dtype=bool),
        "fleet_owner": np.asarray(state.fleet_owner, dtype=np.int8),
        "fleet_x": np.asarray(state.fleet_x, dtype=np.float32),
        "fleet_y": np.asarray(state.fleet_y, dtype=np.float32),
        "fleet_angle": np.asarray(state.fleet_angle, dtype=np.float32),
        "fleet_ships": np.asarray(state.fleet_ships, dtype=np.int32),
        "fleet_mask": np.asarray(state.fleet_mask, dtype=bool),
        "step": np.asarray(state.step, dtype=np.int32).reshape(()),
        "angular_velocity": np.asarray(state.angular_velocity, dtype=np.float32).reshape(()),
        "planet_orbit_radius": np.asarray(state.planet_orbit_radius, dtype=np.float32),
        "planet_orbit_phase": np.asarray(state.planet_orbit_phase, dtype=np.float32),
        "planet_is_orbiting": np.asarray(state.planet_is_orbiting, dtype=bool),
        "home_planet_idx": np.zeros(constants.NUM_PLAYERS, dtype=np.int32),
    }


def flip_owners(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Swap player 0 ↔ player 1 owners; learner always ends up as player 0."""
    out = {k: v.copy() for k, v in d.items()}
    for field in ("planet_owner", "fleet_owner"):
        arr = out[field].copy()
        is0 = arr == np.int8(0)
        is1 = arr == np.int8(1)
        arr[is0] = np.int8(1)
        arr[is1] = np.int8(0)
        out[field] = arr
    h = out["home_planet_idx"].copy()
    h[0], h[1] = h[1], h[0]
    out["home_planet_idx"] = h
    return out


def remap_4p_to_2p(d: dict[str, np.ndarray], perspective: int) -> dict[str, np.ndarray]:
    """Collapse a 4P board snapshot into 2P ids for our NUM_PLAYERS=2 trainer.

    ``perspective`` becomes player 0; every other live player id becomes player 1.
    Neutral (-1) and padding (-2) are unchanged.
    """
    out = {k: v.copy() for k, v in d.items()}
    pers = np.int8(perspective)
    for field in ("planet_owner", "fleet_owner"):
        arr = out[field].copy()
        is_pers = arr == pers
        is_other_player = (arr >= np.int8(0)) & (~is_pers)
        arr[is_pers] = np.int8(0)
        arr[is_other_player] = np.int8(1)
        out[field] = arr
    return out


def stack_samples(samples: List[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("no samples to stack")
    keys = list(samples[0].keys())
    return {k: np.stack([s[k] for s in samples], axis=0) for k in keys}


def write_npz(stacked: dict[str, np.ndarray], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **stacked)


def print_buffer_stats(stacked: dict[str, np.ndarray], label: str = "") -> None:
    n = stacked["planet_ships"].shape[0]
    steps = stacked["step"]
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}total states : {n}")
    print(f"{prefix}step range   : [{steps.min()}, {steps.max()}] mean={steps.mean():.1f}")
    hist, edges = np.histogram(steps, bins=10)
    for i, c in enumerate(hist):
        print(f"{prefix}  step [{edges[i]:.0f},{edges[i+1]:.0f}): {c} ({100*c/n:.1f}%)")


def subsample_rows(stacked: dict[str, np.ndarray], n: int, seed: int = 0) -> dict[str, np.ndarray]:
    total = stacked["step"].shape[0]
    if n >= total:
        return stacked
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in stacked.items()}


def merge_balanced(
    paths: Iterable[str],
    out: Path,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Load each .npz and subsample to min(count) so sources are equally represented."""
    parts: List[dict[str, np.ndarray]] = []
    counts: List[int] = []
    for p in paths:
        data = dict(np.load(p))
        parts.append(data)
        counts.append(int(data["step"].shape[0]))
        print_buffer_stats(data, label=Path(p).name)

    cap = min(counts)
    print(f"\n[merge] balancing each source to {cap} states (min pool size)")
    subsampled = [subsample_rows(data, cap, seed=seed + i) for i, data in enumerate(parts)]
    keys = list(subsampled[0].keys())
    stacked = {k: np.concatenate([d[k] for d in subsampled], axis=0) for k in keys}
    write_npz(stacked, out)
    print_buffer_stats(stacked, label=out.name)
    return stacked
