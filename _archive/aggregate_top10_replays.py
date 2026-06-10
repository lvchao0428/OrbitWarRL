"""Aggregate behaviour metrics across top-10% Orbit Wars replay JSONs.

Reads ``top10_episodes_*/episodes/episodes/*.json`` + ``manifest.csv``,
reuses ``analyze_expert_replay.analyze_episode``, and emits cohort stats
(winner vs loser, first-80 vs full, Day5-relevant gates).

Usage:
    python -m orbit_wars_rl.scripts.aggregate_top10_replays \
        --dataset-dir top10_episodes_2026-05-04 \
        --out logs/top10_aggregate_2026-05-04.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from orbit_wars_rl.scripts.analyze_expert_replay import analyze_episode


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[k])


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _player_extra_metrics(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics not in analyze_expert_replay aggregates."""
    if not series:
        return {}

    emit_counts = [r["num_launches"] for r in series]
    n = len(emit_counts)
    emit2 = sum(1 for v in emit_counts if v >= 2) / n
    emit3 = sum(1 for v in emit_counts if v >= 3) / n
    emit1_only = sum(1 for v in emit_counts if v == 1) / n

    launch_sizes: List[int] = []
    for r in series:
        if r["num_launches"] > 0 and r["ships_per_fleet_max"] > 0:
            # upper bound: at least one launch of max size; for fractions use max per turn
            launch_sizes.append(int(r["ships_per_fleet_max"]))
            if r["num_launches"] > 1 and r["ships_launched"] > 0:
                # approximate mean launch size this turn
                avg = r["ships_launched"] / r["num_launches"]
                for _ in range(r["num_launches"] - 1):
                    launch_sizes.append(int(avg))

    large100 = sum(1 for r in series if r["ships_per_fleet_max"] >= 100) / n
    large500 = sum(1 for r in series if r["ships_per_fleet_max"] >= 500) / n

    garr = [r["garrison_total"] for r in series]
    peak_garr = max(garr) if garr else 0
    peak_garr_turn = garr.index(peak_garr) if garr else 0

    fleets = [r["fleet_count_alive"] for r in series]

    first80 = series[:80]
    rest = series[80:]

    def _prod_mean(rows: List[Dict[str, Any]]) -> float:
        return _mean([r["production_share"] for r in rows]) if rows else 0.0

    return dict(
        emit2_rate=round(emit2, 4),
        emit3_rate=round(emit3, 4),
        emit1_only_rate=round(emit1_only, 4),
        turns_with_launch_ge100=round(large100, 4),
        turns_with_launch_ge500=round(large500, 4),
        peak_garrison=int(peak_garr),
        peak_garrison_turn=int(peak_garr_turn),
        mean_fleets_in_flight=round(_mean(fleets), 2),
        prod_share_turn80=round(_prod_mean(first80), 3) if len(first80) >= 80 else None,
        prod_share_after80=round(_prod_mean(rest), 3) if rest else None,
        garrison_turn80=int(first80[-1]["garrison_total"]) if len(first80) >= 80 else None,
        planet_count_turn80=int(first80[-1]["planet_count"]) if len(first80) >= 80 else None,
    )


def _cohort_row(values: List[float]) -> Dict[str, float]:
    if not values:
        return dict(n=0, mean=0.0, p50=0.0, p95=0.0)
    return dict(
        n=len(values),
        mean=round(_mean(values), 2),
        p50=round(_pct(values, 0.50), 2),
        p95=round(_pct(values, 0.95), 2),
    )


def _extract_player_snapshot(
    ep: Dict[str, Any], pp: Dict[str, Any], window: str
) -> Dict[str, Any]:
    w = pp["first80"] if window == "first80" else pp["full"]
    series = ep["per_turn_series"][pp["player"]]
    extra = _player_extra_metrics(series[:80] if window == "first80" else series)
    return dict(
        spf_mean=w["spf_mean"],
        spf_max=float(w.get("spf_max_global", 0)),
        zero_emit_rate=w["zero_emit_rate"],
        mean_garrison=w["mean_garrison"],
        mean_planets=w["mean_planets"],
        prod_share=w["mean_production_share"],
        max_launch=pp["max_launch_size"],
        p95_launch=pp["p95_launch_size"],
        decisive_turn=pp["decisive_turn"],
        decisive_ships=pp["decisive_ships"],
        **extra,
    )


def aggregate_dataset(dataset_dir: Path, max_episodes: Optional[int] = None) -> Dict[str, Any]:
    replay_dir = dataset_dir / "episodes" / "episodes"
    manifest_path = dataset_dir / "manifest.csv"
    if not replay_dir.is_dir():
        raise FileNotFoundError(replay_dir)

    manifest_rows: List[Dict[str, str]] = []
    if manifest_path.is_file():
        with manifest_path.open() as f:
            manifest_rows = list(csv.DictReader(f))
    score_by_id = {r["episode_id"]: float(r["sum_score"]) for r in manifest_rows}

    paths = sorted(replay_dir.glob("*.json"))
    if max_episodes:
        paths = paths[:max_episodes]

    winner_snaps: Dict[str, List[Dict[str, float]]] = {"first80": [], "full": []}
    loser_snaps: Dict[str, List[Dict[str, float]]] = {"first80": [], "full": []}
    # flat keys for cohort
    metric_keys = [
        "spf_mean", "spf_max", "zero_emit_rate", "mean_garrison", "mean_planets",
        "prod_share", "max_launch", "p95_launch", "emit2_rate", "emit3_rate",
        "mean_fleets_in_flight", "peak_garrison",
    ]
    cohort_w: Dict[str, Dict[str, List[float]]] = {
        w: {k: [] for k in metric_keys} for w in ("first80", "full")
    }
    cohort_l: Dict[str, Dict[str, List[float]]] = {
        w: {k: [] for k in metric_keys} for w in ("first80", "full")
    }

    episode_summaries: List[Dict[str, Any]] = []
    errors: List[str] = []
    n_ep = 0
    n_4p = 0
    lengths: List[int] = []

    for path in paths:
        try:
            ep = analyze_episode(path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path.name}: {e}")
            continue

        n_ep += 1
        if ep["num_players"] == 4:
            n_4p += 1
        lengths.append(ep["episode_length"])

        rewards = ep["final_rewards"] or []
        max_r = max(rewards) if rewards else 0
        winners = [i for i, r in enumerate(rewards) if r == max_r and r > 0]

        ep_summary = dict(
            episode_id=ep.get("episode_id") or path.stem,
            path=str(path),
            sum_score=score_by_id.get(path.stem),
            episode_length=ep["episode_length"],
            num_players=ep["num_players"],
            winners=winners,
            rewards=rewards,
            players=[],
        )

        for pp in ep["per_player"]:
            for window in ("first80", "full"):
                snap = _extract_player_snapshot(ep, pp, window)
                is_winner = pp["player"] in winners
                bucket = cohort_w if is_winner else cohort_l
                for k in metric_keys:
                    v = snap.get(k)
                    if v is not None and isinstance(v, (int, float)):
                        bucket[window][k].append(float(v))

            ep_summary["players"].append(dict(
                player=pp["player"],
                reward=pp["final_reward"],
                is_winner=pp["player"] in winners,
                first80=_extract_player_snapshot(ep, pp, "first80"),
                full=_extract_player_snapshot(ep, pp, "full"),
                garrison_decile=pp["garrison_decile_curve"],
            ))

        episode_summaries.append(ep_summary)

    def _build_cohort(cohort: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for window in ("first80", "full"):
            out[window] = {k: _cohort_row(v) for k, v in cohort[window].items()}
        return out

    # Top episodes by manifest sum_score
    ranked = sorted(
        [e for e in episode_summaries if e.get("sum_score") is not None],
        key=lambda x: x["sum_score"],
        reverse=True,
    )[:15]

    top_episode_table = []
    for e in ranked:
        w = next(p for p in e["players"] if p["is_winner"])
        top_episode_table.append(dict(
            episode_id=e["episode_id"],
            sum_score=round(e["sum_score"], 1),
            turns=e["episode_length"],
            winner_p=w["player"],
            first80=w["first80"],
            full=w["full"],
        ))

    return dict(
        dataset=str(dataset_dir),
        n_files=len(paths),
        n_parsed=n_ep,
        n_errors=len(errors),
        errors=errors[:20],
        n_4p=n_4p,
        episode_length=_cohort_row([float(x) for x in lengths]),
        winners=_build_cohort(cohort_w),
        losers=_build_cohort(cohort_l),
        top_episodes=top_episode_table,
        episode_summaries=episode_summaries,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args(argv)

    result = aggregate_dataset(args.dataset_dir, max_episodes=args.max_episodes)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out} ({result['n_parsed']} episodes)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
