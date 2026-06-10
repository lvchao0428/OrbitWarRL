#!/usr/bin/env python3
"""Hyperparameter sensitivity report from v14d search state."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "logs" / "v14d_search.state.json"


def _param_rows(trials: dict) -> list[dict]:
    rows = []
    for tid, t in trials.items():
        if t.get("status") not in ("probed", "passed", "failed", "aborted"):
            continue
        params = {**(t.get("env_params") or {}), **(t.get("ppo_params") or {})}
        rows.append(
            {
                "trial_id": tid,
                "phase": t.get("phase"),
                "kind": t.get("kind"),
                "score": float(t.get("score", 0)),
                "status": t.get("status"),
                "beats_v13c": t.get("beats_v13c", False),
                "gate": t.get("gate"),
                **params,
            }
        )
    return rows


def _sensitivity(rows: list[dict], param: str) -> list[dict]:
    buckets: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if param not in r:
            continue
        buckets[float(r[param])].append(r["score"])
    out = []
    for val, scores in sorted(buckets.items()):
        out.append(
            {
                "param": param,
                "value": val,
                "n": len(scores),
                "score_mean": sum(scores) / len(scores),
                "score_max": max(scores),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "v14d_sensitivity.md")
    ap.add_argument("--json-out", type=Path, default=ROOT / "logs" / "v14d_sensitivity.json")
    args = ap.parse_args()

    if not args.state.is_file():
        print(f"no state file: {args.state}")
        return 1

    state = json.loads(args.state.read_text())
    rows = _param_rows(state.get("trials", {}))
    if not rows:
        print("no completed trials")
        return 1

    params = sorted({k for r in rows for k in r if k.startswith(("ORBITWARS_", "lr_", "ent_", "clip_"))})
    report: dict = {"search_id": state.get("search_id"), "n_trials": len(rows), "params": {}}

    lines = [
        f"# v14d sensitivity — {state.get('search_id', '?')}",
        "",
        f"Trials: {len(rows)} | pipeline: {json.dumps(state.get('pipeline', {}), indent=None)[:200]}",
        "",
    ]

    for phase in ("a", "b", "c"):
        phase_rows = [r for r in rows if r["phase"] == phase]
        if not phase_rows:
            continue
        best = max(phase_rows, key=lambda r: r["score"])
        lines += [
            f"## Phase {phase.upper()}",
            "",
            f"Best probe/trial: `{best['trial_id']}` score={best['score']:.1f} status={best['status']}",
            "",
            "| param | value | n | mean score | max score |",
            "|-------|-------|---|------------|-----------|",
        ]
        for p in params:
            sens = _sensitivity(phase_rows, p)
            if not sens:
                continue
            report["params"].setdefault(phase, {})[p] = sens
            for s in sens:
                lines.append(
                    f"| {p} | {s['value']} | {s['n']} | {s['score_mean']:.1f} | {s['score_max']:.1f} |"
                )
        lines.append("")

    pipeline = state.get("pipeline", {})
    if pipeline.get("c"):
        c = pipeline["c"]
        lines += [
            "## Final vs v13c",
            "",
            f"- Phase C winner: `{c.get('trial_id')}`",
            f"- beats_v13c: **{c.get('beats_v13c', False)}**",
            f"- gate: {c.get('gate')}",
            "",
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.json_out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
