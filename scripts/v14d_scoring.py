#!/usr/bin/env python3
"""Unified scoring for v14d search — targets beating v13c_final baseline."""

from __future__ import annotations

from pathlib import Path
from statistics import median

# v13c_final vs v20 (4 games): WLD=1/4, spf≈64, flip≈11%, z0≈45%
V13C_BASELINE = {
    "wins": 1,
    "losses": 3,
    "draws": 0,
    "spf": 64.0,
    "flip": 11.0,
    "z0": 45.0,
    "garr": 25.0,
}


def _import_gate():
    from v14d_gate_check import _read_last_eval, _read_train_metrics, gate_a, gate_b, gate_c

    return _read_last_eval, _read_train_metrics, {"a": gate_a, "b": gate_b, "c": gate_c}


def score_trial(phase: str, log_path: Path, *, probe: bool = False) -> dict:
    """Return {score, gate_ok, metrics, beats_v13c, detail}."""
    _read_last_eval, _read_train_metrics, gates = _import_gate()
    ok, gate_msg = gates[phase](log_path)
    ev = _read_last_eval(log_path)
    rows = _read_train_metrics(log_path, 30)

    score = 0.0
    detail: list[str] = []
    beats_v13c = False

    if ok:
        score += 1000.0
        detail.append("gate_pass")

    if phase == "a" and rows:
        garr = median(r["garr"] for r in rows)
        z0 = median(r["z0"] for r in rows)
        spf = median(r["spf"] for r in rows)
        emits = median(r["emits"] for r in rows)
        score += max(0.0, 1.0 - abs(garr - 70.0) / 70.0) * 25
        score += max(0.0, 1.0 - abs(z0 - 0.45) / 0.45) * 25
        score += max(0.0, 1.0 - abs(spf - 40.0) / 40.0) * 20
        score += max(0.0, 1.0 - abs(emits - 0.55) / 0.55) * 15
        if ev:
            score += min(ev["spf"], 80.0) * 0.15
            score += ev["flip"] * 0.2
            if ev["spf"] >= 3.0:
                score += 10.0
        metrics = {"garr": garr, "z0": z0, "spf": spf, "emits": emits}
    elif ev:
        score += ev["wins"] * 120.0
        score += ev["flip"] * 1.5
        score += min(ev["spf"], 120.0) * 0.4
        score += max(0.0, 40.0 - abs(ev["z0"] - V13C_BASELINE["z0"])) * 0.3
        if ev["wins"] > V13C_BASELINE["wins"]:
            score += 80.0
            beats_v13c = True
        elif ev["wins"] == V13C_BASELINE["wins"]:
            better = (
                ev["flip"] > V13C_BASELINE["flip"]
                or ev["spf"] > V13C_BASELINE["spf"]
            )
            if better:
                score += 40.0
                beats_v13c = True
        if ev["wins"] < V13C_BASELINE["wins"]:
            score -= 30.0
        metrics = dict(ev)
    else:
        metrics = {}

    if probe and not ok:
        score *= 0.85

    return {
        "score": score,
        "gate_ok": ok,
        "gate_msg": gate_msg,
        "metrics": metrics,
        "beats_v13c": beats_v13c,
        "detail": detail,
    }


def beats_v13c_strict(ev: dict | None) -> bool:
    if ev is None:
        return False
    if ev["wins"] > V13C_BASELINE["wins"]:
        return True
    if ev["wins"] < V13C_BASELINE["wins"]:
        return False
    return ev["flip"] > V13C_BASELINE["flip"] or ev["spf"] > V13C_BASELINE["spf"]
