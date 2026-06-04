"""Mini replay gate vs v20 during training (P2: train/replay alignment).

Runs ``scripts/quick_replay.sh`` on CPU with a small game count, then parses
the first-80 summary line. Intended for ``eval_every`` checkpoints only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


_SUMMARY_LINE = re.compile(
    r"spf=([\d.]+)\s+garr=([\d.]+)\s+flip=([\d.]+)%\s+"
    r"z0=([\d.]+)%\s+e8=([\d.]+)%\s+e2\+=([\d.]+)%\s+WLD=([\d/]+)"
)


def parse_gate_summary(summary_path: str | Path) -> dict[str, Any]:
    """Parse ``logs/replay_analyze/<tag>_vs_v20.summary.txt`` first line."""
    path = Path(summary_path)
    if not path.is_file():
        raise FileNotFoundError(summary_path)
    for line in path.read_text().splitlines():
        if line.startswith("tag="):
            m = _SUMMARY_LINE.search(line)
            if not m:
                raise ValueError(f"cannot parse summary line: {line!r}")
            w, l, d = (int(x) for x in m.group(7).split("/"))
            return {
                "spf": float(m.group(1)),
                "garr": float(m.group(2)),
                "flip_pct": float(m.group(3)),
                "z0_pct": float(m.group(4)),
                "e8_pct": float(m.group(5)),
                "e2_plus_pct": float(m.group(6)),
                "wins": w,
                "losses": l,
                "draws": d,
                "summary_line": line.strip(),
            }
    raise ValueError(f"no tag= line in {path}")


def run_v20_mini_gate(
    ckpt_path: str | Path,
    tag: str,
    *,
    project_root: str | Path | None = None,
    num_games: int = 3,
    seed_base: int = 0,
    python: str | None = None,
    timeout_sec: int = 3600,
) -> dict[str, Any]:
    """Export ckpt + replay vs v20; return parsed first-80 metrics."""
    root = Path(project_root or Path(__file__).resolve().parents[2])
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    env = os.environ.copy()
    env["NUM_GAMES"] = str(num_games)
    env["SEED_BASE"] = str(seed_base)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    if python:
        env["PYTHON"] = python

    script = root / "scripts" / "quick_replay.sh"
    proc = subprocess.run(
        ["bash", str(script), str(ckpt_path), tag],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    summary = root / "logs" / "replay_analyze" / f"{tag}_vs_v20.summary.txt"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(
            f"v20 mini gate failed (rc={proc.returncode}): {tail}"
        )
    out = parse_gate_summary(summary)
    out["tag"] = tag
    out["summary_path"] = str(summary)
    return out
