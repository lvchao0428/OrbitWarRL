"""Run a kaggle orbit_wars game and save native JSON + HTML replay.

Kaggle Environments supports ``render(mode="html")`` — an interactive step-through
player (same widget as competition notebooks). This script runs one episode,
writes ``.json`` (episode dump) and ``.html`` (open in browser).

Usage (5090 or local CPU, ~30-120s per game):

    # After export:
    python -m orbit_wars_rl.scripts.replay_html \
        --agent-a submission_rl_v11_k8_no_emit.py \
        --agent-b submission_v20_0513.py \
        --seed 0 \
        --out-dir logs/replay_html/v11_k8_no_emit_seed0

    # Open replay:
    #   logs/replay_html/v11_k8_no_emit_seed0/replay.html

Reload JSON → HTML only (no re-run):

    python -m orbit_wars_rl.scripts.replay_html \
        --from-json logs/replay_html/v11_k8_no_emit_seed0/replay.json \
        --out-dir logs/replay_html/v11_k8_no_emit_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kaggle_environments import make


def _write_replay(env, out_dir: Path, tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}.json"
    html_path = out_dir / f"{tag}.html"

    episode_json = env.render(mode="json")
    if isinstance(episode_json, str):
        json_path.write_text(episode_json, encoding="utf-8")
    else:
        json_path.write_text(json.dumps(episode_json, indent=2), encoding="utf-8")

    html = env.render(mode="html")
    if not isinstance(html, str):
        raise RuntimeError(f"render(mode=html) returned {type(html)}, expected str")
    html_path.write_text(html, encoding="utf-8")

    return json_path, html_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Save kaggle native JSON + HTML replay.")
    ap.add_argument("--agent-a", type=str, default="", help="player 0 submission .py")
    ap.add_argument("--agent-b", type=str, default="submission_v20_0513.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="replay")
    ap.add_argument(
        "--from-json",
        type=str,
        default="",
        help="skip game run; load existing episode JSON and render HTML only",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.from_json:
        raw = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        steps = raw.get("steps") if isinstance(raw, dict) else raw
        env = make("orbit_wars", steps=steps, debug=False)
    else:
        if not args.agent_a:
            print("ERR: --agent-a required unless --from-json", file=sys.stderr)
            return 2
        for p in (args.agent_a, args.agent_b):
            if not Path(p).exists():
                print(f"ERR: agent not found: {p}", file=sys.stderr)
                return 2
        env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
        print(f"running seed={args.seed}  A={args.agent_a}  B={args.agent_b} ...", flush=True)
        env.run([args.agent_a, args.agent_b])
        last = env.steps[-1]
        r0, r1 = last[0]["reward"], last[1]["reward"]
        print(f"done: steps={len(env.steps)}  p0_reward={r0}  p1_reward={r1}", flush=True)

    json_path, html_path = _write_replay(env, out_dir, args.tag)
    print(f"[saved] {json_path}")
    print(f"[saved] {html_path}")
    print(f"Open in browser: file://{html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
