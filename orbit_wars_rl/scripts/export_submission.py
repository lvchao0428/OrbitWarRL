"""Inject a trained checkpoint into ``submission_rl_v1.py``.

Pipeline:
  1. Load ``ckpt_XXXXXX.pkl`` from training side.
  2. Flatten flax params to ``{str: np.ndarray}``.
  3. Run a one-shot parity check (jax forward vs numpy forward) so we never
     ship a submission whose embedded numpy weights disagree with what was
     actually trained.
  4. Serialize as npz, zlib-compress, base64-encode, and substitute the
     ``WEIGHTS_B64 = "__WEIGHTS_B64__"`` placeholder in ``submission_rl_v1.py``
     (or a copy under ``--out``) with the encoded payload.

Usage:
    python -m orbit_wars_rl.scripts.export_submission \
        --ckpt ckpt_mvp/ckpt_000049.pkl \
        --template submission_rl_v1.py \
        --out submission_rl_v1_filled.py
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import zlib
from pathlib import Path

import numpy as np

from orbit_wars_rl.inference.test_parity import run_parity
from orbit_wars_rl.inference.weights import (
    assert_expected_keys,
    infer_arch_from_flat,
    load_flat_params,
)


def _serialize(flat: dict[str, np.ndarray]) -> str:
    buf = io.BytesIO()
    np.savez(buf, **flat)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    return base64.b64encode(compressed).decode("ascii")


def _inject(template_path: str, out_path: str, payload_b64: str) -> None:
    """Replace the ``WEIGHTS_B64 = "..."`` assignment line in the template.

    Uses a regex anchored at start-of-line so a docstring or comment that
    happens to contain the literal placeholder string is NOT substituted
    (we hit that exact bug while iterating on the v4 template, which
    silently produced a placeholder-only submission that imported fine
    but errored at first ``agent()`` call, returning ``[]`` via the
    ``try/except`` swallower).
    """
    import re

    txt = Path(template_path).read_text(encoding="utf-8")
    new_line = f'WEIGHTS_B64 = "{payload_b64}"'
    # Anchored at start of line; only matches top-level assignment.
    new_txt, n_sub = re.subn(
        r'^WEIGHTS_B64 = "[^"]*"', new_line, txt, count=0, flags=re.MULTILINE,
    )
    if n_sub == 0:
        raise RuntimeError(
            f"could not find a top-level 'WEIGHTS_B64 = \"...\"' line in "
            f"{template_path}"
        )
    if n_sub > 1:
        raise RuntimeError(
            f"template {template_path} has {n_sub} top-level 'WEIGHTS_B64 = "
            f"\"...\"' lines; expected exactly one. Refusing to inject "
            f"ambiguously."
        )
    # Cheap post-write verification: re-read and confirm the placeholder is
    # gone from the assignment lines (a docstring containing the literal
    # string is fine; we only care about top-level assigns).
    Path(out_path).write_text(new_txt, encoding="utf-8")
    written = Path(out_path).read_text(encoding="utf-8")
    bad = re.findall(r'^WEIGHTS_B64 = "__WEIGHTS_B64__"', written, flags=re.MULTILINE)
    if bad:
        raise RuntimeError(
            f"after inject, {out_path} still has a placeholder assignment "
            f"line; injector is broken."
        )


def _read_template_dims(template_path: str) -> dict[str, int]:
    """Parse PLANET/GLOBAL feat dims and K from submission template source."""
    import re

    txt = Path(template_path).read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for name, pat in (
        ("planet_feat_dim", r"^PLANET_FEAT_DIM\s*=\s*(\d+)"),
        ("global_feat_dim", r"^GLOBAL_FEAT_DIM\s*=\s*(\d+)"),
        ("max_fleets_per_turn", r"^MAX_FLEETS_PER_TURN\s*=\s*(\d+)"),
    ):
        m = re.search(pat, txt, flags=re.MULTILINE)
        if not m:
            raise RuntimeError(
                f"template {template_path} missing {name} constant "
                f"(expected line like PLANET_FEAT_DIM = 22)"
            )
        out[name] = int(m.group(1))
    # has_pair: True iff template defines SUN_BLOCK_THRESH (only f26+ does).
    has_pair = re.search(r"^SUN_BLOCK_THRESH\s*=", txt, flags=re.MULTILINE) is not None
    out["has_pair"] = 1 if has_pair else 0
    return out


def _assert_template_matches_ckpt(template_path: str, arch: dict[str, int]) -> None:
    tpl = _read_template_dims(template_path)
    mismatches = []
    if tpl["planet_feat_dim"] != arch["planet_feat_dim"]:
        mismatches.append(
            f"PLANET_FEAT_DIM template={tpl['planet_feat_dim']} "
            f"ckpt={arch['planet_feat_dim']}"
        )
    if tpl["global_feat_dim"] != arch["global_feat_dim"]:
        mismatches.append(
            f"GLOBAL_FEAT_DIM template={tpl['global_feat_dim']} "
            f"ckpt={arch['global_feat_dim']}"
        )
    if tpl["max_fleets_per_turn"] != arch["max_fleets_per_turn"]:
        mismatches.append(
            f"MAX_FLEETS_PER_TURN template={tpl['max_fleets_per_turn']} "
            f"ckpt={arch['max_fleets_per_turn']}"
        )
    ckpt_has_pair = bool(arch.get("has_pair", False))
    if bool(tpl["has_pair"]) != ckpt_has_pair:
        mismatches.append(
            f"HAS_PAIR_FEATS template={bool(tpl['has_pair'])} "
            f"ckpt={ckpt_has_pair}"
        )
    if mismatches:
        if arch["planet_feat_dim"] == 28 and ckpt_has_pair:
            hint = "submission_rl_v11_f26.py"
        elif arch["planet_feat_dim"] == 28:
            hint = "submission_rl_v11_f25.py"
        elif arch["planet_feat_dim"] == 22:
            hint = "submission_rl_v11.py"
        else:
            hint = "submission_rl_v4.py"
        raise RuntimeError(
            "template/ckpt architecture mismatch — refusing to export:\n  "
            + "\n  ".join(mismatches)
            + f"\n  Use --template {hint}"
        )


def _smoke_obs_v11() -> dict:
    """Dummy obs with orbital motion + inbound foe fleet (v11 A1' not all zeros)."""
    return {
        "player": 0,
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 120, 3],
            [1, -1, 80.0, 20.0, 1.0, 40, 2],
            [2, -1, 20.0, 80.0, 1.0, 25, 2],
            [3, 1, 80.0, 80.0, 1.0, 80, 3],
        ],
        # foe fleet heading toward player home (planet 0)
        "fleets": [
            [0, 1, 55.0, 35.0, 0.785, 0, 25],
        ],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 120, 3],
            [1, -1, 80.0, 20.0, 1.0, 40, 2],
            [2, -1, 20.0, 80.0, 1.0, 25, 2],
            [3, 1, 80.0, 80.0, 1.0, 80, 3],
        ],
        "angular_velocity": 0.04,
    }


def _smoke_obs_v4() -> dict:
    return {
        "player": 0,
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 30, 3],
            [1, -1, 80.0, 20.0, 1.0, 15, 2],
            [2, -1, 20.0, 80.0, 1.0, 15, 2],
            [3, 1, 80.0, 80.0, 1.0, 30, 3],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 30, 3],
            [1, -1, 80.0, 20.0, 1.0, 15, 2],
            [2, -1, 20.0, 80.0, 1.0, 15, 2],
            [3, 1, 80.0, 80.0, 1.0, 30, 3],
        ],
        "angular_velocity": 0.04,
    }


def main() -> int:
    # Parity only needs one forward pass; avoid grabbing GPU while training runs.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to ckpt_XXXXXX.pkl")
    ap.add_argument(
        "--template",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "submission_rl_v4.py"),
        help="single-file submission template (defaults to submission_rl_v4.py at repo root)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="output submission file. defaults to <template>",
    )
    ap.add_argument("--skip-parity", action="store_true",
                    help="skip jax-vs-numpy parity test (NOT recommended)")
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="logit drift threshold (informational; argmax must always match)")
    ap.add_argument("--num-states", type=int, default=16)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[export] ckpt not found: {args.ckpt}", file=sys.stderr)
        return 2

    out_path = args.out or args.template
    flat = load_flat_params(args.ckpt)
    arch = infer_arch_from_flat(flat)
    assert_expected_keys(flat, n_layers=arch["n_layers"])
    print(f"[export] loaded {len(flat)} param tensors; "
          f"total floats = {sum(v.size for v in flat.values()):,}")
    print(f"[export] arch: planet={arch['planet_feat_dim']} global={arch['global_feat_dim']} "
          f"K={arch['max_fleets_per_turn']} d_model={arch['d_model']}")

    try:
        _assert_template_matches_ckpt(args.template, arch)
    except RuntimeError as e:
        print(f"[export] {e}", file=sys.stderr)
        return 5

    if not args.skip_parity:
        print(f"[export] running parity test against {args.ckpt} "
              f"(tol={args.tol}, num_states={args.num_states})")
        status = run_parity(
            args.ckpt,
            tol=args.tol,
            num_states=args.num_states,
            max_fleets_per_turn=arch["max_fleets_per_turn"],
            d_model=arch["d_model"],
            n_layers=arch["n_layers"],
            n_heads=arch["n_heads"],
            ff_dim=arch["ff_dim"],
        )
        if status != 0:
            print("[export] parity FAILED (argmax disagreement); "
                  "refusing to write submission.", file=sys.stderr)
            return 3
    else:
        print("[export] WARNING: skipping parity test")

    payload = _serialize(flat)
    raw_size = sum(v.nbytes for v in flat.values())
    enc_size = len(payload)
    print(f"[export] raw weights ~{raw_size/1024:.1f}KB, "
          f"compressed+base64 ~{enc_size/1024:.1f}KB")

    _inject(args.template, out_path, payload)
    print(f"[export] wrote {out_path}")

    # Smoke test: forward must not raise; empty moves only warn (v11 may z0 on calm obs).
    print("[export] importing and smoke-testing agent()...")
    import importlib.util as iu
    spec = iu.spec_from_file_location("_submission_rl_test", out_path)
    mod = iu.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    fake_obs = (
        _smoke_obs_v11() if arch["planet_feat_dim"] >= 22 else _smoke_obs_v4()
    )
    # Direct forward (surfaces shape bugs); agent() swallows exceptions → [].
    try:
        W = mod._load_weights()
        enc = mod.encode_obs(fake_obs, player=0, step=0, episode_steps=500)
        # f26 templates accept home_idx; older templates don't.
        try:
            src, dst, pct = mod.greedy_multi_action(W, enc, home_idx=0)
        except TypeError:
            src, dst, pct = mod.greedy_multi_action(W, enc)
        print(f"[export] smoke forward: {len(src)} launch(es) "
              f"src={src[:3]} dst={dst[:3]} pct={pct[:3]}")
    except Exception as e:
        print(f"[export] smoke forward FAILED: {e}", file=sys.stderr)
        return 4

    moves = mod.agent(fake_obs, {"episodeSteps": 500})
    print(f"[export] smoke agent() moves: {moves}")
    if not moves:
        print("[export] WARNING: agent() returned [] (forward ok — may be emit-stop on calm obs)")
        print("  Verify with replay_analyze vs v20.")
    if not isinstance(moves, list):
        print("[export] agent() did not return a list!", file=sys.stderr)
        return 4
    for m in moves:
        if m and (not isinstance(m, list) or len(m) != 3):
            print(f"[export] agent() returned malformed move: {m}", file=sys.stderr)
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
