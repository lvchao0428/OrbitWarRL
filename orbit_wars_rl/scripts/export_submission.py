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


def _rewrite_top_level_int_assign(path: str, name: str, value: int) -> None:
    import re

    txt = Path(path).read_text(encoding="utf-8")
    pat = rf"^{name}\s*=\s*\d+"
    repl = f"{name} = {int(value)}"
    new_txt, n_sub = re.subn(pat, repl, txt, count=1, flags=re.MULTILINE)
    if n_sub != 1:
        raise RuntimeError(
            f"could not find top-level '{name} = <int>' in {path} for override"
        )
    Path(path).write_text(new_txt, encoding="utf-8")


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


def _read_template_mask_flags(template_path: str) -> dict[str, bool]:
    """Parse EMIT_HARD_STOP / F31_HARD_MASKS / emit-hold flags from template."""
    import re

    txt = Path(template_path).read_text(encoding="utf-8")
    emit_m = re.search(r"^EMIT_HARD_STOP\s*=\s*(\d+)", txt, flags=re.MULTILINE)
    emit_min_step_m = re.search(
        r"^EMIT_HARD_STOP_MIN_STEP\s*=\s*(\d+)", txt, flags=re.MULTILINE
    )
    flip_m = re.search(
        r"^F(?:31|32)_HARD_MASKS\s*=\s*(\d+)", txt, flags=re.MULTILINE
    )
    allow_hold_m = re.search(r"^ALLOW_HOLD\s*=\s*(\d+)", txt, flags=re.MULTILINE)
    force_worth_m = re.search(
        r"^FORCE_EMIT_WORTH_IT\s*=\s*(\d+)", txt, flags=re.MULTILINE
    )
    min_pct_m = re.search(r"^MIN_PCT_BIN\s*=\s*(\d+)", txt, flags=re.MULTILINE)
    return {
        "emit_hard_stop": bool(int(emit_m.group(1))) if emit_m else False,
        "emit_hard_stop_min_step": int(emit_min_step_m.group(1)) if emit_min_step_m else 1,
        "flip_hard_mask": bool(int(flip_m.group(1))) if flip_m else False,
        "allow_hold": bool(int(allow_hold_m.group(1))) if allow_hold_m else False,
        "force_emit_worth_it": bool(int(force_worth_m.group(1))) if force_worth_m else False,
        "min_pct_bin": int(min_pct_m.group(1)) if min_pct_m else 0,
    }


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
        if arch["planet_feat_dim"] == 41 and arch["global_feat_dim"] == 427 and ckpt_has_pair:
            hint = "submission_rl_v17.py"
        elif arch["planet_feat_dim"] == 39 and arch["global_feat_dim"] == 27 and ckpt_has_pair:
            hint = "submission_rl_v15.py"
        elif arch["planet_feat_dim"] == 39 and arch["global_feat_dim"] == 24 and ckpt_has_pair:
            hint = "submission_rl_v13c.py"
        elif arch["planet_feat_dim"] == 33 and arch["global_feat_dim"] == 18 and ckpt_has_pair:
            hint = "submission_rl_v11_f37.py or submission_rl_v11_f38.py"
        elif arch["planet_feat_dim"] == 33 and ckpt_has_pair:
            hint = "submission_rl_v11_f35.py"
        elif arch["planet_feat_dim"] == 28 and ckpt_has_pair:
            dpd = arch.get("dst_pair_dim", 0)
            if dpd == 7:
                hint = "submission_rl_v11_f32.py"
            elif dpd == 5:
                hint = "submission_rl_v11_f31.py"
            elif dpd == 4:
                hint = "submission_rl_v11_f26.py"
            else:
                hint = "submission_rl_v11_f27.py"
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
    ap.add_argument(
        "--emit-hard-stop-min-step",
        type=int,
        default=None,
        help="override EMIT_HARD_STOP_MIN_STEP in the output submission",
    )
    ap.add_argument(
        "--emit-hard-stop",
        type=int,
        choices=(0, 1),
        default=None,
        help="override EMIT_HARD_STOP in parity and the output submission",
    )
    ap.add_argument(
        "--flip-hard-mask",
        type=int,
        choices=(0, 1),
        default=None,
        help="override F31_HARD_MASKS/F32_HARD_MASKS in parity and the output submission",
    )
    ap.add_argument(
        "--allow-hold",
        type=int,
        choices=(0, 1),
        default=None,
        help="override ALLOW_HOLD in parity and the output submission",
    )
    ap.add_argument(
        "--force-emit-worth-it",
        type=int,
        choices=(0, 1),
        default=None,
        help="override FORCE_EMIT_WORTH_IT in parity and the output submission",
    )
    ap.add_argument(
        "--min-pct-bin",
        type=int,
        default=None,
        help="override MIN_PCT_BIN in parity and the output submission",
    )
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
        mask_flags = _read_template_mask_flags(args.template)
        if args.emit_hard_stop is not None:
            mask_flags["emit_hard_stop"] = bool(args.emit_hard_stop)
        if args.flip_hard_mask is not None:
            mask_flags["flip_hard_mask"] = bool(args.flip_hard_mask)
        if args.allow_hold is not None:
            mask_flags["allow_hold"] = bool(args.allow_hold)
        if args.force_emit_worth_it is not None:
            mask_flags["force_emit_worth_it"] = bool(args.force_emit_worth_it)
        if args.min_pct_bin is not None:
            mask_flags["min_pct_bin"] = int(args.min_pct_bin)
        emit_hard_stop_min_step = (
            args.emit_hard_stop_min_step
            if args.emit_hard_stop_min_step is not None
            else mask_flags["emit_hard_stop_min_step"]
        )
        print(f"[export] running parity test against {args.ckpt} "
              f"(tol={args.tol}, num_states={args.num_states})")
        print(
            f"[export] emit flags: allow_hold={mask_flags['allow_hold']} "
            f"force_emit_worth_it={mask_flags['force_emit_worth_it']} "
            f"min_pct_bin={mask_flags['min_pct_bin']}"
        )
        status = run_parity(
            args.ckpt,
            tol=args.tol,
            num_states=args.num_states,
            max_fleets_per_turn=arch["max_fleets_per_turn"],
            d_model=arch["d_model"],
            n_layers=arch["n_layers"],
            n_heads=arch["n_heads"],
            ff_dim=arch["ff_dim"],
            emit_hard_stop=mask_flags["emit_hard_stop"],
            emit_hard_stop_min_step=emit_hard_stop_min_step,
            flip_hard_mask=mask_flags["flip_hard_mask"],
            allow_hold=mask_flags["allow_hold"],
            force_emit_worth_it=mask_flags["force_emit_worth_it"],
            min_pct_bin=mask_flags["min_pct_bin"],
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
    if args.emit_hard_stop is not None:
        _rewrite_top_level_int_assign(out_path, "EMIT_HARD_STOP", args.emit_hard_stop)
    if args.flip_hard_mask is not None:
        hard_name = "F31_HARD_MASKS"
        try:
            _rewrite_top_level_int_assign(out_path, hard_name, args.flip_hard_mask)
        except RuntimeError:
            _rewrite_top_level_int_assign(out_path, "F32_HARD_MASKS", args.flip_hard_mask)
    if args.emit_hard_stop_min_step is not None:
        _rewrite_top_level_int_assign(
            out_path,
            "EMIT_HARD_STOP_MIN_STEP",
            args.emit_hard_stop_min_step,
        )
    if not args.skip_parity:
        _rewrite_top_level_int_assign(
            out_path, "ALLOW_HOLD", int(mask_flags["allow_hold"])
        )
        try:
            _rewrite_top_level_int_assign(
                out_path, "FORCE_EMIT_WORTH_IT", int(mask_flags["force_emit_worth_it"])
            )
        except RuntimeError:
            pass
        _rewrite_top_level_int_assign(
            out_path, "MIN_PCT_BIN", int(mask_flags["min_pct_bin"])
        )
    elif args.allow_hold is not None:
        _rewrite_top_level_int_assign(out_path, "ALLOW_HOLD", args.allow_hold)
        if args.force_emit_worth_it is not None:
            _rewrite_top_level_int_assign(
                out_path, "FORCE_EMIT_WORTH_IT", args.force_emit_worth_it
            )
        if args.min_pct_bin is not None:
            _rewrite_top_level_int_assign(out_path, "MIN_PCT_BIN", args.min_pct_bin)
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
