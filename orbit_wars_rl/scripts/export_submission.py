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
    load_flat_params,
)


def _serialize(flat: dict[str, np.ndarray]) -> str:
    buf = io.BytesIO()
    np.savez(buf, **flat)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    return base64.b64encode(compressed).decode("ascii")


def _inject(template_path: str, out_path: str, payload_b64: str) -> None:
    txt = Path(template_path).read_text(encoding="utf-8")
    placeholder = 'WEIGHTS_B64 = "__WEIGHTS_B64__"'
    if placeholder not in txt:
        # also accept the case where it was already filled (rebuild).
        import re
        new_line = f'WEIGHTS_B64 = "{payload_b64}"'
        new_txt, n = re.subn(r'^WEIGHTS_B64 = "[^"]*"', new_line, txt, count=1, flags=re.MULTILINE)
        if n == 0:
            raise RuntimeError(
                f"could not find a WEIGHTS_B64 = \"...\" line in {template_path}"
            )
        Path(out_path).write_text(new_txt, encoding="utf-8")
        return
    new_line = f'WEIGHTS_B64 = "{payload_b64}"'
    Path(out_path).write_text(txt.replace(placeholder, new_line, 1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to ckpt_XXXXXX.pkl")
    ap.add_argument(
        "--template",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "submission_rl_v1.py"),
        help="single-file submission template (defaults to submission_rl_v1.py at repo root)",
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
    assert_expected_keys(flat, n_layers=2)
    print(f"[export] loaded {len(flat)} param tensors; "
          f"total floats = {sum(v.size for v in flat.values()):,}")

    if not args.skip_parity:
        print(f"[export] running parity test against {args.ckpt} "
              f"(tol={args.tol}, num_states={args.num_states})")
        status = run_parity(args.ckpt, tol=args.tol, num_states=args.num_states)
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

    # Smoke test: import the freshly-written submission and call agent() on a
    # dummy obs to verify it doesn't crash on import or first call.
    print("[export] importing and smoke-testing agent()...")
    import importlib.util as iu
    spec = iu.spec_from_file_location("_submission_rl_test", out_path)
    mod = iu.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    fake_obs = {
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
    }
    moves = mod.agent(fake_obs, {"episodeSteps": 500})
    print(f"[export] smoke moves: {moves}")
    if not isinstance(moves, list):
        print("[export] agent() did not return a list!", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
