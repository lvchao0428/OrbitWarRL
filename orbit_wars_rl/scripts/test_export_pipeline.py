"""Unit test for the submission export pipeline.

What this guards against (this is the test we WISH we had on Day 2):

    A. ``_inject`` must completely replace ``WEIGHTS_B64 = "__WEIGHTS_B64__"``
       in the v4 template -- the placeholder line MUST NOT remain.
    B. ``_inject`` must refuse to write if there is more than one top-level
       ``WEIGHTS_B64 = "..."`` assignment in the template (this is exactly
       the regression that lost us a training run, see DAY2 §9.10: a docstring
       containing the literal placeholder string fooled ``str.replace`` into
       overwriting the docstring instead of the real assignment).
    C. The post-inject file must import cleanly and ``agent()`` must return
       a ``list`` (even an empty one is acceptable on the smoke obs; the
       point is "no silent ``except Exception`` swallow").

This test does **not** require any real training checkpoint -- it builds
``params`` from ``model.init`` so it can run in CI / on the dev mac.

Usage:
    python -m orbit_wars_rl.scripts.test_export_pipeline
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.features import encode
from orbit_wars_rl.inference.weights import flatten_params
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.scripts.export_submission import _inject, _serialize


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "submission_rl_v4.py"

_PLACEHOLDER_LINE = 'WEIGHTS_B64 = "__WEIGHTS_B64__"'
_ANCHORED_RE = re.compile(r'^WEIGHTS_B64 = "[^"]*"', flags=re.MULTILINE)


def _build_fresh_init_flat() -> dict[str, np.ndarray]:
    """Return a ``flatten_params``-style dict from fresh ``model.init``."""
    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    state = env.reset(jax.random.PRNGKey(0))
    obs = encode(state, 0, 60)
    model = ActorCritic(max_fleets_per_turn=constants.MAX_FLEETS_PER_TURN)
    params = model.init(
        jax.random.PRNGKey(0), obs, jax.random.PRNGKey(1), state.planet_ships,
    )
    return flatten_params(params)


def _make_smoke_obs() -> dict:
    """A minimal in-distribution obs for the v4 smoke call.

    ``angular_velocity`` is drawn from the training distribution
    [0.025, 0.05]; 0.04 is a comfortable midpoint. Four corner planets with
    a couple of pieces of ship mass each so the policy actually has
    decisions to make.
    """
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


def _import_submission_from(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _check_template_match_count() -> int:
    """A. The v4 template must have EXACTLY one anchored WEIGHTS_B64 line."""
    txt = TEMPLATE_PATH.read_text(encoding="utf-8")
    matches = _ANCHORED_RE.findall(txt)
    n = len(matches)
    if n != 1:
        print(f"  [FAIL] template has {n} top-level WEIGHTS_B64 assigns, expected 1")
        for m in matches:
            print(f"         -> {m!r}")
        return 1
    if matches[0] != _PLACEHOLDER_LINE:
        print(f"  [FAIL] template's WEIGHTS_B64 line is not the placeholder")
        print(f"         got: {matches[0]!r}")
        return 1
    print("  [OK ] template has exactly one placeholder assignment")
    return 0


def _check_inject_replaces_placeholder() -> int:
    """B. ``_inject`` produces a file where the placeholder is gone."""
    flat = _build_fresh_init_flat()
    payload = _serialize(flat)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        _inject(str(TEMPLATE_PATH), str(out_path), payload)
        written = out_path.read_text(encoding="utf-8")
        if _PLACEHOLDER_LINE in written:
            placeholder_lines = [
                (i + 1, line) for i, line in enumerate(written.splitlines())
                if line.strip() == _PLACEHOLDER_LINE
            ]
            print(f"  [FAIL] placeholder line still present after inject:")
            for ln, txt in placeholder_lines[:3]:
                print(f"         L{ln}: {txt!r}")
            return 1
        new_matches = _ANCHORED_RE.findall(written)
        if len(new_matches) != 1:
            print(f"  [FAIL] post-inject expected 1 assignment, got {len(new_matches)}")
            return 1
        if new_matches[0] == _PLACEHOLDER_LINE:
            print("  [FAIL] post-inject assignment still equals placeholder")
            return 1
        print(f"  [OK ] post-inject placeholder removed; "
              f"new line is {len(new_matches[0])} chars")
        # C. The post-inject file imports + agent() returns a list.
        mod = _import_submission_from(out_path, "_test_export_pipeline_v4")
        moves = mod.agent(_make_smoke_obs(), {"episodeSteps": 500})
        if not isinstance(moves, list):
            print(f"  [FAIL] agent() did not return list, got {type(moves)}")
            return 1
        for m in moves:
            if not (isinstance(m, list) and len(m) == 3):
                print(f"  [FAIL] agent() returned malformed move: {m!r}")
                return 1
        print(f"  [OK ] imported submission, agent() returned list of {len(moves)} moves")
        return 0
    finally:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass


def _check_inject_refuses_ambiguous() -> int:
    """B2. If template has 2 anchored WEIGHTS_B64 lines, ``_inject`` raises.

    This is the §9.10 regression guard: we synthesize an ambiguous template
    on-the-fly and verify the injector REFUSES to overwrite ambiguously,
    rather than silently picking the wrong one.
    """
    base = TEMPLATE_PATH.read_text(encoding="utf-8")
    # Inject a second top-level assignment at file start so an unanchored
    # str.replace would have hit the wrong one first (this exactly mimics
    # the docstring-placeholder bug pattern).
    poisoned = 'WEIGHTS_B64 = "fake_value"\n' + base
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tf:
        tf.write(poisoned)
        poisoned_template = Path(tf.name)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        try:
            _inject(str(poisoned_template), str(out_path), "dummy_payload")
        except RuntimeError as exc:
            if "expected exactly one" in str(exc) or "ambiguously" in str(exc):
                print(f"  [OK ] _inject correctly refused ambiguous template: "
                      f"{type(exc).__name__}: {str(exc)[:60]}...")
                return 0
            print(f"  [FAIL] _inject raised but for wrong reason: {exc!r}")
            return 1
        else:
            # Worst case: inject "succeeded" on ambiguous template.
            print("  [FAIL] _inject silently accepted an ambiguous template "
                  "(would have repeated the §9.10 bug)")
            return 1
    finally:
        for p in (poisoned_template, out_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    print("export pipeline test (template = submission_rl_v4.py)")
    failed = 0
    print(" [1] template placeholder count")
    failed += _check_template_match_count()
    print(" [2] inject replaces + import + agent() returns list")
    failed += _check_inject_replaces_placeholder()
    print(" [3] inject refuses ambiguous template (§9.10 regression guard)")
    failed += _check_inject_refuses_ambiguous()
    if failed == 0:
        print("\n[OK] all export pipeline checks passed")
        return 0
    print(f"\n[FAIL] {failed} export pipeline check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
