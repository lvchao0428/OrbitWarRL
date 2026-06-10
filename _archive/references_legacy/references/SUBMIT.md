# Kaggle submission paths

The graded bot must expose `agent(obs, config=None)` returning a list of moves.

- **Default (current):** submit [`submission_v20.py`](../submission_v20.py) as a single file, or use the pack script below. Optional re-export: [`kaggle_submit_entry.py`](../kaggle_submit_entry.py).

- **Older versions (archived):** [`archive/legacy/submissions/`](../archive/legacy/submissions/) — e.g. v6–v18, `submission.py`. Resolve paths in scripts with [`submission_resolve.py`](../submission_resolve.py).

- **Examples of archived baselines:**
  - Planet Wars–style scaffold: `archive/legacy/submissions/submission_v8.py` — compare locally with `scripts/eval_compare_v6_v7.py` (pass explicit paths or extend scripts to use `resolve_submission_path`).
  - Stable heuristic lineage: `submission_v6.py` … `submission_v7.py` under `archive/legacy/submissions/`.

**One-shot pack:** from `kaggle/`, `./scripts/package_submission.sh` (defaults to `submission_v20.py`). Examples: `./scripts/package_submission.sh archive/legacy/submissions/submission_v8.py`. Outputs `dist/main.py`, runs `py_compile`, and `dist/submission.tar.gz` containing only `main.py`.

**Local A/B:** `python3 scripts/eval_compare_v6_v7.py --seeds 0 1 2`

Packaged helpers under [`orbit_wars_bot/`](../orbit_wars_bot/) are for **local** training and evaluation only; they are not required on the server unless you vendor them into the submission archive.
