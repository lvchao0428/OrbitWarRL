#!/usr/bin/env python3
"""v14d hyperparameter search — binary coordinate descent + gate inheritance.

Default mode (binary): per phase, for each shaping/PPO dimension:
  probe lo / mid / hi → keep best → shrink range → repeat passes → confirm run.
Only the confirmed winner inherits to the next phase.

Resume: re-run the same command; state in logs/v14d_search.state.json.

Usage:
  python scripts/v14d_curriculum_search.py --space scripts/v14d_search_space.yaml
  python scripts/v14d_curriculum_search.py --dry-run
  python scripts/v14d_sensitivity_report.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from v14d_gate_check import gate_a, gate_b, gate_c  # noqa: E402
from v14d_scoring import beats_v13c_strict, score_trial  # noqa: E402

GATES = {"a": gate_a, "b": gate_b, "c": gate_c}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


def _mid(lo: float, hi: float) -> float:
    return (lo + hi) / 2.0


def _fmt_val(v: float, integer: bool = False) -> float:
    if integer:
        return float(int(round(v)))
    return round(v, 6)


def _init_ranges(spec: dict) -> dict:
    out = {}
    for name, cfg in spec.items():
        lo = float(cfg["low"])
        hi = float(cfg["high"])
        integer = bool(cfg.get("integer", False))
        mid = _fmt_val(_mid(lo, hi), integer)
        out[name] = {"lo": lo, "hi": hi, "integer": integer, "mid": mid}
    return out


def _init_current(ranges: dict) -> dict:
    return {k: v["mid"] for k, v in ranges.items()}


def _latest_ckpt(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pkl"))
    return ckpts[-1] if ckpts else None


def _run_early_abort(phase: str, log_path: Path, min_updates: int) -> int:
    cmd = [
        sys.executable,
        str(SCRIPTS / "v14d_early_abort.py"),
        phase,
        "--log",
        str(log_path),
        "--min-updates",
        str(min_updates),
    ]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode


def _write_trial_config(
    base_config: Path,
    out_config: Path,
    *,
    ckpt_dir: Path,
    seed: int,
    ppo_overrides: dict | None = None,
) -> Path:
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("train", {})
    cfg.setdefault("ppo", {})
    cfg["train"]["ckpt_dir"] = str(ckpt_dir)
    cfg["train"]["seed"] = seed
    cfg["train"]["resume_ckpt"] = ""
    for k, v in (ppo_overrides or {}).items():
        cfg["ppo"][k] = v
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return out_config


def _train_once(
    *,
    py: str,
    config: Path,
    log_path: Path,
    num_updates: int,
    env: dict,
    resume: Path | None,
) -> subprocess.Popen:
    args = [
        py,
        "-m",
        "orbit_wars_rl.scripts.train",
        "--config",
        str(config),
        "--log-dir",
        str(log_path.with_suffix("")),
        "--num-updates",
        str(num_updates),
    ]
    if resume is not None:
        args += ["--resume-from", str(resume)]
    proc_env = os.environ.copy()
    proc_env.update({k: str(v) for k, v in env.items()})
    proc_env["PYTHONPATH"] = str(ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    return subprocess.Popen(
        args,
        cwd=ROOT,
        env=proc_env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _split_params(combined: dict, ppo_keys: set[str]) -> tuple[dict, dict]:
    env = {k: v for k, v in combined.items() if k not in ppo_keys}
    ppo = {k: v for k, v in combined.items() if k in ppo_keys}
    return env, ppo


def _run_trial_loop(
    *,
    py: str,
    phase: str,
    phase_cfg: dict,
    trial_id: str,
    env_params: dict,
    ppo_params: dict,
    parent_ckpt: Path | None,
    poll_seconds: int,
    num_updates: int,
    probe: bool,
) -> dict:
    ckpt_dir = ROOT / "ckpt_search" / trial_id
    log_path = ROOT / "logs" / "search" / f"{trial_id}.log"
    cfg_dir = ROOT / "logs" / "search" / "configs"
    base_config = ROOT / phase_cfg["base_config"]
    extend = int(phase_cfg.get("extend_updates", 400))
    max_extends = int(phase_cfg.get("max_extends", 2))
    min_abort = int(phase_cfg.get("early_abort_min_updates", 80))

    fixed = phase_cfg.get("fixed", {})
    env = {**fixed, **env_params}
    env.setdefault("ORBITWARS_SHAPING_SCALE", "0.0")
    combined = {**env_params, **ppo_params}
    seed = int(phase_cfg.get("seed_base", 1600)) + int(_slug(combined), 16) % 100
    trial_config = _write_trial_config(
        base_config,
        cfg_dir / f"{trial_id}.yaml",
        ckpt_dir=ckpt_dir,
        seed=seed,
        ppo_overrides=ppo_params or None,
    )

    result = {
        "trial_id": trial_id,
        "phase": phase,
        "kind": "probe" if probe else "confirm",
        "env_params": env_params,
        "ppo_params": ppo_params,
        "parent_ckpt": str(parent_ckpt) if parent_ckpt else None,
        "status": "running",
        "ckpt": None,
        "gate": None,
        "score": 0.0,
        "beats_v13c": False,
        "started_at": _utcnow(),
        "finished_at": None,
    }

    if not log_path.is_file() or (parent_ckpt is None and probe):
        log_path.write_text("")

    total_budget = num_updates
    extends_done = 0
    resume = parent_ckpt
    spent = 0

    while spent < total_budget:
        chunk = total_budget - spent
        header = (
            f"\n=== [{_utcnow()}] {trial_id} phase={phase} kind={'probe' if probe else 'confirm'} "
            f"chunk={chunk} env={json.dumps(env_params)} ppo={json.dumps(ppo_params)} ===\n"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(header)

        proc = _train_once(
            py=py,
            config=trial_config,
            log_path=log_path,
            num_updates=chunk,
            env=env,
            resume=resume,
        )
        try:
            while proc.poll() is None:
                time.sleep(poll_seconds)
                if _run_early_abort(phase, log_path, min_abort) == 2:
                    print(f"[search] ABORT {trial_id}")
                    _stop_proc(proc)
                    sc = score_trial(phase, log_path, probe=probe)
                    result.update(
                        status="aborted",
                        score=sc["score"],
                        gate=sc["gate_msg"],
                        beats_v13c=sc["beats_v13c"],
                        finished_at=_utcnow(),
                    )
                    return result
        finally:
            if proc.poll() is None:
                _stop_proc(proc)
            proc.wait()

        spent += chunk
        ckpt = _latest_ckpt(ckpt_dir)
        sc = score_trial(phase, log_path, probe=probe)
        result.update(score=sc["score"], gate=sc["gate_msg"], beats_v13c=sc["beats_v13c"])

        if not probe and sc["gate_ok"]:
            if phase_cfg.get("require_beats_v13c"):
                from v14d_gate_check import _read_last_eval

                ev = _read_last_eval(log_path)
                if not beats_v13c_strict(ev):
                    print(f"[search] {trial_id} gate OK but below v13c — extend/continue")
                else:
                    result.update(
                        status="passed",
                        ckpt=str(ckpt) if ckpt else None,
                        finished_at=_utcnow(),
                    )
                    return result
            else:
                result.update(
                    status="passed",
                    ckpt=str(ckpt) if ckpt else None,
                    finished_at=_utcnow(),
                )
                return result

        if probe:
            result.update(
                status="probed",
                ckpt=str(ckpt) if ckpt else None,
                finished_at=_utcnow(),
            )
            return result

        if extends_done >= max_extends:
            result.update(
                status="failed",
                ckpt=str(ckpt) if ckpt else None,
                finished_at=_utcnow(),
            )
            return result
        if ckpt is None:
            result.update(status="failed", finished_at=_utcnow())
            return result
        extends_done += 1
        total_budget += extend
        resume = ckpt

    result.update(status="failed", finished_at=_utcnow())
    return result


class BinarySearchOrchestrator:
    def __init__(self, space_path: Path, state_path: Path, py: str, dry_run: bool = False):
        with open(space_path) as f:
            self.space = yaml.safe_load(f)
        self.state_path = state_path
        self.py = py
        self.dry_run = dry_run
        self.poll = int(self.space.get("poll_seconds", 90))
        self.search_id = self.space.get("search_id", "v14d_binary")
        self.passes = int(self.space.get("coordinate_passes", 2))
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.is_file():
            st = json.loads(self.state_path.read_text())
            if st.get("search_id") == self.search_id:
                return st
        binary: dict = {}
        for phase in ("a", "b", "c"):
            pcfg = self.space["phases"][phase]
            env_spec = pcfg.get("binary", {})
            ppo_spec = pcfg.get("ppo_binary", {})
            env_ranges = _init_ranges(env_spec)
            ppo_ranges = _init_ranges(ppo_spec)
            current = {**_init_current(env_ranges), **_init_current(ppo_ranges)}
            binary[phase] = {
                "env_ranges": env_ranges,
                "ppo_ranges": ppo_ranges,
                "current": current,
                "pass_idx": 0,
                "param_idx": 0,
                "param_order": sorted(current.keys()),
                "step_key": None,
                "step_pending": [],
                "step_results": {},
                "best": None,
                "ready_confirm": False,
                "confirmed": None,
                "done": False,
            }
        return {
            "search_id": self.search_id,
            "mode": "binary",
            "started_at": _utcnow(),
            "trials": {},
            "binary": binary,
            "pipeline": {},
        }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = _utcnow()
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _ppo_keys(self, phase: str) -> set[str]:
        return set(self.space["phases"][phase].get("ppo_binary", {}).keys())

    def _parent_ckpt(self, phase: str) -> Path | None:
        inherit = self.space["phases"][phase].get("inherit_from")
        if inherit is None:
            return None
        prev = self.state["pipeline"].get(inherit)
        if prev and prev.get("ckpt"):
            return Path(prev["ckpt"])
        return None

    def _trial_id(self, phase: str, tag: str, params: dict) -> str:
        return f"{self.search_id}_{phase}_{tag}_{_slug(params)}"

    def _schedule_probe(self, phase: str, pname: str, value: float) -> dict | None:
        bst = self.state["binary"][phase]
        current = dict(bst["current"])
        current[pname] = value
        env, ppo = _split_params(current, self._ppo_keys(phase))
        tag = f"p{bst['pass_idx']}i{bst['param_idx']}_{pname}_{value}"
        tid = self._trial_id(phase, tag, current)
        if tid in self.state["trials"]:
            return None
        return {
            "trial_id": tid,
            "phase": phase,
            "params": current,
            "env_params": env,
            "ppo_params": ppo,
            "pname": pname,
            "value": value,
            "probe": True,
        }

    def _next_job(self) -> dict | None:
        for phase in ("a", "b", "c"):
            bst = self.state["binary"][phase]
            pcfg = self.space["phases"][phase]
            if bst["done"]:
                continue
            parent = self._parent_ckpt(phase)
            if phase != "a" and parent is None:
                print(f"[search] phase {phase}: waiting for parent pipeline")
                return None

            if bst["ready_confirm"] and bst["confirmed"] is None:
                best = bst["best"]
                if best is None:
                    bst["done"] = True
                    continue
                current = best["params"]
                env, ppo = _split_params(current, self._ppo_keys(phase))
                tid = self._trial_id(phase, "confirm", current)
                if tid in self.state["trials"]:
                    bst["confirmed"] = self.state["trials"][tid]
                    if bst["confirmed"]["status"] == "passed":
                        self.state["pipeline"][phase] = bst["confirmed"]
                        bst["done"] = True
                    elif bst["confirmed"]["status"] in ("failed", "aborted"):
                        bst["done"] = True
                    continue
                return {
                    "trial_id": tid,
                    "phase": phase,
                    "env_params": env,
                    "ppo_params": ppo,
                    "parent_ckpt": parent,
                    "probe": False,
                    "num_updates": int(pcfg["num_updates"]),
                }

            order = bst["param_order"]
            if bst["param_idx"] >= len(order):
                if bst["pass_idx"] + 1 < self.passes:
                    bst["pass_idx"] += 1
                    bst["param_idx"] = 0
                    continue
                bst["ready_confirm"] = True
                continue

            pname = order[bst["param_idx"]]
            step_key = f"p{bst['pass_idx']}_i{bst['param_idx']}_{pname}"
            if bst["step_key"] != step_key:
                bst["step_key"] = step_key
                bst["step_pending"] = []
                bst["step_results"][step_key] = []
                spec = bst["env_ranges"].get(pname) or bst["ppo_ranges"].get(pname)
                lo, hi = spec["lo"], spec["hi"]
                integer = spec.get("integer", False)
                mid = _fmt_val(_mid(lo, hi), integer)
                for v in (lo, mid, hi):
                    vv = _fmt_val(v, integer)
                    job = self._schedule_probe(phase, pname, vv)
                    if job is not None:
                        bst["step_pending"].append(job)

            while bst["step_pending"]:
                job = bst["step_pending"][0]
                tid = job["trial_id"]
                if tid in self.state["trials"]:
                    rec = self.state["trials"][tid]
                    bst["step_results"][step_key].append(
                        {"value": job["value"], "score": rec.get("score", -1e9), "trial_id": tid}
                    )
                    bst["step_pending"].pop(0)
                    continue
                return {
                    **job,
                    "parent_ckpt": parent,
                    "num_updates": int(pcfg.get("probe_updates", 350)),
                }

            results = bst["step_results"].get(step_key, [])
            if len(results) < 3:
                bst["param_idx"] += 1
                continue
            winner = max(results, key=lambda r: r["score"])
            bst["current"][pname] = winner["value"]
            best_rec = {
                "params": dict(bst["current"]),
                "score": winner["score"],
                "trial_id": winner["trial_id"],
            }
            if bst["best"] is None or winner["score"] > bst["best"]["score"]:
                tr = self.state["trials"].get(winner["trial_id"], {})
                bst["best"] = {**best_rec, "ckpt": tr.get("ckpt")}

            spec = bst["env_ranges"].get(pname) or bst["ppo_ranges"].get(pname)
            lo, hi = spec["lo"], spec["hi"]
            mid = _fmt_val(_mid(lo, hi), spec.get("integer", False))
            wv = winner["value"]
            if abs(wv - lo) < 1e-9:
                spec["hi"] = mid
            elif abs(wv - hi) < 1e-9:
                spec["lo"] = mid
            else:
                spec["lo"] = _fmt_val(_mid(lo, mid), spec.get("integer", False))
                spec["hi"] = _fmt_val(_mid(mid, hi), spec.get("integer", False))

            bst["param_idx"] += 1
        return None

    def estimate_trials(self) -> dict:
        est = {}
        for phase in ("a", "b", "c"):
            pcfg = self.space["phases"][phase]
            n_params = len(pcfg.get("binary", {})) + len(pcfg.get("ppo_binary", {}))
            probes = n_params * self.passes * 3
            est[phase] = {"probes": probes, "confirm": 1}
        est["total"] = sum(v["probes"] + v["confirm"] for v in est.values())
        return est

    def run(self) -> None:
        est = self.estimate_trials()
        print(f"[search] {self.search_id} binary mode — ~{est['total']} trials (est.)")
        for ph, e in est.items():
            if ph != "total":
                print(f"  phase {ph}: {e['probes']} probes + {e['confirm']} confirm")

        if self.dry_run:
            print("[search] dry-run — no trials executed")
            return

        while True:
            job = self._next_job()
            if job is None:
                if all(self.state["binary"][p]["done"] for p in ("a", "b", "c")):
                    break
                print("[search] blocked — waiting for parent phase")
                break

            tid = job["trial_id"]
            if tid in self.state["trials"]:
                continue

            print(f"[search] START {tid}")
            result = _run_trial_loop(
                py=self.py,
                phase=job["phase"],
                phase_cfg=self.space["phases"][job["phase"]],
                trial_id=tid,
                env_params=job["env_params"],
                ppo_params=job["ppo_params"],
                parent_ckpt=job.get("parent_ckpt"),
                poll_seconds=self.poll,
                num_updates=job["num_updates"],
                probe=job.get("probe", False),
            )
            self.state["trials"][tid] = result
            self._save_state()

        best_c = self.state["pipeline"].get("c")
        if best_c:
            self.state["best_pipeline"] = best_c
            self._save_state()
            print(f"[search] DONE best={best_c['trial_id']} ckpt={best_c.get('ckpt')}")
            print(f"[search] beats_v13c={best_c.get('beats_v13c')}")
        else:
            print("[search] DONE — no phase-C winner yet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=Path, default=SCRIPTS / "v14d_search_space.yaml")
    ap.add_argument("--state", type=Path, default=ROOT / "logs" / "v14d_search.state.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    args = ap.parse_args()
    BinarySearchOrchestrator(args.space, args.state, args.python, dry_run=args.dry_run).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
