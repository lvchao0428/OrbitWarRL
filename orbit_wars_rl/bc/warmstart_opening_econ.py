"""Pre-train dst_economics_head on early-game states (ROI teacher CE).

Loads v29 u3999, shape-adapts to v30, runs supervised CE on the economics
head only (src = home, teacher = argmax pair_roi). Saves a warm ckpt for v30b PPO.

Usage:
  python -m orbit_wars_rl.bc.warmstart_opening_econ \\
      --init ./ckpt_multi_action_v29_aim/ckpt_003999.pkl \\
      --out-dir ./ckpt_multi_action_v30b_warm
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.actions import noop_multi_action
from orbit_wars_rl.env.env import OrbitWarsEnv
from orbit_wars_rl.features.encode import EncodedObs, encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.parity.kaggle_bridge import kaggle_obs_to_envstate
from orbit_wars_rl.ppo.runner import _merge_ckpt_params, load_checkpoint, save_checkpoint


def _collect_from_replay(replay_path: Path, *, max_turn: int = 80) -> list[dict]:
    with open(replay_path) as f:
        replay = json.load(f)
    samples: list[dict] = []
    home_id = 12
    for t, step in enumerate(replay.get("steps") or []):
        if t > max_turn:
            break
        obs = step[0].get("observation")
        if not obs:
            continue
        state = kaggle_obs_to_envstate(obs)
        enc = encode(state, player=0, episode_steps=500, wins_needed=1)
        home_idx = int(np.asarray(state.home_planet_idx[0]))
        if home_idx != home_id:
            continue
        home_ships = int(state.planet_ships[home_idx])
        if home_ships < 8:
            continue
        samples.append({
            "enc": enc,
            "planet_ships_raw": np.asarray(state.planet_ships, dtype=np.int32),
            "planet_x": np.asarray(state.planet_x, dtype=np.float32),
            "planet_y": np.asarray(state.planet_y, dtype=np.float32),
            "home_idx": np.int32(home_idx),
            "planet_orbit_phase": np.asarray(state.planet_orbit_phase, dtype=np.float32),
            "planet_orbit_radius": np.asarray(state.planet_orbit_radius, dtype=np.float32),
            "planet_is_orbiting": np.asarray(state.planet_is_orbiting, dtype=np.bool_),
            "angular_velocity": np.float32(state.angular_velocity),
            "turn": t,
        })
    return samples


def _collect_from_env(*, num_seeds: int = 16, max_turn: int = 80) -> list[dict]:
    env = OrbitWarsEnv(num_groups=5, episode_steps=500, wins_needed=1)
    hold = noop_multi_action()
    samples: list[dict] = []
    for seed in range(num_seeds):
        key = jax.random.PRNGKey(seed)
        s = env.reset(key)
        for t in range(max_turn):
            enc = encode(s, player=0, episode_steps=500, wins_needed=1)
            home_idx = int(np.asarray(s.home_planet_idx[0]))
            home_ships = int(s.planet_ships[home_idx])
            if home_ships >= 8:
                samples.append({
                    "enc": enc,
                    "planet_ships_raw": np.asarray(s.planet_ships, dtype=np.int32),
                    "planet_x": np.asarray(s.planet_x, dtype=np.float32),
                    "planet_y": np.asarray(s.planet_y, dtype=np.float32),
                    "home_idx": np.int32(home_idx),
                    "planet_orbit_phase": np.asarray(s.planet_orbit_phase, dtype=np.float32),
                    "planet_orbit_radius": np.asarray(s.planet_orbit_radius, dtype=np.float32),
                    "planet_is_orbiting": np.asarray(s.planet_is_orbiting, dtype=np.bool_),
                    "angular_velocity": np.float32(s.angular_velocity),
                    "turn": t,
                })
            key, k1, k2 = jax.random.split(key, 3)
            s, _ = env.step(s, (hold, hold))
            if bool(jnp.asarray(s.done).any()):
                break
    return samples


def _stack_batch(items: list[dict]) -> dict:
    def stack_field(name, dtype=None):
        arrs = [it[name] for it in items]
        if name == "enc":
            return EncodedObs(
                planet_feats=jnp.stack([a.planet_feats for a in arrs]),
                planet_mask=jnp.stack([a.planet_mask for a in arrs]),
                fleet_feats=jnp.stack([a.fleet_feats for a in arrs]),
                fleet_mask=jnp.stack([a.fleet_mask for a in arrs]),
                global_feats=jnp.stack([a.global_feats for a in arrs]),
                my_planet_mask=jnp.stack([a.my_planet_mask for a in arrs]),
                enemy_planet_mask=jnp.stack([a.enemy_planet_mask for a in arrs]),
                neutral_planet_mask=jnp.stack([a.neutral_planet_mask for a in arrs]),
            )
        return jnp.stack([jnp.asarray(x) for x in arrs]) if dtype is None else jnp.stack(
            [jnp.asarray(x, dtype=dtype) for x in arrs]
        )

    return {
        "enc": stack_field("enc"),
        "planet_ships_raw": stack_field("planet_ships_raw"),
        "planet_x": stack_field("planet_x"),
        "planet_y": stack_field("planet_y"),
        "home_idx": stack_field("home_idx"),
        "planet_orbit_phase": stack_field("planet_orbit_phase"),
        "planet_orbit_radius": stack_field("planet_orbit_radius"),
        "planet_is_orbiting": stack_field("planet_is_orbiting"),
        "angular_velocity": stack_field("angular_velocity"),
    }


def _zero_non_econ_grads(grads):
    def mask(path, g):
        path_str = "/".join(str(getattr(p, "key", p)) for p in path)
        if "dst_economics_head" in path_str:
            return g
        return jnp.zeros_like(g)

    return jax.tree_util.tree_map_with_path(mask, grads)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="./ckpt_multi_action_v29_aim/ckpt_003999.pkl")
    ap.add_argument("--out-dir", type=str, default="./ckpt_multi_action_v30b_warm")
    ap.add_argument("--replay", type=str, default="logs/replay_html/v29_u3999_s0/replay.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()

    init_path = Path(args.init)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCritic(
        d_model=256,
        n_layers=4,
        n_heads=8,
        ff_dim=1024,
        max_fleets_per_turn=constants.MAX_FLEETS_PER_TURN,
        flip_hard_mask=True,
        zero_sum_value=True,
        allow_hold=True,
    )
    env = OrbitWarsEnv(num_groups=5, episode_steps=500)
    s0 = env.reset(jax.random.PRNGKey(0))
    enc0 = encode(s0, player=0, episode_steps=500)
    params = model.init(
        jax.random.PRNGKey(1),
        enc0,
        jax.random.PRNGKey(2),
        s0.planet_ships,
        s0.planet_x,
        s0.planet_y,
        jnp.int32(0),
    )

    ckpt = load_checkpoint(str(init_path))
    merged = _merge_ckpt_params(params, ckpt["params"], label="warmstart")
    params = merged if merged is not None else ckpt["params"]
    print(f"[warmstart] loaded {init_path}")

    replay_path = Path(args.replay)
    if replay_path.is_file():
        samples = _collect_from_replay(replay_path)
        print(f"[warmstart] {len(samples)} states from replay {replay_path.name}")
    else:
        samples = _collect_from_env()
        print(f"[warmstart] {len(samples)} states from env seeds (no replay)")

    if len(samples) < args.batch_size:
        extra = _collect_from_env(num_seeds=32)
        samples = samples + extra
        print(f"[warmstart] padded to {len(samples)} states")

    K = constants.MAX_FLEETS_PER_TURN
    rng = jax.random.PRNGKey(42)

    def loss_fn(params, batch):
        obs = batch["enc"]
        B = obs.planet_feats.shape[0]
        src = batch["home_idx"]
        src_idx = jnp.broadcast_to(src[:, None], (B, K))
        dst_idx = jnp.zeros((B, K), dtype=jnp.int32)
        pct_bin = jnp.zeros((B, K), dtype=jnp.int32)
        emit_mask = jnp.zeros((B, K), dtype=jnp.bool_)
        emit_mask = emit_mask.at[:, 0].set(True)
        out = model.apply(
            params,
            obs,
            src_idx,
            dst_idx,
            pct_bin,
            emit_mask,
            batch["planet_ships_raw"],
            batch["planet_x"],
            batch["planet_y"],
            batch["home_idx"],
            batch["planet_orbit_phase"],
            batch["planet_orbit_radius"],
            batch["planet_is_orbiting"],
            batch["angular_velocity"],
            None,
            freeze_dst_attn=True,
            method=ActorCritic.evaluate,
        )
        logp = jax.nn.log_softmax(out.dst_econ_logits, axis=-1)
        teacher = out.roi_teacher
        picked = jnp.take_along_axis(logp, teacher[..., None], axis=-1)[..., 0]
        mask = emit_mask.astype(jnp.float32)
        n = jnp.maximum(mask.sum(), 1.0)
        loss = -(picked * mask).sum() / n
        acc = ((jnp.argmax(out.dst_econ_logits, axis=-1) == teacher).astype(jnp.float32) * mask).sum() / n
        return loss, acc

    def loss_scalar(params, batch):
        loss, _ = loss_fn(params, batch)
        return loss

    grad_fn = jax.value_and_grad(loss_scalar)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    n = len(samples)
    for step in range(args.steps):
        rng, sub = jax.random.split(rng)
        idx = np.random.default_rng(int(sub[0])).choice(n, size=args.batch_size, replace=n < args.batch_size)
        batch = _stack_batch([samples[int(i)] for i in idx])
        loss, grads = grad_fn(params, batch)
        grads = _zero_non_econ_grads(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step % 50 == 0 or step == args.steps - 1:
            _, acc = loss_fn(params, batch)
            print(f"  step {step:4d}  loss={float(loss):.4f}  acc={float(acc):.3f}")

    out_ckpt = out_dir / "ckpt_warm.pkl"
    save_checkpoint(str(out_ckpt), params, opt_state, step=args.steps, meta={"warmstart": True})
    print(f"[warmstart] saved {out_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
