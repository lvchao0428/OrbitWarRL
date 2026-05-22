# DAY3 PLAN — BC bootstrap from v20, then RL fine-tune

> Written 2026-05-22 ~midnight after exhausting pure-RL options. **Read this
> first before touching any code on Day 3.** It captures the full lesson from
> v4.x / v5.x runs and lays out the unambiguous next path.

---

## 0. TL;DR — why we are switching strategy

After Day 2 we had a healthy RL pipeline by every classic metric and a
working multi-action policy that beat v3.2 and v1, but **0/20 vs v20**. Day 2
late-night runs (`v4.2 → v5.0 → v5.1 → v5.2 → v5.3`) showed:

* `v4.2` (the strongest pure-RL run): `mean_emits` plateau at ~1.55, 0/20 vs v20.
* `v5.2` (kill reward shaping, shorten episodes for dense terminals):
  upd 0-29 advantage signal recovered (adv_std 0.20-0.32, pg_loss -0.0025),
  then **value head outran policy** → adv_std fell to 0.13, learning stopped.
* `v5.3` (value_coef 0.5 → 0.20, lr_decay 800 → 2400 — *single-variable diff
  from v5.2*): **`pg_loss` stayed strong (−0.0036), `WRr` reached 0.91** —
  the value-head fix was correct. But `mean_emits` STAYED LOCKED at 1.4
  through 144 updates. Policy was learning, just not emitting bigger fleets.

**Root cause** (confirmed against `top_players_rl.txt`):

> Pure-RL from scratch **cannot solve emit credit assignment** within our
> time budget. The reward signal for "emit 5 ships instead of 1" arrives
> ~800 env steps later as a single terminal reward. With our K=8 emit
> autoregression, that's at minimum 8×800 = 6400 credit-assignment hops for
> a 1-bit reward. `top_players_rl.txt` §75-§109 says all top RL teams
> bootstrapped from BC (behavior clone) on a strong heuristic first.

**Plan**: BC clone `submission_v20_0513.py` (our strongest available bot)
into our existing multi-action `ActorCritic` network, then RL fine-tune
from that warm start. Day 3 deliverable target: BC policy that scores
**>= 5/20 vs v20** (a clear, falsifiable goal). RL fine-tune is Day 4.

---

## 1. What we have working already (no work needed)

* `orbit_wars_rl/parity/kaggle_bridge.py::kaggle_obs_to_envstate(obs)` —
  given a raw kaggle observation dict, returns our `EnvState`. **Crucially:
  `slot == kaggle_planet.id`**, so v20's `src_id=7` is already our slot 7
  (the brief comment at line 82-83 confirms this).
* `orbit_wars_rl/features/encode.py::encode(state, player, episode_steps)`
  → produces `EncodedObs`, the network's input.
* `orbit_wars_rl/net/model.py::ActorCritic` — already has src/dst/pct/emit
  heads with K=8 autoregression and value head. **No architecture change
  needed for BC**: we just use the heads' raw logits for CE loss.
* `orbit_wars_rl/scripts/export_submission.py` — converts a JAX ckpt to
  numpy and injects into `submission_rl_v4.py` template. Already tested and
  validated; BC ckpts will export the same way.

---

## 2. What we have to build for BC (4 small files)

### 2.1 `orbit_wars_rl/bc/action_inverse.py` (the only non-trivial piece)

Given:
* `state: EnvState` (kaggle-bridged)
* `player: int`
* `kaggle_actions: List[List[float]]`  -- v20's output, list of `[src_id, angle, ships]`

Return target labels for supervised training:
```
src_targets : (K,) int32 -- slot index of each emitted fleet, K = MAX_FLEETS_PER_TURN = 8
dst_targets : (K,) int32 -- slot index, recovered by closest-angle match
pct_targets : (K,) int32 -- discretized into 4 bins {0.25, 0.5, 0.75, 1.0}
emit_targets: (K,) bool  -- True for steps 0..len(actions)-1, False after
emit_free   : (K,) bool  -- True for steps >= 1 (step 0 is forced-emit, doesn't supervise)
loss_mask   : (K,) bool  -- True where the step has a real target
```

**Algorithm** (numpy, called once per data point at train time):

```python
def kaggle_action_to_targets(state, player, actions):
    K = MAX_FLEETS_PER_TURN  # 8
    src = np.zeros(K, np.int32)
    dst = np.zeros(K, np.int32)
    pct = np.zeros(K, np.int32)
    emit = np.zeros(K, np.bool_)
    loss = np.zeros(K, np.bool_)

    # Cap at K (v20 might emit more, but our network can't represent it)
    actions = actions[:K]

    px = np.asarray(state.planet_x)
    py = np.asarray(state.planet_y)
    pmask = np.asarray(state.planet_mask)
    pships = np.asarray(state.planet_ships)

    for k, (src_id, angle, ships) in enumerate(actions):
        src_id = int(src_id)
        ships = int(ships)
        src[k] = src_id
        # ships at src after k-1 emits already deducted -- approximate by
        # using initial ships (good enough for BC; the policy will learn to
        # be precise via RL fine-tune). v20 emits one move per ACTION turn,
        # so within a single agent() call this is fine.
        src_ships = max(int(pships[src_id]), 1)
        pct_frac = ships / src_ships
        # Snap to nearest pct bin
        pct[k] = int(np.argmin(np.abs(np.array([0.25, 0.5, 0.75, 1.0]) - pct_frac)))

        # Recover dst by angle match. v20 computes angle = atan2(dy, dx)
        # where (dx, dy) = dst.center - src.center. We mirror that.
        sx, sy = px[src_id], py[src_id]
        dx_all = px - sx
        dy_all = py - sy
        angles = np.arctan2(dy_all, dx_all)
        # circular distance
        adiff = np.abs(((angles - angle) + np.pi) % (2 * np.pi) - np.pi)
        # Disqualify src and padding planets
        adiff = np.where(pmask, adiff, np.float32(1e9))
        adiff[src_id] = 1e9
        dst[k] = int(np.argmin(adiff))

        emit[k] = True
        loss[k] = True

    # Steps after len(actions) — agent stopped emitting at this step.
    # We supervise emit=False on exactly one extra step (the "stop" decision)
    # if there's room. After that, no signal.
    if len(actions) < K:
        loss[len(actions)] = True  # the explicit "stop" supervision
        # emit[len(actions)] already False -> correct target
    emit_free = np.ones(K, np.bool_)
    emit_free[0] = False  # step 0 is forced-emit in our env; no supervision
    return src, dst, pct, emit, emit_free, loss
```

**Why this is correct**:
* v20's `src_id` is already our slot index (see §1).
* v20's `angle = atan2(dst.y - src.y, dst.x - src.x)` (see `submission_rl_v1.py:351`
  — every submission computes angle the same way the kaggle env consumes it).
* The closest-angle planet (excluding src and padding) is uniquely determined
  by v20's choice unless two planets are co-linear from src (negligible in
  randomly-generated maps).
* `pct_frac = ships / src_ships` is exact at emit time because v20 emits
  before its other fleets fly off (single-step decision).

**Edge cases to handle**:
* v20 emits 0 fleets that turn → `loss[0] = True, emit[0] = False`.
* v20 emits more than K fleets → cap at K (rare; v20 usually emits ≤ 6).
* `src_ships <= 0` → guard divide-by-zero (shouldn't happen for a valid
  v20 action but be defensive).

### 2.2 `orbit_wars_rl/bc/collect_data.py`

Drives a kaggle env with v20 on both sides, dumps `(EnvState_pkl, player, action_list)` tuples.

```bash
python -m orbit_wars_rl.bc.collect_data \
    --num-games 200 \
    --agent submission_v20_0513.py \
    --opponent submission_v20_0513.py \
    --out data/bc_v20_self.npz
```

Implementation sketch:
```python
import kaggle_environments as ke
env = ke.make("orbit_wars", debug=False)
# loop num_games:
#   env.reset()
#   while not env.done:
#       obs_p0 = env.state[0]['observation']
#       obs_p1 = env.state[1]['observation']
#       action_p0 = agent_p0(obs_p0, env.configuration)
#       action_p1 = agent_p1(obs_p1, env.configuration)
#       # save both perspectives
#       data.append((kaggle_obs_to_envstate(obs_p0), 0, action_p0))
#       data.append((kaggle_obs_to_envstate(obs_p1), 1, action_p1))
#       env.step([action_p0, action_p1])
```

**Target dataset size**: 200 games × ~120 turns × 2 players ≈ 48k (obs, action)
pairs. Should fit in a few hundred MB pickled.

**Speed estimate**: kaggle env is slow (~100ms per turn with v20 thinking),
so 200 games × 120 turns × 0.1 s = 2400 s ≈ 40 min on 5090. Acceptable.

### 2.3 `orbit_wars_rl/bc/train_bc.py`

Standard supervised training loop, reusing existing `ActorCritic` apply.

* Load `data/bc_v20_self.npz` → numpy arrays of states, players, action targets.
* Encode in bulk (jit over batch): `encode_batch = jax.vmap(encode, in_axes=(0, 0, None))`
* Forward pass: produce src_logits[K, MAX_PLANETS], dst_logits[K, MAX_PLANETS],
  pct_logits[K, 4], emit_logits[K, 2] per sample.
* Loss = sum over heads of `softmax_cross_entropy(logits, target_label)`,
  masked by `loss_mask`. Weight `emit_loss` 2x to push policy toward emitting more.
* Adam with `lr=3e-4`, batch_size=256, train for 5 epochs (~30 min on 5090).
* Save ckpt in same format as RL ckpt (so export_submission accepts it).

**Sanity check before exporting**: pick 10 random training samples, predict
greedy actions, compute *exact match rate* on (src, dst, pct, emit) → must
be >= 60% (lower bound for "BC actually learned the heuristic").

### 2.4 `orbit_wars_rl/bc/export_check.py`

Validate that the BC submission file works under the same gauntlet as RL ckpts:

```bash
python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_bc_v0/ckpt_final.pkl \
    --out submission_rl_bc_v0.py

python -m orbit_wars_rl.scripts.h2h_gauntlet \
    --agent submission_rl_bc_v0.py \
    --opponents submission_v20_0513.py submission_rl_v1.py submission_rl_v4p2.py \
    --num-games 10 --seeds 0,1,2,3,4,5,6,7,8,9 \
    > logs/h2h_bc_v0.log 2>&1
```

---

## 3. Success criteria (Day 3)

| Metric                                            | Target               | Why                                              |
|---------------------------------------------------|----------------------|--------------------------------------------------|
| Exact-match rate on validation set                | ≥ 60% over all heads | BC actually mimicked v20's decisions             |
| `mean_emits` on val obs (greedy decode)           | ≥ 2.5                | We're no longer locked at 1.4 — policy emits big |
| H2H BC vs v20 (10 games, both sides)              | ≥ 5/20               | BC clone is at least 25% of v20's strength       |
| H2H BC vs v4.2 final ckpt                         | ≥ 14/20              | BC beats Day 2's best pure-RL agent              |

If H2H vs v20 is ≥ 5/20, RL fine-tune (Day 4) becomes meaningful (start
from a non-degenerate policy that knows mass attacks exist).

---

## 4. Order of operations (Day 3 morning, in order)

1. **Read §1 + §2.1 carefully**. The angle → dst recovery is the only
   non-mechanical step; everything else is glue code.
2. Write `bc/action_inverse.py` and unit-test it against a single
   hand-constructed example (`src=2, dst=5, ships=10` from a known state
   round-trips to the same src/dst/pct).
3. Write `bc/collect_data.py`, smoke-test with `--num-games 2` first
   (should produce ~500 samples in <30s).
4. Run real collection: `--num-games 200`. While it runs (~40 min), write
   `bc/train_bc.py`.
5. Train: 5 epochs at batch 256, monitor exact-match rate per head every epoch.
6. Export, H2H against v20/v1/v4.2.

**Hard kill criteria**:
* Exact-match rate < 30% after epoch 1 → action_inverse has a bug, debug.
* `mean_emits` on val < 1.5 after training → BC didn't even learn to emit;
  check `emit_loss` weighting and `loss_mask` for emit head.
* H2H vs v4.2 < 10/20 → BC is worse than pure RL was; abort BC, go back to
  RL with different hparams (re-examine v5.3 ckpt at upd 144).

---

## 5. Status at end of Day 2

* **v5.3 ckpt at ~upd 144 retained** at `ckpt_multi_action_v5p3/`. WRr 0.91
  vs random, WRf 0.81 vs frozen. Best pure-RL ckpt we have if BC fails.
* `v5p3.yaml` documents the value_coef fix (0.5 → 0.20) that actually
  worked. Keep this for any future RL experiments.
* All v5.x configs (v5p0, v5p1, v5p2, v5p3) are checked in for traceability.
* **No `bc/` directory exists yet** — created tomorrow as the first thing.

## 6. Files to leave alone tomorrow morning

Do not modify these unless something is provably wrong:

* `orbit_wars_rl/env/*` — env parity is good (kaggle bridge works, lead-target
  features are correct).
* `orbit_wars_rl/net/heads.py::EmitHead.continue_bias` — DAY2 §3.3 already
  litigated this; keep at 0.0. Emit issue is reward-signal, not bias.
* `orbit_wars_rl/scripts/export_submission.py` — fixed in DAY2 §9.10.
* `submission_rl_v4.py` template — same.
