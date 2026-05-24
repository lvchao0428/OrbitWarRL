# DAY3 PROGRESS — v7 architectural fix for the K-step "reserved-blind" bug

> Written 2026-05-23 evening. DAY3_PLAN.md described a BC bootstrap route. We
> took a different path instead: we tried one more pure-RL iteration after
> diagnosing the **architectural** root cause of v6p3's failure against v20.
> BC remains Plan B if v7 ≥ 5k updates does not break 0/20 vs v20.

---

## 0. TL;DR

* **Diagnosed** the actual reason v6p3 finished 0/20 vs v20 despite healthy
  training metrics: **all 4 action heads were blind to the within-turn
  reserved-ships state** in the K-step (max=8) autoregressive sample loop,
  so the policy collapsed to "emit K identical (src, dst, pct) triples"
  every turn, mostly 1-ship fleets that drained the mother planet within
  one turn.
* **Architectural fix (v7)**: each head now takes a reserved-aware input —
  `remaining_norm`, `reserved_norm`, `src_remaining_norm`,
  `total_remaining_norm` — so it can refine its decision as the K-loop
  consumes ships from the mother planet.
* **Also fixed a long-standing float32-rounding bug** in inference: every
  fleet was silently emitting 1 fewer ship than training intended (e.g.
  `pct=0.70, garrison=10` → `floor(10 * 0.6999..) = 6` instead of `7`).
  This affected every submission from v2 onwards.
* **Submission template was the actual blocker.** I forgot that
  `submission_rl_v4.py` is a *standalone* copy of the inference code that
  Kaggle uses verbatim — it had its own copies of the head functions, not
  imports from `numpy_forward.py`. v7's first export looked perfect in JAX
  parity but every inference call crashed with shape mismatch. Fixed by
  syncing the template too.
* **v7 u499 final result** (with template fix): pct now uses bins 4-6 not 0,
  dst no longer collapses, emits drops from 4.0 to 1.65 (1 fleet/turn, but
  *meaningful* fleets). H2H pending — train metrics suggest strong vs v1
  but unclear vs v20.

---

## 1. The full v6p3 → v7 detective story

### 1.1 v6p3 finished at u4999

v6p3 was launched at the end of Day 2 to validate the OPT A fix (DstHead
gets `src_idx` and masks src out so dst ≠ src). It ran the full 5000
updates over ~6 hours on the 5090.

Train metrics tail (last few hundred updates):
```
upd 4999   WRr 0.92    WRf 0.78    mean_emits 2.10   ent_dst 1.31   clip 0.04
upd 4500   WRr 0.92    WRf 0.77    mean_emits 2.30   ent_dst 1.31   clip 0.05
upd 4000   WRr 0.91    WRf 0.78    mean_emits 2.58   ent_dst 1.40   clip 0.05
upd 3000   WRr 0.91    WRf 0.76    mean_emits 3.42   ent_dst 1.51   clip 0.06
upd 2000   WRr 0.90    WRf 0.74    mean_emits 4.10   ent_dst 2.09   clip 0.10
```

`mean_emits` peaked at 4.10 around upd 2000 and slowly decayed to 2.10 at
the end. PPO `clip_frac` fell from 0.10 to 0.04 — clear sign of policy
convergence. Self-play WRf rose to 0.78 (strong dominance over its own
older snapshots).

### 1.2 v6p3 u4999 H2H gauntlet

```
vs submission_v1.py         W= 17/20  (vs v6p2 u999's 8/20  -- big jump)
vs submission_rl_v4p2.py    W= 15/20  (vs v6p2 u999's 10/20)
vs submission_v20_0513.py   W=  0/20  (unchanged from v6p2)
overall: 32/60 = 0.533
```

Beating v1 and v4p2 confirmed OPT A worked (dst no longer collapses to
src; fleets actually launch). But **0/20 vs v20** — same as every prior
RL run.

My first read: "policy converged to a v4p2-class local optimum; can't
break v20 from pure RL self-play in 5000 updates". This was wrong. The
*metric* dynamics looked like convergence, but the *policy itself* was
broken in a way the metrics couldn't see.

### 1.3 replay_dump exposed the real behaviour

I ran `replay_dump.py` to print every turn's (state, action) for v6p3 vs
v20. The mother planet went from 10 ships to 1 ship in the first turn:

```
[turn=  0] v6p3 sP=10 -> emit 8 fleets, all 1 ship each, dst spread across
                        8 different planets   ("spray of mosquitoes")
[turn=  1] v6p3 sP= 3  -> emit 3 fleets, all 1 ship
[turn=  2] v6p3 sP= 1  -> emit 1 fleet, 1 ship
[turn=  5] v6p3 sP= 1  -> emit 1 fleet, 1 ship
[turn= 20] v6p3 sP=10  -> emit 4 fleets, all 1 ship   (briefly recovered, then re-drained)
```

Meanwhile v20 sat on its planet, accumulated 5-15 ships, then sent a
single decisive 8-ship fleet. v6p3 lost every game.

### 1.4 diag_real_game_obs confirmed the K-step lock

I wrote a new diagnostic that prints the *raw* `greedy_multi_action`
output (src_list, dst_list, pct_list before `decode_multi_to_kaggle_moves`
filters anything). v6p3 turn 0:

```
v6p3 u4999 player=0 seed=0:
  turn 0: src=[12,12,12,12,12,12,12,12]   <- K=8 steps, identical src
          dst=[3,3,3,3,3,3,3,3]            <- identical dst
          pct=[0,0,0,0,0,0,0,0]            <- identical pct_bin=0 (= 0.10)
          ships per fleet: floor(10*0.10) = 1, so 8 × 1-ship fleets

v4p2 u4999 player=0 seed=0:
  turn 0: src=[12], dst=[19], pct=[3]      <- K=1, only 1 emit, pct_bin=3 (= 0.40)
          ships per fleet: floor(10*0.40) = 4 ship
```

Inside the K=8 autoregressive loop, every head produced the **same**
output across all 8 steps. The agent committed to one src, one dst, one
pct at t=0 and never refined.

### 1.5 Root cause: heads were blind to within-turn reserved state

Looking at `ActorCritic.__call__` (the sample loop), each head's input
was *deterministic given the turn-start state*:

```python
for t in range(K):
    src_logits_t = self.src_head(planet_emb, eff_mask)        # planet_emb is turn-start
    src_t = argmax(src_logits_t)                                # always same src
    dst_logits_t = self.dst_head(planet_emb, src_emb_t, ...)   # src_emb_t fixed
    dst_t = argmax(dst_logits_t)                                # always same dst
    pct_logits_t = self.pct_head(src_emb_t, dst_emb_t, global_emb)  # all fixed
    pct_t = argmax(pct_logits_t)                                # always same pct
    # reserved gets updated, but only used in `_ships_to_send` -- not passed back to any head
```

`reserved` was bookkept correctly for ship arithmetic but **never fed
back into the heads**. So argmax across K iterations gave 8 identical
(src, dst, pct) triples by construction.

For `emit_head`, only `step_oh` changed within the loop — emit could in
principle stop at later steps via `step_oh`-conditioned logits, but it
had no idea whether the mother planet had ships left.

This was an **architectural** bug, not a training-signal bug. No amount
of RL fine-tuning, BC bootstrap, or reward shaping would fix it without
changing the network input topology.

---

## 2. v7 architectural changes

Each head gains one new reserved-aware input (~320 params total):

| Head | New input | Shape | Source |
|---|---|---|---|
| SrcHead | `remaining_norm[p]` | (P,) | `log1p(ships - reserved) / 8` |
| DstHead | `reserved_norm[p]` | (P,) | `log1p(reserved) / 8` |
| PctHead | `src_remaining_norm` | scalar | `remaining_norm[src_t]` |
| EmitHead | `total_remaining_norm` | scalar | `log1p(sum(remaining * my_mask)) / 8` |

`log1p/8` matches the planet-feature normalisation in `features/encode.py`
so the heads see comparable magnitudes to the existing `log_ships`
feature.

Implementation:
* `orbit_wars_rl/net/heads.py` — 4 heads each take an optional new arg.
* `orbit_wars_rl/net/model.py` — added `_remaining_features()` helper;
  both `evaluate()` (training-time logp recompute) and `__call__()`
  (sample loop) compute these features per-step and pass them in.
* `orbit_wars_rl/inference/numpy_forward.py` — mirrored the JAX changes.
* `orbit_wars_rl/inference/test_parity.py` — still 16/16 OK (the parity
  contract held).

After implementation, fresh-init smoke test:
```
[init] total params: 676,557      (v6: 676,237, delta = +320)
[parity_check] arch d_model=128 n_layers=2 n_heads=4 ff_dim=512
               whole-turn action list match: 16/16
```

### 2.1 Side fix: float32 rounding in ships_t

While checking parity I noticed numpy inference dropping 1 ship per fleet
for several pct bins:

```python
# OLD (buggy):
pct = float(_PCT_BIN_TABLE_NP[pct_t])   # f32 -> python float64
ships_t = max(1, int(np.floor(avail_at_src * pct)))
# For pct=0.70 (stored as 0.6999998 f32), 10 * 0.6999998 (in f64) = 6.999998
# floor(6.999998) = 6   <-- expected 7

# NEW:
mult = np.float32(avail_at_src) * _PCT_BIN_TABLE_NP[pct_t]   # all f32
ships_t = max(1, int(np.floor(mult)))
# 10.0 * 0.6999998 (in f32) = 7.0
# floor(7.0) = 7    OK
```

JAX silently did this correctly via XLA fusion. Numpy didn't. **Every
exported submission from v2 through v6 sent 1 fewer ship per fleet than
training intended** for pct ∈ {0.30, 0.50, 0.70, 0.85}. That's the
*single biggest train/eval gap we've ever had*, and we missed it for
weeks because the parity_check tolerance was on logits, not on
ship-count outputs.

Applied to both `numpy_forward.py` and `submission_rl_v4.py` template.

---

## 3. v7 yaml + launch

`orbit_wars_rl/configs/multi_action_v7.yaml`:
* From scratch (no resume — architecture changed by `+320` params).
* Same arch as v6/v6p3: `d_model=128, n_layers=2, n_heads=4, ff_dim=512`.
* `lr_warmup_steps=2000`, `lr_decay_steps=100000` — same fix as v6p2
  (these are *optimizer* steps not PPO updates, scaled 16x).
* `ent_coef_dst=0.001` — same as v6p3 (10x v6p2 to prevent re-collapse).
* `num_updates=5000`, expected wallclock ~6 h on 5090.
* `selfplay.warmup_updates=50`, frozen pool joins after warmup.

Launched, ran fine. SPS hit 5200 by upd 200 (vs v6p3's 3000 — the per-step
feature compute is essentially free).

---

## 4. v7 training metrics through u575

```
upd      0   adv_std 0.272  pg -0.0002  emits 1.55  ent[s/d/p/e] 0.31/2.89/1.97/0.67
upd     30   adv_std 0.197  pg -0.0026  emits 1.57  ent       0.27/2.82/1.98/0.66
upd    100   adv_std 0.159  pg -0.0036  emits 1.62  ent       0.30/2.74/1.92/0.64       WRr none
upd    150   adv_std 0.165  pg -0.0056  emits 1.58  ent       0.33/2.66/1.86/0.59       WRr 0.72
upd    200   adv_std 0.154  pg -0.0066  emits 1.36  ent       0.37/2.53/1.82/0.53       WRr 0.91 WRf 0.78
upd    300   adv_std 0.181  pg -0.0058  emits 1.50  ent       0.54/2.35/1.77/0.45       WRr 0.84 WRf 0.88
upd    400   adv_std 0.106  pg -0.0070  emits 1.50  ent       0.61/2.12/1.73/0.44       WRr 0.78 WRf 0.88
upd    500   adv_std 0.114  pg -0.0077  emits 1.65  ent       0.59/2.04/1.77/0.43       WRr 0.75 WRf 0.97
upd    575   adv_std 0.120  pg -0.0079  emits 1.56  ent       0.70/1.94/1.77/0.41       WRr 0.75 WRf 0.88
```

### 4.1 What the v7 metrics actually mean (vs v6p3)

| Metric | v6p3 u500 | v7 u500 | What changed |
|---|---|---|---|
| `WRf` | 0.45 | **0.97** | v7 beats every frozen snapshot 97% of the time |
| `tR` | +0.30 | +0.72 | self-play return strongly positive |
| `mean_emits` | 4.0+ | **1.65** | v7 emits 1-2 fleets per turn, not 8 |
| `ent_dst` | 1.31 (collapsed) | 2.13 | dst entropy stays high — head still has options |
| `ent_pct` | 1.50 | 1.78 | pct distribution covers multiple bins |
| `ent_src` | 0.50 | 0.69 | src not commit-locked |
| `clip_frac` | 0.04 (converged) | 0.17 | v7 still actively learning |

**Sign-of-life interpretation**:
* Entropy triangulation: none of the 4 head entropies collapsed.
* `WRf 0.97` is unprecedented — it means the *current* v7 policy beats
  every snapshot in the frozen pool nearly perfectly, sustained across
  hundreds of updates.
* `mean_emits 1.65` is *lower than v6p3* but each fleet is now a real,
  considered launch, not a duplicate spray.

### 4.2 But we still don't know if v7 beats v20

The training signal is "v7 beats its own past selves". That's a healthy
self-improvement signal but it does not prove the policy reaches v20-class
play. We need real H2H to know.

---

## 5. The export bug — and why we lost 30 minutes

First u499 H2H run:
```
vs submission_rl_v1.py        W= 1/40   <-- impossible, should be ~17
```

`diag_real_game_obs` revealed:
```
diag-ERR: matmul: Input operand 1 has a mismatch in its core dimension 0,
          with gufunc signature (n?,k),(k,m?)->(n?,m?)
          (size 265 is different from 264)
```

Root cause: I had updated `orbit_wars_rl/inference/numpy_forward.py` but
**not** `submission_rl_v4.py`. The submission template is a *standalone*
copy of all the head functions — it has to be, because Kaggle imports
only what's in the submission file. With template heads still at v6
shape (264) and ckpt weights at v7 shape (265), every inference call
crashed → `agent() returned []` → empty actions → games lost.

**Fix applied** to template: synced all 4 head functions and
`greedy_multi_action` with the v7 numpy_forward logic. Parity confirmed
end-to-end:

```
$ python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_v7/ckpt_000499.pkl --out submission_rl_v7_u499.py
[export] running parity test against ckpt ..._u499.pkl (tol=0.005, num_states=16)
[OK ] whole-turn action list match: 16/16
[INFO] emit count match: 16/16
[export] smoke moves: [[0, 0.0, 6]]    <-- non-empty: inference works
```

### 5.1 Process lesson

The kaggle deploy model forces this duplication: any inference logic
needed at deploy time must live in *exactly one* template file we
maintain by hand. Going forward I will:

1. Make every change to `submission_rl_v4.py` template *first* (it's
   what Kaggle runs).
2. Then mirror to `orbit_wars_rl/inference/numpy_forward.py` (used only
   by parity test).
3. The parity test should compare *the submission template's* heads
   against JAX, not the numpy_forward heads.

I'll refactor parity test on Day 4 to consume the template directly so
this kind of drift can't recur.

---

## 6. v7 u499 real behaviour (the actual win)

After the template fix, diag_real_game_obs vs v20 (seed 0, player 0):

```
turn |  src    | dst   | pct_bin       | ships sent  | mother sP
   0 |  [12]   | [19]  | [4]=0.55      | 5           | 10 -> 6
   1 |  [12]   | [19]  | [4]=0.55      | 3           |  6 -> 4
   2 |  [12]   | [19]  | [6]=0.85      | 3           |  4 -> 1
   5 |  [12]   | [19]  | [6]=0.85      | 1           |  2 -> 1
  10 |  [12]   | [19]  | [6]=0.85      | 1           |  2 -> 1
```

Comparison with v6p3 turn 0 (same seed, same player):

```
v6p3:  src=[12]×8  dst=[3]×8   pct=[0]×8    -> 8 fleets, all 1 ship
v7  :  src=[12]   dst=[19]    pct=[4]      -> 1 fleet, 5 ship
```

**Bug-fix scoreboard**:

| Original v6p3 bug | v7 u499 status |
|---|---|
| (1) src commit-lock across K | unchanged — only 1 my-planet exists, lock is correct |
| (2) pct stuck at min (0.10) → 1 ship fleets | **fixed** — pct now 0.55-0.85 |
| (3) dst spray (8 different dst at t=0) | **fixed** — only 1 emit at t=0 |
| (4) emit max (K=8 at turn 0) | **fixed** — emits 1.65/turn |
| (5) mother planet drained in 1 turn | **fixed** — drained over ~3-5 turns |

4 of the 5 bugs are demonstrably gone. (1) was never a bug — it's the
only legal choice given the early-game state.

### 6.1 New observation: v7 still doesn't "accumulate then strike"

v7 emits 1 fleet per turn, every turn. That keeps mother planet near
empty. v20's winning trick is the opposite: *let mother planet
accumulate to 5-15 ships, then send one big fleet*. v7 has the *micro*
right (one good fleet at a time) but not the *macro* (when to stockpile
vs spend).

This is a much narrower gap than "always sprays 1-ship fleets". It
might be closable by:
* Adding v20 itself to the opponent pool so the policy meets a stockpile
  strategy in self-play.
* BC warmstart from v20 (the DAY3_PLAN.md path).
* A reward bonus on "ships sent per fleet" to push the policy away from
  tiny fleets.

We won't know which is needed until the full 5000-update v7 run + H2H.

---

## 7. Status snapshot at end of Day 3 (2026-05-23 22:55)

* `v7` training running on 5090: at u575 / 5000.  WRf 0.88-0.97, WRr
  0.72-0.91, emits 1.5-1.7, all 4 ent above their v6p3 collapse points.
* `submission_rl_v7_u499.py` exported with template fix; smoke moves
  non-empty; parity 16/16 against ckpt.
* `diag_real_game_obs` confirms v7 emits *meaningful* fleets (pct=0.55,
  ship counts 3-5) — first time any RL run has done this.
* H2H gauntlet for v7_u499 is the next blocker — number to watch is
  "vs v20" line. Even **3/20 means the architectural fix worked end to
  end**, after which we can decide what to add for the v20-class gap.

### 7.1 Decision tree pending H2H

```
v7_u499 vs v20 result
│
├── ≥ 3/20  → architecture fix succeeded.
│              Plan: let v7 finish 5000 updates, re-evaluate at u4999.
│              If u4999 vs v20 ≥ 5/20: keep going with RL.
│              If u4999 vs v20 stuck at 3-4: try v7p1 with v20 in opponent pool.
│
├── 1-2/20  → fix worked but RL self-play can't bridge to v20.
│              Plan: BC warmstart (DAY3_PLAN.md §2) on top of v7 arch.
│
└── 0/20    → architecture isn't enough. Something else is wrong.
              Plan: deeper diagnosis with replay_dump in mid-game,
              probably reward shaping or BC.
```

---

## 8. What we learned from top_players_rl.txt today

The §65-§109 BC bootstrap discussion was the obvious motivation but the
**more useful insight** turned out to be §230:

> "Action head is pretty much standard."

That sentence implied top teams' action heads see *enough state* to
make different decisions across the K loop. We were not. We'd implicitly
assumed our heads were standard because they were named the same way.
Re-reading the architecture in context made it clear that "action head"
on top of "transformer encoder" implies the head sees a *dynamic* view
of the planet/fleet state, not the static turn-start tokens.

The architectural fix in §2 is what we should have built in v3. We
spent 4 versions (v3 → v6p3) tweaking hyperparameters around a
fundamentally K-blind network.

### 8.1 What we will check from top_players_rl.txt for Day 4

If v7_u4999 still loses to v20, the next priorities from the doc:

| Section | Topic | Why now |
|---|---|---|
| §75-109 | BC from heuristic | Plan B (DAY3_PLAN.md ready to execute) |
| §170-200 | Reward shaping for fleet size | If emits stays at 1/turn, reward bigger fleets |
| §230 onwards | Action head details | Re-read to find anything else we missed |
| §145-165 | Opponent pool curation | Inject v20 to learn anti-stockpile play |

---

## 9. Status of v7 in context

| Run | Final result vs v20 | Diagnosis |
|---|---|---|
| v4.2 | 0/20 | mean_emits stuck at 1.4 (couldn't credit-assign K-step) |
| v5.3 | 0/20 | same — pure-RL credit assignment time budget too small |
| v6p2 u999 | 0/20 | dst collapsed to src ("smoke moves: []") |
| v6p3 u4999 | 0/20 | K-step heads blind to reserved state — *the actual bug* |
| **v7 u499** | **?** | K-step heads see remaining/reserved; pct now uses high bins |

The unbroken 0/20 streak across 5 versions was misleading us. It looked
like a strategy ceiling. It was actually a series of distinct, increasingly
subtle bugs — each one masked the next by producing the same end metric
(0/20 vs v20). v7 may or may not be the version that finally lifts the
ceiling, but it's the first one whose *policy itself* is non-degenerate
according to replay analysis.

---

## 10. Files touched today

* `orbit_wars_rl/net/heads.py` — 4 heads now optionally take reserved-aware inputs.
* `orbit_wars_rl/net/model.py` — added `_remaining_features()`; sample + evaluate updated.
* `orbit_wars_rl/inference/numpy_forward.py` — mirror; float32 ships_t fix.
* `submission_rl_v4.py` — **same** changes mirrored (this was the gotcha).
* `orbit_wars_rl/scripts/diag_real_game_obs.py` — fixed dataclass module-loading bug for python 3.13.
* `orbit_wars_rl/configs/multi_action_v7.yaml` — new run, ent_dst 0.001, warmup_steps 2000.
* `docs/DAY2_PROGRESS.md` §12 — full v6p3 → v7 chain documented.
* `docs/DAY3_PROGRESS.md` — this doc.

## 11. v7_u499 H2H result + BC parallel track

### 11.1 The result

```
vs submission_rl_v1.py     W=14/20   WR=0.70   avg_steps=424
vs submission_rl_v4p2.py   W=19/20   WR=0.95   avg_steps=500
vs submission_v20_0513.py  W= 0/20   WR=0.00   avg_steps=159
                                   overall  0.55
```

Comparison vs v6p3 u4999 (the previous best):

| Opponent | v6p3 u4999 | v7 u499 | delta | interpretation |
|---|---|---|---|---|
| v1   | 17/20 (0.85) | 14/20 (0.70) | −0.15 | 1-fleet-per-turn means less brute-force pressure vs v1 |
| v4p2 | 15/20 (0.75) | 19/20 (0.95) | **+0.20** | architectural fix pays off vs our own family |
| v20  | 0/20  (0.00) | 0/20  (0.00) | 0.00  | strategy ceiling not lifted |

**v4p2 +20%** confirms the v7 fix made the policy strictly stronger at the
*micro* level (one good fleet > eight 1-ship fleets). But the *macro*
strategy — stockpile then strike — was never learned because:

1. v7's self-play opponents (frozen v7 snapshots) all emit 1 fleet/turn,
   so the v7 actor never *sees* a stockpile-then-strike opponent during
   training.
2. v20 ends games at avg_steps=159, i.e. v20 kills v7's mother planet around
   turn 80 — too early for v7 to compete on long-horizon production.

That's textbook §75-§109 of `top_players_rl.txt`: **pure-RL self-play
can't credit-assign across 800-step delayed rewards**. We need BC from
v20 to seed the macro strategy.

### 11.2 Why we did not just continue with v7

If we ran v7 to u5000 we would burn 5h GPU. Even an optimistic +3/20 vs v20
at u5000 would still leave us at 0.55 overall (= u499). The marginal
expected value of waiting is low.

Instead we kept v7 training in the background and committed to BC in
parallel — this is the **Plan D** mentioned at top of §0.

### 11.3 BC pipeline status (this evening)

Implemented and smoke-tested today:

| File | Status | Notes |
|---|---|---|
| `bc/action_inverse.py` | **done** | 5 unit tests pass (single move, two moves same src reserved, empty moves, invalid src, K-overflow) |
| `bc/collect_data.py`   | **done** | 2-game smoke produced 466 samples — see §11.4 for the v20 stats |
| `bc/train_bc.py`       | **done** | 1-epoch smoke training drops loss 10.06 → 8.93; val acc src/dst/pct/emit climbing; ckpt export+inject works |
| 100-game collection    | **running** | Background, PID 13902, ~50 min total |
| BC training (real)     | TBD       | After collection finishes, ~30 min of training |
| BC validation H2H      | TBD       | Goal: ≥ 5/20 vs v20 from BC alone |
| RL fine-tune (Day 4)   | TBD       | If BC reaches 5/20 vs v20, fine-tune to push higher |

### 11.4 v20 self-play statistics (2-game smoke)

The data tells us why v7's strategy is structurally wrong:

```
total samples: 466 (= 2 games × ~120 turns × 2 perspectives)
total emits  : 406 (avg 0.87 per sample)

emits-per-sample histogram:
  0 emits: 64.4%   ← v20 STOCKPILES the majority of turns!
  1 emits: 15.0%
  2 emits:  8.8%
  3 emits:  4.3%
  4 emits:  3.6%
  5 emits:  0.6%
  6 emits:  0.9%
  7 emits:  0.2%
  8 emits:  2.1%   ← v20 occasionally swings hard (8 fleets in one turn)

pct_bin distribution (across actually-emitted fleets):
  bin 0 (0.10):  6.7%   ← small scouts/feints
  bin 1 (0.20):  8.1%
  bin 2 (0.30): 11.3%
  bin 3 (0.40): 17.5%
  bin 4 (0.55): 26.1%   ← single most common
  bin 5 (0.70): 20.2%
  bin 6 (0.85):  8.9%
  bin 7 (1.00):  1.2%   ← all-in is almost never used
```

Two-thirds of v20 turns are *no-action*. That is fundamentally
incompatible with any policy that learns "emit something useful every
turn" via self-play. The 0-emit rate is the macro behaviour the BC must
clone, and it is exactly the behaviour our RL training never sees a
reward signal for (a 0-emit turn produces neither immediate progress
nor an obvious downside in self-play, since both sides do nothing).

`top_players_rl.txt` §75-§109 says all top-scoring RL teams bootstrapped
from a strong heuristic for exactly this reason.

---

## 12. Course correction — re-read top1's forum posts

Late evening, user shared an updated version of `top_players_rl.txt` with
new replies from Lin Myat Ko (top1) to forum questions. Three statements
flip our plan upside down:

### 12.1 What top1 actually said

* §70 — "Do you need to write heuristic agent? **I don't.**"
  → Top1 does **not** use BC warmstart at all. Pure self-play.

* §82 + §409 — "Best model took ~3 days, self-play from the start" =
  **600M steps**. v6p3 ran ~80M. v7 u499 ran ~8M.
  → We are not under-architected, we are **under-trained**.

* §92–§96 — "Add one architecture delta at a time. Always. I shipped 7
  changes vs F12 in two days. ... lost 1 baseline."
  → We've been shipping 5+ deltas per version (v6→v7 changed 4 heads,
  added float32 fix, added src_idx mask). Top1's F-series took a year
  to evolve in tiny single-deltas.

* §102 — "**clip_frac creep 0.10 → 0.30+ is the most reliable warning
  sign**. Cut lr or revert capacity. Don't wait."
  → We watch clip_frac but never set a hard threshold for action.

* §307 — "**explained_variance should hit at least 0.8 in 100 iters**,
  0.9 in 20 iters. If it never gets past 0.5, your obs representation
  or architecture is wrong."
  → **We never measured this.** We've been flying blind on the
  single most important value-head health metric.

* §216 — "RL policy is robust enough to learn the fraction"
  → confirms our pct head with bins is correct, not a heuristic.

* §73 — "+1 −1 is enough for 2p mode"
  → confirms our `ORBITWARS_SHAPING_SCALE=0.0` choice.

### 12.2 What changed in this conversation

| Before | After |
|---|---|
| Plan: BC bootstrap (DAY3_PLAN.md, §2.1–§2.4) | **Cancelled BC route.** Top1 explicitly does not use BC. |
| Plan: kill v7, write v8 with heuristic seed | **v8 = "v7 + log explained_variance + 2x training budget".** No architecture delta. |
| Metric: clip_frac shown, no threshold | **`clip_frac > 0.25` sustained 50 updates ⇒ cut lr 2x. > 0.35 ⇒ kill.** |
| Metric: no explained_variance | **`explained_variance` added to PPO loss and train log column.** |
| BC pipeline (`bc/action_inverse.py`, `bc/collect_data.py`, `bc/train_bc.py`) | **Code kept on disk as Plan B**, not used. Tests still pass. |

### 12.3 v8 spec

* `configs/multi_action_v8.yaml`:
  * SAME architecture as v7 (reserved-aware heads, single delta from v6p3).
  * SAME hparams (lr peak, ent_coefs, clip_eps, all from v7).
  * `num_updates` 5000 → 10000 (= ~160M extra steps).
  * `lr_decay_steps` 100000 → 200000 (matches longer run).
  * `ckpt_every` 50 → 100 (operational change, not learning).
  * Resume from v7 final ckpt instead of from scratch so we keep
    the 80M steps already invested.
  * `seed` 70 → 80 (new RNG since resuming).

* Code changes (one delta: instrumentation only):
  * `orbit_wars_rl/ppo/update.py`:
    `explained_variance = 1 - Var(returns - value) / Var(returns)`
    added to metrics dict and the static `_ZERO_METRICS_KEYS` tuple.
  * `orbit_wars_rl/ppo/runner.py`: print format now includes `ev +0.42`
    next to `v 0.012`.
  * Local smoke (3 updates): `ev +0.13 → +0.24 → +0.49`. Curve climbs
    as expected — code works.

### 12.4 Sign-of-life expectations for v8

| Update | Metric | Expected | Action if not met |
|---|---|---|---|
| upd 50  | explained_variance | ≥ 0.6 | If ≤ 0.3, value head broken — debug obs |
| upd 100 | explained_variance | ≥ 0.8 | If < 0.5, obs/arch problem (top1 §307) |
| upd 1000 | clip_frac | < 0.20 | If ≥ 0.25 sustained, cut lr 2x |
| upd 3000 | WRr | > 0.90 | (matches v6p3 plateau) |
| upd 5000 | H2H vs v20 | ≥ 3/20 | first sign macro strategy learned |
| upd 10000 | H2H vs v20 | ≥ 8/20 | top1 trajectory scaled by 1/6 |

### 12.5 What v8 still cannot test

If v8 u10000 is **still 0/20 vs v20**, the four remaining hypotheses
(in order of cost):

1. **More training**: bring it to 20k updates (= 320M steps, still
   half top1's 600M).
2. **v20 in opp pool**: inject v20 ckpt as a frozen opponent at upd 200.
   Forces policy to actually see stockpile-then-strike behaviour. This
   is a single delta vs v8 — we can do it.
3. **F12 architectural sweeps** documented in §92–§96 of top1's post:
   TypedInputProjection, sun mask, MLP FireHead, per-source TargetMix,
   multi-query ValueHead. Each is a single delta, testable.
4. **BC as Plan B**: code already on disk (`bc/`); if the previous
   three fail, BC clone v20, then RL fine-tune.

Top1's lesson on this exact path:
  > "F12's single-Dense FireHead, global head_mix_logits, missing sun
  > mask — those weren't bugs to fix. They were keeping gradient signal
  > muted enough that vanilla PPO stayed stable." (§95)

i.e. we should be *very suspicious* of each architectural delta — they
can blow up training even when each looks correct on paper.

---

## 13. Open todos

* User launches v8 on 5090: resume from latest v7 ckpt, log to
  `logs/multi_action_v8.log`, watch `ev` and `clip` columns.
* Whoever monitors: alert at clip_frac ≥ 0.25 sustained 50 updates,
  or explained_variance < 0.5 at upd 100.
* Day 4: refactor parity test against submission template (prevents
  §5 drift recurrence).
* Day 4: write `monitor_train.py` — tails the log, computes rolling
  averages of clip_frac and EV, prints an alert when thresholds break.
* Day 4: add `mean_ships_per_fleet` and `pct_bin_distribution` to log
  (currently only `mean_emits` — useful but not sufficient).
* BC pipeline kept on disk as Plan B; do not delete `bc/`.
  Test suite: `python -m orbit_wars_rl.bc.test_action_inverse` still
  passes (verified).

---

## 14. Pre-v8 Reward Audit (2026-05-23 night)

User pushback before v8 launch: *"现在这个阶段我觉得还没到 one-at-a-time
的时候,属于基本的 reward 信号还存在问题,修一个没有意义,先按照现在没对
齐的部分改彻底把."*

Audited `orbit_wars_rl/env/rewards.py` against
`kaggle_environments/envs/orbit_wars/orbit_wars.py:684-715`. **Four
silent mismatches found.** All four had been there since at least v3.
All four are now fixed in a single PR (Day 4 morning) before v8 launches.

### 14.1  terminal_reward: tie was 0/0, should be +1/+1

Kaggle (line 710-715):
```python
max_score = max(scores)
for i in range(num_agents):
    if scores[i] == max_score and max_score > 0:
        state[i].reward = 1
    else:
        state[i].reward = -1
```

So a 75-vs-75 ship tie gives **+1 to BOTH players**, not 0/0.

Our code (pre-Day-4):
```python
win = me > opp
loss = me < opp
return (win - loss)  # tie -> 0
```

**Why this matters.** With 80-turn episodes and the v6p3/v7 policy class,
~10-15% of episodes ended in a near-tie (both home planets still alive,
mid-game). Old reward marked those as 0; new reward marks both as +1.
The value head was being told "I might end up in a tie state, no signal";
it should have been told "ties are wins, plan for them or lose."

### 14.2  Double wipeout: 0/0 should be -1/-1

Kaggle's `max_score > 0` clause: if both players are reduced to 0 ships,
**both get -1** (neither is "the winner of nothing").

Our code: tie at 0 → 0/0.

In 80-turn episodes this is rare but possible (suicide trades into the
final turn). Fix: aligned with kaggle.

### 14.3  Termination at step >= episodeSteps - 2

Kaggle (line 686): `if step >= configuration.episodeSteps - 2: terminated = True`.

Our code: `state.step >= episode_steps`. Off by 2.

Minor practical impact (episode is 2 ticks shorter than I thought) but
**meaningful for log interpretation**: when we said "episodes end at step
78" we should have said step 78 (= 80 − 2). Now corrected so the env's
behaviour matches the spec exactly.

### 14.4  SHAPING_SCALE default 0.1 → 0.0

`rewards.py` had `SHAPING_SCALE = float(os.environ.get(..., "0.1"))`.

Every launch since v5p2 set `ORBITWARS_SHAPING_SCALE=0.0` on the command
line because top1 §73 says +1/-1 is enough. But the *default* if you
forget the env var was 0.1 — a hidden trap.

Fixed: default is now `0.0`. Set env var to `0.1` if you want shaping
back. The startup banner now echoes `SHAPING_SCALE=X.X` so it's auditable.

### 14.5  Audit verification

* New unit test: `orbit_wars_rl/env/test_rewards.py`. Run with
  `python -m orbit_wars_rl.env.test_rewards`. **15 / 15 pass**.
* New runtime banner in `ppo/runner.py:train()`:
  `[reward] kaggle-aligned: terminal +1 win/+1 tie>0/-1 loss/-1 double-wipeout; ...`
* Env-parity (30 steps, seed 42) still clean — only reward/done logic
  changed, dynamics unchanged.
* Rollout smoke (4 envs × 128 steps): no NaN, all done rewards in
  {-1, +1}, all non-done rewards = 0 (since SHAPING_SCALE=0).

### 14.6  What we deliberately did NOT change

* `episode_steps: 80` (not 500). This is a **hyperparameter**, not a
  reward bug. 80 steps gives terminal signal in every rollout
  (rollout_length=128 > 80) which is critical for GAE credit assignment.
  Top1 used 500 + 600M training steps; we use 80 + ~100M steps. Trade-off
  is documented in `multi_action_v8.yaml` header comments.
* `shaping_potential` math. Not used by default; kept so we can A/B it
  later if pure +1/-1 stalls.

### 14.7  Re-run plan for v8

After these fixes, v8 launches with:
* New reward landscape that matches kaggle exactly.
* Resume from v7 final ckpt → value head must re-learn ties=win (~50
  updates of value-head adjustment expected before metrics stabilise).
* Same architecture (v7 reserved-aware K-step heads).
* Same hyperparams (lr schedule, ent_coefs, episode_steps=80).
* Banner in log confirms reward config at every launch.

This is now "one delta at a time" *properly*: the architecture delta
(v6p3 → v7) is held fixed; only the **reward function is being aligned
with the spec it should have been aligned with from day 1**. After v8
we can resume "one architecture delta at a time" with confidence that
the reward signal is correct.
