# DAY4 进展 — Reward 信号深度审计 + Track 1/2/3 路线

> 写于 2026-05-24。Day 3 把 v7 reserved-aware 架构修了，把 reward function 与
> kaggle 对齐了，跑了 v8（7400 updates / ~121M steps）。**v8 干净停顿、未崩，
> 但策略在 v7-class plateau**。Day 4 不再加架构 delta，而是回到 top1 §66-§70
> 「reward shaping is important; you want to see the architecture react to
> reward shape — signs of life」的原则：先把游戏机制对应的 reward 信号补到位，
> 然后让纯 self-play 跑足够长的时间。

---

## 0. TL;DR

* **v8 没崩**。121M steps、EV 0.88-0.92、clip 0.10-0.13、4 个 ent 都没坍缩。
  WRr 0.78-0.91、WRf 0.84-0.97 健康振荡。**指标全绿，但策略停留在 v7-class**。
* **新发现：v7/v8 在真实 500-turn kaggle 游戏里行为远比训练 log 显示的差**。
  v7 vs v20 实测：v7 母星 garrison 平均 **9.3 舰** vs v20 **212 舰**（22x 差距）；
  v7 每支舰队 **1.98 舰** vs v20 **15.7 舰**（8x 差距）。
* **核心机制盲点**：`speed = 1 + 5 * (log(ships)/log(1000))^1.5`。**舰队规模直接
  决定速度**——1 舰飞 1 单位/turn，500 舰飞 5 单位/turn。我们训练用 80 turn
  episode，对角线 ~70 单位，**1 舰小舰队飞不到 v20 就输了**。
* **`episode_steps=80` 是 train-eval gap 的根源**。kaggle 全程 500，实战 replay
  显示游戏经常打到 300+ 回合。**Day 4 改为 350**。
* **Day 4 三轨**：Track 1 已落地（behaviour metric + replay analyzer）；
  Track 2（pct_bin 加 0.0）**暂不做**，新数据显示不是 root cause；
  Track 3（reward shaping family + episode_steps 350）是主线。
* **v9 = v7 架构（不动）+ episode_steps 350 + HOMEWORLD_KEEP & FLEET_SIZE
  shaping + 更长训练**。从 v7 ckpt resume。

---

## 1. v8 训练复盘（u0 → u7391, ~121M steps）

### 1.1 健康指标全程 OK

| 指标 | u499 | u2999 | u4999 | u7391 | top1 阈值 | 判定 |
|---|---|---|---|---|---|---|
| `explained_variance` | 0.89 | 0.88 | 0.86 | 0.90 | ≥ 0.8 @ u100 | ✅ value head 健康 |
| `clip_frac` | 0.18 | 0.17 | 0.16 | 0.10-0.13 | < 0.25 持续 | ✅ |
| `approx_kl` | 0.012 | 0.012 | 0.014 | 0.008 | < 0.02 | ✅ |
| `WRr` (vs random) | 0.81 | 0.91 | 0.91 | 0.88 | > 0.85 | ✅ 但不再升 |
| `WRf` (vs frozen) | 0.84 | 0.78 | 0.91 | 0.84 | 振荡 | ✅ 健康 self-play |
| `mean_emits` | 2.52 | 2.20 | 2.89 | 2.45 | — | ⚠️ 窄区间 |
| `ent_dst` | 1.90 | 1.64 | 1.62 | 1.47 | 不坍缩 | ✅ 缓慢漂移 |

**结论**：训练机制本身完全健康，**这是 self-play Nash plateau**，不是 bug。

### 1.2 v8 干净停止，未崩

最后一行 `upd 7391  steps 121110528  ...  emits 2.45  clip 0.11`。
没有 traceback、没有 OOM、没有 NaN。可能是 nohup session 断开或手动 kill。
**继续硬跑到 10k/20k 不大可能改变 plateau**——指标已横向 5000 updates。

---

## 2. Replay 实战数据 — 第一次揭穿训练 log 的假象

`scripts/replay_analyze.py`（Track 1 新工具）跑了 3 组 kaggle env 真实游戏（500 turn），
每组 3 局，按「前 80 turn」与「全程」两个时间窗口聚合 6 个行为指标：

### 2.1 三组对比（每格 `首80turn / 全程`）

| 指标 | v7 vs v7 | v7 (vs v20) | v20 (vs v7) | v20 vs v20 |
|---|---|---|---|---|
| outcome (3 局 W/L/D) | 平 | **0/3/0** | **3/0/0** | 平 |
| emits/turn | 3.7 / 3.7 | **2.62 / 1.45** | 0.66 / 0.55 | 0.76 / 1.60 |
| zero_emit_rate | 0.2% / 0.3% | **4% / 49%** | 36% / 72% | 17% / 38% |
| mean_ships_per_fleet | 17.9 / 17.9 | **1.98 / 1.97** | 15.7 / 17.99 | 16.0 / 15.1 |
| mean_garrison_my | 401 / 401 | **9.3 / 5.1** | 212 / 1375 | 105.7 / 280.7 |
| p95_garrison_my | 2942 | **34 / 20** | 777 / 6092 | 281 / 857 |
| fleet_flip_rate | 1.14% / 1.24% | 23% | 7.5% | 7.5% |

### 2.2 五个颠覆性发现

#### 发现 1：v7 micro 完全没对（DAY3 §6.1 是误判）

之前的判断：「v7 micro 对了（每次发一支好舰队），macro 不对（不囤积）」。
**新数据完全推翻**：

* v7 在 v20 面前 `mean_ships_per_fleet = 1.98` —— **每支舰队不到 2 舰**
* v20 是 15.7 舰
* 差距 8x

`diag_real_game_obs` 在 DAY3 §6 看到 v7 发了 `pct=4(0.55), 5 ships` 是因为
那时母星刚开局有 10 舰。**而 v7 vs v20 几个 turn 后母星就被打回 1 舰，
然后 v7 的所有舰队都是 1 舰小舰队**。

#### 发现 2：v7 母星速崩（first 80 turn 内就死）

v7 vs v20 在 first 80 turn 内：

* v7 母星 garrison 平均 **9.3 舰**，p95 仅 **34 舰**
* v20 母星 garrison 平均 **212 舰**，p95 **777 舰**
* 差距 22x

`diag_real_game_obs` 在 DAY3 §6 看到 v7 母星 turn 0 从 10 → 6 → 4 → 1 → 1，
**当时以为这是好行为「3-5 回合慢慢抽干」，实际是灾难**。

#### 发现 3：v7 self-play 看起来好，是「双方都不打」的废涡

v7 vs v7 `fleet_flip_rate = 1.14%` —— 1000 个舰队里只有 11 个让行星易主。
意味着 v7 自对弈里 99% 的舰队都在打废仗，**没有任何 capture 信号驱动学习**。
而它的 `mean_ships_per_fleet = 17.9` 看起来漂亮，**只是因为对手太弱，没人打到它的母星，
母星可以慢慢累积到 400 舰**。`v7 vs v20` 时这个数立刻塌到 1.98。

#### 发现 4：episode_steps=80 的真实代价

旧假设：「80 太短，学不到囤积」。
新数据：v20 在 **first 80 turn** 已经在囤积（garr 105.7，z0 17%）。
真实问题是：**v7 在前 80 turn 母星就被打到 garr<10**，根本没有「囤积选择」的样本。

意味着：

* 改 episode_steps 80 → 350 + reward shaping 必须 **同时做**
* 单改 episode_steps 没用：80 turn 内 v7 都死了，给它 350 turn 更死得彻底
* 单加 shaping 没用：v7 训练时还是只看到 80 turn 内的小样本，没机会演练 mid-game

#### 发现 5：fleet 速度机制 — 我们一直忽视的关键

`overview.txt §121-127`：

```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
1 ship → speed 1.0/turn
~500 ships → speed ~5/turn  
~1000 ships → max 6.0/turn
```

我们 env 的 `dynamics.fleet_speed` 与之 1:1 对齐（已验证）。**但 RL 策略不知道这件事**。

* 1 舰小舰队跨越对角 ~70 单位需要 **70 turn**
* 500 舰大舰队需要 **14 turn**
* 5x 速度差距

**v7 在 v20 面前发 1 舰小舰队**：飞 70 turn 才到，**那时 v20 已经早就摧毁了 v7 母星**。
舰队规模不是「好看」的事，**是决定能不能在被摧毁前到达目标**。

这给 `FLEET_SIZE` shaping 提供了硬机制依据，不是 ad-hoc bonus。

---

## 3. v8 真正缺的 reward 信号清单

基于 §2 数据，按重要性排序：

### 缺陷 A：「保住母星 garrison」无 reward 信号 — **致命**

**证据**：v7 母星 garr < 10 几乎全程，被 v20 一发 8-20 舰冲塌就输。

**为什么 +1/-1 看不到这个**：

* episode_steps=80 + GAE γ=0.97 → 母星在 turn 20 被打塌，terminal 在 turn 50，
  γ^30 ≈ 0.40，信号衰减一半
* 而且 v7 母星被打塌的所有训练 sample 都得到同样的 -1，PPO 无法区分
  「turn 20 失去母星 = 早 lost」 vs 「turn 50 失去母星 = 几乎平局」

**修复**：`reward_step += α · tanh(log1p(my_homeworld_ships) / 8)`，每步小正 reward。
让 value head 学到「母星 ships 多 ≈ 好」。`α ≤ 0.01`。

### 缺陷 B：「单舰队规模 → 速度」无 reward 信号 — **关键**

**证据**：v7 spf=1.98 vs v20 spf=15.7。1 舰飞 70 turn，500 舰飞 14 turn。

**修复**：发射时 `reward += β · (min(ships,20)/20 − threshold)`。`β ≤ 0.01`。
强迫 RL 学到「发就发大的」。

### 缺陷 C：「fleet 浪费 / capture 成功」无信号 — 次要

**证据**：v7 vs v7 flip rate 1.14% — 99% 的舰队都在白干。

**修复**：成功 capture 加小正 reward，但实现成本高（需要追踪 fleet → outcome）。
**Day 4 不做**，留作 Day 5+ 备选。

### 缺陷 D：episode_steps=80 vs 500 的本质失配 — **结构性**

**证据**：v20 vs v20 first80 vs full 数据差异巨大；v7 在前 80 turn 母星都已经
被速杀。

**修复**：episode_steps 80 → 350（多人 replay 录像显示游戏经常打到 300+，
350 是「足够覆盖典型 mid-late game 决战」的最小值）。

**代价分析**：

* 80 → 350：单 episode 长 4.4x，但 rollout_length 不变（128），
  所以 **terminal 信号密度下降 4.4x**
* 缓解：用 shaping 在每步给信号 → 不需要靠 terminal 频率
* GPU 时间：理论上不增加（同样的 env steps / update），但 H2H eval 单局变 4.4x 慢

---

## 4. Track 1 已完成：可观测性

### 4.1 训练 log 6 个新 metric（已合入 `ppo/update.py`）

| metric | 含义 | top1 阈值 / v20 参考 |
|---|---|---|
| `mean_ships_per_fleet` (`spf`) | 每支已发射舰队的平均舰数 | v20 ≈ 15-17 |
| `zero_emit_rate` (`z0`) | 0-emit 回合占比 | v20 ≈ 17-40% |
| `mean_garrison_my` (`garr`) | 我方所有行星平均 garrison | v20 ≈ 100-280 |
| `pct_bin0..7` | pct head 选择 8 个 bin 的分布 | v20 偏 bin 4/7 |
| `explained_variance` (`ev`) | DAY3 已有 | ≥ 0.8 |
| `mean_emits_per_turn` | DAY3 已有 | — |

训练 log 列扩展（已合入 `ppo/runner.py`）：
```
upd 0  ...  ev +0.13  ...  emits 2.41  spf 2.7  z0 0.03  garr 7.0  clip 0.14
upd 4  ...  ev -0.24  ...  emits 2.11  spf 7.2  z0 0.06  garr 17.1 clip 0.01
```

Smoke train 5 updates，确认 metric 真实产出。本地 CPU 小规模也能看到 `spf` 从
2.7 在 4 个 update 内升到 7.2，**说明 metric 有判别力**。

### 4.2 `scripts/replay_analyze.py`（已落地）

吃两个 submission，跑 N 局 kaggle env 真实游戏，按「前 80 turn」+「全程」两个
时间窗口聚合 6 个 metric，导出 JSON。**唯一能桥接「训练 log 数字」与
「kaggle 真实行为」的工具**。

已经跑了 3 组对比并存档：

* `logs/replay_analyze/v7_vs_v7.json`
* `logs/replay_analyze/v7_vs_v20.json`
* `logs/replay_analyze/v20_vs_v20.json`

`fleet_loss_rate` 字段含义注意：**不是 sun loss**，是「fleet 消失但没造成所有权
翻转」的占比。包含 combat 互抵、reinforce 已有行星、sun loss、出界。**只用于
agent 之间对比**，绝对值不可解读为单一物理事件。

---

## 5. Track 2（pct_bin 加 0.0）— Day 4 暂不做

原计划：把 `PCT_BIN_VALUES = (0.10, ..., 1.00)` 改为 `(0.0, 0.10, ..., 1.00)`，
让 pct head 自己学「不发」。

**§2 新数据后取消**：

* v7 在 v7 vs v7 时 `mean_ships_per_fleet=17.9`，证明 pct head **已经能在
  garrison 高时发大舰队**
* v7 在 v7 vs v20 时跌到 1.98，**不是因为 pct bin 太粗，是因为 garrison 已经 < 10**
* 加 0.0 bin 修的是「不该发时还发」，但 v7 的问题是「garrison 太低，发啥都不到」
* `bin0=0.10` × 1 舰 garrison = 1 舰舰队，与 `bin0=0.0` × 1 舰 garrison = 0 舰，
  在 garrison=1 时几乎等价

**结论**：Track 2 推迟到 Day 5+。Track 3 是真问题，先动它。

---

## 6. Track 3 计划（Day 4 主线）

### 6.1 v9 配置（一次只动 reward + episode，不动架构）

| 参数 | v8 | v9 | 理由 |
|---|---|---|---|
| 架构 | v7 reserved-aware | **same** | top1 §92 one-delta-at-a-time |
| episode_steps | 80 | **350** | overview / 多人 replay 显示典型对局 300+ turn |
| 起始 ckpt | v7_u4999 | **v7 final** | 保留 80M 已投入的训练 |
| num_updates | 10000 | 10000 | 同 v8 |
| rollout_length | 128 | **256** | episode 变长 4.4x，至少让一个 episode 能完整在 rollout 内结束（350 > 256 仍跨段，但比 80 < 128 好） |
| shaping family | OFF | **HOMEWORLD_KEEP + FLEET_SIZE** | §3 缺陷 A + B |
| `ent_coef_emit` | 0.003 | 0.003 | 同 |
| 其余 PPO | 同 v8 | 同 v8 | |

### 6.2 Reward shaping 数学（写入 `env/rewards.py`）

```python
# step_reward = SHAPING_KEEP * tanh(log1p(my_homeworld_ships) / 8)
#             + SHAPING_FLEET_BONUS * sum_launches(min(ships,SHIP_NORM)/SHIP_NORM - 0.2)
#             + (existing potential-based shaping, default 0)
```

`SHAPING_KEEP` 和 `SHAPING_FLEET_BONUS` 都由环境变量控制，**默认 0**（向后兼容）。
`v9` 启动时设 `SHAPING_KEEP=0.01 SHAPING_FLEET_BONUS=0.005`。

**关键约束**：

* 全部 shaping 累计每 step 上界 < 0.02
* 80-step episode 累计 shaping ≤ 1.6，**仍小于** terminal ±1 的绝对值之和（2.0）
* 350-step episode 累计 shaping ≤ 7.0 —— 这超过 ±1，**需要重新平衡**
* 实际选 `SHAPING_KEEP=0.003, SHAPING_FLEET_BONUS=0.002`：
  350-step 累计上界 ≈ 1.05 + 0.7 = 1.75 ≈ 与 terminal 同量级

启动 banner 必须 echo 全部 shaping 系数。

### 6.3 Ablation 设计

| run | episode_steps | KEEP | FLEET | 评估窗口 |
|---|---|---|---|---|
| v9.a (baseline) | 350 | 0 | 0 | 500 updates |
| v9.b (KEEP only) | 350 | 0.003 | 0 | 500 updates |
| v9.c (FLEET only) | 350 | 0 | 0.002 | 500 updates |
| v9.d (both) | 350 | 0.003 | 0.002 | 500 updates |

**判定指标**（500 updates 末）：

* **必须满足**：`ev > 0.7`（500 updates 后），`clip_frac < 0.25` 持续
* **正向信号**：`mean_garrison_my` 较 v8 翻倍以上（v8 ≈ 7-17 → v9 目标 ≥ 30）
* **正向信号**：`mean_ships_per_fleet` ≥ 5（v8 ≈ 5-7 → v9 目标 ≥ 8）
* **决定性指标**：v7_u499 H2H 在新 ckpt 下，vs v20 至少破 0/20（哪怕 1/20）

选出最有「sign of life」的 1 个 shaping config，跑完整 10k updates 作为 v9 主线。

### 6.4 episode_steps 350 的工程影响

* **rollout_length 必须随之调整**：v8 是 128，350 turn 跨 ~2.7 个 rollout
  → 大部分 rollout 看不到 terminal 信号 → GAE bootstrap 不稳
  → 提案 256，仍跨段但只 1.4 段，与 v8 的 128/80=1.6 接近
* **H2H eval 慢 4.4x**：单 eval 局从 ~80 turn 变 ~350 turn。`eval_every=25` 不变，
  但 eval_num_envs 32→16 把 eval 时间总量控制住
* **reward banner 必须明示**：「episode_steps=350; kaggle=500; mismatch=150 turn」

---

## 7. Day 4 执行顺序

1. ✅ **Track 1.1 训练 log 加 metric**（`ppo/update.py`, `ppo/runner.py`）
2. ✅ **Track 1.2 `replay_analyze.py`**
3. ✅ **Track 1.3 跑 3 组对比 + 诊断**（v7-vs-v7, v7-vs-v20, v20-vs-v20）
4. 🔄 **写 Day 4 progress 文档（本文）**
5. ⏳ **Track 3.1 实现 reward shaping family**：`env/rewards.py` 加
   `homeworld_keep_reward`、`fleet_size_reward`，env vars 控制，banner echo，
   单元测试覆盖 OFF / KEEP-only / FLEET-only / both 4 种 config
6. ⏳ **Track 3.2 写 `multi_action_v9a.yaml ~ v9d.yaml`**（4 个 ablation 配置）
7. ⏳ **本地 smoke**（10 updates each）确认 shaping 数学正确、不爆 NaN
8. ⏳ **GPU 启动 4 个 500-update ablation**，看 `garr` / `spf` / `ev` / `clip`
9. ⏳ **挑选最优 shaping，10k updates 主线训练**
10. ⏳ **训练完毕 H2H gauntlet vs v1 / v4p2 / v20** + replay_analyze 复测

### Day 4 杀手指标（任一触发即 abort 当前路线）

* shaping 让 `ev` 永远 < 0.5（top1 §307）
* shaping 让 `clip_frac` 长期 > 0.30
* 4 个 ablation 都没让 `mean_garrison_my` 翻倍

如果都触发，回退到 v8（v7 ckpt）的 reward，**改投 Day 5 的 architecture sweep**：
TypedInputProjection、sun mask、MLP FireHead（top1 §92 列表）。

---

## 8. Day 4 不做的事（明确排除）

* ❌ BC（top1 §70 明确不用，DAY3 §12.1 已排除）
* ❌ 把 v20 注入 frozen pool（变相 BC）
* ❌ 大改架构（top1 §92 one-delta-at-a-time，§95 删除「stupider」regularization 危险）
* ❌ pct_bin 加 0.0（§5 暂时不动）
* ❌ 增加 transformer 容量（v8 EV 0.9 → 架构容量足够）

---

## 9. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| shaping 系数太大 → value head 学 shaping 而非 win/loss | 中 | 高 | 单步上界 0.02，accumulated ≤ terminal |
| episode_steps 350 → rollout cross-episode → GAE 不稳 | 中 | 中 | rollout_length 256；不稳就回 80 |
| v7 ckpt resume + 新 reward → 头 100 updates ev 跌至 0 | 高 | 低 | top1 §307 说 100 updates 内回升 0.8 即可 |
| 4 个 ablation 都没 sign of life → Day 4 烧 4× GPU 时间无果 | 低 | 高 | 每个 ablation 早停阈值（200 updates 无信号即 kill） |

---

## 10. 待整理（次要）

* `DAY3_PROGRESS.md` 还有未提交的 zh→en 同步翻译修改
* `DAY3_PROGRESS.zh.md` 已写好但 untracked
* `monitor_train.py` 还没写（DAY3 §13 留下的）—— Track 3 启动前必须有，
  否则 4 个 500-update ablation 没人盯

---

## 11. Track 3.1 落地总结（2026-05-24 PM）

### 11.1 Homeworld 判定 —— 方案 A 确定

用户提示「state 里看不到么」 → 答案是**之前没有，但很容易加**。
落地：

* `env/state.py` 加 `home_planet_idx: int32[NUM_PLAYERS]`
  + ckpt 只存 params/opt_state，schema 改动**不影响 resume**
* `env/init.py` reset 时：
  + Pre-permutation 时 home 在 slot 0 (p0) 和 slot 3 (p1)
  + Shuffle 之后用 `inv_perm = jnp.argsort(perm)` 推出 home 的新 slot
* 3 处手写 EnvState（parity_check / kaggle_bridge / test_action_inverse）补 0 占位
  + 这些都是 parity / test 用，shaping 不在路径上

**判定精确度**：home 被攻占后 `home_planet_idx` 仍指向原 home slot，
但 owner 变成对方 → `keep_home_reward` 自动归 0（设计正确：
丢失 home = 停止 reward 流，正确的梯度方向）。

### 11.2 Fleet_size_reward 触发时机 —— 用「实际发射」

按用户「长期 self-play 更强」原则：

* 在 `dynamics.launch_fleets_with_info`（新加的 API）返回 `(valid_mask[K], ships[K])`
* `env.py:step` 拿 p0 的 valid+ships → 喂给 `rewards.fleet_size_reward`
* 设计：`SHAPING_FLEET_SIZE * (clip(ships/20, 0, 1) - 0.2) * valid`
  + 1 ship: -0.15 * coef（惩罚）
  + 4 ships: 0（保本）
  + 20+ ships: +0.8 * coef（最大奖励）
  + invalid emit: 0（防作弊）

**长期 self-play 适配性**：
模型如果发垃圾 invalid 动作只能拿 0，
而想要赚 reward 必须真发出去 → 强制学「reserved-aware + valid src/dst」。

### 11.3 单元测试 26/26 通过

`env/test_rewards.py` 在 Day-3 的 15 个 case 基础上新增 11 个：

* SHAPING_KEEP_HOME / SHAPING_FLEET_SIZE 默认 = 0
* keep_home (coef=0/1, owned/lost)
* fleet_size (coef=0/1, 10+5 ships, no valid, 1-ship 惩罚)

### 11.4 Smoke train 3 updates 行为正常

`ORBITWARS_SHAPING_KEEP_HOME=0.003 SHAPING_FLEET_SIZE=0.002`：

| upd | spf | z0 | garr | ev |
|---|---|---|---|---|
| 0 | 2.7 | 0.03 | 7.0  | +0.15 |
| 1 | 5.5 | 0.03 | 14.4 | +0.19 |
| 2 | 6.5 | 0.04 | 16.2 | +0.24 |

3 步内 spf 从 2.7 → 6.5，garr 从 7 → 16.2，**shaping 在正确方向上引导**。
完整训练时 spf 应能 stabilize 到 10-15，garr 30-80（看 v20 是 212，
新模型不会一上来就那么高，但越来越大才是健康轨迹）。

### 11.5 高手 replay 数据分析（2026-05-24 PM）

用户提供了 `top10_episodes_2026-05-04/`，共 2633 个 JSON replay。
出于速度，写了 `scripts/analyze_expert_replay.py` 解析两个高分 episode：

* **Episode 75873267**：4P 全场打满 500 turn，sum_score=6385（榜单第一）
* **Episode 75862353**：4P 274 turn 提前结束，winner 用绝对优势横扫

#### 11.5.1 关键数据对比

| 指标 | v7 vs v7 | v20 vs v20 | **高手 winner** |
|---|---|---|---|
| spf_mean (full) | 1.6 | 70 | **40-80** |
| spf_max (decisive strike) | ~10 | ~200 | **575**（!） |
| z0_rate (full game) | 0.03 | 0.38 | **0.46-0.79** |
| mean_garrison | 9 | 212 | **407-1048** |
| mean_planets | n/a | n/a | **8-13** |
| prod_share | n/a | n/a | **0.26-0.55** |

#### 11.5.2 五个颠覆性发现

1. **z0 不区分胜负**：赢家 p0 (75873267) z0=0.52，输家 p3 z0=0.50。
   **关键不是「发不发」而是「发对方向」**。
   → 我们之前以为 v20 的 38% z0 是问题，**其实不是**

2. **production_share 是赢家最强特征**（我们目前 reward 完全没建模）
   - Ep 75862353 winner prod_share = 0.55（4P 基线 = 0.25，超 2 倍）
   - Ep 75873267 winner prod_share = 0.26（base + 4%）
   - 输家 prod_share 多数 < 0.15

3. **planet_count 是强单调信号**（比 garrison 更稳定）
   - Ep 75862353 winner: 13 planets vs 输家 1.6-3.3
   - garrison 受 stockpile/release 周期影响，noisy；planet_count 单调累积

4. **决战 ships 数量级 575**（!）
   - Ep 75873267 winner max launch = **575**，p99 = 227
   - 我们 v7 max launch ≈ 10，**两个数量级差距**
   - **fleet_size_reward NORM=20 太小**——20 ships 就饱和，
     模型学到 20 就停，永远学不到 575 的决战气魄

5. **stockpile-then-release 周期模式**（Ep 75862353 winner）：
   garrison decile curve = `10 → 118 → 222 → 455 → 208 → 402 → 308 → 860 → 1667 → 3701 → 5519`
   - **囤 → 放 → 囤 → 放** 反复（不是单纯一直囤）
   - keep_home_reward 是「持续囤」信号，**奖励错了行为**
   - 应该奖励**领土增长**（planet/prod share）而非绝对 garrison

#### 11.5.3 v9 reward 设计修正

基于 11.5.2 的发现，**修正 Track 3.1 的 shaping family**：

| reward | 公式 | 系数 | 状态 |
|---|---|---|---|
| ~~keep_home~~ | tanh(log1p(home_ships)/8) | 0 | **取消**：奖励错行为（囤而不放） |
| **prod_share_reward** | (my_prod_owned / total_prod) - 1/N | 0.005 | **新增**：占领高产行星 |
| **planet_share_reward** | my_planets / total_planets | 0.003 | **新增**：领土增长 |
| **fleet_size_reward** v2 | log1p(ships) - log1p(threshold) | 0.001 | **改公式**：log scale，不饱和 |
| SHAPING_SCALE | terminal-only | 0 | 不变 |

**为什么这个组合更对**：

* `prod_share - 1/N` 自然零和：N=2 时 baseline=0.5，
  超过对方就 +，被超就 −。**自动产生「赢家持续涨、输家持续跌」的梯度**
* `planet_share` 是单调累积，对应高手「占地盘」的核心策略
* fleet_size 改 log scale 后，从 1 ship 到 500 ships 的 reward 是连续上升的
  （不再 20 ships 饱和），允许模型学到「决战时机 100-500 ships」

**Track 3.1 的两个 reward 现状**：
* keep_home：暂时保留代码（已通过测试），但 v9 默认系数 = 0
* fleet_size：保留接口，但要把 NORM 改成 log-based 公式

#### 11.5.4 下一步

1. 重写 fleet_size_reward 为 log-scale（不动接口，只改公式）
2. 新增 prod_share_reward + planet_share_reward 到 env/rewards.py
3. 单元测试覆盖三个新 reward
4. smoke train 验证 metric 走向（spf 应该能涨到 50+ 而不是 10-15）
5. v9a/b/c/d ablation 设计：四个配置在新 reward 上扫不同系数

---

## 12. 高手 replay 5-episode 深度分析（2026-05-24 PM）

用户提供 `top10_episodes_2026-05-04/`（2633 个 JSON）。
看了 5 个最高 sum_score：75873267 / 75862353 / 75864613 / 75853309 / 75864896。

### 12.1 五场总览

| Ep | turns | winner | spf_mean | spf_max | mean_garr | mean_plnt | **prod_share** | z0 | decile_garr 末 |
|---|---|---|---|---|---|---|---|---|---|
| 75873267 | 500 | p0 | 82  | 575  | 407  | 8.1 | **0.26** | 0.52 | 1384 |
| 75862353 | 274 | p2 | 38  | 139  | 1048 | 13.2 | **0.55** | 0.46 | 5519 |
| 75864613 | 255 | p0 | 125 | 506  | 427  | 7.4 | **0.41** | 0.70 | 2103 |
| 75853309 | 408 | p0 | **329** | **2076** | **2827** | 13.6 | **0.55** | 0.74 | **9256** |
| 75864896 | 235 | p3 | 36  | 245  | 1194 | 12.5 | **0.49** | 0.56 | 6061 |

输家共性（5 × 3 = 15 个）：
* mean_garr 全 < 280
* mean_plnt 全 < 7
* **prod_share 全 < 0.18**

### 12.2 prod_share 是 PERFECT 5/5 separator

5 个赢家 prod_share = [0.26, 0.41, 0.49, 0.55, 0.55]，中位数 **0.49**
15 个输家 prod_share <= 0.18
**判别率 100%。这是最强的单一胜负预测指标。**

### 12.3 五个颠覆性发现

1. **z0 不区分胜负**：赢家平均 z0=0.60，输家平均 z0=0.78。差距小。
   v7（z0=0.03）"乱发"是问题，但 v20 (z0=0.38) **接近高手**，不是问题。

2. **决战 ships 数量级 575-2076**！比 v7 (max ≈ 10) 高两个数量级。
   75853309 winner p99=2059, max=2076。
   → fleet_size_reward NORM=20 **彻底错了**，必须 log scale 不饱和。

3. **「囤足够 → 一波带走」压倒性**（看 75853309 winner trace）：
   ```
   turn  50: 0 launch, garr  137
   turn 100: 0 launch, garr  580
   turn 150: 0 launch, garr 1315
   turn 200: 0 launch, garr 2847
   turn 250: 0 launch, garr 4397   ← 一直囤
   turn 300: 2 launch (705+24)     ← 第一次出手
   turn 400: 0 launch, garr 8917   ← 比赛结束前还在囤
   ```
   episode_steps=80 训练根本看不到这种策略。

4. **「占地」先于「囤兵」**（75853309 winner: turn 100 planet=9 garr=580；
   turn 200 planet=13 garr=2847；turn 350 planet=23 garr=6129）
   → planet_share 是比 prod_share 更早期的密集 signal。

5. **stockpile-then-release 周期模式**（75862353 winner decile）：
   `10 → 222 → 455 → 208 → 402 → 308 → 860 → 1667 → 3701 → 5519`
   赢家是「囤 → 放 → 囤 → 放」节奏，不是单纯一直囤。
   → keep_home_reward（"持续囤"）奖励错了行为，**v9 默认 0**。

### 12.4 v2 shaping family（落地）

修正 §11 的 v1 family。三个新 reward 已实现 + 测试通过：

| reward | 公式 | env var | 默认 | v9 推荐 |
|---|---|---|---|---|
| **prod_share** | `α * (my_prod/total_prod - 1/N)` | `SHAPING_PROD_SHARE` | 0 | **0.01** |
| **planet_share** | `β * (my_planets/total_planets - 1/N)` | `SHAPING_PLANET_SHARE` | 0 | **0.005** |
| **fleet_size_log** | `γ * (clip(log1p(ships)/log1p(REF),0,1) - FLOOR)` per valid launch | `SHAPING_FLEET_LOG` | 0 | **0.002** |
| ~~keep_home~~ | （v1 保留代码，默认 0） | `SHAPING_KEEP_HOME` | 0 | 0 |
| ~~fleet_size~~ v1 | （v1 保留代码，默认 0） | `SHAPING_FLEET_SIZE` | 0 | 0 |

`prod_share` 自然零和：N=2 时 baseline=0.5，
赢方持续 +reward，输方持续 -reward，无需额外 SHAPING_SCALE。

`fleet_size_log` 不饱和测试（REF=500, FLOOR=0.3，γ=1）：
* 1 ship = -0.19（轻惩罚）
* 100 ships = +0.44
* 500 ships = +0.70
* 2000 ships = +0.70（clipped，不过度奖励）

### 12.5 单元测试 35 / 35 通过

`env/test_rewards.py` 在 v1 family 26 个 case 基础上新增 9 个：
* SHAPING_PROD_SHARE / SHAPING_PLANET_SHARE / SHAPING_FLEET_LOG 默认 0
* prod_share / planet_share 零和（p0=+ ↔ p1=−）
* fleet_size_log: 1 ship / 500 ships / 2000 ships clipped / no valid

### 12.6 Smoke train 验证

`PROD_SHARE=0.01 PLANET_SHARE=0.005 FLEET_LOG=0.002` × 3 updates:

| upd | spf | z0 | garr | ev |
|---|---|---|---|---|
| 0 | 2.7 | 0.03 | 7.0  | -0.38 |
| 1 | 5.5 | 0.03 | 14.3 | +0.25 |
| 2 | 6.3 | 0.04 | 15.8 | +0.28 |

3 步内 spf 增长正常，env 没崩。准备进入 Track 3.2（写 v9a/b/c/d configs）。

### 12.7 v9 ablation 设计

四个配置扫 prod/planet/fleet_log 的不同组合：

| config | prod | planet | fleet_log | 目的 |
|---|---|---|---|---|
| v9a (control) | 0 | 0 | 0 | 纯 sparse +1/-1（baseline） |
| **v9b (prod-only)** | **0.01** | 0 | 0 | prod_share 单 reward 是否足够 |
| v9c (full v2) | 0.01 | 0.005 | 0.002 | 全 v2 family |
| v9d (no fleet_log) | 0.01 | 0.005 | 0 | 验证 fleet_log 是否必要 |

episode_steps 改 350（不是 80），rollout_length=256，gamma 0.97 → 0.99。

**关键预期**：v9b 应该在 200 updates 内出现 sign of life
（prod_share 自带零和 → value head 可以稳定学）；
v9c 应该有更高的 spf（fleet_log 起作用）。

---

## 13. v9 启动操作手册（5090）

### 13.1 已交付的代码 / 产物

| 文件 | 作用 |
|---|---|
| `env/rewards.py` | v1 (keep_home, fleet_size) + v2 (prod_share, planet_share, fleet_size_log) 全部接入 |
| `env/env.py` | step() 累加 5 个 shaping term（除 terminal 步外） |
| `env/state.py` | 加 `home_planet_idx[2]` 字段（不影响 ckpt schema） |
| `env/init.py` | reset 用 inv_perm 跟踪 home idx |
| `env/dynamics.py` | 新 API `launch_fleets_with_info` 返回 valid+ships |
| `ppo/rollout.py` | Rollout 加 `planet_prod_raw` / `planet_owner_raw` |
| `ppo/update.py` | 新 metric: `prod_share` / `planet_share` / `fleet_log_score` |
| `ppo/runner.py` | banner 显示 5 个 shaping coef，log 加 `pS / ptS / fLog` |
| `configs/multi_action_v9{a,b,c,d}.yaml` | 4 个 ablation 配置 |
| `scripts/run_v9_ablation.sh` | 启动 4 个或选定子集，并发到后台 |
| `scripts/monitor_train.py` | tail 多个 log，EV/clip/kl 阈值告警 |
| `scripts/analyze_expert_replay.py` | 解析 kaggle JSON replay（5 episode 已分析） |
| `env/test_rewards.py` | 35 个 case 全部通过（v1 26 + v2 9） |

### 13.2 关键超参的来源

- **episode_steps = 350**：高手 75853309 winner 在 turn 300 才打第一波，
  episode_steps=80 完全看不到这种 stockpile-then-release 模式
- **rollout_length = 256**：350-turn ep 平均跨越 0.73 episode/rollout；
  比 v8 的 rollout=128（1.6 ep/rollout）显著改善 GAE 切断噪声
- **gamma = 0.99**：horizon 从 80 → 350，effective discount horizon
  1/(1-γ) 要从 33 提到 100；用 0.99 而非 0.999 是为了避免 advantage 爆炸
- **lr_decay_steps = 100k**：4000 updates × 4 epochs × 4 mbs = 64k 优化步
- **num_updates = 4000**：预算约 4000 × 256 × 128 = ~130M env steps；
  与 v8 (10000 × 128 × 128 = 164M) 同量级但 episode 更长 → 实际 transitions 略多

### 13.3 4 个 shaping 系数为什么是这个数

`prod_share = 0.01`：
* 上界 (1-1/N) = 0.5 → 单步 reward 上界 ±0.005
* 350 turn 累积 ≤ ±1.75 → 与 terminal ±1 同数量级 ✓

`planet_share = 0.005`：
* 同样 ±0.5 上界 → 单步 ±0.0025
* 累积 ≤ ±0.875；意图低于 prod_share 因为部分信号重叠

`fleet_log = 0.002`：
* 单 launch 最大 1.0 - 0.3 = 0.7 → 单 launch ±0.0014
* K=8 launches/turn × 350 turns × 0.0014 = ±3.9 上界
* 实际平均 spf < 50 → 实际 fleet_log ≈ 0.3 → 累积 ~1.0 ✓

### 13.4 启动顺序（5090）

```bash
# 1. 启动 4 个 ablation（同机器并行）
cd /workspace/OrbitWarRL  # 或对应路径
PYTHON=/opt/anaconda3/bin/python bash scripts/run_v9_ablation.sh

# 2. 等 30 秒后 sanity check banner
sleep 30
grep -h '\[reward\]' logs/multi_action_v9*.log

# 3. 拉起 monitor 后台跑
nohup python -m orbit_wars_rl.scripts.monitor_train \
  --interval 60 \
  logs/multi_action_v9*.log \
  > logs/monitor.log 2>&1 &

# 4. 看 monitor 实时状态
tail -f logs/monitor.log
```

### 13.5 Sign of life 阈值（按 update 段）

| 节点 | upd 50 | upd 500 | upd 1500 | upd 3000 |
|---|---|---|---|---|
| ev (全 4 config) | > 0.4 | > 0.7 | > 0.8 | > 0.85 |
| clip_frac | < 0.20 | < 0.20 | < 0.20 | < 0.15 |
| approx_kl | < 0.05 | < 0.05 | < 0.03 | < 0.02 |
| spf (v9c) | > 5 | > 15 | > 30 | > 50 |
| pS (v9b/c/d) | > 0.50 | > 0.55 | > 0.60 | > 0.65 |
| ptS (v9c/d) | > 0.50 | > 0.55 | > 0.60 | > 0.65 |

### 13.6 Kill 准则（任何 config 触发即停 + 调查）

* ev < 0.3 持续 100 updates → obs/arch 问题（DAY3 §6 v6p3-bug 重发）
* clip_frac > 0.35 持续 50 updates → lr 减半重启
* approx_kl > 0.10 持续 20 updates → 单 update 太大，clip_eps 减到 0.10
* loss 出现 |x| > 5.0 → shaping 系数过大，减半重启
* spf < 3 after upd 500（v9c）→ fleet_log 没起作用，调高到 0.005

### 13.7 决策（v9 跑完后）

预期 4000 updates 之后：
* 主指标：H2H vs v8 last-ckpt（20 局 seed 0..19）的赢率
* 次指标：mean_emits_per_turn, spf, pS, ptS 走向

决策规则：
* v9c 显著 > v9a 且 > v9d → v9c 上 production（fleet_log 必要）
* v9c ≈ v9d > v9a → v9d 上 production（fleet_log 多余，简洁优先）
* v9b ≈ v9c → v9b 上 production（prod_share 单独足够）
* 全部 ≈ v9a → 都失败；需要回到 §6 重新看 obs/arch

### 13.8 已确认无回归

* `python -m orbit_wars_rl.env.test_rewards` → 35/35 PASS
* `python -m orbit_wars_rl.bc.test_action_inverse` → 1/1 PASS
* local smoke train 5 updates with full v2 shaping → spf 2.7 → 7.0, pS 0.06 → 0.17
* monitor_train --once 解析 log → 字段全提取，warning/alert 触发正常

### 13.9 OOM 调试与修复（2026-05-24 PM）

**症状**：5090（32GB VRAM）跑 v9a：
```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying
to allocate 28.11GiB. [executable_name='jit_train_step']
```

**根因**：单个 PPO update 的 activation peak 远超预算。计算：
* rollout flatten → `N = T*B = 256 × 128 = 32768`
* minibatch size = `N / num_minibatches = 32768 / 4 = 8192`
* Transformer attention activation: `[8192, n_heads=4, tokens=169, tokens=169]`
  = 8192 × 4 × 169² × 4 bytes = **3.75 GB per layer per direction**
* 2 layers × forward + backward + K=8 autoregressive scan
  → activation peak ≈ **28 GB**（匹配报错）

**修复**：把 `num_minibatches` 从 4 改为 16，单 minibatch 从 8192 → 2048。
PPO 数学上等价（`update_epochs * num_minibatches` 决定优化步数，
单 mb 大小只影响梯度噪声），但 activation peak 缩小 4x → **~7GB**。

**4 个 v9 config 同步改动**：
* `num_minibatches: 4` → `num_minibatches: 16`
* launch script `JAX_FRAC`：
  - `N_CONCURRENT=1`: 0.50 → **0.85**（27GB cap，单 run 全力）
  - `N_CONCURRENT=2`: 0.40 → **0.42**（13.4GB/job，2 jobs ~ 26.8GB）
  - `N_CONCURRENT=4`: 0.20 → **0.21**（6.7GB/job，要求 minibatch fix）

**重启**：
```bash
pkill -f "multi_action_v9"
sleep 5
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
bash scripts/run_v9_ablation.sh v9a v9b
```
