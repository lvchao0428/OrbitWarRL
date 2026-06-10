# Day5 训练动作清单 — 基于 Top-10% Replay 的可落地建议

> 写于 2026-05-25。源数据：[`TOP10_REPLAY_METRICS.zh.md`](TOP10_REPLAY_METRICS.zh.md)（2630 局 4P 高手 replay）。
>
> **本文目标**：把「能让模型更能打 v20」翻译成 **具体的 reward 公式 / obs 增项 / curriculum 配置**，每条都给：
>
> 1. **观测证据**（top10 数据 + v9 训练/eval gap）
> 2. **公式 / 实现位置**（env var、代码文件）
> 3. **系数推算**（保证 ∑shaping ≲ terminal）
> 4. **FAST 500-upd 验证 metric** 与 KILL 规则
> 5. **风险与对应**
>
> **不再做**：扫 v9 已有 prod/planet/fleet_log 系数；keep_home；BC；4P env（移到 Week 2+）。

---

## 0. 一句话总策略

> **v9 的 macro shaping 方向对，但是「水平 reward」而不是「行为 reward」。**
> Day5 不再加更多水平 shaping，而是改成 **delta / event / 局部** 信号，并把 **威胁感知 (obs)** 与 **v20 级对手 (curriculum)** 同时引入。

每条改动都遵循：
* **One delta at a time**（每个 FAST run 只动 1 处）
* **从 frozen base ckpt resume**（除 A2 改坐标）
* **500 upd ≈ 3h 决策**（替代旧的 4000 upd 闭眼跑）

---

## 1. Reward 改动（4 项，按 ROI 排序）

### R1. **prod_share_DELTA**（最高优先，先做）

**为什么**

* Top10 全场 prod_share 增长：赢家 **+0.286**，输家 **−0.020**（2630 局）。
* 当前 `prod_share_reward` 是 **水平** `(share − 0.5)` × 0.01：
  * 早期 share≈0.5 → reward≈0 → 没有「我刚抢了 +0.05 share」的信号
  * 中后期赢者 share=0.6 → 持续 +0.001/step，但**不是因为这一步**，是历史累积，credit assignment 弱
* delta 形式 `Δshare × α` 把 +0.05 ship-flip 一瞬产生 +0.05α reward，**梯度直接打到导致这次 flip 的动作**

**实现位置**

```python
# orbit_wars_rl/env/rewards.py 新增
def prod_share_delta_reward(prev_state, next_state, player):
    share_prev = my_prod_share(prev_state, player)
    share_next = my_prod_share(next_state, player)
    return SHAPING_PROD_SHARE_DELTA * (share_next - share_prev)
```

需要 env.step 传 `prev_state` 给 reward layer；`env/env.py` 已有 prev_state（用于现 shaping），改动很小。

**env var**

```
ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.5
```

**系数推算**

* 单步 |Δshare| 上界 ≈ 1 行星被夺 → ~0.07（按 8 行星 4P，2P 时 ~0.10）
* `α=0.5` → 单步上界 ±0.035；350 turn 累积上界 ±12，**远超 terminal ±1**
* **真实分布** mean |Δshare| ≈ 0.286/350 ≈ 0.0008 → 平均累积 ≈ 0.286 × 0.5 = **±0.14**（不撑爆）
* 退路：若 EV 跌，先降到 `0.2`，再 `0.1`

**FAST 验证**

| metric | 目标 |
|---|---|
| ev@500 | > 0.6（delta reward 噪声大些） |
| prod_share@500（训练 log） | ≥ frozen base@500 + 0.02 |
| spf@500 | ≥ 15（不退化） |
| replay vs v20 garr80 | > 50（基线 v9b 31.9） |

**风险**

* delta 受 noise 大（行星归属抖动） → 用 `(share_next − share_prev)`，不是 `gain only` ReLU（保留对称信号）
* 与现有 `prod_share_reward`（水平）**互斥跑**，不要叠加

---

### R2. **release_bonus（gated launch 奖励）— 替代 fleet_log 升级**

**为什么**

* Top10 赢家 **peak/mean garrison ratio = 4.28**（p95 **6.6**），说明赢家 garr 时常 ≥ 4× 均值，**然后释放**
* decile 曲线非单调（DAY4 §12 + 本报告 §6）：455→208→860 这种「囤一波 → 放一波 → 再囤」
* 当前 `fleet_size_log_reward` 奖**任何时候**的大 launch；正确的应该是「**garr 高时**的大 launch」

**公式**

```
release_bonus = β × Σ_valid_launches [
    clip(log1p(ships)/log1p(REF), 0, 1)
    × tanh(src_garrison_before_launch / RELEASE_THRESH - 1)
    × valid
]
```

`tanh(g/T − 1)` 在 g=T 时 = 0；g=2T 时 ≈ 0.76；g<T 时为负 → 自然 gated。

**env var**

```
ORBITWARS_SHAPING_RELEASE=0.003     # 单 launch 上界 ≈ 1 × 1 × 0.003 = 0.003
ORBITWARS_SHAPING_RELEASE_THRESH=80 # 高手 first-80 garr p50 = 137；half-life 取 80
ORBITWARS_SHAPING_RELEASE_REF=500   # 同 fleet_log_log_ref
```

**系数推算**

* K=8 launches × 350 turn × 0.003 × 0.7（实际 spf 不会顶满）≈ ±5（与现 fleet_log 同量级）
* 与 `SHAPING_FLEET_LOG` **二选一**（FAST B3 同时关 fleet_log）

**FAST 验证**

| metric | 目标 |
|---|---|
| spf@500（训练） | ≥ 30（base 21–27；释放奖励应推高） |
| z0@500（训练） | **0.10–0.30**（base ≈ 0；学囤的间歇） |
| garr@500 | ≥ frozen base（不应崩） |
| peak/mean garr ratio（新加 metric） | ≥ 2.0 |

**风险**

* THRESH 太高 → 模型学不到囤够；THRESH 太低 → 等于 fleet_log
* 退路：用 frozen base 的 garr p75（≈40-60）做 THRESH，RETRY 时调 ±50%

**新增训练 metric**（必加，否则 FAST 没法 gate）：

```python
# ppo/update.py 加 peak_garr_per_episode（每个 env 内 max - mean）/ mean
```

---

### R3. **planet_share — 修 4P baseline bug**（修代码不加 reward）

**为什么**

* 当前 `planet_share_reward` 写死 `n_players = 2.0`
* 计划 Track C 把 v20 替身（4P bot）加进 frozen pool → 训练分布变 N=4 但 reward 仍用 N=2 baseline → **梯度方向错**
* 修法：从 `state` 推（用 `constants.NUM_PLAYERS`），或加 env var

**实现**

```python
# orbit_wars_rl/env/rewards.py:225
n_players = jnp.float32(constants.NUM_PLAYERS)  # 不再硬编码
```

`prod_share_reward` 同样修。

**FAST 验证**：当前 2P 训练数值不变（NUM_PLAYERS=2）；上 C1 4P 时正确。

**风险**：单元测试要补一条 4P case；env constants 改动需 parity check。

---

### R4. **不奖 multi_emit，奖「有效 launch 数」**

**为什么 NOT B1**

* Top10 赢家 first-80 emit2 分布：6.8% =0, 41.3% < 10%, 28.4% [10,20%), 23.4% ≥20%
* **multi-emit 不是必要条件**；强行 +bonus 会推策略到分布**右尾**（≥40% 那 4.3%）
* 真正的问题是 v9b 几乎**只 emit 1**（95.3% turn=1）→ 应该奖「**愿意 emit ≥ 1**」而不是「emit ≥ 2」

**B1' 替代方案：valid_launch_count_log**

```
valid_emit_bonus = γ × Σ_valid_launches log(1 + n_valid_this_turn) / log(3)  per launch
```

* 1 launch → 每 launch ×0.63
* 2 launches → 每 launch ×1.00 → 总 ×2.0
* 4 launches → 每 launch ×1.46 → 总 ×5.8

让模型学「多发 valid 就好」，**但**因为是 per-launch sum，单 launch 5 舰仍优于 2 launch 1 舰各（spf 决定大头还是 fleet_log/release）。

**env var**

```
ORBITWARS_SHAPING_EMIT_LOG=0.001
```

**系数**：单 turn 上界 ≈ 8 × log(9)/log(3) × 0.001 ≈ 0.016；350 turn ≈ ±5.6（与现 shaping 同量级）。

**FAST 验证**

| metric | 目标 |
|---|---|
| mean_emits_per_turn@500 | ≥ 1.5（base ~2.0，不应跌） |
| replay vs v20 emit2 占比 | > 5%（gate） |

**风险**：与 fleet_log 双向叠加 → 模型学「小 launch 多次」走捷径；用 release_bonus 替代 fleet_log 可缓解。

---

## 2. Obs 改动（A1' 重写）

### A1'. **planet threat 三特征**

**为什么**

* `encode.py` 已有 `in_friend_norm`, `in_foe_norm`（粗 ray 归因），但 **policy「看得见数字、不知道紧迫度」**
* 高手赢家 `mean_fleets_in_flight = 11` vs 输家 3.7 → 高手能感知/调度多线威胁

**新增 3 维 planet feat**

```python
# 在 features/encode.py _encode_planets 内
threat_ratio = in_foe / jnp.maximum(planet_ships, 1.0)        # > 1 = 必失守
net_inbound  = (in_foe - in_friend) / jnp.maximum(planet_ships, 1.0)
eta_foe_min  = approx_min_eta(state, planet_xy)  # 用 ship_speed(ships) + 距离
```

**ETA 近似**（避免精确 sim）：

```
eta = ||fleet_xy - planet_xy|| / fleet_speed(fleet_ships)
min over enemy fleets that "could reach" this planet (projection > 0)
```

`_inbound_ships` 已有 ray 归因；同样的 ray 算到一个 fleet 后取距离 + speed 即可。

**Resume**

* obs 维度 19 → **22**（planet feat）
* **必须重训 InputProjection**：要么 from scratch（推荐），要么用 [-pad zeros, +zero init new 3] resume

**推荐**：from scratch 500 upd FAST，因为已有 v9c spf 27 self-play 基线，**A1' from scratch 应该 200 upd 内追上**（前期没 v20 信号，only obs 更丰富）。

**FAST 验证**

| metric | 目标 |
|---|---|
| ev@500 | > 0.6 |
| replay vs v20 garr80 | > 50 |
| replay vs v20 flip_proxy | > 5% |
| 自对弈双色 WR | 40–60%（修 asymmetry 间接证据） |

**风险**

* `_inbound_ships` ray 归因近似可能错配 → 加 unit test：手工 1 fleet → 1 planet 路径，verify 归到对的 idx
* obs 改动破坏 parity → 走 `--skip-parity-on-mismatched-cols`（待加）或重写 parity check

---

## 3. Curriculum 改动（C1 精细版）

### C1. **v20 进 frozen pool**

**为什么**

* v9 self-play flip < 4% → 双方都不打 → 没有 capture 梯度
* Top10 = 强对强，flip 频繁
* BC 已排除 → **v20 作为对手** ≠ BC

**实现**

* `selfplay/pool.py`（如已存在）：扩 `--external-agent submission_v20_0513.py --weight 0.25`
* 训练时 25% rollout 用 v20，75% 用 frozen self-snapshot
* `train.py` 启动 banner 必须打印 `[curriculum] v20_ratio=0.25 weighted`

**FAST 验证**

| metric | 目标 |
|---|---|
| WRf@500 | 0.20–0.40（vs frozen pool 含 v20；不再 0.88 假高） |
| replay vs v20 garr80 | > 50 |
| ev@500 | > 0.5（v20 对手会拉 ev down，给宽松） |

**风险**

* v20 比例过高 → policy collapse（学不动）→ FAST RETRY 用 0.15 / 0.10
* v20 推理速度慢（python submission）→ rollout sps 可能掉 30%，纳入预算

---

## 4. FAST 实验排程（修订）

按 ROI（容易实现 × 期望增益）排：

| 顺序 | ID | 类型 | 实现成本 | base | 期望 FAST 信号 |
|---|---|---|---|---|---|
| 1 | **R1 prod_share_delta** | reward 公式 | 低（rewards.py +1 函数） | resume frozen base | training prod gain↑, spf 稳 |
| 2 ∥ | **C1 v20 pool** | curriculum | 中（pool 加 external） | resume frozen base | replay vs v20 garr80↑ |
| 3 | **R2 release_bonus** | reward 公式 | 中（需 src_garrison from launch） | R1 或 base | spf↑, z0 升到 0.1–0.3 |
| 4 | **A1' threat obs** | encode 加 3 维 | 高（_inbound_ships 重审 + parity） | **from scratch** | flip_proxy↑, garr↑ |
| 5 | **R4 emit_log** | reward 公式 | 低 | R1 后 | emit2↑ |
| 6 | **R3 baseline fix** | bug fix | 极低 | 与任意 C1 同跑 | 不变（2P）/对 4P 正 |
| defer | A2 relative coords / A4 容量 | 架构 | 高 | scratch | 修 asymmetry |
| defer | B2 capture event | reward 事件 | 高 | — | 不依赖 obs |

**典型 3 天计划**

| 天 | FAST (并行 2) | overnight 候选 |
|---|---|---|
| Day1 | R1, C1 | R1 PROMOTE → R1 4000 upd |
| Day2 | R2, R4 | C1 PROMOTE → C1+R1 4000 |
| Day3 | A1' (scratch) | best of Day1–2 |

---

## 5. KILL / 退路矩阵

| 信号 | 立即动作 |
|---|---|
| 任意 R 系列 ev@200 < 0.3 | 系数减半 RETRY 1 次 |
| 任意 R 系列 clip@200 > 0.40 | lr×0.5 RETRY 1 次 |
| R1 prod_gain@500 ≤ base@500 | KILL，回 R2 |
| C1 replay vs v20 garr80 ≤ base | KILL，比例降到 0.15 RETRY |
| A1' from scratch ev@200 < 0.4 | obs 实现可能错，加 unit test |
| 三连 R/C 全 KILL | 进 A2 / A4 架构主线（Day6+） |

---

## 6. 监控指标新增（必须先做）

在 `ppo/update.py` 加 2 个 metric（不加无法 gate）：

```python
# 1. 每个 env 内 peak_garrison / mean_garrison (release 强度)
peak_over_mean = peak_garr_per_env / mean_garr_per_env  # mean over batch

# 2. prod_share delta per step
prod_share_delta = (share_t - share_{t-1})  # rolling mean
```

打印 column 在 runner log 加 `pkR pdΔ`。

---

## 7. 与现有 v9c shaping 的关系

| 项 | 当前 v9c | Day5 计划 |
|---|---|---|
| SHAPING_PROD_SHARE (水平) | 0.01 | **R1 启用时设 0.0**（互斥） |
| SHAPING_PLANET_SHARE | 0.005 | 保留（修 baseline bug） |
| SHAPING_FLEET_LOG | 0.002 | **R2 启用时设 0.0**（互斥） |
| SHAPING_PROD_SHARE_DELTA | — | R1：0.5 |
| SHAPING_RELEASE | — | R2：0.003 |
| SHAPING_EMIT_LOG | — | R4：0.001 |

**互斥规则**：任意 Day5 R-* 启用时，关掉它的「水平 / 不分时机」对应项。banner 必须 echo 全部 shaping coef。

---

## 8. 成功定义（Day5 末）

按 ROI 倒推，**两条以上达到** 即视为 Day5 成功：

1. **R1 (delta) 比 v9c (水平) 在 replay vs v20 garr80 提升 ≥ 30%**（base 31.9 → 42+）
2. **C1 让 vs v20 gauntlet 破 0/40**（≥ 1/40）
3. **R2 release_bonus 让 train spf > 40 且 replay spf > 8**
4. **A1' threat obs 让 flip_proxy > 6%**

**任何一条 = sign of life**，叠加（R1+C1+R2）overnight → Production 10k。

---

## 9. 相关文档

* [`TOP10_REPLAY_METRICS.zh.md`](TOP10_REPLAY_METRICS.zh.md) — 数据源
* [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md) — Track A/B/C 战术（本文是其精细落地版）
* [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md) — Phase 0 现状
* [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md) — 启动命令模板
* `orbit_wars_rl/env/rewards.py` — 所有 reward 函数（R1/R2/R3/R4 的目标修改位置）
