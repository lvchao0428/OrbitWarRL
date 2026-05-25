# DAY5 规划 — 激进快速迭代 + 架构主线

> 写于 2026-05-25。Day4 v9 ablation 已给出足够证据：**macro reward shaping 在 self-play 内有效，但对 v20 不够**。
> Day5 目标不是「再训一个 4000-update 的 v9x」，而是 **2–4 小时筛 hypothesis → 每晚 1 条 overnight 主线**，
> 同时并行推进 **obs/架构 + 局部 reward + opponent curriculum**。

---

## 0. TL;DR

| 问题 | Day4 答案 | Day5 做法 |
|---|---|---|
| prod/planet/fleet shaping 有用吗？ | 有。v9b vs v9a **17/20** | **冻结为 v9c base**，不再扫 shaping 系数 |
| 能打赢 v20 吗？ | **不能**。0/20；replay spf 21→4.8 | 换主战场：**架构 + 威胁感知 + curriculum** |
| 迭代太慢？ | 4000 upd ≈ 12h/条 | **500 upd gate（2–3h）** + replay 3 局 vs v20 |
| multi-launch / 换家 / 防守留兵？ | 未建模或 obs 看不见 | **Track A 架构** 为主，Track B 小 reward 为辅 |
| 4P Kaggle？ | 全程 2P 训练 | **Track C 独立立项**，不阻塞 2P 打 v20 |

**Day5 杀手指标（vs v20，replay 3 局 first-80 窗口）**：

- `mean_ships_per_fleet` **> 10**（v9b 实测 4.76）
- `mean_garrison_my` **> 60**（v9b 实测 31.85）
- `fleet_flip_rate` **> 6%**（v9b 实测 3.88%）
- emit≥2 的 turn 占比 **> 5%**（v9b ≈ 0%）

**WR 不是 2–4h 的主 gate**；破 **1/20 gauntlet** 留给 overnight winner。

---

## 1. Day4 证据摘要（决策依据，不重复训练）

### 1.1 已证实

- **训练机制健康**：ev 0.97+、clip 收敛、无 NaN/OOM（num_minibatches=32 后稳定）
- **shaping sign of life**：v9b gauntlet vs v9a **17/20**；prod_share 单独足够（v9b ≈ v9d 中期 macro）
- **fleet_log 推 spf**：v9c @ upd1158 spf **21.5** > v9b final 21.2
- **episode_steps=350** 必要：v4 自对弈 500 步全 tie；v9 能早终、能打仗

### 1.2 已证伪 / 触顶

- **「训练 log 外推 vs v20」**：v9b 训练 spf=21.2，replay vs v20 **spf=4.76**
- **「再加 global macro reward 就能赢」**：v9 全家桶 ≈ v9c，预期仍难破 0/20
- **keep_home / 纯 garrison shaping**：DAY4 高手 replay 已证方向错（囤而不放）
- **h2h_local 单色 eval**：player 位偏差可达 25/75；**正式 eval 只用 gauntlet 双色**

### 1.3 v9b vs v20 replay 暴露的根因（Day5 要打这些）

| 现象 | 数据 | 根因类别 |
|---|---|---|
| 前 80 turn 就被压制 | garr 32 vs v20 132 | early game 失守 |
| 舰队绝对规模小 | spf 4.76 vs v20 15.88 | garrison 基数低 + bin7×小garr |
| 几乎不 multi-launch | 95% turn 只 emit 1 | emit 头 + 无 multi 信号 |
| 舰队大多白干 | flip_rate 3.9% | 目标选择 / 时机 |
| 中后期停手 | full-game z0 41% | losing spiral |
| 无换家/转移 | （未量化） | dst 偏攻、缺 threat obs |

---

## 2. Day5 迭代哲学（激进版）

### 2.1 两条速度档

```
┌─────────────────────────────────────────────────────────┐
│  FAST LANE（2–4h）                                       │
│  500 updates · 1 delta · export@500 · replay×3 vs v20     │
│  并行 2–3 条（5090 上 N_CONCURRENT=2）                     │
│  输出：行为 metric gate → PROMOTE / KILL                   │
└─────────────────────────────────────────────────────────┘
                          │ PROMOTE（≤1 条/天）
                          ▼
┌─────────────────────────────────────────────────────────┐
│  OVERNIGHT LANE（8–12h）                                 │
│  4000 updates · 从 FAST 胜出 config · gauntlet 20×2色    │
│  输出：vs v9c / vs v20 WR · 是否进 production 10k        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 激进原则

1. **默认 500 updates，不是 4000** — 4000 只给 overnight winner
2. **并行 smoke，不是 serial ablation** — 5090 上 2 路 FAST 同时跑
3. **replay vs v20（3 局）> gauntlet WR** — 3 局 ~15min，metric 比 0/20 敏感
4. **one delta at a time** — 架构 / obs / reward / curriculum 不混在一个 run
5. **v20 进训练分布，不是 BC** — frozen pool 或 periodic eval 驱动 curriculum
6. **先修「看得见威胁」，再加 reward** — 换家/留兵是 obs 问题，不是第三个 tanh bonus

### 2.3 明确不做（Day5 排除）

- ❌ 再扫 prod/planet/fleet 系数（v9c 已定）
- ❌ 4000-update 四路并行 ablation（Day4 已做完）
- ❌ BC / 把 v20 权重注入 policy
- ❌ 在 2P 训练里硬塞 4P reward（Track C 单列）
- ❌ 用 h2h_local 单色数字做决策

---

## 3. 战术缺口 → 工程映射

用户观察的 top 选手行为，按 **obs → 架构 → reward → curriculum** 分解：

### 3.1 Multi-launch（同一 turn 多路出击）

| 层 | 现状 | Day5 方向 |
|---|---|---|
| 架构 | K=8 emit slots 已有，策略学「发 1 次」 | 保持；可选 **emit 条件头** 单独加强 |
| Reward | 无 | **v3-R1**：valid launch≥2 时小 bonus（per-turn cap） |
| Eval | replay `emit_count≥2` 占比 | FAST gate **> 5%** |

### 3.2 换家 / 兵力转移（威胁下 friendly dst）

| 层 | 现状 | Day5 方向 |
|---|---|---|
| Obs | 缺「incoming threat per planet」 | **v10-A1**：每星 `nearest_enemy_fleet_ships`, `eta_turns`, `threat_ratio` |
| 架构 | dst head 全局 argmax，偏攻 | 可选 **threat-gated dst bias**（只架构，不改规则） |
| Reward | 无 | 暂不主做；**先看 A1 是否抬 garr/降 early loss** |

### 3.3 留守军 vs 敌方舰队规模

| 层 | 现状 | Day5 方向 |
|---|---|---|
| Obs | fleet_feats 可能有，需 audit 是否含 **到每星的威胁聚合** | A1 覆盖 |
| Reward | 无 local 信号 | **v3-R2**（次要）：`min(garr_p, α×threat_p) − β`  per owned planet，小系数 |
| Eval | replay p95_garrison、early garr | gate **> 60** |

### 3.4 占近处弱星 / 扩张节奏

| 层 | 现状 | Day5 方向 |
|---|---|---|
| Reward | prod_share + planet_share（v9c 已有） | **不再加强**；用 **capture/flip** 作 eval |
| Eval | fleet_flip_rate | gate **> 6%** |

### 3.5 囤→放周期（stockpile-then-release）

| 层 | 现状 | Day5 方向 |
|---|---|---|
| Reward | prod/planet 奖「持续占着」→ z0≈0 | **v3-R3**（实验性）：fleet_log 只在 `garr > θ` 时生效；或 late-episode shaping 加权 |
| Eval | z0 应升到 **15–40%**（不是 0%，也不是 v7 的乱发） | 与 spf 联看 |

### 3.6 4P Kaggle

| 层 | 现状 | Day5 方向 |
|---|---|---|
| 训练 | 2P only | **Track C**：4P env 或 2P+2 bot；**Week 2+**，不挡 Track A/B |
| 分析 | 高手 replay 已有 | 继续 `analyze_expert_replay` 抽 4P **threat/transfer** 统计 |

---

## 4. 三条 Track 与优先级

### Track A — 架构 / Obs（**Day5 主线，最激进**）

**假设**：v9 触顶 because policy 对 **绝对坐标 + 无威胁特征** 过拟合 self-play；player 位不对称是症状。

| ID | Delta | FAST 500 upd | 2–4h 看什么 | 风险 |
|---|---|---|---|---|
| **A1** | Planet threat features（敌舰队 ships/ETA 聚合到每星） | base=v9c | vs v20 replay：garr↑ spf↑ | obs 维度变，需 parity |
| **A2** | Relative / egocentric coords（去绝对 x,y 或加 relative fleet/planet） | base=v9c | 自对弈 gauntlet 双色 **WR→50/50** | 大改 encode |
| **A3** | Sun mask / 遮挡感知（overview 机制） | base=v9c | flip_rate↑ | 实现成本 |
| **A4** | TypedInputProjection / MLP FireHead（top1 §92 列表） | base=v9c | ev 不降 + spf↑ | 容量↑ |

**推荐顺序**：**A1 → A2**（A1 更贴「换家/留兵」，A2 修 asymmetry）；A3/A4 看 A1/A2 结果。

**Overnight 条件**：A* 在 FAST 中 **至少 2/4 replay metric 过 gate**，且 ev@500 > 0.7。

---

### Track B — Reward v3（**小步、500 upd 并行筛**）

base = v9c shaping 系数不变，**只加一个 term**。

| ID | Term | 目的 | Kill if |
|---|---|---|---|
| **B1** | multi_emit bonus | multi-launch | emit≥2 仍 < 2% @ upd500 |
| **B2** | capture_success | flip_rate | flip_rate 无 +2pp vs v9c@500 |
| **B3** | fleet_log gated（garr>θ 才奖大 launch） | 囤放周期 | z0 仍 < 5% 且 spf 无升 |
| **B4** | local_defense（可选，依赖 A1 threat） | 留兵 | 无 A1 时不跑 |

**B Track 不与 A 同 run**；A1 promote 后再跑 B4。

---

### Track C — Opponent Curriculum（**激进，与 A 并行**）

**假设**：self-play 废涡（v7 vs v7 flip 1.14%）because 对手太弱；**必须定期打 v20**。

| ID | 做法 | FAST 验证 |
|---|---|---|
| **C1** | frozen_ratio 提高 + pool 加入 **v20-bot 替身**（规则 bot 非 BC） | 500 upd 后 vs v20 replay 是否优于 v9c@500 |
| **C2** | 每 100 upd eval vs v20，**WR<0.05 则 lr×0.5 不 kill**（仅 log） | 看曲线是否「对 v20 更抗揍」 |
| **C3** | League：保留「对 v20 撑过 150 step」的 snapshot | overnight 指标 |

**注意**：v20 作 **对手** ≠ BC；不反向传播 v20 动作。

**C1 可与 A1 并行**（一个改 obs，一个改 selfplay config，不同文件）。

---

## 5. FAST LANE 标准操作（2–4 小时）

### 5.1 配置模板（概念层，实现时写 yaml）

| 参数 | FAST | OVERNIGHT |
|---|---|---|
| num_updates | **500** | 4000 |
| eval_every | 50 | 50 |
| ckpt_every | 100 | 100 |
| base shaping | v9c 全套 | 同 FAST winner |
| 架构 / reward / curriculum | **+1 delta** | 同 FAST |

### 5.2 时间线（单条 FAST，5090 单 job）

| 时刻 | 动作 |
|---|---|
| T+0 | 启动 train，grep banner 确认 delta |
| T+30min | upd~50：ev>0.4, clip<0.30 |
| T+1.5h | upd~200：spf/pS vs v9c@200 |
| T+2.5h | upd~500：export ckpt_000499 |
| T+2.5–3h | replay×3 vs v20 + replay×3 v9c vs v9c（可选） |
| T+3h | **PROMOTE / KILL / RETRY** 决策 |

### 5.3 PROMOTE / KILL 规则

**PROMOTE → overnight（需满足）**

- ev@500 **> 0.70**
- clip@500 **< 0.20**
- vs v20 replay（3 局 first-80）**至少 2/4**：
  - spf > 10
  - garr > 60
  - flip_rate > 6%
  - emit≥2 占比 > 5%

**KILL（立即停，不 export）**

- ev@200 **< 0.30** 持续
- clip@200 **> 0.40** 持续 50 upd
- vs v20 replay：**4/4 metric 劣于 v9c@500 基线**（同 seed）

**RETRY（同 hypothesis，调参一次）**

- metric 方向对但未过线 → 系数 ±50% 或 lr×0.5，**只 retry 1 次**

### 5.4 并行策略（5090）

```
Day5 典型一天：
  上午：启动 FAST-A1 + FAST-C1（N_CONCURRENT=2）
  下午：读 gate → 杀 1 留 1 → 启动 FAST-A2 或 FAST-B1
  晚上：1 条 PROMOTE → overnight 4000
  次日：gauntlet + replay×5 vs v20
```

**吞吐目标**：**3–4 个 hypothesis / 天**（FAST），**1 个 overnight / 天**。

---

## 6. OVERNIGHT 与 gauntlet

### 6.1 仅对 FAST PROMOTE 的 config 跑 4000

- export `@3999`（`--skip-parity` 前跑 `test_parity --num-states 128`）
- gauntlet **双色**：
  - vs `v9c_u3999`（10×2 局）
  - vs `v20`（10×2 局）
- replay×5 vs v20 → JSON 存档

### 6.2 Overnight 成功标准（比 Day4 更贴近 v20）

| 级别 | 条件 |
|---|---|
| **Sign of life** | vs v20 gauntlet **≥ 1/20** |
| **Meaningful** | vs v20 replay spf **> 12**, garr **> 80** |
| **Production** | vs v20 **≥ 5/20** 或 vs v9c **≥ 18/20** → 上 10k |

---

## 7. Day5 执行顺序（建议）

### Phase 0 — 收尾 v9（≤1 天，不阻塞 FAST）

1. v9c/v9d 跑完 4000（已在跑）
2. export + gauntlet + replay v9c vs v20
3. **书面结论**：v9c vs v9b vs v9d 选一个 **frozen base tag**（预期 v9c）
4. **不再**开 v9e/v9f 长训

### Phase 1 — FAST 架构（Day5 主战场，Day 1 起）

1. **A1** threat features + v9c base，500 upd
2. 并行 **C1** curriculum v20 替身，500 upd
3. 读 gate → overnight 最多 1 条

### Phase 2 — FAST reward v3（Day 5–7，与 Phase 1 交错）

1. **B1** multi_emit（若 A1 未解决 emit≥2）
2. **B2** capture
3. **B3** gated fleet_log（若 z0 仍≈0）

### Phase 3 — 坐标 / 容量（Day 6–10）

1. **A2** relative coords（若 A1 promote 但 asymmetry 仍存）
2. **A4** MLP FireHead（若 ev 健康但 spf 仍顶）

### Phase 4 — 4P Track C（Week 2+）

1. 高手 replay 统计：4P 下 transfer / threat 频率
2. env 4P 可行性评估（不与 2P FAST 混）

---

## 8. 风险登记

| 风险 | 概率 | 缓解 |
|---|---|---|
| FAST 500 upd 噪声大，误杀好 hypothesis | 中 | 同 config retry 1 次；replay 3 局 + 固定 seed |
| A1/A2 改动破坏 ckpt resume | 高 | **from scratch 500**，不 resume v9c weights（one-delta） |
| C1 v20 对手导致 ev 崩溃 | 中 | v20 比例 20–30%，不是 100% |
| 并行 FAST 抢 GPU | 低 | N_CONCURRENT=2，JAX_FRAC=0.42 |
| 2P 练成只能打 2P | 中 | Track C 单列；Kaggle 提交前 4P smoke |

---

## 9. 文档与工具（Day5 待补，仍不写代码）

| 产物 | 用途 |
|---|---|
| `docs/FAST_ITER_RUNBOOK.md` | 500 upd 命令、gate 表、kill 规则 |
| `docs/DAY5_PROGRESS.zh.md` | 每日 FAST 结果 log |
| `H2H_EVAL_RUNBOOK.md` 增补 | gauntlet-only、parity skip 流程、replay gate |
| monitor 扩展（可选） | 打印 vs-v20 replay 阈值告警 |

---

## 10. 成功定义（Day5 末）

**最低成功**：

- 完成 ≥6 个 FAST hypothesis（A/B/C 合计）
- 至少 1 条 overnight
- vs v20 replay：**spf 从 4.76 提到 >10**（3 局均值）

**目标成功**：

- vs v20 gauntlet **≥ 1/20**
- 自对弈 gauntlet 双色 WR **45–55%**（修 asymmetry）
- 明确 **A1 vs C1** 谁对 v20 更关键 → Day6 10k 方向

**未达成则**：

- 不回到 v9 shaping 扫参
- 进入 **A2/A4 架构 deep sweep** + 考虑 **4P 最小 env**

---

## 附录 A — v9 基线数字（复制粘贴用）

```
v9b gauntlet:  vs v9a 17/20, vs v20 0/20
v9b replay vs v20 (first-80): spf=4.76, garr=31.85, flip=3.88%, z0=0.98%
v9b replay vs v20 (full):     spf=4.81, garr=21.68, z0=41.52%
v9b train final:              spf=21.2, pS=0.41, garr=49.9
v9c @1158:                   spf=21.5, fLog=0.40, pS=0.38
```

## 附录 B — FAST gate 速查

```
PROMOTE if: ev@500>0.7 AND (2 of 4): spf>10, garr>60, flip>6%, emit2+>5%
KILL if:    ev@200<0.3 OR all 4 metrics worse than v9c@500 baseline
OVERNIGHT:  only PROMOTE configs, 4000 upd, then gauntlet
```
