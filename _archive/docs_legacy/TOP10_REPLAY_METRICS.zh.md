# Top-10% 高手 Replay 关键指标 — 2026-05-04

> 数据源：[`top10_episodes_2026-05-04/`](../top10_episodes_2026-05-04/)（Kaggle 当日对局 rating 总和 Top 10%，**2630 局 4P replay JSON**）。
> 聚合脚本：[`orbit_wars_rl/scripts/aggregate_top10_replays.py`](../orbit_wars_rl/scripts/aggregate_top10_replays.py) +
> [`analyze_expert_replay.py`](../orbit_wars_rl/scripts/analyze_expert_replay.py)。
> 原始聚合 JSON：[`logs/top10_aggregate_2026-05-04.json`](../logs/top10_aggregate_2026-05-04.json)。
>
> 写于 2026-05-25。对照 Day5 规划见 [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md)，我方基线见 [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md)。

---

## 0. TL;DR

| 发现 | 数据 | 对 Day5 的含义 |
|---|---|---|
| **prod_share 是最强胜负指标** | 全场赢家 mean **0.48** vs 输家 **0.12**；阈值 **≥0.35** 时赢家 95% / 输家 0.5% | v9c 的 prod shaping 方向对；**不要再扫系数** |
| **高手也大量 z0（不发）** | 赢家全场 z0 mean **51%**，first-80 也有 **62%** | z0 目标应是 **15–50%**，不是 v9 训练的 ~0% |
| **multi-emit 是常态** | 赢家 first-80 emit≥2 **14%**（p50 10%） | v9b vs v20 仅 **2.8%** → **B1 multi_emit** 或行为/obs 修复 |
| **囤兵量级差两个数量级** | 赢家 max launch p50 **181**，p95 **1125**；top1 局 **6743** | fleet_log log-scale 方向对；v9 训练 spf~27 仍远低于决战规模 |
| **garrison = 总驻防（全行星合计）** | 赢家 first-80 mean **152**，全场 mean **1717** | v9b vs v20 replay **31.9** → early game 失守是主因 |
| **扩张先于囤满** | 赢家全场 mean **13.5 星** vs 输家 **3.35 星** | planet_share shaping 方向对 |
| **典型节奏：囤→放→囤** | decile 曲线非单调（见 §6） | 不要 keep_home；可选 **B3 gated fleet_log** |

**与 v9b vs v20 的差距（first-80，player_0）**：

| metric | 高手赢家 (cohort) | v9b | v20 (对手) | Day5 FAST gate |
|---|---|---|---|---|
| spf | **28.7** | 4.76 | 15.9 | > 10 |
| garr（总驻防） | **152** | 31.9 | 132 | > 60 |
| emit≥2 占比 | **14%** | 2.8% | ~55%² | > 5% |
| prod_share | **0.19**¹ | — | — | — |
| z0 | **62%**¹ | 1.0% | 24% | 15–40%（全场） |

¹ first-80 窗口；² v20 为 5 局 replay 中 player_1 近似值。

---

## 1. 数据集说明

| 项 | 值 |
|---|---|
| 路径 | `top10_episodes_2026-05-04/episodes/episodes/*.json` |
| 局数 | **2631** 文件，**2630** 局成功解析（**4P**） |
| manifest | `manifest.csv`：按 `sum_score`（四选手 rating 之和）降序 |
| 典型长度 | mean **268** turn，p50 **222**，p95 **500**（满局） |
| 选取方法 | 当日所有对局 → 按双方 UpdatedScore 之和取 **Top 10%** → 优先下载至 20GiB 上限 |

**注意**：

* 全是 **4P**；我方 2P 训练/env 与高手 4P 有分布差（`prod_share` 基线 0.25 vs 2P 的 0.5）。
* replay JSON **不含** fleet 到达/flip 仿真；**无 `flip_proxy`**，需跑 `replay_analyze.py` 对局才有。
* `submission_ids` 在 manifest 中，本报告按 **赢家/输家 cohort** 聚合，未按 submission 拆选手。

---

## 2. 指标定义（读表前必读）

与 [`replay_analyze.py`](../orbit_wars_rl/scripts/replay_analyze.py) / 训练 log 对齐说明：

| 指标 | 本报告含义 | 与训练 log `garr` 的区别 |
|---|---|---|
| **garrison / garr** | 每回合初，该选手**所有占有行星驻防舰之和**（总囤兵量） | 训练 log 是 **每星平均驻防**；数值不可直接比 |
| **spf** | 该回合所有 launch 的舰数均值（只统计有发射的 turn 内各 launch） | 与 replay_analyze 一致 |
| **prod_share** | 该选手占有行星 `production` 之和 / 全局 production | 4P 公平基线 **0.25** |
| **planet_count** | 占有行星数；4P 开局约 **8** | — |
| **z0 / zero_emit_rate** | 本回合 launch 数 = 0 的占比 | 高 z0 = 主动囤兵，不一定是坏事 |
| **emit2_rate** | 本回合 launch 数 **≥ 2** 的占比 | Day5 称 emit≥2 |
| **emit3_rate** | launch 数 **≥ 3** | multi-launch 强度 |
| **max_launch / p95_launch** | 单局内单次 launch 最大值 / 95 分位 | 「决战气魄」 |
| **decisive_turn / decisive_ships** | 全场最大单次 launch 所在 turn / 舰数 | 决战时机 |
| **garrison_decile** | 局内 0%–100% 十等分节点的总驻防 | 囤→放节奏 |
| **mean_fleets_in_flight** | 在途舰队数量时间均值 | 多线操作强度 |

---

## 3.  cohort 汇总：赢家 vs 输家（2630 局 × 1 赢家 + 3 输家）

### 3.1 First-80 turns（Day5 FAST gate 窗口）

| 指标 | 赢家 mean | 赢家 p50 | 赢家 p95 | 输家 mean | 输家 p50 | **分离度** |
|---|---|---|---|---|---|---|
| **spf** | 28.7 | 25.8 | 53.6 | 23.2 | 20.7 | 中（赢家 +24%） |
| **garr（总驻防）** | **152** | 137 | 293 | 98 | 86 | **高（+55%）** |
| **prod_share** | 0.19 | 0.19 | 0.29 | 0.13 | 0.14 | 中（early 都低于 0.25） |
| **planet_count** | 5.3 | 5.1 | 8.2 | 3.8 | 3.8 | **高** |
| **z0** | 0.62 | 0.64 | 0.86 | 0.67 | 0.69 | 低（输家更常不发） |
| **emit2_rate** | **0.14** | 0.10 | 0.38 | 0.10 | 0.06 | 中 |
| **emit3_rate** | 0.06 | 0.03 | 0.21 | 0.04 | 0.01 | 中 |
| **max_launch** | 429 | 181 | 1125 | 117 | 77 | **极高（3.7×）** |
| **p95_launch** | 136 | 89 | 336 | 68 | 51 | **高（2×）** |
| **peak_garrison** | 442 | 394 | 877 | 260 | 225 | **高** |

### 3.2 Full game（全场）

| 指标 | 赢家 mean | 赢家 p50 | 赢家 p95 | 输家 mean | 输家 p50 |
|---|---|---|---|---|---|
| **prod_share** | **0.48** | **0.47** | 0.69 | **0.12** | 0.10 |
| **spf** | 46.3 | 35.3 | 102.6 | 27.9 | 22.8 |
| **garr（总驻防）** | **1717** | 716 | 7362 | 239 | 79 |
| **planet_count** | **13.5** | 13.1 | 21.0 | 3.4 | 2.9 |
| **z0** | **0.51** | 0.51 | 0.84 | 0.75 | 0.78 |
| **emit2_rate** | **0.27** | 0.23 | 0.62 | 0.09 | 0.05 |
| **max_launch** | 429 | 181 | 1125 | 117 | 77 |
| **mean_fleets_in_flight** | **11.0** | 7.8 | 29.6 | 3.7 | 2.6 |

### 3.3 prod_share 胜负阈值（全场，最强单一指标）

| 阈值 | 赢家中 ≥ 阈值 | 输家中 ≥ 阈值 |
|---|---|---|
| ≥ 0.18 | **99.9%** | 19.7% |
| ≥ 0.25（4P 基线） | **99.4%** | 6.2% |
| ≥ 0.35 | **95.1%** | 0.5% |
| ≥ 0.45 | **59.4%** | **0.0%** |
| ≥ 0.55 | 17.2% | 0.0% |

**结论**：全场 `prod_share ≥ 0.35` 几乎完美分离胜负；这与 Day4 5 局 deep dive 一致，在 **2630 局** 上复核成立。

---

## 4. Day5 关心方向 — 逐项对照

### 4.1 Multi-launch（Track B1）

| 观察 | 数据 |
|---|---|
| 赢家 first-80 emit≥2 | mean **14%**，p95 **38%** |
| 赢家全场 emit≥2 | mean **27%**，p50 **23%** |
| v9b vs v20 first-80 | **2.8%**（几乎从不 multi-emit） |
| v20 对手 first-80 | ~**55%** emit≥2（5 局样本） |

**启示**：高手并非每 turn 都 multi-launch，但 **显著多于 v9b**；FAST gate **>5%** 合理，目标区间 **10–25%**（first-80）。

### 4.2 囤兵 / garrison / 决战规模（Track A1' + fleet_log）

| 观察 | 数据 |
|---|---|
| 赢家 first-80 总驻防 | mean **152**，p95 **293** |
| 赢家 peak 总驻防 | mean **442**，p95 **877** |
| 赢家 max 单次 launch | p50 **181**，p95 **1125** |
| v9b vs v20 first-80 总驻防 | **31.9** |
| v9 训练 log garr | ~50（**每星平均**，不可直接比） |

**启示**：

* early game 总驻防 **<60** 基本不够（v9b 31.9 vs 高手 152）。
* 决战 launch **100–2000+** 是常态；linear NORM=20 的 fleet_size v1 完全不够，**log-scale fleet_log 正确**。
* decile 曲线见 §6：**非单调上升** → 不要 keep_home 式「持续囤」reward。

### 4.3 扩张 / prod_share / planet_share（v9c shaping，已冻结）

| 观察 | 数据 |
|---|---|
| 全场 planet_count | 赢家 **13.5** vs 输家 **3.4** |
| 全场 prod_share | 赢家 **0.48** vs 输家 **0.12** |
| first-80 prod_share | 赢家 **0.19** vs 输家 **0.13**（early 差距较小） |

**启示**：**planet/prod shaping 方向正确**；early 窗口 prod 差距不大，单靠 macro reward 可能不够解释 v9b 对 v20 early 溃败 → 还需 **威胁感知 / curriculum**。

### 4.4 换家 / 威胁 / 留兵（Track A1'）

Replay JSON **无法直接观测**「威胁下向 friendly dst 转移」或「按威胁留兵」——需要：

* 解析 launch 目标 planet（angle → planet 匹配），或
* 在 env 加 `threat_ratio` / `eta`（Day5 A1'）

**间接信号**：

* 赢家 `mean_fleets_in_flight` **11** vs 输家 **3.7** → 多线调度是高手特征。
* 赢家 first-80 garr 高但 z0 也高 → **囤的时候真囤**，不是 v7 式乱发。

### 4.5 对手分布 / curriculum（Track C1）

| 观察 | 数据 |
|---|---|
| 数据集本身 | Top 10% **强对强**（sum_score 高） |
| 我方 self-play | v7 vs v7 flip **1.14%**（Day4）— 弱对弱废涡 |
| v9b vs v20 | **0/40** — 分布外对手 |

**启示**：高手数据证明「强对强」局里 flip/对抗频繁；**必须把 v20 级对手注入训练分布**（C1），不是再加 global reward。

### 4.6 z0（不发 turn）— 修正训练目标

| 群体 | first-80 z0 | 全场 z0 |
|---|---|---|
| 高手赢家 | **62%** | **51%** |
| 高手输家 | 67% | 75% |
| v9b vs v20 | 1.0% | 41% |
| v9 训练 | ~**0–4%** | — |

**启示**：

* **高 z0 是囤兵期正常行为**；v9 训练 z0≈0 说明策略「每 turn 都要做点什么」，与高手相反。
* v9b 对 v20 **全场** z0=41% 说明 losing spiral 里才停手；高手赢家全场 z0=51% 是 **主动节奏**。
* Day5 **B3 gated fleet_log**（garr>θ 才奖大 launch）有助于学囤放周期，而非追求 z0→0。

---

## 5. 与我方 agent 基线对照

| metric | 高手赢家 first-80 | v9b vs v20 | v7 vs v20 (Day4) | Day5 gate |
|---|---|---|---|---|
| spf | 28.7 | 4.76 | 1.98 | > 10 |
| garr（总驻防） | 152 | 31.9 | 9.3 | > 60 |
| emit2_rate | 14% | 2.8% | — | > 5% |
| prod_share (full) | 0.48 | — | — | — |
| gauntlet vs v20 | — | 0/40 | 0/3 (3局) | overnight ≥1/40 |

**Train–eval gap（v9b）**：训练 spf **21.2** → replay **4.76**（4.5× 高估）；说明 self-play 指标 **不能** 外推到 v20。

---

## 6. Top-10 局深度表（按 sum_score）

| episode | turns | prod_full | spf_80 | garr_80 | emit2_80 | z0_full | max_launch | decile garr (0→100%) |
|---|---|---|---|---|---|---|---|---|
| 75873267 | 500 | 0.26 | 46.7 | 288 | 0% | 0.52 | 575 | 10→…→**1384** |
| 75862353 | 274 | **0.55** | 38.1 | 158 | 10% | 0.46 | 355 | 10→222→455→**208**→…→**5519** |
| 75864613 | 255 | 0.41 | 42.5 | 75 | 0% | 0.70 | 506 | …→**2103** |
| 75853309 | 408 | **0.55** | 37.6 | 91 | 2.5% | 0.74 | **2076** | 囤到 turn~250 才大 launch |
| 75864896 | 235 | 0.49 | 21.9 | 172 | **24%** | 0.56 | 245 | …→**6061** |
| 75854445 | 500 | **0.56** | 27.0 | 228 | 5% | 0.28 | 1000 | …→**12006** |
| 75863901 | 231 | 0.44 | 20.5 | 109 | 6% | 0.43 | 220 | …→3985 |
| 75859613 | 190 | 0.36 | 36.7 | 57 | 2.5% | 0.55 | 344 | 短局速推 |
| 75867019 | 500 | 0.51 | 50.9 | 223 | 16% | 0.48 | **4359** | …→**10989** |
| 75854230 | 500 | **0.73** | **126.5** | **300** | 4% | **0.86** | **6743** | 超大规模囤兵 |

**可读模式**：

1. **75853309**：turn 50–250 几乎纯囤（z0 高），turn 300 才 705+24 launch — stockpile-then-release 教科书。
2. **75862353**：decile 曲线 455→**208**→860 — 「放完再囤」，非单调。
3. **75854230**：全场 z0 **86%**，max launch **6743** — 极端囤→一波流。
4. **75864896**：first-80 emit2 **24%** — multi-launch 型赢家也存在，不是唯一解。

---

## 7. 其他值得关注的方向

### 7.1 局时长与早终

* cohort mean **268** turn；Top 局 190–500 turn 均有。
* 短局赢家（75859613，190 turn）仍靠 **prod_share 0.36 + max 344** — 存在「快攻型高手」子风格。

### 7.2 在途舰队（多线操作）

* 赢家全场 mean **11** 条在途 vs 输家 **3.7**。
* 与 emit2/multi-launch 一致：**不是单线 ping-pong**。

### 7.3 单次 launch 体量分布

* 赢家 p95_launch **136**（全场统计），max 可达 **6743**。
* 输家 p95_launch **68** — 决战规模是质的不同，不只是 spf 略高。

### 7.4 first-80 vs full 的 prod_share 跃迁

* 赢家 first-80 prod **0.19** → full **0.48**：中后期扩张/吞并是拉开差距的主阶段。
* 解释 v9 **episode_steps=350** 的必要性；也解释为何 first-80 gate 对 prod 不敏感。

### 7.5 4P vs 2P 训练差异

| 项 | 4P 高手 | 2P 训练 (v9) |
|---|---|---|
| prod_share 公平基线 | 0.25 | 0.50 |
| 典型 planet_count（赢家 full） | ~13 | ~2 开局 |
| 对手数 | 3 | 1 |

Kaggle 提交是 **4P**；2P 练出的 macro 策略可能 **scale 不对**，Track C（4P env）是 Week 2+ 项，但应提前知道此 gap。

### 7.6 本数据集做不到的

* ❌ `flip_proxy` / capture 成功率（需仿真）
* ❌ 按 submission_id 拆选手风格（需另做 join）
* ❌ homeworld 单独 garrison（需知 home planet id）
* ❌ sun loss / 遮挡（需仿真或更细解析）

---

## 8. 对 Day5 优先级的影响（建议）

详细落地（公式 + env var + FAST gate）见 **[`DAY5_TRAINING_ACTIONS.zh.md`](DAY5_TRAINING_ACTIONS.zh.md)**。

**精炼版**（按 ROI 排序）：

| 优先级 | Track / 动作 | 数据依据 | 关键点 |
|---|---|---|---|
| **P0** | Phase 0 收尾 v9c + replay vs v20 | 必须有 frozen base | — |
| **P1** | **R1 prod_share_DELTA** | 赢家 +0.29 vs 输家 −0.02 | 水平 → delta（credit 强 10×） |
| **P1 ∥** | **C1 v20 进 frozen pool 25%** | 强对强才有 flip 信号 | ≠ BC |
| **P2** | **R2 release_bonus** | peak/mean garr ratio 4.28 | 替代 fleet_log，gated by garr |
| **P2** | **A1' threat obs**（threat_ratio, eta, net_inbound） | fleets_in_flight 11 vs 3.7 | 修 ray 归因 + unit test |
| **P3** | **R4 emit_log**（不是 multi_emit bonus） | emit2 6.8% =0，分布右偏 | 奖「有效 launch 数」非「越多越好」 |
| **P3** | **R3 baseline bug fix**（N=2 硬编码） | C1 上 4P bot 后梯度方向错 | 修代码不加 reward |
| **defer** | B2 capture / A4 容量 | 实现成本高 | — |
| **defer** | 再扫 prod/planet/fleet 系数 | 分离度已验证 | — |

---

## 9. 复现命令

```bash
# 全量聚合（~4–5 min，2630 局）
python -m orbit_wars_rl.scripts.aggregate_top10_replays \
  --dataset-dir top10_episodes_2026-05-04 \
  --out logs/top10_aggregate_2026-05-04.json

# 单局可读摘要
python -m orbit_wars_rl.scripts.analyze_expert_replay \
  --replay-glob 'top10_episodes_2026-05-04/episodes/episodes/75853309.json' \
  --print-summary --print-trace

# 快速 smoke（前 50 局）
python -m orbit_wars_rl.scripts.aggregate_top10_replays \
  --dataset-dir top10_episodes_2026-05-04 \
  --max-episodes 50 \
  --out logs/top10_aggregate_smoke.json
```

---

## 10. 相关文档

* [`DAY5_TRAINING_ACTIONS.zh.md`](DAY5_TRAINING_ACTIONS.zh.md) — **Day5 训练动作清单**（R1/R2/R3/R4/A1'/C1 公式+系数）
* [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md) — 我方 v9 现状与 Phase 0
* [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md) — FAST / Track A/B/C 规划
* [`DAY4_PROGRESS.zh.md`](DAY4_PROGRESS.zh.md) §11–§12 — 早期 5 局 deep dive（与本报告一致）
* [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md) — replay gate 阈值
