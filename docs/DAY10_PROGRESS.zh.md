# DAY10 进展 — f36a/f36b 分析 + f37 方案启动

> **2026-06-02 晚更新**
> 接续 [`DAY9_PLAN.zh.md`](DAY9_PLAN.zh.md)。
> Day10 完成 f36a/f36b 分析、f37/f38 训满与 replay、f38s1b 对照实验，并启动 **f40 Expert-Seeded Gated League** 基建与 smoke 验证。
> 详见 [`DAY11_PLAN.zh.md`](DAY11_PLAN.zh.md)。

---

## 0. TL;DR — 当前状态

| 维度 | 状态 |
|---|---|
| **f36a** | 🔴 **WLD 0/5 全 ckpt**；`emit_hard_stop_min_step=2` 未打破单路 emit |
| **f36b** | 🔴 **strong opponent 被静默禁用**（shape mismatch 33 vs 28）；实际退化为 f33-like 短训 |
| **根因诊断** | 问题不在动作机制或训练分布，而是 **四个 action head 缺乏足够直接的输入信号** |
| **Vadasz 对弈分析** | 顶级选手 z0=34%、e2+=37%、spf 中位 25；我们 z0<1%、e2+=0%、spf≈5-17 |
| **f37 方案** | 训满 3000 upd；@199 e2+=14.8% 峰值，@499+ 塌缩 |
| **f38 s1** | CAPTURE=0.05 完成；@199 captures=19/e2+=15.2%，@499 塌缩 → **不进入 s2/s3** |
| **f38s1b** | 🔴 **500 upd 完成**；@199 **劣于 s1**（captures 8 vs 19）；@499 e2+=20.2% 但 flip/captures 仍 fail → **f38 全系关闭** |
| **f40** | 🟡 **基建 + smoke 完成**；BC seed / PPO@4 replay 未过 promote；**新主线** |
| **提交候选** | **f29 @599** 仍为基线 |

---

## 1. f36a 结果分析 — 动作机制 delta

### 1.1 实验设计

- **唯一 delta**：`emit_hard_stop_min_step: 1 -> 2`（允许 step 0 和 step 1 不被 hard stop）
- **其他全部冻结**：PPO/特征/selfplay 均同 f33

### 1.2 replay 结果（vs v20，first_80turns）

| ckpt | WLD | emits | z0 | spf | flip | bin7 |
|------|-----|-------|----|-----|------|------|
| @149 | 0/5/0 | 0.99 | 0.58% | 5.0 | 3.72% | 75.2% |
| @199 | 0/5/0 | 0.99 | 0.60% | 5.9 | 5.55% | 71.1% |
| @249 | 0/5/0 | 0.99 | 0.71% | 10.4 | 4.98% | 71.4% |
| @299 | 0/5/0 | 0.99 | 0.64% | 17.1 | 5.08% | 65.1% |
| @349 | 0/5/0 | 0.99 | 0.65% | 16.6 | 3.18% | 76.8% |

### 1.3 结论

- `emits` 始终≈0.99，`z0` 始终<1%：**单路 emit attractor 完全没有被打破**
- `spf` 从 5 涨到 17（@299），但这是 self-play 膨胀带来的 garrison 增长，不是策略改善
- `emit_hard_stop_min_step=2` 的效果：允许第二步不被 hard stop，但模型自己学会了在 step 1 就选择 stop
- **结论**：动作机制闸门不是瓶颈。模型缺乏的不是"发第二次的权利"，而是"知道该发第二次的信号"

---

## 2. f36b 结果分析 — 训练分布 delta

### 2.1 关键错误

训练日志第一行暴露了致命问题：

```
WARN strong ckpt shape mismatch, disabling:
  shapes: (33, 128) != (28, 128)
```

f36b 尝试用 `f29@599`（28 维 planet 特征）作为 strong opponent，但 f36b 自身使用 f35 的 33 维特征。Shape 不兼容导致 strong opponent 被**静默禁用**。

### 2.2 实际运行状态

- `strong_ratio=0.35` 被禁用 -> 实际 strong_ratio=0
- `frozen_ratio=0.25` 正常
- 训练分布与 f33 几乎相同，只是更短（350 upd vs 2000 upd）
- 末期 `spf=235, garr=204` 再次出现 self-play 膨胀

### 2.3 结论

- f36b 实验**无效**：设计变量（strong opponent）实际未生效
- 但也确认了一个事实：即使 shape 兼容问题解决，f36b 的核心假设（"训练分布是瓶颈"）可能也不成立，因为 f36a 的结果已表明问题在于信号缺失

---

## 3. 顶级选手对弈分析 — Vadasz vs 我们的模型

### 3.1 episode 75829347 分析

从 Kaggle 对弈数据中，分析了 Vadasz（顶级 RL 选手）的行为模式：

| 指标 | Vadasz | 我们的模型 | 差距 |
|------|--------|-----------|------|
| **空回合率 (z0)** | 34.4% | <1% | 战略等待 vs 永不停歇 |
| **多路 emit (e2+)** | 37% of active turns | 0% | 同时进攻多个目标 vs 单路 |
| **舰队规模 (spf)** | 中位 25 ships/emit | ≈5 ships | 有效舰队 vs 无效 trickle |
| **进攻策略** | 蓄力 -> 爆发 -> 等待 | 每回合 trickle 1-3 艘 | 节奏对比 |

### 3.2 核心洞察

Vadasz 的优势来自三个"基础决策"做对了：

1. **什么时候发（z0 高）**：不是每回合都发，而是蓄力到足够再发 -> 需要 "蓄力够不够" 的信号
2. **发多少条（e2+ 高）**：打到软目标时同时发 2-3 路 -> 需要 "能打几个目标" 的信号
3. **发多大（spf 高）**：每支舰队都足以翻转目标 -> 需要 "最小有效舰队" 的信号

---

## 4. 根因诊断 — 为什么 f36a/f36b 都失败

### 4.1 问题定位

| 基础决策 | v20 启发式做法 | 当前 RL 模型看到的信号 | 缺口 |
|----------|--------------|---------------------|------|
| **从哪发** | `avail >= ABS_MIN_BATCH(8)` + 按可用兵力排序 | `remaining_norm = log1p(remaining)/8`（压缩后差异太小） | 分不清兵库 vs 空壳 |
| **发多少** | `capture_need()` 精确计算 -> 找最小足够 bin | `min_bin_norm` + `pair_flip_bin5` | 缺少 "need 几艘才够翻" 的连续信号 |
| **往哪打** | `target_score()` = prod_value * distance_decay - need | `v20_target_score`（f35 已加） | 已有但未训够 |
| **发不发** | `safe_surplus >= ABS_MIN_BATCH` + 有合格目标才发 | `emit_worth_it`（二值）+ `total_remain_ratio` | 缺少 "当前蓄力是否足够" 的连续信号 |

### 4.2 结论

**问题不在动作机制（f36a）也不在训练分布（f36b）**，而是四个 action head 在做最基础的决策时，缺乏足够直接的输入信号。

---

## 5. f37 方案 — 强化基础动作信号

### 5.1 三管齐下

#### 一、特征强化（保留 33 维 + 新增关键信号）

| 改动 | 位置 | 内容 |
|------|------|------|
| EmitHead 信号增强 | `pair.py` | `emit_pair_globals` 4 -> 6 维：+`feasible_target_count_norm`（能翻几个目标）、+`surplus_ratio`（打完还剩多少） |
| 全局特征增强 | `encode.py` | `GLOBAL_FEAT_DIM` 17 -> 18：+`min_effective_fleet_norm`（ABS_MIN_BATCH/avg_garrison） |

#### 二、Reward 简化

所有中间 shaping 设为 0，只留 +1/-1 终局：

- `RELEASE=0.0`（去掉！这个直接奖励发兵 -> 强化 trickle）
- `PLANET_SHARE=0.0`、`PROD_SHARE_DELTA=0.0`、`CAPTURE=0.0`

#### 三、训练配置

- `num_updates=3000`（~100M steps，顶级选手建议的基线量）
- `ckpt_every=100`
- `frozen_ratio=0.40`，不用 strong opponent（避免 shape 兼容问题）
- PPO 配方同 f33

### 5.2 实施状态

| 文件 | 改动 | 状态 |
|------|------|------|
| `orbit_wars_rl/features/pair.py` | `emit_pair_globals` 4->6 维 + `EMIT_PAIR_DIM=6` | ✅ |
| `orbit_wars_rl/features/encode.py` | `GLOBAL_FEAT_DIM=18` + `min_effective_fleet_norm` | ✅ |
| `orbit_wars_rl/net/heads.py` | EmitHead 文档更新 | ✅ |
| `orbit_wars_rl/inference/numpy_forward.py` | `_emit_pair_globals_np` 同步 6 维 | ✅ |
| `submission_rl_v11_f37.py` | 新提交模板 (GLOBAL=18, EMIT_PAIR=6) | ✅ |
| `orbit_wars_rl/configs/multi_action_v11_f37.yaml` | 新配置 | ✅ |
| `scripts/run_v11_f37.sh` | 训练脚本（全部 shaping=0） | ✅ |
| `scripts/run_f37_eval.sh` | 评估脚本 | ✅ |
| `orbit_wars_rl/scripts/export_submission.py` | f37 模板识别 | ✅ |

### 5.3 验证

- **Parity test**：16/16 通过 ✅
- **CPU smoke test**：2-update 训练成功完成，exit_code=0 ✅
- **训练启动**：已在 5090 服务器上通过 `bash scripts/run_v11_f37.sh` 启动 ✅

### 5.4 评估计划

重点 checkpoint：`@199 / @499 / @999 / @1999 / @2999`

早期关注指标：
- `z0`：是否从 <1% 拉到 >3%（模型学会等待）
- `e2+`：是否从 0% 拉到 >3%（模型学会多路 emit）
- `spf`：是否稳定在 >10（不是膨胀带来的）

**风险**：纯 +1/-1 在早期可能 reward 太稀疏。如果 @499 的 WR vs random 跌破 0.5，需要加回少量 `CAPTURE=0.02`（只奖励翻转星球，不奖励发兵本身）。

---

## 6. Day10 执行决策树

```
f37 训练中 (3000 upd, ~100M steps)
│
├── @199: 看 z0/e2+/spf 是否有信号
│   ├── 有改善 -> 继续训练
│   └── 完全不动 + WR vs random < 0.3 -> 加回 CAPTURE=0.02 重训
│
├── @499: 第一个正式 replay gate
│   ├── 多指标改善 -> 继续到 @999
│   └── 仍 WLD 0/5 且指标不动 -> 止损
│
├── @999: 中期 checkpoint
│   └── 若出现 WLD >= 1/5 -> 进入提交候选
│
└── @2999: 终点
    └── 全面 replay 评估
```

---

## 7. 与 Day9 计划的偏差说明

| Day9 计划 | Day10 实际 | 原因 |
|-----------|-----------|------|
| f36a -> f36b -> f36c 串行 | f36a/f36b 都跑完，跳过 f36c | 两条支线均 fail，f36c 组合无意义 |
| 不优先做新特征工程 | f37 重新做特征工程 | 根因诊断发现问题在信号缺失，不是动作机制 |
| 短训 ≤350 upd | f37 长训 3000 upd | 纯终局 reward 需要更多样本量学习 |
| 不碰 reward shaping | f37 去掉所有 shaping | 发现 RELEASE=0.05 是 trickle attractor 的直接原因 |

关键反思：Day9 的假设（"问题在 turn-level 动作结构"）方向正确，但药方（"改闸门/改训练分布"）开错了。正确的药方是**给 action head 足够直接的信号**让它知道"什么时候蓄力够了"、"当前能打几个目标"。

---

## 8. f37 Replay 结果分析 — 策略塌缩确认

### 8.1 Replay 结果（vs v20，first_80turns）

| ckpt | WLD | emits | z0 | e2+ | spf | garr | flip | captures | bin7 | one_ship_rate |
|------|-----|-------|----|-----|-----|------|------|----------|------|---------------|
| @199 | 0/5/0 | 1.28 | 4.2% | **14.8%** | 2.65 | 4.87 | 1.53% | 9 | 64.1% | 32.8% |
| @499 | 0/5/0 | 0.99 | 0.6% | 0.2% | 5.26 | 65.0 | 3.47% | — | — | 28.5% |
| @999 | 0/5/0 | 0.99 | 1.2% | 0.0% | 5.01 | 60.2 | 2.45% | — | — | 33.8% |
| @1999 | 0/5/0 | 0.99 | 0.6% | **0.0%** | 6.89 | 96.1 | 4.96% | 26 | 92.9% | 39.0% |

### 8.2 @199 vs @1999 详细对比

@199 是 f37 的**最佳时刻**，也是整个 f33-f37 系列中唯一一个 `e2+` 通过 5% gate 的 ckpt：

| 指标 | @199 | @1999 | 趋势 |
|------|------|-------|------|
| e2+ | 14.8% | 0.0% | 从"会多路进攻"退化到"只会单路" |
| z0 | 4.2% | 0.6% | 从"偶尔等待"退化到"永不停歇" |
| emit_count 分布 | 2路=5.8%, 3路=4.2%, 8路=1.0% | **1路=98.8%**, 其余全 0% | 多样性完全消失 |
| spf | 2.65 | 6.89 | 舰队变大但 garrison 涨更多（膨胀） |
| bin7 (100%) | 64.1% | 92.9% | 从部分兵力分配退化到永远全押 |
| one_ship_rate | 32.8% | 39.0% | trickle 比例反而增加 |
| captures | 9 | 26 | 翻转稍多但仍远不及 v20 的 123 |

### 8.3 训练指标 vs Replay 指标 — 严重脱节

| 指标 | 训练日志 @u2000 | Replay @u1999 vs v20 | 差异来源 |
|------|-----------------|---------------------|---------|
| e2+ | 0.27-0.33 (27-33%) | **0.0%** | 自我对弈中 garrison 膨胀，轻松多路；面对 v20 被压制后无法多路 |
| z0 | 0.03-0.06 (3-6%) | **0.6%** | 自我对弈中偶有空回合；面对 v20 的进攻压力下每回合都被迫行动 |
| emits | 1.41-1.56 | 0.99 | 自我对弈 garrison 高，有能力多发；实战只能单路 |
| spf | 19-23 | 6.89 | 自我对弈 garrison=30+ 所以 spf 自然高；实战 garrison 被 v20 压制 |
| garr | 27-34 | 96 (replay) | 自我对弈的是友好生态下的 garrison，不代表对外战力 |
| WR vs random | 0.88 | — | 能打 random 但对 v20 WLD=0/5/0 |

**核心问题**：训练指标完全不可信。`e2+=30%` 和 `spf=20` 都是自我对弈膨胀的假象。唯一可信的评估是 replay vs 外部对手。

### 8.4 塌缩时间线

```
u0-u49:   warmup（vs random），模型探索阶段
u50-u199: 自我对弈开始，策略多样性最高 -> @199 是峰值（e2+=14.8%）
u200-u499: 策略快速退化，e2+ 从 14.8% 暴跌到 0.2%
u500-u2000+: 完全锁死在 "单路 trickle + bin7 全押" attractor
```

f37 特征增强（feasible_target_count_norm / surplus_ratio / min_effective_fleet_norm）在 @199 确实产生了效果（首次突破 e2+ > 5%），但这个信号无法在自我对弈中存活——两个弱策略互相 trickle 形成的均衡是一个**稳定但退化的纳什均衡**。

---

## 9. 根因分析 — 为什么 f33-f37 反复塌缩

### 9.1 实验时间线回顾

| 实验 | 核心改动 | 结果 | 是否打破 trickle |
|------|---------|------|-----------------|
| f33 | 基线配置 | WLD 0/5 全 ckpt | 否 |
| f35 | 33 维特征 + v20 target score | 同上 | 否 |
| f36a | emit_hard_stop_min_step=2 | 同上 | 否 |
| f36b | strong opponent（被禁用） | 同上（无效实验） | — |
| f37 | 18 维 global + 6 维 pair + 纯终局 | @199 有信号，随后塌缩 | **瞬间打破，无法维持** |

五次实验全部失败，收敛到同一个 attractor：单路 trickle + bin7 全押 + z0<1% + e2+=0%。

### 9.2 鸡生蛋问题 — 为什么纯 +1/-1 在这里不够

Lin Myat Ko 说"+1/-1 is enough for 2p mode"。但这有一个隐含前提：**模型的特征/架构足以让它在早期自然学会翻转星球**。

我们面临的困境：

```
纯 +1/-1 reward
    |
    v
早期模型不知道怎么翻转星球
    |
    v
自我对弈两方都在 trickle -> 谁也翻不了
    |
    v
胜负完全取决于地图初始布局 -> reward 信号 ≈ 随机噪声
    |
    v
PPO 找不到梯度 -> 策略不变 -> 继续 trickle
    |
    v
(循环)
```

对比 Lin Myat Ko 的情况：他的 feature engineering 更成熟（"even with basic features, we can train it to have a decent performance"），模型在早期就能偶然翻转足够多星球产生有效的 reward 信号。而我们的模型在 500 步 episode 中翻转率只有 1.5-5%（vs v20 的 13-17%），梯度信号太弱。

### 9.3 自我对弈塌缩机制

基于 AlphaStar League、OpenAI Five、Lux AI 冠军方案、以及最新研究 (arxiv:2605.22217, SPIRAL) 的调研：

**机制一：退化纳什均衡**

两个同水平的弱策略自我对弈时，"双方都 trickle" 是一个**稳定均衡**：
- 任何一方单独切换到"蓄力+爆发"不会立即受益（因为对方在 trickle，蓄力期间不会被有效攻击，但 trickle 也不会翻转）
- 终局 reward 主要由地图对称性决定（≈50/50），PPO 看到的 advantage 几乎为零
- 结果：策略停留在退化均衡，训练指标看起来"正常"但对外战力为零

**机制二：对手池封闭导致漂移**

当前系统 `pool_capacity=8, snapshot_every=50`，对手全部是自身的最近快照。如果当前策略已退化，pool 中全是退化策略的变体，形成正反馈循环：
- 退化策略 vs 退化策略 = 退化继续
- 没有外部锚点打破循环

AlphaStar 用 League（Main Agent + Main Exploiter + League Exploiter）解决此问题。OpenAI Five 用 80% 最新策略 + 20% 历史策略。Lux AI 冠军用 teacher-student + KL 约束。

**机制三：数据管门缺失**

最新研究（2026, arxiv:2605.22217）发现："自我对弈稳定性由数据层决定，不由 reward 层决定。严格的数据过滤足以在任何 reward 设计下防止塌缩；而没有数据过滤时，没有任何 reward 设计能防止塌缩。" 当前系统没有对自我对弈产生的 trajectory 做任何质量过滤。

### 9.4 为什么 f37 的特征增强不够

f37 加了 `feasible_target_count_norm`、`surplus_ratio`、`min_effective_fleet_norm`。在 @199 确实产生了效果（e2+=14.8%），说明**特征方向是对的**。但问题在于：

1. 这些特征提供了"能打几个目标"的信息，但 **reward 不提供"翻转星球"的梯度**
2. 模型学会了利用这些特征在自我对弈中多路 emit，但当 garrison 膨胀后（两方都不有效进攻），多路 emit 变得 trivial 而不是 strategic
3. 随着 self-play 进行，模型发现"单路 trickle + bin7 全押"在退化生态中也能获得 ≈50% 胜率，于是策略简化

---

## 10. f38 方案 — Curriculum + Capture Shaping + External Anchor

### 10.1 核心思路

从 Lux AI 冠军方案和 AlphaStar 中提取三个关键教训：

1. **Shaped -> Sparse Curriculum**：先用 dense reward 教会基础技能（翻转星球），再退火到 sparse reward 优化终局胜率
2. **External Anchor**：在对手池中保留一个不随训练漂移的外部对手，防止 population 整体退化
3. **只奖励翻转，不奖励发兵**：CAPTURE shaping 奖励"成功翻转星球"这个关键事件，而不是 RELEASE（奖励发兵本身，这是 trickle 的直接诱因）

### 10.2 三阶段课程

#### 阶段一：Shaped Reward Bootstrap（update 0-500）

| 参数 | 值 | 作用 |
|------|----|------|
| `CAPTURE` | 0.05 | 成功翻转星球时的奖励，提供"翻转=好"的直接梯度 |
| `PROD_SHARE_DELTA` | 0.02 | 夺取产能星球的 delta 奖励，credit assignment 紧密 |
| 其余 shaping | 0.0 | 不奖励发兵本身（避免 RELEASE 诱导 trickle） |
| 对手 | random（warmup=50）+ frozen self-play | 先打 random 学会基础翻转 |

#### 阶段二：Anneal（update 500-1000）

| 参数 | 变化 | 说明 |
|------|------|------|
| `CAPTURE` | 0.05 -> 0.0 线性退火 | 逐步转向纯终局 reward |
| `PROD_SHARE_DELTA` | 0.02 -> 0.0 线性退火 | 同上 |
| self-play | 正常 frozen_ratio=0.40 | 模型此时已学会翻转，可以安全进入 self-play |

#### 阶段三：Pure Terminal + Anchor（update 1000-3000）

| 参数 | 值 | 说明 |
|------|----|------|
| 全部 shaping | 0.0 | 纯 +1/-1 终局 reward |
| strong_ratio | 0.20 | 20% 对局 vs f29@599（外部锚点） |
| frozen_ratio | 0.40 | 40% 对局 vs frozen self |
| random | 40% | 剩余 vs random |

### 10.3 External Anchor — Shape Adapter 方案

f29@599（28 维 planet, 17 维 global, 4 维 emit pair）与 f38（33 维 planet, 18 维 global, 6 维 emit pair）存在 shape 不兼容。解决方案：

在 `runner.py` 的 strong checkpoint 加载逻辑中实现 shape adapter：
- 对 strong ckpt 中 planet feature 相关的权重矩阵做 zero-padding（28 -> 33 维）
- 对 global feature 相关的权重矩阵做 zero-padding（17 -> 18 维）
- 对 emit pair 相关的权重做 zero-padding（4 -> 6 维）
- 新增维度填零 = 这些新特征对 strong opponent 的推理不产生影响

### 10.4 Shaping Anneal 实现方案

通过环境变量实现分阶段 shaping 不够灵活（进程启动时就固定了）。方案：

**在训练脚本中分段启动**：
- 第一阶段（0-500 upd）：带 CAPTURE=0.05 + PROD_SHARE_DELTA=0.02 启动，`num_updates=500`
- 第二阶段需要从 ckpt 500 resume，减半 shaping 系数，再跑 500 upd
- 第三阶段从 ckpt 1000 resume，shaping=0，跑到 3000

优点：不需要改 reward.py 代码，用现有的 `--resume-from` 机制即可。

### 10.5 配置要点

| 参数 | 值 | 说明 |
|------|----|------|
| 特征维度 | 33 planet + 18 global + 6 emit pair | 沿用 f37 |
| num_updates | 3000（分三段） | ~100M steps |
| strong_ckpt_path | f29@599 (第三阶段启用) | 外部锚点 |
| strong_ratio | 0.20 (第三阶段) | 外部对手比例 |
| frozen_ratio | 0.40 | 自我对弈 |
| PPO 配方 | 同 f37 | 不变 |

### 10.6 风险评估

| 风险 | 概率 | 应对 |
|------|------|------|
| CAPTURE shaping 再次诱导 spam 发兵 | 中 | CAPTURE 只奖励翻转（不奖励发兵），且只用 500 upd 就退火 |
| shape adapter 导致 strong opponent 行为异常 | 低 | 新维度填零不影响原有推理；可通过 parity test 验证 |
| 三段训练中间 resume 失败 | 低 | resume 机制已在 f33 验证过 |
| 退火后仍塌缩 | 中 | 如果阶段一学会翻转但退火后丢失，考虑保留微量 CAPTURE=0.005 |

---

## 11. Top 选手经验总结 + 竞赛方案对比

### 11.1 Lin Myat Ko（第 7 名）关键信息

| 主题 | 原文要点 | 对我们的启示 |
|------|---------|------------|
| **Reward** | "+1 -1 is enough for 2p mode" | 终局 reward 足够，但前提是模型能在早期学会翻转 |
| **Self-play** | "About 100M samples with pure self-play should beat all public agents by 90%" | 我们已跑 ~70M steps（f37 @u2083），还未 beat v20 |
| **Architecture** | "entity transformer", "600K params" -> SPS 10K | 我们 680K params, SPS 16K — 架构规模相当 |
| **Feature engineering** | "put as many inductive bias as possible" | 特征方向正确，但需要配合 reward curriculum |
| **Stability** | Opus (AI) "shipped 7 changes... couldn't tell which one was responsible" | 一次只改一个变量 |
| **clip_frac** | "your most reliable warning sign... 0.10 -> 0.30+ over a few million" | 我们 clip_frac 始终 <0.02，训练稳定但策略不进步 |
| **Explained variance** | "should go up to at least 0.8 in 100 iters" | 我们 @u49 已达 0.92 — value head 正常 |

### 11.2 其他竞赛参与者的经验

**alekh（第 135 名）** 遇到与我们相同的问题：
> "entropy stays fine, its more the winrate against the eval set that collapses… could be something about the distribution shift when new self-play snapshots enters the opponent pool"

这与我们的"训练指标正常但 replay 塌缩"完全一致。Lin Myat Ko 的回应只是"self-play should work fine for this game"——说明他的系统有某些我们没有的东西（可能是更好的特征让早期策略不退化）。

### 11.3 Lux AI 冠军方案对比

| 方案 | Reward 策略 | Self-play 策略 | 关键技术 |
|------|-----------|---------------|---------|
| **Lux AI S1 冠军** (IsaiahPressman) | Shaped reward 前 20M steps -> sparse reward | Teacher-student + KL 约束 | IMPALA + UPGO + TD-lambda |
| **Lux AI S3 第 5 名** (Kiwis) | 分三阶段：小地图 dense -> 大地图 dense -> 大地图 sparse | 先 teacher 后 self-play | xLSTM backbone |
| **Lux AI S3 第 1 名** (tonykozlovsky) | 动态 reward scaling + adaptive entropy | Teacher-student + opponent pool | ResNet + ConvLSTM |
| **我们 (OrbitWar f37)** | 纯 +1/-1 从头开始 | Self-play, 无外部锚点 | Entity Transformer + PPO |

共同模式：**所有成功方案都用了 shaped -> sparse 的 curriculum**。没有任何冠军方案从 pure +1/-1 cold start。

### 11.4 学术研究启示

| 研究 | 核心发现 | 对 f38 的指导 |
|------|---------|-------------|
| **AlphaStar** (Nature 2019) | PFSP + League（Main/Exploiter 角色）防止策略循环 | 我们资源不支持 league，用 external anchor 模拟 |
| **OpenAI Five** (2019) | 80% 最新 + 20% 历史策略 | 已有 frozen_ratio=0.40，但缺外部锚点 |
| **SPIRAL** (2026) | Role-conditioned Advantage Estimation 防止 "thinking collapse" | 对称零和博弈，RAE 可简化为标准 advantage |
| **Data Gating** (arxiv:2605.22217, 2026) | 数据管门 > reward 设计 | 考虑在 f38 之后加 trajectory 质量过滤 |
| **Entropy Preservation** (2026) | PPO clipping 隐式约束 entropy 变化率 | 我们 entropy 稳定，非当前瓶颈 |

---

## 12. Day10 修订后执行决策树

```
f38 三阶段课程训练
│
├── 阶段一（0-500 upd）: CAPTURE=0.05 + PROD_SHARE_DELTA=0.02
│   ├── @199: 看 flip > 3%，captures > 20
│   │   ├── 有改善 -> 继续到 @499
│   │   └── 无改善 -> 加大 CAPTURE=0.10
│   └── @499: 看是否学会翻转
│       ├── flip > 5% + captures > 40 -> 进入阶段二
│       └── 否 -> 止损，重新审视特征/架构
│
├── 阶段二（500-1000 upd）: Anneal shaping + self-play
│   ├── @750: 看 shaping 减半后是否保持翻转能力
│   └── @999: 进入阶段三前的 replay gate
│       ├── WLD >= 1/5 vs v20 -> 继续
│       └── WLD 0/5 + 翻转能力丢失 -> 保留微量 CAPTURE=0.005
│
├── 阶段三（1000-3000 upd）: 纯终局 + f29 anchor
│   └── @1999, @2999: 全面 replay 评估
│       ├── WLD >= 2/5 -> 进入提交候选
│       └── WLD 0/5 -> 考虑 data gating / trajectory filtering
│
└── 提交候选
    └── 与 f29@599 对比，取更强者提交
```

---

## 13. f38 Stage 1 Replay 结果 + Gate 决策（2026-06-02）

### 13.1 f38 s1 replay（vs v20，first-80）

| ckpt | WLD | z0 | e2+ | spf | flip | captures | bin0 | one_ship |
|------|-----|----|-----|-----|------|----------|------|----------|
| @199 | 0/5/0 | 4.5% | **15.2%** | 3.14 | 2.13% | **19** | 2.4% | 29.5% |
| @499 | 0/5/0 | 1.4% | 0.8% | 3.56 | 2.07% | 10 | 1.3% | 34.5% |

对比 f37 同 ckpt：

| ckpt | 指标 | f38 s1 | f37 | 趋势 |
|------|------|--------|-----|------|
| @199 | e2+ | 15.2% | 14.8% | f38 略优 |
| @199 | captures | **19** | 9 | f38 明显优 |
| @199 | flip | 2.13% | 1.53% | f38 优 |
| @499 | e2+ | 0.8% | 0.2% | 两者均塌缩 |
| @499 | captures | 10 | 20 | f38 更差 |

### 13.2 Stage 1 Gate 判定

| Gate | @199 标准 | @199 实际 | @499 标准 | @499 实际 |
|------|----------|----------|----------|----------|
| flip | > 3% | 2.13% **FAIL** | > 5% | 2.07% **FAIL** |
| captures | > 20 | 19 **FAIL** | > 40 | 10 **FAIL** |
| e2+ | > 5% | 15.2% OK | > 5% | 0.8% FAIL |

**结论**：@199 接近达标（captures 差 1、flip 差 0.9pp），特征+shaping 方向正确；@499 再次塌缩，**不进入 Stage 2/3**（从 @499 resume 无意义）。

### 13.3 决策：Path B — 启动 f38s1b

按决策树「@199 有改善但 @499 未达标 → CAPTURE=0.10 重训」：

| 项 | 值 |
|---|---|
| 实验 | **f38s1b** |
| CAPTURE | 0.10（s1 的 2x） |
| PROD_SHARE_DELTA | 0.02（不变） |
| 配置 | [`multi_action_v11_f38s1b.yaml`](orbit_wars_rl/configs/multi_action_v11_f38s1b.yaml) |
| 脚本 | [`scripts/run_v11_f38s1b.sh`](scripts/run_v11_f38s1b.sh) |
| Gate | @199/@499 replay，目标 captures@199>20、flip@499>5% |

### 13.4 配套改动

- 新增 [`submission_rl_v11_f38.py`](submission_rl_v11_f38.py) 提交模板（arch 同 f37）
- 修复 [`scripts/quick_replay.sh`](scripts/quick_replay.sh) 空 `EXPORT_ARGS` 在 `set -u` 下崩溃
- Stage 2/3 **暂缓**，待 f38s1b gate 通过后再 resume 课程

---

## 14. f38s1b 结果 + f38 路线关闭（2026-06-02 晚）

### 14.1 实验设计

| 项 | f38 s1 | f38s1b |
|---|---|---|
| CAPTURE | 0.05 | **0.10** |
| PROD_SHARE_DELTA | 0.02 | 0.02 |
| updates | 500 | 500 |
| ckpt_dir | `ckpt_multi_action_v11_f38` | `ckpt_multi_action_v11_f38s1b` |

### 14.2 replay（vs v20，first-80）

| ckpt | 线 | WLD | z0 | e2+ | spf | flip | captures | bin0 | one_ship |
|------|-----|-----|----|-----|-----|------|----------|------|----------|
| @199 | f38 s1 | 0/5/0 | 4.5% | **15.2%** | 3.14 | 2.13% | **19** | 2.4% | 29.5% |
| @199 | f38s1b | 0/5/0 | 5.0% | 5.3% | 2.99 | 1.63% | **8** | 0.7% | 36.1% |
| @499 | f38 s1 | 0/5/0 | 1.4% | 0.8% | 3.56 | 2.07% | 10 | 1.3% | 34.5% |
| @499 | f38s1b | 0/5/0 | 0.6% | **20.2%** | 2.73 | 1.68% | 14 | 1.2% | 23.3% |

训练 log 末段（f38s1b @499）：`emits≈1.49 / spf≈21 / e2≈0.30 / z0≈0.03` — 与 replay 再次脱节。

### 14.3 Gate 判定

| Gate | @199 目标 | f38s1b @199 | @499 目标 | f38s1b @499 |
|------|----------|-------------|----------|-------------|
| captures | > 20 | 8 **FAIL** | > 40 | 14 **FAIL** |
| flip | > 3% | 1.63% **FAIL** | > 5% | 1.68% **FAIL** |
| e2+ | > 5% | 5.3% OK | > 5% | 20.2% OK |

**结论**：加大 CAPTURE **未改善 @199 峰值**，反而丢失 s1 的 captures/e2+ 信号；@499 虽 e2+ 升高，但 **flip/captures/WLD 全线 fail**。**f38 全系归档**，不再 resume Stage 2/3，也不再调 shaping 系数。

**可保留资产**：f38 s1 @199 ckpt 仍可作为 f40 的行为多样性 anchor 候选（e2+=15.2%, captures=19）。

---

## 15. f40 Expert-Seeded Gated League — 基建与 smoke（2026-06-02 晚）

### 15.1 已实现

| 模块 | 改动 |
|------|------|
| BC 采集 | [`collect_data.py`](../orbit_wars_rl/bc/collect_data.py) 保存 `planet_x/y_raw`、`home_idx_raw` |
| BC 训练 | [`train_bc.py`](../orbit_wars_rl/bc/train_bc.py) 真实几何 + masked-label CE + emit 正类权重 |
| Runner | [`runner.py`](../orbit_wars_rl/ppo/runner.py) 增加 `pool_seed_paths`、`snapshot_current`、`buffer_rollout_ratio` |
| 配置/脚本 | `multi_action_v11_f40_buffer.yaml`、`collect_f40_expert_data.sh`、`run_f40_bc.sh`、`run_v11_f40_buffer.sh` |
| Export | BC replay 可 `--emit-hard-stop 0 --flip-hard-mask 0` 导出 |

### 15.2 数据与 ckpt

| 产物 | 规模 |
|------|------|
| `data/bc_f40_v20_self.npz` | 20 局 v20，4972 samples |
| `data/f40_mixed_states.npz` | 2160 states（v20 + top10 均衡采样） |
| `ckpt_bc_f40/ckpt_final.pkl` | BC seed（emit_pos_weight=4.0） |
| `ckpt_multi_action_v11_f40_buffer_smoke/ckpt_000004.pkl` | PPO smoke 5 upd |

### 15.3 replay gate（vs v20，first-80）

| tag | WLD | z0 | e2+ | spf | flip | captures | 判定 |
|-----|-----|----|-----|-----|------|----------|------|
| v11_f40_bc_seed | 0/5/0 | 3.3% | 3.0% | 4.61 | 2.44% | ~16 | **未过 BC gate**（目标 captures>40, e2+>10%） |
| v11_f40_buffer_smoke_u4 | 0/5/0 | 3.2% | **9.8%** | 5.91 | 2.94% | ~18 | e2+ OK；flip/captures fail |

PPO smoke 训练 5 upd 内：`emits≈2.56 / e2≈0.50 / spf≈19.4` — buffer curriculum 能恢复多路 emit，但 replay 翻转仍弱。

### 15.4 尚未完成（移交 Day11）

- BC 数据扩至 **200 局**；BC gate 过关后再长训
- replay 驱动的 **gated pool 自动入池** + @100 回退
- buffer step 20–120 分桶
- f40 长训 500–1000 upd + 每 100 upd replay

---

## 16. Day10 总结与路线决策

### 16.1 已证伪（Day10 追加）

| 假设 | 反证 |
|------|------|
| CAPTURE 加倍可推过 f38 @199 gate | f38s1b @199 captures 8 < s1 的 19 |
| f38 三阶段 curriculum 可继续 | s1 + s1b 均在 @499 前后 lose 有效翻转信号 |
| 仅靠 shaping + self-play 可打破 trickle | f33–f38s1b 全线 **WLD 0/5** |

### 16.2 仍有效的正信号

| 信号 | 证据 |
|------|------|
| f37/f38 @199 的 e2+ 峰值 | 14.8%–15.2%，说明特征方向对 |
| f38 s1 @199 captures | 19，全系列最高之一 |
| buffer curriculum | f40 smoke 训练 e2≈50%，replay e2+=9.8% |
| f29 @599 | 仍是动作结构最健康的提交基线 |

### 16.3 路线决策

| 线 | 决策 |
|----|------|
| **f38/f37 shaping 线** | **归档**；最多保留 s1@199 / f37@199 作 f40 anchor |
| **f40** | **新主线**：BC seed → buffer PPO → gated league |
| **Gate 口径** | 唯一 promote：**replay vs v20**；训练 log 仅健康检查 |

### 16.4 下一步

见 [`DAY11_PLAN.zh.md`](DAY11_PLAN.zh.md)。
