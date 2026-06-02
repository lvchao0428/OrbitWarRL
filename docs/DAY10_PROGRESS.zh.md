# DAY10 进展 — f36a/f36b 分析 + f37 方案启动

> **2026-06-02 更新**
> 接续 [`DAY9_PLAN.zh.md`](DAY9_PLAN.zh.md)。
> f36a/f36b 训练 + replay 完成；两条支线均未达到 promote 标准。
> 根据对弈分析和根因诊断，提出并启动 f37 方案。

---

## 0. TL;DR — 当前状态

| 维度 | 状态 |
|---|---|
| **f36a** | 🔴 **WLD 0/5 全 ckpt**；`emit_hard_stop_min_step=2` 未打破单路 emit |
| **f36b** | 🔴 **strong opponent 被静默禁用**（shape mismatch 33 vs 28）；实际退化为 f33-like 短训 |
| **根因诊断** | 问题不在动作机制或训练分布，而是 **四个 action head 缺乏足够直接的输入信号** |
| **Vadasz 对弈分析** | 顶级选手 z0=34%、e2+=37%、spf 中位 25；我们 z0<1%、e2+=0%、spf≈5-17 |
| **f37 方案** | ✅ 已实施并启动训练：**特征强化 + 纯终局 reward + 3000 upd** |
| **提交候选** | **f29 @599** 仍为基线；f37 待训练完成后评估 |

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
