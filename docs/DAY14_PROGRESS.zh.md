# DAY14 — 训练进展追踪 & 代码聚焦整理

> **2026-06-10 Day14**  
> 三条训练线并行推进，v14 新架构正式上线训练。  
> 代码归档整理完毕，项目聚焦于 v12/v13/v14 三条活跃路线。

---

## 0. Day13 回顾与成就

Day13 完成了从 Frog Parade (Lux S3 #2) 参考架构的全面重构：
- ✅ v12_lux_b 大模型 (256d/4L/8H, ~3-5M params) 对称 self-play 稳定训练
- ✅ 纯 sparse ±1 reward + zero-sum value head 方案验证可行
- ✅ clip_frac 从 0.47 修正到 0.10 区间，训练稳定
- ✅ v13c_hold 实验完成 10000 updates，获取 hold action 基线数据
- ✅ v14 新配置 (16 PCT bins, 精细兵力分配) 设计并上线

---

## 1. 远端训练进展 (截至 2026-06-10 20:30)

### 1.1 v12_lux_b — 主力训练线 (进行中)

| 指标 | 当前 (u7428) | u7399 | u8199 | u8999 | u9199 |
|------|-------------|-------|-------|-------|-------|
| update | 7428 | 7399 | 8199 | 8999 | 9199 |
| steps | 30.4M | 30.3M | 33.6M | 36.9M | 37.7M |
| sps | 1290 | 1293 | — | — | — |
| ev | 0.92~0.97 | 0.95 | — | — | — |
| spf (self) | 22~27 | 26.0 | 26.9 | 25.6 | 20.1 |
| garr (self) | 35~48 | 47.4 | 44.7 | 45.5 | 36.4 |
| z0 (self) | 0.02~0.08 | 0.03 | 0.04 | 0.01 | 0.04 |
| e2 rate | 0.30~0.36 | 0.30 | 0.30 | 0.35 | 0.29 |
| clip | 0.08~0.11 | 0.09 | — | — | — |
| kl | 0.007~0.022 | 0.008 | — | — | — |

**vs v20 评测趋势 (训练中自动 3 局)**:

| Checkpoint | spf | garr | flip% | e2+% | bin0% | WLD |
|-----------|-----|------|-------|------|-------|-----|
| u5899 | 4.89 | 33.4 | 1.94 | 3.8 | 1.2 | 0/3/0 |
| u5999 | 4.50 | 36.6 | 2.31 | 10.0 | 6.4 | 0/3/0 |
| u6399 | 3.96 | 20.9 | 3.05 | 5.4 | 1.8 | 0/3/0 |
| u6999 | 3.50 | 7.2 | 2.35 | 7.9 | 2.5 | 0/3/0 |
| u7199 | 3.89 | 14.6 | 2.17 | 7.1 | 2.0 | 0/3/0 |
| u7399 | 3.36 | 7.4 | 1.49 | 10.4 | 11.1 | 0/3/0 |

**分析**:
- 自我对弈指标健康: spf 20~27, garr 35~48, ev > 0.92
- 但 vs v20 始终全输 (WLD=0/3/0)，说明 **self-play 策略空间与 v20 策略差距大**
- vs v20 的 spf~3.5 (太低), garr~7-15 (太少) → 模型被 v20 压制
- 可能需要阶段二 shaping 介入，或尝试 vs v20 mixed training

### 1.2 v13c_hold — 已完成 ✅

| 指标 | 最终 (u9999) |
|------|-------------|
| 训练总量 | 20.5M steps |
| WR vs random | 90.6% |
| spf (self) | 79.0 |
| z0 (self) | 0.75 |
| garr (self) | 197.3 |
| emits | 0.58 |
| entropy [s/d/p/e] | 0.15/0.43/0.14/0.42 |

**分析**:
- z0=0.75 表明模型 75% 回合选择跳过 (hold)，极端保守
- spf=79 (超大舰队)、garr=197 (超高囤兵) → 典型的"囤到爆"策略
- vs v20 评测失败 (template/ckpt 架构不匹配)
- **结论**: hold action 导致策略坍缩为过度保守，不适合当前路线
- **处置**: 保留 checkpoint 数据，不继续训练

### 1.3 v14 — 精细化兵力分配 (新启动)

| 指标 | 当前 (u1999) |
|------|-------------|
| 训练总量 | 4.1M steps |
| sps | 2782 |
| WR vs random | 91% |
| spf (self) | 60~90 |
| z0 (self) | 0.69~0.80 |
| garr (self) | 138~210 |
| emits | 0.45~0.56 |
| entropy [s/d/p/e] | 0.98/0.74/1.71/0.51 |
| clip | 0.14 (偏高) |

**核心变化 (vs v12_lux_b)**:
- NUM_PCT_BINS: 8 → 16 (5%~100%, 低区间 5% 等距)
- min_pct_bin: 2 → 0 (移除 30% 下限硬规则)
- pct pair features: 2 → 6 dims
- PctHead hidden: 64 → 128
- 轻量 reward shaping: PROD_SHARE_DELTA + CAPTURE + RELEASE
- lr: 5e-5 (更保守), update_epochs: 1

**已知问题**:
- ⚠️ vs v20 评测全部失败: `pct_head/fc1 input dim=775` 不匹配
  - 原因: v14 的 pct pair features 从 2→6 dims，export_submission 的 `infer_arch_from_flat` 未适配
  - 需要修复 `orbit_wars_rl/inference/weights.py` 中的维度推断逻辑
- z0=0.69~0.80 偏高 (类似 v13c 的保守倾向)，可能因为 hold=true + min_pct_bin=0 导致
- clip=0.14 (略高于 0.10 目标), 训练还在早期阶段

---

## 2. 评测命令速查

### 2.1 v12_lux_b 评测 (当前只有 u7399 的 pkl)

```bash
# 1. 同步最新 checkpoint 到本地
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/ckpt_multi_action_v12_lux_b/ ckpt_multi_action_v12_lux_b/

# 2. 在服务器上评测最新 checkpoint vs v20
ssh charlie@www.ultrapp.online
cd ~/project/OrbitWarRL
LATEST=$(ls -1 ckpt_multi_action_v12_lux_b/ckpt_*.pkl | sort | tail -1)
NUM_GAMES=5 bash scripts/v12_eval_gate.sh "$LATEST" v12b_latest

# 3. 批量评估所有 checkpoint
bash scripts/v12_milestone_eval.sh ckpt_multi_action_v12_lux_b

# 4. 快速 replay (本地可以跑)
bash scripts/quick_replay.sh ckpt_multi_action_v12_lux_b/ckpt_007399.pkl v12_u7399

# 5. 生成 HTML replay 可视化
bash scripts/v12_replay_html.sh ckpt_multi_action_v12_lux_b/ckpt_007399.pkl 0
bash scripts/v12_replay_html.sh ckpt_multi_action_v12_lux_b/ckpt_007399.pkl 42
```

### 2.2 v14 评测 (需先修复 export 适配)

```bash
# v14 当前 eval_vs_v20 失败，需要先修复 weights.py
# 修复后可用:
ssh charlie@www.ultrapp.online
cd ~/project/OrbitWarRL
LATEST=$(ls -1 ckpt_multi_action_v14/ckpt_*.pkl | sort | tail -1)
NUM_GAMES=5 bash scripts/v12_eval_gate.sh "$LATEST" v14_latest
```

### 2.3 远程监控

```bash
# 查看 v12_lux_b 最新训练行
ssh charlie@www.ultrapp.online "tail -5 ~/project/OrbitWarRL/logs/v12_lux_b.log"

# 查看 v14 最新训练行
ssh charlie@www.ultrapp.online "tail -5 ~/project/OrbitWarRL/logs/v14.log"

# 查看 v14 eval 报错
ssh charlie@www.ultrapp.online "grep 'eval_vs_v20' ~/project/OrbitWarRL/logs/v14.log | tail -3"

# 查看所有在跑的训练进程
ssh charlie@www.ultrapp.online "ps aux | grep 'orbit_wars_rl.scripts.train' | grep -v grep"

# 同步所有日志到本地
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/logs/ logs/
```

---

## 3. 代码归档整理

已将与当前路线无关的代码归档到 `_archive/` 目录:

| 归档类别 | 文件数 | 说明 |
|---------|--------|------|
| scripts_legacy | 54 | v11 评测/训练脚本, BC pipeline, v9 消融 |
| configs_legacy | 64 | MVP ~ v11 系列 YAML 配置 |
| submissions_legacy | 80 | v1~v11 导出的 submission agent |
| docs_legacy | 16 | Day1~Day9 进展日记, 旧 runbook |
| references_legacy | — | halite3, planet-wars, lux-s2 参考代码 |
| kaggle-lux-2024 | — | Frog Parade 参考代码 (已提取完) |
| bc_module | — | Behavior Cloning (Day3 取消) |
| 旧 checkpoint 目录 | 6 | mvp, selfplay, v11_f26/f29/f38/f44 |

**精简后项目结构**:

```
OrbitWarRL/
├── orbit_wars_rl/           # 核心 RL 代码
│   ├── env/                 # JAX 游戏环境
│   ├── features/            # 特征编码 (planet 28d, fleet 10d, global 17d)
│   ├── net/                 # 模型 (EntityTransformer + heads)
│   ├── ppo/                 # PPO (rollout, rollout_symmetric, update, runner)
│   ├── selfplay/            # Self-play pool
│   ├── inference/           # NumPy 推理 (Kaggle 提交用)
│   ├── eval/                # v20 mini gate
│   ├── parity/              # 环境一致性检查
│   ├── scripts/             # 核心工具脚本
│   └── configs/             # 4 个活跃配置 (v12_lux, v12_lux_b, v13_hold, v14)
├── scripts/                 # Shell 启动脚本 (11 个)
├── docs/                    # 活跃文档 (Day10+, OVERVIEW, 评测 runbook)
├── ckpt_multi_action_v12_*/ # v12 checkpoints
├── ckpt_multi_action_v13*/  # v13 checkpoints
├── logs/                    # 训练日志
├── _archive/                # 归档代码 (不影响训练)
└── submission_rl_*.py       # 活跃 submission + 训练 checkpoint exports
```

---

## 4. 下一步计划

### 4.1 紧急: 修复 v14 的 eval_vs_v20

v14 的 pct_head 使用了 6 维 pair features (vs v12 的 2 维)，导致 `infer_arch_from_flat` 维度推断失败。需要:
1. 在 `orbit_wars_rl/inference/weights.py` 的 `infer_arch_from_flat` 中增加 pct pair features=6 的分支
2. 创建 v14 专用 submission template
3. 验证 export → parity → eval 全链路

### 4.2 v12_lux_b 决策点

当前 u7428/10000, 已跑 74%。vs v20 始终 0 胜：
- **选项 A**: 继续到 10000 看是否有突破
- **选项 B**: 从当前 best ckpt resume, 引入微量 shaping (CAPTURE=0.02)
- **选项 C**: 引入 v20 作为 opponent mixture (10% v20 + 90% symmetric)

### 4.3 v14 观察窗口

u1999 仍处于早期，需要等到 u3000~5000 才能判断:
- 关注 z0 是否从 0.75 下降 (否则重蹈 v13c 覆辙)
- 关注 clip_frac 是否稳定到 < 0.10
- 16 bin PCT 分布是否有意义分化 (不只聚集在 bin0)

---

## 5. 关键发现

1. **v13c_hold 已证明 hold action 有害**: z0=75%, spf=79, garr=197 → 极端保守策略坍缩
2. **v12_lux_b 的 self-play 与 v20 策略鸿沟**: 模型在 self-play 中发展出独特策略 (spf~25, garr~40), 但面对 v20 的压制完全不适应 (spf 降到 3-4)
3. **v14 是当前最有前景的方向**: 更精细的兵力分配 (16 bins) + 轻量 shaping, 有望弥合 self-play 与 v20 的差距
4. **eval export 链路需要跟进架构变更**: 每次改动 action space / pair features 后都要同步更新 inference 和 submission template
