# DAY14 — 训练进展追踪 & 代码聚焦整理

> **2026-06-10 Day14（23:00 更新）**  
> v14 系列经历 v14 → v14b → **v14c** 三轮迭代；export 链路已修复。  
> **当前主线：v14c**（worth_it-gated hold），5090 训练中。  
> **对比基线：v13c_final**（唯一曾赢 v20 的版本）+ v20。

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

### 1.3 v14 系列 — 精细化兵力分配（三轮迭代）

#### 架构共性（v14 / v14b / v14c）

- NUM_PCT_BINS: 16 (5%~100%)
- pct pair: 6-dim；PctHead hidden=128；d_model=256 / 4L
- 39-dim planet / 24-dim global（同 v13c_final 编码）
- symmetric self-play + zero_sum_value + emit_hard_stop + flip_hard_mask
- shaping: CAPTURE=0.08, ONE_SHIP_PENALTY=0.03, min_pct_bin=5

#### v14 — 已停（u~12299，策略坍缩）

| 阶段 | 自博弈 spf | z0 | 行为 |
|------|-------------|-----|------|
| u8599 乱射期 | ~5 | ~1% | 每回合 1 艘 spam |
| u12199 囤兵期 | ~128 | ~67% | allow_hold 坍缩，bin7=100% |

vs v20（u10199，修正 export 前）: **0W/4L**，囤兵特征明显。

#### v14b — 已停（u~1599，forced emit 失败）

改动: `allow_hold=false`（有兵必发）+ min_pct_bin=5。

| 指标 | 自博弈 (u1385) | vs v20 u1199（修正 export） |
|------|---------------|---------------------------|
| spf | ~55 | **4.65** |
| z0 | ~0.02 | **0.6%** |
| garr | ~50 | 12.6 |
| flip | — | **3.67%** |
| WLD | — | **0/4/0** |

**结论**: 反囤兵成功，但 forced t=0 导致微 spam（spf≈3–5），对 v20 完全无效。

#### v14c — 当前主线 🔄（worth_it-gated hold）

改动: `allow_hold=true` + **`force_emit_worth_it=true`**（有好目标才强制 t=0 发射）。

| 指标 | u86 (自博弈) | u99 vs v20 |
|------|-------------|------------|
| spf | 30~42 | **13.76** |
| z0 | 0.27~0.38 | **28.4%** |
| garr | 24~32 | **37.55** |
| flip | — | **10.88%** |
| emits | 0.63~0.75 | — |
| WLD | — | **0/3/0** |

启动: 2026-06-10 22:56，seed=1420，日志 `logs/v14c.log`，ckpt `ckpt_multi_action_v14c/`。

**u99 初判（vs v20）**:
- flip=10.88% 已接近 v13c（11%），远高于 v14b（3.7%）
- spf=13.76 仍偏低（v13c=64），但比 v14b 的 4.65 好 3×
- z0=28.4% 合理（可 hold，非 v14b 式强制 spam 也非 v13c 式 45% 囤兵）
- 仍 0 胜，需继续观察到 u500~1000

#### 三线对比 vs v20（first-80，修正 export 后）

| 版本 | W/L/D | spf | garr | flip% | z0% | 备注 |
|------|-------|-----|------|-------|-----|------|
| **v13c_final** | **1/4/0** | 64.4 | 81.7 | 11.0 | 44.7 | 唯一胜场；战术有瑕疵（射太阳等） |
| v14b u1199 | 0/4/0 | 4.7 | 12.6 | 3.7 | 0.6 | forced emit 微 spam |
| **v14c u99** | 0/3/0 | 13.8 | 37.6 | **10.9** | 28.4 | 早期；flip 已对齐 v13c |
| v20 (对手) | — | ~20 | ~1700+ | ~24 | ~70 | 参考 |

**解读**: v13c 是「能赢但粗糙」的上界参考；v14c u99 在 flip 上已追上 v13c，但 spf/胜场仍差。目标是在 v14c 框架下学到 v13c 的进攻力度，同时避免 v13c 的战术 bug。

#### Export 链路修复 ✅（Day14 下午）

根因: `submission_rl_v13c.py` 模板默认 `ALLOW_HOLD=1`，与 v14b 训练配置不一致 → 训练内 `[eval_vs_v20]` 从 u299 起 spf=0 全假。

修复:
1. `export_submission.py` 支持 `--allow-hold` / `--force-emit-worth-it` / `--min-pct-bin`
2. `quick_replay.sh` 从 ckpt `.meta.json` 的 `export` 块自动读取
3. `runner.py` 保存 ckpt 时写入 export meta
4. `submission_rl_v13c.py` 新增 `FORCE_EMIT_WORTH_IT`；16-bin 不再重置 MIN_PCT_BIN=0

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

### 2.2 v14c 评测（当前主线）

```bash
# 远程监控训练 + 自动 eval
ssh charlie@www.ultrapp.online "tail -3 ~/project/OrbitWarRL/logs/v14c.log"
ssh charlie@www.ultrapp.online "grep eval_vs_v20 ~/project/OrbitWarRL/logs/v14c.log | tail -5"

# 手动评测最新 ckpt vs v20（export flags 从 .meta.json 自动读取）
ssh charlie@www.ultrapp.online
cd ~/project/OrbitWarRL
LATEST=$(ls -1 ckpt_multi_action_v14c/ckpt_*.pkl 2>/dev/null | sort | tail -1)
NUM_GAMES=4 bash scripts/quick_replay.sh "$LATEST" v14c_latest

# 与 v13c 基线对比（同脚本，换 agent-a）
python3 -m orbit_wars_rl.scripts.replay_analyze \
  --agent-a submission_rl_v13c_final.py \
  --agent-b submission_v20_0513.py \
  --num-games 4 --seed-base 0 \
  --out logs/replay_analyze/v13c_final_vs_v20.json
```

### 2.3 v13c 基线评测

v13c_final 是目前**唯一对 v20 有胜场**的版本，作为 v14 系列的战术参考上界：

| 指标 | v13c_final | 含义 |
|------|-----------|------|
| WLD | 1/4/0 | 25% 胜率 |
| spf | 64.4 | 大舰队进攻 |
| flip | 11.0% | 有效 flip |
| z0 | 44.7% | 偏保守但非极端囤兵 |

已知缺陷: 射太阳、离屏发射、节奏混乱 — v14c 目标是在 flip 接近的前提下修复这些。

### 2.4 远程监控

```bash
# 查看 v14c 最新训练行
ssh charlie@www.ultrapp.online "tail -5 ~/project/OrbitWarRL/logs/v14c.log"

# 查看 v14c eval 趋势
ssh charlie@www.ultrapp.online "grep 'eval_vs_v20' ~/project/OrbitWarRL/logs/v14c.log"

# 查看 v12_lux_b 最新训练行（已停或低速）
ssh charlie@www.ultrapp.online "tail -5 ~/project/OrbitWarRL/logs/v12_lux_b.log 2>/dev/null"

# 一键观测 v14c + v13c 基线 + 最新 eval
bash scripts/monitor_v14c.sh
bash scripts/monitor_v14c.sh --watch   # 每 10 分钟轮询

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
│   └── configs/             # 活跃配置 (v12_lux_b, v13_hold, v14b, v14c)
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

### 4.1 v14c 观察窗口（当前优先）

每 100 updates 自动 `[eval_vs_v20]`，同时对照 v13c 基线：

| 里程碑 | 关注指标 | 决策 |
|--------|---------|------|
| u500 | spf>20, flip>8%, 出现胜场 | 继续 |
| u1000 | spf>30, WLD≥1/4 | 可延长训练 |
| u2000 | 仍 0 胜且 spf<15 | 考虑 resume + shaping 微调 |

### 4.2 v12_lux_b

已跑至 u~9200，vs v20 始终 0 胜。资源让给 v14c，暂不续训。

### 4.3 持续观测

5090 上 v14c 训练进行中，eval 每 u100 自动写入 `logs/v14c.log` 的 `[eval_vs_v20]` 行。  
观测命令见 §2.4；新 eval 出现后更新本文 §1.3 v14c 表格。

---

## 5. 关键发现

1. **v13c_final 是战术参考上界**: 1/4 胜 v20，flip=11%，但 spf=64/z0=45% 偏极端，且有射太阳等 bug
2. **v14 坍缩路径**: 乱射 (u8599) → 囤兵 (u12199)；allow_hold 无约束时策略不稳定
3. **v14b forced emit 走不通**: z0 修好但 spf≈4、flip≈3%，对 v20 0 胜
4. **v14c worth_it-gated hold 早期信号积极**: u99 flip=10.9% 对齐 v13c，spf 仍低但方向正确
5. **export 必须与训练配置一致**: ALLOW_HOLD / FORCE_EMIT_WORTH_IT / MIN_PCT_BIN 写入 ckpt meta，否则 eval 全假
6. **对比评测应三角测量**: v14c vs v20（实战）+ v13c vs v20（上界）+ 自博弈指标（训练健康度）
