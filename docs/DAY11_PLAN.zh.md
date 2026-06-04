# DAY11 进展 — f40/f41/f42 完成 + f44 已启动

> **2026-06-03 晚更新** · 后续计划见 [`DAY12_PLAN.zh.md`](DAY12_PLAN.zh.md)
> 接续 [`DAY10_PROGRESS.zh.md`](DAY10_PROGRESS.zh.md)。
> **f40** 500 upd 完成；**f41** 500 upd 完成；**f42** 500 upd 完成；**f43** 方案已落地，待远程训练。

---

## 0. TL;DR

| 维度 | 状态 |
|---|---|
| **f40** | ✅ 500 upd 完成；@99 **flip=5.26% / e2+=24.8% / spf=8.51**（全系列综合最佳）；后续 spf 退化 |
| **f41** | ✅ 500 upd 完成（从 f40 @99 resume）；**spf/flip 大幅改善，e2+ 塌缩** |
| **f41 最佳** | @99: **spf=13.23 / flip=6.23%**（首次 replay 过 flip>6% promote gate）|
| **f42** | ✅ replay 完成；**@49 最佳**（e2+=6.5%, spf=11.47）；@99 训练峰值在 replay 塌缩 |
| **f43** | ⏸ 暂缓（等 f44 验证对齐管线） |
| **f44_align** | 🟡 待训：P0–P4 训练–replay 对齐基建 |
| **核心矛盾** | **spf/flip vs e2+ 此消彼长**：f40 强在 e2+，f41 强在 spf/flip |
| **提交基线** | 仍为 **f29 @599** |
| **WLD** | f40/f41 全 ckpt 0/5/0；f42 replay 待补 |

---

## 1. f40 最终结果（500 upd 完成）

### 1.1 Replay vs v20 (first-80 turns)

| ckpt | WLD | spf | flip | e2+ | z0 | one_ship_rate | min_game_spf |
|------|-----|-----|------|-----|-----|---------------|-------------|
| BC seed | 0/5/0 | 8.30 | 4.01% | 9.3% | 0.5% | 18.7% | 1.14 |
| **@99** | 0/5/0 | **8.51** | **5.26%** | **24.8%** | 0.6% | **5.8%** | 1.90 |
| @199 | 0/5/0 | 6.17 | 5.06% | 28.0% | 0.6% | 23.3% | 1.30 |
| @299 | 0/5/0 | 5.24 | 3.02% | **36.8%** | 0.6% | 14.2% | 1.24 |
| @499 | 0/5/0 | 5.96 | 3.83% | 19.2% | 0.5% | 21.7% | 1.14 |

### 1.2 f40 关键结论

- **@99 综合最佳**：spf/flip/one_ship_rate 三项最优
- **e2+ 先升后降**：@299 达 36.8% 峰值，但 spf 同时退化到 5.24
- **spf 持续退化**：8.51→5.24，模型学会多路但每路变成 trickle
- **z0 始终冻结 <1%**：模型完全没学会等待蓄力
- **flip 先升后降**：@99 的 5.26% 接近 6% promote gate
- **one_ship_rate U 型**：@99 只有 5.8%（从 BC seed 继承），后续退化

---

## 2. f41 实验 — 从 f40 @99 resume + 强化 shaping

### 2.1 设计动机

f40 @99 的 **e2+ 很强（24.8%）** 但 **spf 退化、flip 不够、z0 不动**。f41 三管齐下：

### 2.2 f41 vs f40 改动对照

| 维度 | f40 | **f41** | 理由 |
|------|-----|---------|------|
| CAPTURE | 0.02 | **0.05** | 强化翻转星球奖励 → 提升 flip |
| ONE_SHIP_PENALTY | 0 | **0.01** | 惩罚 ≤3 艘 trickle → 提升 spf |
| HIGH_PROD_CAPTURE | 0 | **0.02** | 额外奖励翻转高产能星球 |
| PROD_SHARE_DELTA | 0.005 | 0.005 | 不变 |
| strong_ratio | 0.20 | **0.25** | 更多 vs f29 强锚点 |
| frozen_ratio | 0.30 | **0.25** | 微减 |
| buffer_rollout | 0.30 | **0.40** | 更多从专家状态学习 |
| random | 0.20 | **0.10** | 减少无效对局 |
| ckpt_every | 100 | **50** | 更密检查点 |
| snapshot_every | 100 | **50** | 同步 |
| Resume | BC seed | **f40 @99** | 保留 e2+=24.8% 初始化 |
| 架构 | 不变 | **不变** | planet=33, global=18, dst_pair=5, emit_pair=6 |

### 2.3 f41 训练概要

| 项 | 值 |
|----|-----|
| config | `orbit_wars_rl/configs/multi_action_v11_f41.yaml` |
| 脚本 | `scripts/run_v11_f41.sh` |
| resume from | `ckpt_multi_action_v11_f40_buffer/ckpt_000099.pkl` |
| num_updates | 500 |
| ckpts | @49, @99, @149, @199, @249, @299, @349, @399, @449, @499 |
| 训练时长 | ~20 min（5090 GPU, SPS ~14K） |

### 2.4 f41 Replay vs v20 (first-80 turns)

| ckpt | WLD | spf | flip | e2+ | z0 | bin0 |
|------|-----|-----|------|-----|-----|------|
| **@49** | 0/5/0 | 10.09 | **7.19%** | 0.8% | 0.6% | 0.0% |
| **@99** | 0/5/0 | **13.23** | **6.23%** | 2.0% | 0.7% | 1.2% |
| @149 | 0/5/0 | 10.31 | **6.89%** | 0.0% | 0.7% | 0.5% |
| @199 | 0/5/0 | 9.61 | **6.56%** | 0.2% | 0.7% | 0.3% |
| @249 | 0/5/0 | 9.07 | 5.54% | 0.8% | 0.6% | 0.3% |
| @299 | 0/5/0 | 8.78 | 5.40% | 0.0% | 0.6% | 0.0% |
| @349 | 0/5/0 | 9.79 | 4.87% | 0.0% | 0.5% | 0.0% |
| @399 | 0/5/0 | 9.30 | **6.56%** | 0.0% | 0.7% | 0.0% |
| @449 | 0/5/0 | 9.18 | 4.03% | 0.0% | 0.5% | 1.3% |
| @499 | 0/5/0 | 8.42 | 5.73% | 0.0% | 0.7% | 0.0% |

### 2.5 f41 vs f40 @99 对比

| 指标 | f40 @99 | **f41 @99** | 变化 |
|------|---------|-------------|------|
| spf | 8.51 | **13.23** | **+55%** ✅ 首次过 spf>10 gate |
| flip | 5.26% | **6.23%** | **+18%** ✅ 首次过 flip>6% promote gate |
| e2+ | **24.8%** | 2.0% | 🔴 从 24.8% 塌缩到 2% |
| z0 | 0.6% | 0.7% | 无变化 |
| bin0 | 0.8% | 1.2% | 正常 |
| WLD | 0/5/0 | 0/5/0 | 无变化 |

### 2.6 f41 训练 log 关键点

| upd | SPS | WRr | WRf | emits | spf(train) | e2(train) | z0(train) |
|-----|-----|-----|-----|-------|-----------|-----------|-----------|
| @49 | 10K | 0.84 | 0.59 | 1.19 | 30.2 | 0.17 | 0.02 |
| @99 | 12K | 0.59 | 0.41 | 1.33 | 34.4 | 0.25 | 0.07 |
| @199 | 14K | 0.88 | 0.59 | 1.36 | 28.6 | 0.25 | 0.03 |
| @299 | 14K | 0.62 | 0.59 | 1.27 | 30.7 | 0.21 | 0.05 |
| @399 | 15K | 0.75 | 0.38 | 1.43 | 31.9 | 0.27 | 0.01 |
| @499 | 15K | 0.81 | 0.59 | 1.39 | 34.8 | 0.26 | 0.00 |

### 2.7 f41 核心分析

**成功**：ONE_SHIP_PENALTY + CAPTURE 提升显著提升了 spf（8.5→13.2）和 flip（5.3%→7.2%）。
f41 @49/99/149/199 的 flip 全部过了 6% promote gate——这是全系列 f33-f41 的首次突破。

**问题**：e2+ 从 f40 @99 的 24.8% 暴跌到 0-2%。模型在学会发更大舰队后，从多路退化为单路。
ONE_SHIP_PENALTY 可能间接惩罚了多路小舰队（第 2/3 路通常更小），导致模型选择集中一路发更大的。

**spf/flip vs e2+ 此消彼长的 tradeoff**：
- f40：多路小舰队（e2+=24.8%, spf=8.5, flip=5.3%）
- f41：单路大舰队（e2+=2%, spf=13.2, flip=6.2%）

两个策略都无法赢 v20。Vadasz 的模式是"蓄力→多路大舰队爆发"，需要同时做到 e2+ 高 + spf 高。

---

## 3. 执行命令（远程 5090）

> 默认：`cd /home/charlie/project/OrbitWarRL`，`PYTHON=/home/charlie/anaconda3/bin/python`

### 3.1 同步代码

```bash
# 本地
bash sync_mirror_ultrapp.sh
```

### 3.2 f41 训练（已完成）

```bash
# 已于 2026-06-03 16:37 启动并完成
bash scripts/run_v11_f41.sh

# 手动执行等效命令：
cd /home/charlie/project/OrbitWarRL
ORBITWARS_SHAPING_CAPTURE=0.05 \
ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.005 \
ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.01 \
ORBITWARS_SHAPING_ONE_SHIP_THRESH=3.0 \
ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.02 \
ORBITWARS_SHAPING_HIGH_PROD_THRESH=3.0 \
ORBITWARS_SHAPING_EMIT_LOG=0.0 \
ORBITWARS_SHAPING_EMIT_GATED=0 \
ORBITWARS_SHAPING_PLANET_SHARE=0.0 \
ORBITWARS_SHAPING_PROD_SHARE=0.0 \
ORBITWARS_SHAPING_FLEET_LOG=0.0 \
ORBITWARS_SHAPING_RELEASE=0.0 \
ORBITWARS_SHAPING_RELEASE_K=20.0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
/home/charlie/anaconda3/bin/python -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v11_f41.yaml \
  --log-dir logs/v11_f41 \
  --resume-from ckpt_multi_action_v11_f40_buffer/ckpt_000099.pkl
```

### 3.3 f41 评估

```bash
# 批量 eval（@49/99/149/199/249/299/349/399/449/499）
bash scripts/run_f41_eval.sh

# 单 ckpt eval
PYTHON=/home/charlie/anaconda3/bin/python \
  bash scripts/quick_replay.sh \
  ckpt_multi_action_v11_f41/ckpt_000099.pkl \
  v11_f41_u99
```

### 3.4 查看结果

```bash
# 一行 summary
head -1 logs/replay_analyze/v11_f41_u99_vs_v20.summary.txt

# 全部 f41 gate 汇总
for U in 49 99 149 199 249 299 349 399 449 499; do
  SUM="logs/replay_analyze/v11_f41_u${U}_vs_v20.summary.txt"
  [ -f "$SUM" ] && head -1 "$SUM"
done

# 拉结果到本地
rsync -avz charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/logs/replay_analyze/v11_f41_* \
  logs/replay_analyze/
```

### 3.5 监控训练

```bash
tail -f logs/v11_f41.log
bash scripts/check_training_health.sh logs/v11_f41.log
grep '^upd.*WRr' logs/v11_f41.log   # 仅看 eval 行
```

### 3.6 从某个 ckpt resume（后续迭代用）

```bash
# 示例：从 f41 @99 resume 跑 300 upd
RESUME=ckpt_multi_action_v11_f41/ckpt_000099.pkl \
NUM_UPDATES=300 \
bash scripts/run_v11_f41.sh
```

---

## 4. 配置速查

### f40

| 项 | 值 |
|----|----|
| config | `orbit_wars_rl/configs/multi_action_v11_f40_buffer.yaml` |
| resume | `ckpt_bc_f40/ckpt_final.pkl` |
| shaping | CAPTURE=0.02, PROD_SHARE_DELTA=0.005 |
| distro | strong=0.20, frozen=0.30, buffer=0.30, random=0.20 |

### f41

| 项 | 值 |
|----|----|
| config | `orbit_wars_rl/configs/multi_action_v11_f41.yaml` |
| resume | `ckpt_multi_action_v11_f40_buffer/ckpt_000099.pkl` |
| shaping | CAPTURE=0.05, PROD_SHARE_DELTA=0.005, ONE_SHIP_PEN=0.01, HIGH_PROD_CAP=0.02 |
| distro | strong=0.25, frozen=0.25, buffer=0.40, random=0.10 |

### f42

| 项 | 值 |
|----|----|
| config | `orbit_wars_rl/configs/multi_action_v11_f42.yaml` |
| resume | `ckpt_multi_action_v11_f41/ckpt_000049.pkl` |
| shaping | CAPTURE=0.05, **CAPTURE_FLEET_SCALE=0.10**, PROD_SHARE_DELTA=0.005, ONE_SHIP_PEN=**0.005**, HIGH_PROD_CAP=0.02 |
| distro | strong=0.25, frozen=0.25, buffer=0.40, random=0.10 |

---

## 5. f42 实验 — 大舰队翻转 bonus

### 5.1 设计动机

f41 证明 spf/flip 和 e2+ 存在此消彼长的 tradeoff：
- ONE_SHIP_PENALTY 提升 spf，但间接惩罚多路小舰队，杀死 e2+
- CAPTURE 奖励翻转但不区分大小舰队，trickle flip 和 decisive flip 奖励相同

**核心缺失**：模型没有"蓄力再打"的信号（z0 始终 <1%）。Vadasz 的 z0=34%。

**用户洞察**：奖励必须有 0/1 方向感——只有翻转发生时才给蓄力奖励，否则囤兵不发走向另一个极端。

### 5.2 CAPTURE_FLEET_SCALE reward

**公式**：`coef * Σ(prod_i * log1p(garrison_i)) / total_prod / 8`

- **仅在星球翻转时触发**（0/1 gate）
- **翻转后我方 garrison 越大 → bonus 越高**（garrison 是"投入兵力减去防守"的 proxy）
- **梯度链**：大 garrison 翻转 ← 大舰队发送 ← 蓄力等待 ← z0

| 场景 | 效果 |
|------|------|
| 大舰队翻转（garrison=30） | `log1p(30)/8 = 0.43`，高 bonus |
| trickle 翻转（garrison=3） | `log1p(3)/8 = 0.17`，低 bonus |
| 没翻转 | 0，无 bonus |

### 5.3 f42 vs f41 改动对照

| 维度 | f41 | **f42** | 理由 |
|------|-----|---------|------|
| CAPTURE | 0.05 | 0.05 | 不变 |
| **CAPTURE_FLEET_SCALE** | — | **0.10** | 新：大舰队翻转 bonus |
| ONE_SHIP_PENALTY | 0.01 | **0.005** | 减半，避免杀死 e2+ |
| HIGH_PROD_CAPTURE | 0.02 | 0.02 | 不变 |
| PROD_SHARE_DELTA | 0.005 | 0.005 | 不变 |
| Resume | f40 @99 | **f41 @49** | flip 峰值 7.19% |
| 架构 | 不变 | **不变** | planet=33, global=18 |

### 5.4 期待效果

- **z0 提升**：模型发现"等几回合蓄力 → 发大舰队 → 翻转拿高分"
- **e2+ 恢复**：ONE_SHIP_PENALTY 减半，多路小舰队不再被过度惩罚
- **spf 保持**：CAPTURE_FLEET_SCALE 直接奖励大舰队

### 5.5 执行命令

```bash
# 同步代码
bash sync_mirror_ultrapp.sh

# 启动训练（已于 2026-06-03 17:50 启动）
bash scripts/run_v11_f42.sh

# 手动执行等效命令：
cd /home/charlie/project/OrbitWarRL
ORBITWARS_SHAPING_CAPTURE=0.05 \
ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE=0.10 \
ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.005 \
ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.005 \
ORBITWARS_SHAPING_ONE_SHIP_THRESH=3.0 \
ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.02 \
ORBITWARS_SHAPING_HIGH_PROD_THRESH=3.0 \
ORBITWARS_SHAPING_EMIT_LOG=0.0 \
ORBITWARS_SHAPING_EMIT_GATED=0 \
ORBITWARS_SHAPING_PLANET_SHARE=0.0 \
ORBITWARS_SHAPING_PROD_SHARE=0.0 \
ORBITWARS_SHAPING_FLEET_LOG=0.0 \
ORBITWARS_SHAPING_RELEASE=0.0 \
ORBITWARS_SHAPING_RELEASE_K=20.0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
/home/charlie/anaconda3/bin/python -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_v11_f42.yaml \
  --log-dir logs/v11_f42 \
  --resume-from ckpt_multi_action_v11_f41/ckpt_000049.pkl
```

```bash
# 评估
bash scripts/run_f42_eval.sh

# 监控
tail -f logs/v11_f42.log
```

---

## 6. f42 Replay 结果（vs v20，first-80）

### 6.1 全 ckpt 汇总

| ckpt | WLD | spf | flip | e2+ | z0 | garr | Day5 gate |
|------|-----|-----|------|-----|-----|------|-----------|
| **@49** | 0/5/0 | **11.47** | 4.40% | **6.5%** | 0.7% | **71.97** | spf✅ garr✅ e2+✅ |
| @99 | 0/5/0 | 8.68 | 4.12% | 0.8% | 0.5% | 42.30 | 全 FAIL |
| @249 | 0/5/0 | 10.25 | 5.27% | 0.0% | 0.6% | 41.99 | spf✅ |
| @399 | 0/5/0 | 8.77 | **6.45%** | 0.0% | 0.8% | 36.79 | flip✅ |
| @499 | 0/5/0 | 9.57 | 5.18% | 0.0% | 0.5% | 59.03 | — |

对比 f40/f41 @99：

| 指标 | f40 @99 | f41 @99 | **f42 @99** | **f42 @49** |
|------|---------|---------|-------------|-------------|
| spf | 8.51 | **13.23** | 8.68 | 11.47 |
| flip | 5.26% | **6.23%** | 4.12% | 4.40% |
| e2+ | **24.8%** | 2.0% | 0.8% | **6.5%** |
| z0 | 0.6% | 0.7% | 0.5% | 0.7% |

### 6.2 核心结论

1. **CAPTURE_FLEET_SCALE 未达 f41 水平**：replay spf/flip 仍低于 f41 @99（13.2 / 6.23%），未兑现训练侧 spf≈50。
2. **训练–replay 严重脱节**：@99 训练 e2=0.27、spf=49.5 → replay e2+=0.8%、spf=8.68；**不可用训练指标选 ckpt**。
3. **@49 是唯一亮点**：e2+=6.5%（过 5% gate）、spf/garr 过线；emit 分布 92% 单路 + **6% 双路**（@99 仅 0.8% 双路）。
4. **继续训导致 e2+ 归零**：@149 起 e2+≈0；@399 靠单路大 bin7（69%）抬 flip 到 6.45%，但 e2+=0。
5. **z0 全线失败**：全 ckpt z0<1%，CAPTURE_FLEET_SCALE 未带来 replay 级「蓄力等待」。

### 6.3 @99 行为快照（塌缩典型）

- `mean_emits=0.99`，**98% 回合只发 1 路**，e2+=0.8%
- `bin7@1.00` 占 **80.9%** → 单路拉满 bin，非 Vadasz 式多路爆发
- v20 对比：flip 11.7%、z0=25%、spf=18.25

### 6.4 决策

- **不 promote f42**；提交基线仍 f29 @599
- **f43 resume：`ckpt_000049.pkl`**（非 @99）
- 若只要 flip gate：@399 可作对照，但无 e2+，不推荐作主线

---

## 7. f43 实验 — gated multi-emit（恢复 e2+）

### 7.1 设计动机

| 实验 | spf/flip | e2+ | 机制 |
|------|----------|-----|------|
| f40 @99 | 中 | **24.8%** | 多路小舰队，无 anti-trickle |
| f41 @99 | **高** | 2% | ONE_SHIP_PEN 惩罚 ≤3 艘 → 杀死第 2/3 路 |
| f42 @99 (train) | 高 | 0.27 (train) | 减半 PEN + fleet-scaled capture，replay 待证 |

**假设**：用 **正奖励** 鼓励「一路大军 + 多路跟进」，比 **负惩罚** 小舰队更能同时保住 spf 与 e2+。

### 7.2 MULTI_EMIT gated reward

**公式**：`coef * 1{n_valid >= 2 AND max(ships) >= 8}`

- 0/1 gate，与 CAPTURE_FLEET_SCALE 同哲学
- `MIN_SHIPS=8` 对齐 v20 `ABS_MIN_BATCH`，避免双 trickle 刷分
- **不惩罚** 小舰队第 2 路，只奖励「有主攻的多路」

### 7.3 f43 vs f42 改动

| 维度 | f42 | **f43** |
|------|-----|---------|
| CAPTURE | 0.05 | 0.05 |
| CAPTURE_FLEET_SCALE | 0.10 | 0.10 |
| ONE_SHIP_PENALTY | 0.005 | **0** |
| **MULTI_EMIT** | — | **0.02** (min ships 8) |
| Resume | f41 @49 | **f42 @49**（replay e2+=6.5%；@99 已塌） |

### 7.4 期待与 gate

- **e2+ replay > 10%** @99（向 f40 靠拢）
- **spf > 10, flip > 6%** 保持 f41 水平
- **z0 replay > 5%**（训练已见 12%，需 replay 验证）

### 7.5 执行命令

```bash
bash sync_mirror_ultrapp.sh

# 先补 f42 replay（若未跑）
bash scripts/run_f42_eval.sh

# 启动 f43
bash scripts/run_v11_f43.sh
tail -f logs/v11_f43.log

# 评估
bash scripts/run_f43_eval.sh
```

### 7.6 配置速查

| 项 | 值 |
|----|-----|
| config | `orbit_wars_rl/configs/multi_action_v11_f43.yaml` |
| script | `scripts/run_v11_f43.sh` |
| resume | `ckpt_multi_action_v11_f42/ckpt_000049.pkl` |
| shaping | CAPTURE=0.05, CAPTURE_FLEET_SCALE=0.10, MULTI_EMIT=0.02, ONE_SHIP_PEN=0 |

---

## 8. f44_align — 训练–replay 对齐实验（P0–P4）

### 8.1 动机

f42 证明：**训练 log 不能预测 replay**（@99 训练 spf=49 → replay 8.7）。在改 shaping（f43）之前，先让训练过程能观测 **vs v20 真值**。

### 8.2 已实现基建

| 优先级 | 实现 | 文件 |
|--------|------|------|
| P0 | ckpt 附带 `.meta.json`（`opp_tag`, align, eval_vs_v20）；`opp=buf` 存盘 WARN | `orbit_wars_rl/ppo/runner.py` |
| P1 | log 增加 `align[strn+frzn]` 滚动均值（window=20） | 同上 |
| P2 | 每 `eval_every` CPU 跑 3 局 `quick_replay` vs v20，log 行 `v20[spf …]` | `orbit_wars_rl/eval/v20_mini_gate.py` |
| P3 | `buffer_rollout_ratio` 0.40→**0.15** | `multi_action_v11_f44_align.yaml` |
| P4 | `strong_ratio` 0.25→**0.35** | 同上 |
| 诊断 | 按 opp 解析历史 log | `scripts/parse_train_log_by_opp.py` |

### 8.3 f44 vs f42

| 项 | f42 | f44_align |
|----|-----|-----------|
| shaping | f42 | **同 f42** |
| resume | f41@49 | **f42@49** |
| buffer_rollout | 0.40 | **0.15** |
| strong_ratio | 0.25 | **0.35** |
| eval_vs_v20 | 无 | **每 50 upd，3 局** |
| promote 依据 | 训练 spf/e2 | **`eval_vs_v20/*` + align 滚动** |

### 8.4 如何判定「训完有效」

1. 看 log 里 **`[eval_vs_v20]`** 行（first-80 spf/flip/e2+），不是 `opp buf` 后的 spf。
2. 看 ckpt **`ckpt_XXXXXX.meta.json`**：`opp_tag` 优先 `strn`/`frzn`；`eval_vs_v20_e2_plus_pct` 上升。
3. `align[spf/e2]` 应逐步接近 v20 行（仍可能偏高，但差距应缩小）。
4. 早停：若连续 2 次 eval `eval_vs_v20/e2_plus_pct` 下降且 `<3%`，停止长训。

### 8.5 执行命令

```bash
bash sync_mirror_ultrapp.sh

# 长训（~25min + 每 50upd 额外 ~3–5min CPU replay）
bash scripts/run_v11_f44_align.sh
tail -f logs/v11_f44_align.log

# 训练中解析 log
python scripts/parse_train_log_by_opp.py logs/v11_f44_align.log --align-only

# 训后全量 replay（可选，NUM_GAMES=5）
bash scripts/run_f44_eval.sh
```

### 8.6 f43 状态

**暂缓**。f44 若 `eval_vs_v20` 曲线与 replay 一致，再在最佳 ckpt 上叠 f43 的 MULTI_EMIT。
