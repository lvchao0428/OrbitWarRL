# DAY11 进展 — f40/f41 完成 + f42 训练中

> **2026-06-03 晚更新**
> 接续 [`DAY10_PROGRESS.zh.md`](DAY10_PROGRESS.zh.md)。
> **f40** 500 upd 完成；**f41** 500 upd 完成；**f42** 训练中。

---

## 0. TL;DR

| 维度 | 状态 |
|---|---|
| **f40** | ✅ 500 upd 完成；@99 **flip=5.26% / e2+=24.8% / spf=8.51**（全系列综合最佳）；后续 spf 退化 |
| **f41** | ✅ 500 upd 完成（从 f40 @99 resume）；**spf/flip 大幅改善，e2+ 塌缩** |
| **f41 最佳** | @99: **spf=13.23 / flip=6.23%**（首次 replay 过 flip>6% promote gate）|
| **f42** | 🔄 训练中（从 f41 @49 resume）；**CAPTURE_FLEET_SCALE** 新 reward |
| **核心矛盾** | **spf/flip vs e2+ 此消彼长**：f40 强在 e2+，f41 强在 spf/flip |
| **f42 目标** | 打破 spf/flip vs e2+ 的 tradeoff，通过奖励大舰队翻转间接激励 z0 |
| **提交基线** | 仍为 **f29 @599** |
| **WLD** | f40/f41 全 ckpt 0/5/0 |

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
