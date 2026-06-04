# DAY12 计划 — f44_align 训练–replay 对齐主线

> **2026-06-03 启动**
> 接续 [`DAY11_PLAN.zh.md`](DAY11_PLAN.zh.md)（f40–f42 replay 结论 + f44 基建）。
> **当前运行中**：`bash scripts/run_v11_f44_align.sh` → `logs/v11_f44_align.log`

---

## 0. TL;DR

| 维度 | 状态 |
|------|------|
| **主线** | **f44_align** — 先对齐观测与分布，再改 shaping |
| **合理之处** | 用 `eval_vs_v20` 作 ground truth；降 buffer、增 strong；ckpt 带 `opp_tag` |
| **f43** | 暂缓，待 f44 证明 inline v20 曲线可信 |
| **基线** | f42 @49 replay：spf=11.47, e2+=6.5%, flip=4.4%, WLD=0/5/0 |
| **提交** | 仍 **f29 @599**，直至 f44 gate 通过 |
| **盯盘** | `bash scripts/watch_f44_align.sh` + rsync log |

---

## 1. 为什么 f44 是「合理尝试」

### 1.1 问题已定位清楚

f42 暴露的不是「奖励公式还差一口气」，而是 **优化目标与评估目标不一致**：

- 训练 log 的 spf/e2 主要在 **buffer 中期局** 上统计（@99 训练 spf=49 → replay 8.7）。
- 同一策略对 **v20** 会塌成单路 trickle（98% 单 emit）。
- 继续加 shaping（f43）会在错误分布上浪费 500 upd。

### 1.2 f44 的三条合理假设

| 假设 | 机制 | 可证伪方式 |
|------|------|------------|
| **H1 可观测** | 每 50 upd 跑 3 局 vs v20 → `eval_vs_v20` 与训后 replay 同源 | 对比 `train_u49` 与 `quick_replay` 全 5 局，误差应小 |
| **H2 分布** | buffer_rollout 0.40→0.15，strong 0.25→0.35 → 梯度更多在「有压力」局 | `align[spf/e2]` 向 v20 行靠拢；buf 占比下降 |
| **H3 选 ckpt** | `.meta.json` 记录 `opp_tag` + v20 指标 → 不再用 buf 峰值 ckpt | 最佳 ckpt 的 `opp_tag` 应为 strn/frzn |

### 1.3 刻意不变的部分

- **shaping 与 f42 相同**（CAPTURE + CAPTURE_FLEET_SCALE），隔离变量。
- **resume f42 @49**（f42 replay 最佳 e2+ 点）。
- 避免同时改 reward + 分布 + 观测，符合 Day10「一次一个变量」。

### 1.4 已知局限（诚实边界）

- `eval_vs_v20` 仅 **3 局**，方差大；训后仍需 5 局 `run_f44_eval.sh` 确认。
- inline eval 在 **CPU**，每 50 upd 增加 ~3–5 min，总时长 +30–50 min。
- **align** 仍是训练环境指标，只是限制在 strn+frzn；**不能替代** v20 行。
- 未改特征/架构；若 H1/H2 通过但 WLD 仍 0/5，下一步才回到 f37/f43 或 league。

---

## 2. f44 实验配置速查

| 项 | 值 |
|----|-----|
| config | `orbit_wars_rl/configs/multi_action_v11_f44_align.yaml` |
| script | `scripts/run_v11_f44_align.sh` |
| log | `logs/v11_f44_align.log` |
| ckpt | `ckpt_multi_action_v11_f44_align/ckpt_*.pkl` + `.meta.json` |
| resume | `ckpt_multi_action_v11_f42/ckpt_000049.pkl` |
| eval_vs_v20 | 每 50 upd，3 局，first-80 |
| buffer_rollout | **0.15** |
| strong_ratio | **0.35** |

---

## 3. 盯盘与拉取（远程 5090）

```bash
# 远程
tail -f logs/v11_f44_align.log
grep '^\[eval_vs_v20\]' logs/v11_f44_align.log
grep '^\[ckpt\]' logs/v11_f44_align.log

# 本地拉 log / ckpt
rsync -avz charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/logs/v11_f44_align.log logs/
rsync -avz charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/ckpt_multi_action_v11_f44_align/ ckpt_multi_action_v11_f44_align/

# 本地一览
bash scripts/watch_f44_align.sh
python scripts/parse_train_log_by_opp.py logs/v11_f44_align.log --align-only
```

### 3.1 关键 log 行含义

```
upd   12  ...  opp strn  ...  spf 34.2  ...  align[spf 33.1 e2 0.25 z0 0.02 n=8]
[eval_vs_v20] u49 tag=train_u49 spf=11.5 flip=4.2% e2+=6.0% ...
[ckpt] saved ... opp=strn
[ckpt] saved ... opp=buf WARN opp=buf (prefer strn/frzn ...)
```

- **决策只看** `[eval_vs_v20]` 和 `.meta.json` 里的 `eval_vs_v20_*`。
- **忽略** 单行 `opp buf` 后的 spf=40+。

---

## 4. Gate 与决策树（f44 跑完后）

### 4.1 第一次 inline eval（@49，约 upd 49）

| 指标 | f42 @49 全量 replay | f44 期待 |
|------|---------------------|----------|
| spf | 11.47 | inline 与全量相差 <2 |
| e2+ | 6.5% | ≥5% 且不低于 f42 @49 |
| flip | 4.40% | 不劣于 4% |

若 inline `@49` 与 f42 全量 replay **接近** → **H1 成立**，可信任后续 inline 曲线。

### 4.2 训练中期（@99 / @149）

| Gate | 通过 | 失败 → 动作 |
|------|------|-------------|
| `eval_vs_v20/e2+` | ≥5%，且未连续 2 次下降 | 早停；检查是否 resume 点太差 |
| `eval_vs_v20/spf` | ≥10 | 考虑略增 CAPTURE_FLEET_SCALE（f44b） |
| `eval_vs_v20/flip` | ≥6% | 已过 promote 门槛之一 |
| `align/spf` vs v20 spf | 差距 <3×（例 align<25, v20>8） | 仍 >5× 则再降 buffer 到 0.10 |

### 4.3 训练结束（@499）

```
eval_vs_v20 最佳 ckpt
  → bash scripts/run_f44_eval.sh（5 局确认）
  → 与 f41 @99 / f40 @99 / f42 @49 对比表
  → 若 WLD≥1/5 或 flip≥6% 且 e2+≥10%：进入 f43 或 f45 league
  → 否则：记录 f44 为「对齐成功但策略未达标」，改特征或 BC+RL 混合
```

### 4.4 早停规则

- 连续 **2** 次 eval（间隔 50 upd）：`e2+ < 3%` 且较上次下降 → **停止**。
- `clip_frac > 0.20` 持续 20 upd → 检查 lr/ent（沿用 Day8 告警）。

---

## 5. f44 结果记录（训练中更新）

> 从远程 rsync log 后填写。`bash scripts/watch_f44_align.sh` 可自动摘表。
> **最后同步**：upd 25/500，SPS ~7.4K，**尚未到第一次 `[eval_vs_v20]`（upd 49）**。

### 5.0 启动确认（log 头）

- `strong_ratio=0.35`, `buffer_rollout_ratio=0.15` ✅
- resume `f42/ckpt_000049.pkl` ✅
- 已出现 `align[spf … e2 … n=…]` 行（P1 生效）✅

### 5.1 inline eval_vs_v20

| upd | spf | flip | e2+ | z0 | WLD | 备注 |
|-----|-----|------|-----|-----|-----|------|
| @49 | — | — | — | — | — | **下一个里程碑**（~upd 49 后 grep eval_vs_v20） |
| @99 | | | | | | |
| @199 | | | | | | |
| @499 | | | | | | |

### 5.2 与 f42 基线对比（@49）

| 来源 | spf | flip | e2+ |
|------|-----|------|-----|
| f42 replay 5局 | 11.47 | 4.40% | 6.5% |
| f44 inline 3局 | TBD | TBD | TBD |

### 5.3 ckpt meta 摘要

| ckpt | opp_tag | eval_vs_v20_e2+ | align_e2 | 备注 |
|------|---------|------------------|----------|------|
| | | | | |

---

## 6. 后续分支（DAY12+）

```
f44_align 完成
│
├─ H1 成立（inline ≈ full replay）
│   ├─ eval 曲线上升 → f43（MULTI_EMIT）从 f44 最佳 ckpt resume，仍开 eval_vs_v20
│   ├─ eval 平坦、e2+≤f42 → 特征/BC 混合（f40 buffer 路线加强）
│   └─ flip≥6% 但 e2+塌 → 单独拉 e2+（f43 或 emit_log gated）
│
└─ H1 不成立（inline 与全量 replay 偏差大）
    └─ 修 mini_gate（局数 3→5、固定 seed）后再跑 f44b
```

| 候选 | 条件 | 内容 |
|------|------|------|
| **f43** | f44 最佳 ckpt + H1 OK | MULTI_EMIT=0.02, ONE_SHIP_PEN=0 |
| **f44b** | buffer 仍污染 align | buffer_rollout=0.10 |
| **f45** | replay gate 部分通过 | gated pool 自动 promote + 长训 |

---

## 7. 执行命令汇总

```bash
# 已在跑
bash scripts/run_v11_f44_align.sh

# 盯盘
bash scripts/watch_f44_align.sh

# 训后
bash scripts/run_f44_eval.sh
```

---

## 8. 相关文件

| 类型 | 路径 |
|------|------|
| 基建 | `orbit_wars_rl/ppo/runner.py`, `orbit_wars_rl/eval/v20_mini_gate.py` |
| 诊断 | `scripts/parse_train_log_by_opp.py`, `scripts/watch_f44_align.sh` |
| 前日 | [`DAY11_PLAN.zh.md`](DAY11_PLAN.zh.md) §6 f42 replay, §8 f44 设计 |
