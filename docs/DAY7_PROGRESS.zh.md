# DAY7 进展 — buf_mix 结论 → v11_f25 特征工程

> **2026-05-28 更新**  
> 接续 [`DAY6_PROGRESS.zh.md`](DAY6_PROGRESS.zh.md)。  
> **项目综述（架构图 + 全流程）**：[`OVERVIEW.zh.md`](OVERVIEW.zh.md)

---

## 0. TL;DR — 最新方案

| 维度 | 状态 |
|---|---|
| **buf_mix** | ✅ 完成；@299 bin0=59% @499 bin0=57% → **pct 无改善** |
| **诊断** | 参数调参无效；根因 = **特征不够**（pct/emit head 缺 relative capacity 信号） |
| **当前主线** | 🔴 **v11_f25 特征工程 + 从头 self-play** |
| **特征维度** | planet **28** / fleet **10** / global **17**（+11 vs 旧 v11） |
| **训练** | `bash scripts/run_v11_f25.sh`（**不 resume** 旧 ckpt） |
| **决策** | **replay only** vs v20 |

### 策略 pivot（2026-05-28）

1. **停止**：frozen_ratio / buffer_reset / ent_coef 等参数 sweep  
2. **停止**：在旧 22-dim arch 上 resume 续训  
3. **开始**：f25 新特征 → 从头 800 upd → @299/@799 replay gate  
4. **成功后再**：mixed buffer 续训（Plan B 作为第二阶段，不是当前第一步）

---

## 1. 今日策略复述（v11_f25）

### 1.1 要解决什么

| 症状 | replay 证据 |
|---|---|
| bin0 锁死 ~60% | buf_mix @299 59.3% @499 57.3% |
| 1-ship spam | spf 6.8–8.2 vs v20 ~18 |
| emit 行为畸形 | 74% 单发 + 23% 一次发满 8（bin0 spam） |
| buffer 无效 | buf_from4k spf/garr ✅，bin0 仍 61% |

**根因**：pct head 只看目标绝对守备，不知道「相对我的 avg/max stack 该送多少」；emit head 不知道「有几颗软目标可打」。reward 不直接惩罚 bin0，state buffer 只改 garr 分布。

### 1.2 解法：+11 维归纳偏置（不调参）

**大舰队（pct head）** — planet [22–26] + global [14]：

| 特征 | 信号 |
|---|---|
| flip_cost_ratio | 目标守备 / 我方**平均**守备 |
| friendly_surplus | 入轨是否已够翻转 |
| capturable_bin3 | bin3(40%) 能否翻（0/1） |
| needed_pct_norm | 目标守备 / 我方**最大**守备 → 推 bin5+ |
| capturable_bin5 | bin5(70%) 能否翻（0/1） |
| max_garr_norm | 全局最强发射台 |

**多路（emit/dst head）** — planet [27] + global [15–16]：

| 特征 | 信号 |
|---|---|
| weak_target_score | 每颗敌/中立星的软度 |
| n_weak_targets_norm | 本 turn 软目标数量 → emit 继续发 |
| ships_to_capture_all_weak_norm | 打遍软目标总成本 |

**fleet [8–9]**：target_dist / target_garrison — 远程大舰队暗示。

### 1.3 训练配置

| 项 | 值 |
|---|---|
| config | `orbit_wars_rl/configs/multi_action_v11_f25.yaml` |
| script | `scripts/run_v11_f25.sh` |
| ckpt | `ckpt_multi_action_v11_f25/` |
| resume | **无**（arch 维度变了） |
| K | 8（不变；见 OVERVIEW §3.4） |
| seed | 250 |
| updates | 800 |
| ent_coef_pct | 0.006 |
| frozen_ratio | 0.40 |
| buffer | **本 run 不用**（f25 成功后再加 mix） |

### 1.4 执行命令

```bash
# 5090
bash sync_mirror_ultrapp.sh
bash scripts/run_v11_f25.sh
tail -f logs/v11_f25.log

# 中途 @299
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_f25/ckpt_000299.pkl v11_f25_u299

# 完整 @799
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_f25/ckpt_000799.pkl v11_f25_u799
```

### 1.5 Replay gate

**@299 早期信号**（非最终 promote）：

| 指标 | 之前 buf_mix | 希望看到 |
|---|---|---|
| bin0 | 57–59% | **< 50%**（有下降即正向） |
| bin4/5/7 之和 | 低 | **> 40%** |
| emit≥2 | ~20% | **> 30%** 且非 bin0 spam |

**@799 决策**：

| 结果 | 下一步 |
|---|---|
| bin0 < 40%，spf > 4 | **PROMOTE** → mixed buffer 续训 400 upd |
| bin0 < 40%，spf < 4 | 续训 800 upd + buffer |
| bin0 > 50% | **B1 BC pct**（`orbit_wars_rl/bc/train_bc.py`） |

---

## 2. buf_mix 阶段记录（已结束）

### 2.1 配置

| 参数 | buf_from4k | buf_mix |
|---|---|---|
| buffer | v20 only | v20 + top10 50/50 |
| frozen_ratio | 0.50 | 0.35 |
| buffer_reset_ratio | 0.50 | 0.70 |
| resume | 4k @3199 | buf_from4k @799 |

### 2.2 Mid-replay 结果

| 指标 | buf_from4k @799 | @299 | @499 | 目标 |
|---|---|---|---|---|
| bin0 | 61.0% | 59.3% | 57.3% | <40% |
| spf | 10.73 | 6.78 | 8.24 | >10 |
| garr | 107.5 | 91.12 | 91.11 | >60 ✅ |
| flip | 1.90% | 1.65% | 2.09% | >3% |
| emit=1 | ~25% | **74%** | **78%** | ↓ |
| WR | 0/5 | 0/5 | 0/5 | — |

**结论**：spf/garr 有 buffer 增益，**pct 完全没动**；emit 退化为「单发 bin0 或 8 路 bin0 spam」。

### 2.3 根因表（已确认）

| run | bin0 | 备注 |
|---|---|---|
| k8_no_emit @800 | **25.4%** | pct 最健康 ckpt（但 spf 低） |
| k8_4k @3199 | 72.1% | self-play 锁 bin0 |
| buf_from4k @799 | 61.0% | buffer 抬 spf 不修 pct |
| buf_mix @299 | 59.3% | top10 buffer 也不修 pct |

---

## 3. 后续备选（f25 失败时）

| 方案 | 描述 | 触发条件 |
|---|---|---|
| **B1 BC** | v20 self-play 采 (s,a) → BC pct head → PPO | f25 @799 bin0 > 50% |
| B2 | pct reward shaping（bin0 penalty） | B1 前可选 |
| C1 | 去掉 PCT bin0(0.10) | 破坏性，最后手段 |
| A2 | buf_mix from k8_no_emit @800 | **已搁置**（策略 pivot 到特征） |

---

## 4. Checklist

### 4.1 buf_mix（已完成）

- [x] mixed buffer 重建
- [x] 训练完成
- [x] @299/@499 replay → pct 无改善
- [x] **决策：pivot 到 f25**

### 4.2 v11_f25（当前）

- [x] `encode.py`：planet 28 / fleet 10 / global 17
- [x] config + `run_v11_f25.sh`
- [x] 本地 forward smoke test 通过
- [ ] sync 5090
- [ ] 启动训练
- [ ] @299 replay
- [ ] @799 replay → 决策

### 4.3 勿再踩坑

- [x] sync exclude `data/*.npz`
- [ ] **不要** resume 旧 ckpt 到 f25
- [ ] **不要** 再调 buf_mix 参数
- [ ] replay 决策 **不看** 训练 spf

---

## 5. 路径备忘

| 用途 | 路径 |
|---|---|
| 综述文档 | [`docs/OVERVIEW.zh.md`](OVERVIEW.zh.md) |
| 特征 | `orbit_wars_rl/features/encode.py` |
| f25 config | `orbit_wars_rl/configs/multi_action_v11_f25.yaml` |
| f25 log | `logs/v11_f25.log` |
| f25 ckpt | `ckpt_multi_action_v11_f25/` |
| buf_mix replay | `logs/replay_analyze/v11_buf_mix_u499_vs_v20.json` |
| mixed buffer（后续用） | `data/mixed_v20_top10.npz` |
