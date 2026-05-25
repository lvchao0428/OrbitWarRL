# DAY5 进展 — 现状梳理 + Phase 0 收尾

> 写于 2026-05-25。Day4 v9 ablation **代码与 v9a/b 训练已闭环**；v9c/d **仍在 5090 上跑**（~65%）。
> 本文是 Day5 的**单一事实来源（status SSOT）**；战术规划见 [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md)，操作手册见 [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md)。

---

## 0. TL;DR（2026-05-25 快照）

| 维度 | 状态 |
|---|---|
| **Day4 目标** | ✅ shaping v2 落地；✅ v9a/b 4000 upd 完成；🔄 v9c/d ~2620/4000 |
| **shaping 在 self-play** | ✅ v9b gauntlet vs v9a **17/20**；训练 ev 0.97+ |
| **shaping vs v20** | ❌ v9b gauntlet vs v20 **0/40**；replay spf **4.76**（训练 21.2） |
| **Day5 方向** | 冻结 v9c shaping 系数；FAST 500 upd 筛 **obs/curriculum/reward v3** |
| **阻塞 FAST 的唯一项** | Phase 0：v9c/d 跑完 → export → gauntlet + replay vs v20 → 定 frozen base |

**当前 monitor 快照（5090，用户 2026-05-25 提供）：**

```
multi_action_v9c  upd 2620  ev 0.97  clip 0.08  spf 27.2  garr 38.6  pS 0.39  fLog 0.43  live
multi_action_v9d  upd 2619  ev 0.97  clip 0.08  spf 20.9  garr 56.9  pS 0.41  fLog 0.37  live
```

预计剩余 **~1380 updates × ~11s ≈ 4–5h** 后 v9c/d 双双到 4000。

---

## 1. v9 Ablation 全景

### 1.1 四路配置对照

| Config | Shaping | 训练状态 | 末段 / 当前 metric | 备注 |
|---|---|---|---|---|
| **v9a** | 无（sparse ±1） | ✅ u3999 DONE | spf **11.4**, garr **58.5**, pS **0.29** | control |
| **v9b** | prod=0.01 | ✅ u3999 DONE | spf **21.2**, garr **49.9**, pS **0.41** | **唯一完成 H2H+replay 的 run** |
| **v9c** | prod+planet+fleet_log | 🔄 u2620 live | spf **27.2**, garr **38.6**, fLog **0.43** | 训练 spf 最高 |
| **v9d** | prod+planet（无 fleet_log） | 🔄 u2619 live | spf **20.9**, garr **56.9**, pS **0.41** | 训练 garr 最高 |

公共超参：`episode_steps=350`, `rollout_length=256`, `gamma=0.99`, `num_minibatches=32`（OOM fix）。

### 1.2 训练健康度

| 指标 | v9a/b @3999 | v9c/d @2620 | 判定 |
|---|---|---|---|
| `explained_variance` | 0.97–0.98 | 0.97 | ✅ value head 健康 |
| `clip_frac` | 0.09 | 0.08 | ✅ 已从早期 spike 恢复 |
| `approx_kl` | 0.005–0.006 | 0.004–0.005 | ✅ |
| NaN / OOM | 无 | 无 | ✅ num_minibatches=32 稳定 |

**v9c/d 早期告警（已恢复，无需 action）：**

| 事件 | upd | 说明 |
|---|---|---|
| ev 偏低 | v9c u0 ev=0.40, v9d u1 ev=0.44 | resume + 新 reward 正常；u500 前回升即可 |
| clip ALERT | v9c u136/u337 clip≈0.36–0.37 | shaping 梯度冲击；u500 后 clip→0.08 |
| clip WARN | v9d u142/u342 同类 | 同上 |

结论：**不是 v6p3-bug 级崩溃**；monitor 告警在 u400 后消失，与 v9b 曲线一致。

### 1.3 同期对比（@u1158 / @u2000）

| upd | v9b spf | v9c spf | v9d spf | 解读 |
|---|---|---|---|---|
| 1158 | 15.5 | **21.5** | — | fleet_log 早期推 spf |
| 2000 | 17.4 | **28.6** | — | v9c 持续领先 |
| 3999 / 2620 | 21.2 | **27.2** | 20.9 | v9c 训练 spf 全程最高 |

**初步倾向**：frozen base → **v9c**（full v2 family）。**最终需 replay vs v20 确认**——训练 log 外推不可靠（见 §2）。

---

## 2. 已完成 Eval（仅 v9b）

### 2.1 Gauntlet（双色，5090）

来源：`logs/h2h_v9b_u3999_gauntlet.log`（20 seeds × 2 colors）

| 对手 | WR | avg_steps |
|---|---|---|
| v20 | **0/40** | 173 |
| v7_u499 | **39/40** | 248 |
| v1 | **39/40** | 262 |

另：`logs/h2h_v9b_gauntlet.log`（10 seeds × 2 colors）vs v9a **17/20**，vs v20 **0/20**。

### 2.2 Replay Analyze（v9b vs v20，5 局，seed 0–4）

来源：`logs/replay_analyze/v9b_vs_v20.json`

**first-80 turns（player_0 = v9b）：**

| metric | v9b | v20 (p1) | Day5 FAST gate |
|---|---|---|---|
| outcome | **0/5/0** | 5/0/0 | — |
| `mean_ships_per_fleet` | **4.76** | 15.88 | > 10 |
| `mean_garrison_my` | **31.85** | 132.03 | > 60 |
| `fleet_arrival_rate`¹ | **3.88%** | 8.56% | > 6% |
| `zero_emit_rate` | 0.98% | 24.4% | — |
| emit=1 占比² | **95.3%** | 22.5% | — |
| emit≥2 占比² | **2.8%** | ~55% | > 5% |
| pct bin7 (100%) | **51.4%** | 30.8% | — |

**full game（player_0 = v9b）：** spf=4.81, garr=21.68, z0=**41.5%**（中后期停手 spiral）。

¹ `fleet_arrival_rate = fleets_arrived / fleets_launched`；Day5 文档中称 `flip_proxy`，与 `fleet_flip_rate` 同义引用此字段。

² 从 `emit_count_distribution` 推算：`emit≥2 = 1 - dist[0] - dist[1]`。

### 2.3 Train–Eval Gap（核心问题）

| metric | v9b 训练 @u3999 | v9b vs v20 replay first-80 | 倍率 |
|---|---|---|---|
| spf | 21.2 | 4.76 | **4.5× 高估** |
| garr | 49.9 | 31.85 | **1.6× 高估** |
| vs v20 WR | — | 0/40 | — |

**含义**：macro shaping 让 self-play 里「看起来对了」，但对 v20 **分布外** 几乎不 transfer。
Day5 主战场不是再加 shaping 系数，而是 **对手分布 + obs 威胁感知 + 行为（multi-emit）**。

### 2.4 v9b vs v20 行为根因（已量化）

1. **early game 失守**：first-80 garr 32 vs v20 132
2. **小 garrison × bin7**：母星被打低后 51% 发射选 100% pct → 1 舰舰队
3. **几乎不 multi-launch**：95% turn 只 emit 1 次
4. **舰队白干**：arrival rate 3.9% vs v20 8.6%
5. **中后期放弃**：full-game z0 41%

---

## 3. Phase 0 Checklist（Day5 启动前必做）

| # | 任务 | 状态 | 产出 |
|---|---|---|---|
| 0.1 | v9c/d 跑完 4000 upd | 🔄 ~65% | `logs/multi_action_v9{c,d}.log` 末行 `upd 3999` |
| 0.2 | export v9a/b/c/d @3999 | ⏳ v9b 已有 `submission_rl_v9b_u3999.py` | 4 个 submission `.py` |
| 0.3 | gauntlet v9c vs [v9a, v9b, v20] | ⏳ | `logs/h2h_v9c_gauntlet.log` |
| 0.4 | gauntlet v9d vs [v9a, v9b, v20] | ⏳ | `logs/h2h_v9d_gauntlet.log` |
| 0.5 | replay v9c vs v20 ×5 | ⏳ | `logs/replay_analyze/v9c_vs_v20.json` |
| 0.6 | replay v9d vs v20 ×5 | ⏳ | `logs/replay_analyze/v9d_vs_v20.json` |
| 0.7 | **frozen base 书面决策** | ⏳ | 见 §4 规则 |
| 0.8 | 更新本文 §1–§2 最终数字 | ⏳ | — |

**Phase 0 完成前 FAST baseline**：暂用 **v9b@3999** replay 数字（§2.2）；v9c 跑完后替换 FAST gate 基线。

---

## 4. Frozen Base 决策规则（Phase 0.7）

在 v9c/d replay vs v20 完成后执行：

| 条件 | 选择 |
|---|---|
| v9c replay spf **≥ v9b** 且 garr **≥ v9b** | **v9c**（预期） |
| v9c replay spf ↑ 但 garr ↓，vs v20 WR 仍 0/40 | **v9c**（spf 优先；garr 靠 A1/C1 补） |
| v9d replay 全面 ≥ v9c 且 spf 差距 < 10% | **v9d**（更简，无 fleet_log） |
| v9c/d replay 均 ≈ v9b | **v9b**（prod-only 足够，最简） |

**Shaping 系数冻结（不论选哪条 base）：**

```
ORBITWARS_SHAPING_PROD_SHARE=0.01
ORBITWARS_SHAPING_PLANET_SHARE=0.005   # v9b 除外
ORBITWARS_SHAPING_FLEET_LOG=0.002      # v9b/v9d 除外
```

---

## 5. Obs 审计结论（影响 Track A1）

`features/encode.py` 的 planet feats **已有** inbound 聚合：

- `in_friend_norm` / `in_foe_norm`（`_inbound_ships` 速度射线粗归因）

**缺失、且高手行为需要的：**

| 特征 | 用途 |
|---|---|
| `threat_ratio` | `in_foe / (garr + 1)` — 留兵 vs 全出 |
| `eta_foe_min` | 最近敌舰队到达回合 — 换家/撤退时机 |
| `net_inbound` | `(in_foe - in_friend) / (garr + 1)` — 净威胁 |

Day5 Track A1 应做 **A1'（补 ETA/比值）**，而非重复堆 inbound ships。
详见 [`DAY5_PLAN.zh.md` §3.2 / §4 Track A](DAY5_PLAN.zh.md)。

**Resume 策略**：A1' 只加 2–3 维 planet feat → **可 resume frozen base ckpt**；
A2 改坐标系 → **from scratch 500 upd**。

---

## 6. Day5 FAST 实验台账（待填）

**串行 launcher（推荐入口）：**

```bash
# 默认队列：r1_only → r2_release → r4_emit → r1_r2_r4
# 每个 variant 最多 1000 upd；gate 每 POLL_SEC 秒判一次
# 命中 PROMOTE(0) 或 KILL(2) 立即停掉当前 run，继续下一个
bash scripts/run_fast_serial.sh

# 只跑 R1 单项，从 v9b@3999 ckpt 暖启动
RESUME_FROM=ckpt_multi_action_v9b/u003999.pkl \
  bash scripts/run_fast_serial.sh r1_only

# 调整 gate 节奏 / baseline（v9c 跑完后用 v9c 的 spf/garr 替换）
POLL_SEC=180 MIN_UPD=400 BASELINE_SPF=27.0 BASELINE_GARR=38.6 \
  bash scripts/run_fast_serial.sh
```

输出：

- `logs/multi_action_v10_<variant>.log` — 单条 stdout
- `logs/multi_action_v10_<variant>/` — tensorboard
- `ckpt_multi_action_v10_<variant>/` — 每 100 upd 一个 pkl
- `logs/fast_serial_summary.tsv` — 所有判决的 TSV 汇总（一行/variant）

实现：

- `scripts/run_fast_serial.sh` — 串行调度 + 轮询 + kill
- `orbit_wars_rl/scripts/check_fast_gate.py` — 末窗口均值 + 2/3 票决（spf/garr/pdelta）+ 硬 kill（ev/clip）

| ID | 假设 | 状态 | upd | ev@500 | replay vs v20 (first-80) | 决策 |
|---|---|---|---|---|---|---|
| — | Phase 0 收尾 | 🔄 | — | — | — | 阻塞中 |
| r1_only | prod_share_delta 替换 level | ⏳ | — | — | — | — |
| r2_release | release_bonus (src_garr/src_prod) | ⏳ | — | — | — | — |
| r4_emit | emit_log_reward | ⏳ | — | — | — | — |
| r1_r2_r4 | 三项叠加 | ⏳ | — | — | — | — |
| C1 | v20 进 frozen pool | ⏳ | — | — | — | — |
| A1' | threat_ratio + eta | ⏳ | — | — | — | — |

每完成一条 FAST，在本表追加一行；命令与 gate 见 [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md)。

---

## 7. 文档索引

| 文档 | 用途 |
|---|---|
| [`DAY4_PROGRESS.zh.md`](DAY4_PROGRESS.zh.md) | Day4 完整审计 + v9 设计与实现 |
| [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md) | Day5 战术规划（Track A/B/C、优先级） |
| **本文** | 现状 SSOT、Phase 0 checklist、eval 数字 |
| [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md) | 500 upd 启动、gate、PROMOTE/KILL |
| [`TOP10_REPLAY_METRICS.zh.md`](TOP10_REPLAY_METRICS.zh.md) | **2630 局 4P 高手 replay 指标 SSOT**（Day5 方向对照） |
| [`DAY5_TRAINING_ACTIONS.zh.md`](DAY5_TRAINING_ACTIONS.zh.md) | **Day5 训练动作清单**（R1/R2/R3/R4/A1'/C1 落地公式） |
| [`H2H_EVAL_RUNBOOK.md`](H2H_EVAL_RUNBOOK.md) | export / gauntlet / replay 全流程 |

---

## 附录 A — v9 训练末段数字（复制用）

```
# DONE
v9a @3999: spf=11.4  garr=58.5  pS=0.29  ptS=0.28  fLog=0.28  WRr=0.88 WRf=0.59
v9b @3999: spf=21.2  garr=49.9  pS=0.41  ptS=0.39  fLog=0.36  WRr=0.94 WRf=0.88

# LIVE (2026-05-25 monitor)
v9c @2620: spf=27.2  garr=38.6  pS=0.39  ptS=0.38  fLog=0.43
v9d @2619: spf=20.9  garr=56.9  pS=0.41  ptS=0.39  fLog=0.37

# EVAL (v9b only)
v9b gauntlet vs v9a: 17/20 (10×2色) / 17/20 单色 log 亦存在
v9b gauntlet vs v20: 0/40 (20×2色)
v9b replay vs v20 first-80: spf=4.76 garr=31.85 flip_proxy=3.88% emit2=2.8%
v9b replay vs v20 full:     spf=4.81 garr=21.68 z0=41.5%
```

## 附录 B — Phase 0 完成后一键 eval（5090）

```bash
# 1. Export（v9c/d 跑完后）
for v in v9a v9b v9c v9d; do
  python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_${v}/ckpt_003999.pkl \
    --template submission_rl_v4.py \
    --out submission_rl_${v}_u3999.py
done

# 2. Gauntlet
python -m orbit_wars_rl.scripts.h2h_gauntlet \
  --agent submission_rl_v9c_u3999.py \
  --opponents submission_rl_v9a_u3999.py submission_rl_v9b_u3999.py submission_v20_0513.py \
  --num-games 20 \
  > logs/h2h_v9c_gauntlet.log 2>&1

# 3. Replay vs v20
python -m orbit_wars_rl.scripts.replay_analyze \
  --agent-a submission_rl_v9c_u3999.py \
  --agent-b submission_v20_0513.py \
  --num-games 5 --seed-base 0 \
  --out logs/replay_analyze/v9c_vs_v20.json
```

完整流程见 [`H2H_EVAL_RUNBOOK.md`](H2H_EVAL_RUNBOOK.md) §6。
