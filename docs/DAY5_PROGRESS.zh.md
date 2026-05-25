# DAY5 进展 — 现状梳理 + Reward v3 FAST 迭代

> 写于 2026-05-25，**2026-05-26 更新**。Day4 v9 ablation 已闭环；v9c/d 在 ~3200 upd 处 **手动 kill**，改走 **Reward v3 FAST（scratch）**。
> 本文是 Day5 的**单一事实来源（status SSOT）**；战术规划见 [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md)，操作手册见 [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md)。

---

## 0. TL;DR（2026-05-26 快照）

| 维度 | 状态 |
|---|---|
| **Day4 目标** | ✅ shaping v2 落地；✅ v9a/b 4000 upd 完成；⏸️ v9c/d @~3200 **已 kill** |
| **shaping 在 self-play** | ✅ v9b gauntlet vs v9a **17/20**；v9c 训练 spf **30.1**（最高） |
| **shaping vs v20** | ❌ v9b gauntlet vs v20 **0/40**；replay spf **4.76**（训练 21.2） |
| **Day5 代码** | ✅ Reward v3（R1/R2/R3/R4）+ FAST gate + 串行 launcher **已落地** |
| **Day5 训练** | 🔄 **top10 完整配方 `r1_r2_r4` scratch FAST**（见 §6） |
| **Resume 策略** | v3 reward **语义级改动 → scratch**；fork v9c 仅适合作 FAST 探路 |

**v9c/d 末段 monitor（5090，2026-05-26 kill 前）：**

```
multi_action_v9c  upd 3198  ev 0.97  clip 0.08  spf 30.1  garr 44.1  pS 0.37  fLog 0.43  live
multi_action_v9d  upd 3197  ev 0.97  clip 0.08  spf 21.4  garr 61.8  pS 0.40  fLog 0.37  live
```

**当前主推一条命令（top10 完整 reward 版，scratch，后台）：**

```bash
nohup env RESUME_FROM= MIN_UPD=600 BASELINE_SPF=12 BASELINE_GARR=25 \
  bash scripts/run_fast_serial.sh r1_r2_r4 \
  > logs/fast_serial_r1_r2_r4.launcher.log 2>&1 &
```

看日志见 **§7**。

---

## 1. v9 Ablation 全景

### 1.1 四路配置对照

| Config | Shaping | 训练状态 | 末段 / 当前 metric | 备注 |
|---|---|---|---|---|
| **v9a** | 无（sparse ±1） | ✅ u3999 DONE | spf **11.4**, garr **58.5**, pS **0.29** | control |
| **v9b** | prod=0.01 | ✅ u3999 DONE | spf **21.2**, garr **49.9**, pS **0.41** | **唯一完成 H2H+replay 的 run** |
| **v9c** | prod+planet+fleet_log | ⏸️ u3198 killed | spf **30.1**, garr **44.1**, fLog **0.43** | 训练 spf 最高；ckpt 保留 |
| **v9d** | prod+planet（无 fleet_log） | ⏸️ u3197 killed | spf **21.4**, garr **61.8**, pS **0.40** | 训练 garr 最高 |

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
| 0.1 | v9c/d 跑完 4000 upd | ⏸️ @~3200 kill | ckpt 在 `ckpt_multi_action_v9c/` 保留 |
| 0.2 | export v9a/b/c/d @3999 | ⏳ v9b 已有 `submission_rl_v9b_u3999.py` | 4 个 submission `.py` |
| 0.3 | gauntlet v9c vs [v9a, v9b, v20] | ⏳ | `logs/h2h_v9c_gauntlet.log` |
| 0.4 | gauntlet v9d vs [v9a, v9b, v20] | ⏳ | `logs/h2h_v9d_gauntlet.log` |
| 0.5 | replay v9c vs v20 ×5 | ⏳ | `logs/replay_analyze/v9c_vs_v20.json` |
| 0.6 | replay v9d vs v20 ×5 | ⏳ | `logs/replay_analyze/v9d_vs_v20.json` |
| 0.7 | **frozen base 书面决策** | ⏳ | 见 §4 规则 |
| 0.8 | 更新本文 §1–§2 最终数字 | ⏳ | — |

**FAST baseline**：scratch 用 `spf=12, garr=25, min_upd=600`（§6.2）；fork v9c 时用 v9c log 自动读 baseline（`run_day5_fast_from_v9c.sh`）。

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

## 6. Day5 FAST 启动命令

### 6.1 Variant 对照（top10 → reward v3）

| Variant | 含义 | Shaping env vars |
|---|---|---|
| `r1_only` | R1 prod_share **delta** 替换 level | `PROD_SHARE=0`, `PROD_SHARE_DELTA=1.0`, 保留 `planet+fleet_log` |
| `r2_release` | R2 release 替换 fleet_log | `RELEASE=0.05`, `RELEASE_K=20`, 关 `FLEET_LOG` |
| `r4_emit` | R4 valid launch 数 log | `EMIT_LOG=0.01`, 保留 v9c v2 全家桶 |
| **`r1_r2_r4`** | **top10 完整配方** | delta + release + emit；关 level prod + fleet_log |

R3（`NUM_PLAYERS` baseline bug）已在代码里修，无需单独 variant。

### 6.2 推荐：完整配方 scratch + 后台

```bash
# 前台（占终端；train 虽 nohup 但 launcher 轮询 gate 在前台）
RESUME_FROM= MIN_UPD=600 BASELINE_SPF=12 BASELINE_GARR=25 \
  bash scripts/run_fast_serial.sh r1_r2_r4

# 后台（推荐 5090 过夜）
nohup env RESUME_FROM= MIN_UPD=600 BASELINE_SPF=12 BASELINE_GARR=25 \
  bash scripts/run_fast_serial.sh r1_r2_r4 \
  > logs/fast_serial_r1_r2_r4.launcher.log 2>&1 &
echo "launcher pid=$!"

# tmux（可随时 attach）
tmux new -s fast_r124
RESUME_FROM= MIN_UPD=600 BASELINE_SPF=12 BASELINE_GARR=25 \
  bash scripts/run_fast_serial.sh r1_r2_r4
```

**Gate baseline 说明（scratch）：** 不能用 v9c@3200 的 spf=30/garr=44（从头训 upd=300 达不到）。上表用 `spf=12, garr=25, min_upd=600` 作 scratch 合理起点；PROMOTE 后仍需 h2h vs v20 确认。

### 6.3 可选：fork v9c 快速探路（不推荐 combo）

```bash
bash scripts/run_day5_fast_from_v9c.sh r1_r2_r4   # 自动 resume v9c 最新 ckpt
bash scripts/run_day5_fast_from_v9c.sh            # 串行 4 条 ablation
```

fork 只加载 **权重**，upd 从 0 重计；value head 按旧 return 校准，combo 易 clip spike。**PROMOTE 后 overnight 仍应 scratch 4000 upd。**

### 6.4 产出路径

| 路径 | 内容 |
|---|---|
| `logs/multi_action_v10_r1_r2_r4.log` | 训练 stdout（每 upd 一行 metric） |
| `logs/multi_action_v10_r1_r2_r4/` | TensorBoard |
| `ckpt_multi_action_v10_r1_r2_r4/ckpt_XXXXXX.pkl` | 每 100 upd 存盘 |
| `logs/fast_serial_r1_r2_r4.launcher.log` | launcher gate 轮询输出 |
| `logs/fast_serial_summary.tsv` | 各 variant 判决 TSV |

### 6.5 FAST 实验台账

| ID | 假设 | 状态 | upd | ev | replay vs v20 | 决策 |
|---|---|---|---|---|---|---|
| — | v9c/d Phase 0 | ⏸️ @3200 kill | 3198 | 0.97 | — | ckpt 保留，改走 v3 FAST |
| **r1_r2_r4** | top10 完整 reward v3 | 🔄 待跑 | — | — | — | scratch FAST |
| r1_only | prod_share_delta | ⏳ | — | — | — | — |
| r2_release | release_bonus | ⏳ | — | — | — | — |
| r4_emit | emit_log | ⏳ | — | — | — | — |
| C1 | v20 进 frozen pool | ⏳ | — | — | — | reward v3 后再做 |
| A1' | threat_ratio + eta | ⏳ | — | — | — | reward v3 后再做 |

---

## 7. 看日志与监控（操作手册）

### 7.1 训练 health 一览（最常用）

```bash
# 单次快照（推荐每隔 30–60 min 看一次）
python -m orbit_wars_rl.scripts.monitor_train --once \
  logs/multi_action_v10_r1_r2_r4.log

# 实时滚动（多 log）
python -m orbit_wars_rl.scripts.monitor_train \
  logs/multi_action_v10_r1_r2_r4.log
```

**表格列含义：**

| 列 | 含义 | 健康区间 |
|---|---|---|
| `ev` | explained_variance | > 0.5（scratch 前 300 upd 可偏低） |
| `clip` | PPO clip fraction | < 0.25 WARN，> 0.35 ALERT |
| `spf` | mean ships per fleet | scratch 目标 upd600 > 15；combo 目标 > 20 |
| `garr` | mean garrison | 不应长期 < 15 |
| `pS` / `ptS` | prod/planet share | 缓慢上升 |
| `pdΔ` | prod_share_delta（v3 新列） | 正值偏多 = 在抢 share |
| `pkR` | peak/mean garr（v3 新列） | > 2 表示有囤放波动（R2 目标） |

monitor 会对 ev/clip/kl 自动打 `[WARN]` / `[ALERT]`；scratch 前 ~300 upd 的 clip spike **预期内**，u500 后应回落。

### 7.2 训练 log 原文

```bash
# 末 20 行
tail -20 logs/multi_action_v10_r1_r2_r4.log

# 实时跟踪
tail -f logs/multi_action_v10_r1_r2_r4.log

# 确认 reward banner（启动后 ~60s，JIT 编译完）
grep '^\[reward\]' logs/multi_action_v10_r1_r2_r4.log | head -1
grep '^\[resume\]' logs/multi_action_v10_r1_r2_r4.log    # scratch 时应无此行
```

banner 应含 `PROD_SHARE_DELTA=1.0 RELEASE=0.05 EMIT_LOG=0.01`（combo）。

### 7.3 Launcher / Gate 判决

```bash
# 后台 launcher 输出（每 POLL_SEC=120s 一行 gate）
tail -f logs/fast_serial_r1_r2_r4.launcher.log

# 手动跑一次 gate（exit 0=PROMOTE, 1=CONTINUE, 2=KILL）
python -m orbit_wars_rl.scripts.check_fast_gate \
  --log logs/multi_action_v10_r1_r2_r4.log \
  --window 50 --min-upd 600 \
  --baseline-spf 12 --baseline-garr 25

# 历史判决汇总
column -t -s $'\t' logs/fast_serial_summary.tsv
```

**Gate 规则（2/3 票决）：** spf > baseline+2 **或** garr > baseline+5 **或** pdelta > baseline+0.02；硬 KILL：ev < 0.30 或 clip > 0.35（窗口均值）。

### 7.4 进程与 GPU

```bash
# 训练是否在跑
pgrep -af 'multi_action_v10_r1_r2_r4' || echo "not running"

# launcher 是否在跑
pgrep -af 'run_fast_serial.sh' || echo "launcher not running"

# GPU
watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'
```

### 7.5 停止

```bash
# 停当前 FAST 训练
pkill -f 'multi_action_v10_r1_r2_r4'

# 停 launcher（会停 gate 轮询；训练子进程可能仍在，需一并 pkill）
pkill -f 'run_fast_serial.sh r1_r2_r4'
```

### 7.6 PROMOTE 之后

```bash
# export
python -m orbit_wars_rl.scripts.export_submission \
  --ckpt ckpt_multi_action_v10_r1_r2_r4/ckpt_000XXX.pkl \
  --template submission_rl_v4.py \
  --out submission_rl_v10_r1_r2_r4_uXXX.py

# h2h vs v20 — 见 H2H_EVAL_RUNBOOK.md
```

---

## 8. 代码与工具改动清单（2026-05-25 ~ 26）

### 8.1 Reward v3（`orbit_wars_rl/env/rewards.py` + `env.py`）

| ID | 函数 | env var | 说明 |
|---|---|---|---|
| **R1** | `prod_share_delta_reward` | `ORBITWARS_SHAPING_PROD_SHARE_DELTA` | Δshare × α；与 level `prod_share` **互斥** |
| **R2** | `release_bonus_reward` | `RELEASE=0.05`, `RELEASE_K=20` | `tanh(src_garr/src_prod/K - 1)` × log_size；**无硬编码阈值**；与 `fleet_log` **互斥** |
| **R3** | `prod_share_reward` / `planet_share_reward` | — | `n_players` 改 `constants.NUM_PLAYERS`（修 4P bug） |
| **R4** | `emit_log_reward` | `ORBITWARS_SHAPING_EMIT_LOG` | `log1p(n_valid_launches)` per turn |

`env.py`：`non_terminal_r` 接入 R1/R2/R4 call site（player 0）。

`test_rewards.py`：新增 R1/R2/R4 单测 + NUM_PLAYERS 断言；**52/52 pass**。

### 8.2 训练 metric（`ppo/update.py` + `runner.py`）

| metric | log 列 | 含义 |
|---|---|---|
| `prod_share_delta` | `pdΔ` | batch 近似 prod share level（gate 用） |
| `peak_over_mean_garr` | `pkR` | max_garr / mean_garr（R2 FAST 目标） |

`runner.py` banner echo 全部 v3 shaping coef；训练 log 行追加 `pdΔ` / `pkR`。

### 8.3 监控与 Gate（新/改）

| 文件 | 改动 |
|---|---|
| `orbit_wars_rl/scripts/monitor_train.py` | `LINE_RE` 兼容 `pdΔ` / `pkR`（旧 log 仍可读） |
| `orbit_wars_rl/scripts/check_fast_gate.py` | **新增** — 末窗口均值 + 2/3 票决 + ev/clip 硬 KILL |
| `scripts/run_fast_serial.sh` | **新增** — 串行 launcher + gate 轮询 + TSV 汇总 |
| `scripts/run_day5_fast_from_v9c.sh` | **新增** — kill v9c/d + 自动 resume v9c ckpt |

### 8.4 配置

| 文件 | 用途 |
|---|---|
| `orbit_wars_rl/configs/multi_action_v10_fast.yaml` | FAST 模板：`num_updates=1000`，其余对齐 v9 |
| `orbit_wars_rl/configs/day5_smoke.yaml` | CPU smoke（验证 reward 路径无 NaN） |

### 8.5 文档（同日新增/更新）

| 文档 | 内容 |
|---|---|
| [`TOP10_REPLAY_METRICS.zh.md`](TOP10_REPLAY_METRICS.zh.md) | 2630 局 top10% replay 指标 SSOT |
| [`DAY5_TRAINING_ACTIONS.zh.md`](DAY5_TRAINING_ACTIONS.zh.md) | R1/R2/R3/R4/A1'/C1 落地公式与系数 |
| [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md) | FAST lane 操作手册 |
| [`H2H_EVAL_RUNBOOK.md`](H2H_EVAL_RUNBOOK.md) | export / gauntlet / replay 流程 |

### 8.6 分析工具

| 文件 | 用途 |
|---|---|
| `orbit_wars_rl/scripts/aggregate_top10_replays.py` | 聚合 top10 replay JSON → 指标 |
| `logs/top10_aggregate_2026-05-04.json` | 2630 局聚合结果 |

### 8.7 尚未实现（下一批）

- **A1'** threat obs（`threat_ratio`, `eta_foe_min`, `net_inbound`）
- **C1** v20 进 frozen self-play pool
- overnight 4000 upd yaml / launcher（PROMOTE 后手动扩 `num_updates`）

---

## 9. 文档索引

| 文档 | 用途 |
|---|---|
| [`DAY4_PROGRESS.zh.md`](DAY4_PROGRESS.zh.md) | Day4 完整审计 + v9 设计与实现 |
| [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md) | Day5 战术规划（Track A/B/C、优先级） |
| **本文** | 现状 SSOT、§6 启动、§7 看日志、§8 改动清单 |
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

# LIVE → KILLED (2026-05-26)
v9c @3198: spf=30.1  garr=44.1  pS=0.37  ptS=0.36  fLog=0.43
v9d @3197: spf=21.4  garr=61.8  pS=0.40  ptS=0.39  fLog=0.37
# v9c ckpt 保留: ckpt_multi_action_v9c/ckpt_003199.pkl (或目录内最新)

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
