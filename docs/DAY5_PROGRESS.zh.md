# DAY5 进展 — 现状梳理 + v11 失败分析 + Ablation 计划

> 写于 2026-05-25，**2026-05-26 多次更新**。
> Day4 v9 ablation 已闭环；v11 全栈 G1 完成，**replay vs v20 失败（spf 1.50, garr 7.86）**；
> Day5 主线改为 **Ablation 拆解**（K=8 / 关 R4）+ **G2 续跑**双线。
> 战术规划见 [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md)，操作手册见 [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md)。

---

## 0. TL;DR（2026-05-26 PM 快照）

| 维度 | 状态 |
|---|---|
| **v11 G1 训练** | ✅ 完成 800 upd，self-play `tG=1363, e2=85%, spf=79`，`ev=0.99` |
| **v11 G1 vs v20 replay** | ❌ **比 v9b 还差**：spf=**1.50** garr=**7.86** flip=**0.66%** |
| **G2/G3** | 因 launcher bug 秒崩；脚本已修，可 SKIP_PHASE1=1 续跑 |
| **失败根因** | R4 `EMIT_LOG` + K=16 → policy 学成「家里钱多 → 撒小钱」；对 v20 早期 garr 极低分布外退化为 spf=1 |
| **Day5 主线** | ❌ 不再开新的「全栈一次堆 5 信号」实验；改为 Ablation 拆解 |
| **下一步** | (1) G2 续跑 sanity check (2) **K=8 + 关 R4** ablation (3) 必要时上 v20 buffer curriculum |

**今晚两条命令（5090，可顺序也可在 G2 后开 ablation）：**

```bash
# 1) G2 续跑（4h，curriculum sanity check）
nohup env SKIP_PHASE1=1 \
  CKPT_G1=ckpt_multi_action_v11_g1_scratch/ckpt_000799.pkl \
  bash scripts/run_v11_validation.sh \
  > logs/v11_g2g3.launcher.log 2>&1 &

# 2) Ablation（12h，K=8 / 关 R4 三路对照；可在 G2 完后启）
nohup bash scripts/run_v11_ablation.sh \
  > logs/v11_ablation.launcher.log 2>&1 &
```

每条 ckpt 完都用 **CPU 一键 export+replay**：

```bash
bash scripts/quick_replay.sh \
  ckpt_multi_action_v11_<tag>/ckpt_000799.pkl  v11_<tag>
```

看 first-80 gate（spf>10, garr>60, flip>6%, e2+>5%）。

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

## 6. v11 全信号验证（今晚跑这条）

**一条命令（5090 后台，3 组串行 × ~800 upd ≈ 4h/组）：**

```bash
nohup bash scripts/run_v11_validation.sh \
  > logs/v11_validation.launcher.log 2>&1 &
echo "pid=$!"
```

### 6.1 全栈配方（无 ablation）

| 层 | 内容 |
|---|---|
| **Reward** | R1 delta + R2 release + R4 emit(**gated**) + R5 capture + planet_share |
| **Obs** | A1' threat_ratio/eta/net_inbound + global total_garr + fleets_in_flight |
| **Scale** | `episode_steps=500`, `max_fleets_per_turn=16` |
| **C1** | G2/G3：25% vs G1 anchor ckpt + 50% frozen self + 25% random |
| **未含** | 4P env（Week2）；真 v20 Python bot（用同架构 anchor ckpt 代 C1） |

### 6.2 三组验证（同一配方，递进训练）

| 组 | 名称 | 做什么 | 产出 |
|---|---|---|---|
| **G1** | `g1_scratch` | 从头 800 upd | `ckpt_multi_action_v11_g1_scratch/` |
| **G2** | `g2_curriculum` | resume G1 + **C1** strong=G1@399 | `ckpt_multi_action_v11_g2_curriculum/` |
| **G3** | `g3_continue` | resume G2 再 800 upd | `ckpt_multi_action_v11_g3_continue/` |

汇总：`logs/v11_validation_summary.tsv`

### 6.3 明早 ~10:00 检查

```bash
tail -20 logs/v11_validation.launcher.log
python -m orbit_wars_rl.scripts.monitor_train --once \
  logs/multi_action_v11_g1_scratch.log \
  logs/multi_action_v11_g2_curriculum.log \
  logs/multi_action_v11_g3_continue.log
column -t -s $'\t' logs/v11_validation_summary.tsv
```

**Sign-of-life（对照 Top10 first-80）：**

| metric | 目标 | 说明 |
|---|---|---|
| `tG` | **> 60** | 总驻防（Top10 赢家 ~152） |
| `e2` | **> 0.05** | emit≥2 占比 |
| `z0` | **0.15–0.50** | 高手也大量不发 turn |
| `pkR` | **> 2.0** | 囤放波动 |
| `ev` | **> 0.5** @ G3 末段 | 训练健康 |

G3 健康 → export `ckpt_g3` → h2h vs v20（`H2H_EVAL_RUNBOOK.md`）。

### 6.4 旧 FAST 串行（r1_r2_r4 分项）

已被 v11 全栈验证取代；保留 `scripts/run_fast_serial.sh` 供日后单项调试。

### 6.5 v12 4P 并行排期（与 v11 同配方）

**工程双线、GPU 串行**：5090 上不能同时跑两个 JAX 训练；「并行」= v11 先跑（或已在跑），v12 接在后面或单独 `v12_only`。

| Track | 玩家数 | Launcher | 输出 |
|---|---|---|---|
| **A v11** | 2P（默认） | `scripts/run_v11_validation.sh` | `logs/multi_action_v11_g*.log` |
| **B v12** | 4P | `ORBITWARS_NUM_PLAYERS=4 bash scripts/run_v12_validation.sh` | `logs/multi_action_v12_g*.log` |
| **一键 pipeline** | 2P→4P 串行 ~26h | `bash scripts/run_parallel_tracks.sh` | 两份 summary TSV |

**若 v11 已在跑**（不要开 `pipeline`，会抢 GPU）：

```bash
# v11 跑完后单独开 v12
nohup env MODE=v12_only bash scripts/run_parallel_tracks.sh \
  > logs/v12_only.launcher.log 2>&1 &
```

**v12 实现要点（MVP）**：

- `ORBITWARS_NUM_PLAYERS=4` 在 **进程启动前** 设置（import 时读 constants；与 2P 默认互不干扰）
- 4 家 home：anchor group 四象限 slot 0–3；`home_planet_idx` shape `[4]`
- P0 学习；P1–P3 共享 frozen/strong/random（`rollout_4p.py`）
- Obs：foe = 所有非己玩家聚合（threat / global share）；`pdΔ` baseline **0.25**（非 2P 的 0.5）
- Config：`orbit_wars_rl/configs/multi_action_v12_4p.yaml`（`num_envs: 96`，略降 VRAM）

**Smoke（5090 或本机，1 upd/phase）**：

```bash
ORBITWARS_NUM_PLAYERS=4 UPD_PER_PHASE=1 bash scripts/run_v12_validation.sh
```

**4P sign-of-life（G3）**：同 v11 的 `tG/e2/pkR/ev`，另看 `pdΔ > 0`（相对 fair share 0.25）。

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
| **R5** | `capture_flip_reward` | `CAPTURE=0.02` | planet flip 事件 × prod 权重 |
| **R4'** | `emit_log_reward_gated` | `EMIT_GATED=1` | 仅 release_factor>0 时奖 emit |
| **A1'** | `encode.py` planet +3 | — | threat_ratio, net_inbound, eta_foe_min |
| **Global obs** | `encode.py` global +3 | — | total_garr_norm, n_fleets mine/enemy |
| **C1** | `runner.py` strong_ratio | yaml | 25% vs anchor ckpt（phase 2+） |

`env.py`：`non_terminal_r` 接入 R1/R2/R4 call site（player 0）。

`test_rewards.py`：新增 R1/R2/R4 单测 + NUM_PLAYERS 断言；**52/52 pass**。

### 8.2 训练 metric（`ppo/update.py` + `runner.py`）

| metric | log 列 | 含义 |
|---|---|---|
| `prod_share_delta` | `pdΔ` | batch 近似 prod share |
| `peak_over_mean_garr` | `pkR` | 囤放波动（Top10 ~4.3） |
| `total_garrison_my` | `tG` | **全星总驻防**（Top10 gate >60） |
| `emit2_rate` | `e2` | emit≥2 占比（Top10 ~14%） |
| `fleets_in_flight` | `nF` | 在途舰队数（Top10 ~11） |
| `zero_emit_rate` | `z0` | 不发 turn 占比（目标 0.15–0.50） |

`runner.py` log 含 `tG e2 nF`；C1 对手 tag `strn/frzn/rand`。

### 8.3 监控与 Gate（新/改）

| 文件 | 改动 |
|---|---|
| `orbit_wars_rl/scripts/monitor_train.py` | `LINE_RE` 兼容 `pdΔ` / `pkR`（旧 log 仍可读） |
| `orbit_wars_rl/scripts/check_fast_gate.py` | **新增** — 末窗口均值 + 2/3 票决 + ev/clip 硬 KILL |
| `scripts/run_fast_serial.sh` | **新增** — 串行 launcher + gate 轮询 + TSV 汇总 |
| `scripts/run_day5_fast_from_v9c.sh` | **新增** — kill v9c/d + 自动 resume v9c ckpt |
| `scripts/run_v11_validation.sh` | v11 三阶段全栈验证 |
| `scripts/run_v12_validation.sh` | v12 4P 三阶段（需 `ORBITWARS_NUM_PLAYERS=4`） |
| `scripts/run_parallel_tracks.sh` | v11→v12 pipeline / `MODE=v12_only` |

### 8.4 配置

| 文件 | 用途 |
|---|---|
| `orbit_wars_rl/configs/multi_action_v11_full.yaml` | v11 全栈模板（2P） |
| `orbit_wars_rl/configs/multi_action_v12_4p.yaml` | v12 全栈模板（4P，`num_envs: 96`） |
| `orbit_wars_rl/configs/multi_action_v10_fast.yaml` | FAST 模板：`num_updates=1000`，其余对齐 v9 |
| `orbit_wars_rl/configs/day5_smoke.yaml` | CPU smoke（验证 reward 路径无 NaN） |

### 8.4b 4P env（v12 MVP）

| 文件 | 改动 |
|---|---|
| `env/constants.py` | `ORBITWARS_NUM_PLAYERS` env var（2 默认 / 4） |
| `env/init.py` | 4P 四 home slot |
| `env/dynamics.py` | `launch_fleets_with_info` 4-tuple |
| `env/rewards.py` | `terminal_reward` N-player max score；shaping vs strongest opp |
| `features/encode.py` | foe 聚合 inbound + global share |
| `ppo/rollout_4p.py` | P0 学 + P1–3 frozen/random |
| `ppo/runner.py` | `NUM_PLAYERS==4` 分支 |
| `ppo/update.py` | `pdΔ` baseline `1/NUM_PLAYERS` |

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

### 8.7 仍 defer

- **真 v20 Python bot 进 rollout**（需非 JIT 路径或 BC 蒸馏；今晚 C1 用 G1 anchor ckpt）
- overnight 4000 launcher（G3 健康后再扩）
- **4P 进阶**：独立 opponent pool（P1/P2/P3 不同 ckpt）、Kaggle 4P 全图 curriculum — MVP 已 land，见 §6.5

---

## 9. 文档索引

| 文档 | 用途 |
|---|---|
| [`DAY4_PROGRESS.zh.md`](DAY4_PROGRESS.zh.md) | Day4 完整审计 + v9 设计与实现 |
| [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md) | Day5 战术规划（Track A/B/C、优先级） |
| **本文** | 现状 SSOT、§6 v11 全栈验证、§7 看日志、§8 改动清单 |
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
