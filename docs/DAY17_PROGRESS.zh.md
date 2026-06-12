# DAY17 — v17：History×50 + ETA-Lead + Frog Curriculum

> **策略升级**：在 v15 Frog BO3 基线上，针对 v16a replay 暴露的「打移动目标不准 / 游牧空城」与 Lux 时序特征差距，
> 一次性落地 **global hist=50**、**ETA-lead dst**、**safe_emit 特征**、**Frog curriculum capture**，
> 并对齐 Lux Top2 超参。Resume 自 v15 u9999（shape-adapt 新维）。

---

## 1. 动机：v15/v16a 告诉我们什么

### v15 终态 (u9999)

| 维度 | 自博弈 | vs v20 | 结论 |
|------|--------|--------|------|
| WRr vs random | 0.88 | — | 自博弈强 |
| vs v20 WLD | — | **0/10** | 实战零胜 |
| emits / z0 | 0.60 / 0.72 | 0.23 / ~40% | 面对 v20 极度保守 |
| 时序 | 无 history | — | 单帧观测，看不到 orbit 节奏 |

**根因**：纯 sparse terminal + 对称自博弈不足以学「瞄准移动目标 + 守家权衡」；缺少 Lux 式 temporal stack。

### v16a capture extend

| 维度 | v15 u9999 | v16a u3999 | 解读 |
|------|-----------|------------|------|
| vs-v20 spf | 29 | **69** | 舰队规模↑ |
| vs-v20 flip% | 22–32% | **39–50%** | 占点能力↑ |
| vs-v20 WLD | 0/10 | 0/3（u3299/u3499 曾 **1/2/0**） | 首胜但不稳定 |
| 副作用 | — | garr 峰 395、游牧空城 | capture 常驻 0.05 偏置 |

**Replay 诊断（u3299/u3499 seed=1 胜局）**：

```
问题 A — 移动目标打偏
  encode 有 lead_x/y @ t+15/30，但 DstHead pair 仍用当前 planet_x/y
  → fleet 打 orbiting 目标时系统性追旧位置，偶发出界

问题 B — 游牧换家
  capture shaping 奖励「占新球」；anti-hoard 只罚 0-emit，不罚「掏空基地」
  → 大舰队不停留、频繁换家清空 garrison
```

**v17 设计原则**：A/B 分治 —— A 用 **ETA-lead + hist**；B 用 **curriculum capture + defense_empty + safe_emit**；终态仍是 **Frog sparse 赢 match 才奖**。

---

## 2. 方案总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        v17 观测 (encode)                         │
├─────────────────────────────────────────────────────────────────┤
│  Planet 41d = v15 39d + safe_emit_margin + hold_value            │
│  Global 427d = base 27d + hist_stack(50 × 8d = 400d)            │
│  Fleet 10d — 不变                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     决策 (ActorCritic K=8)                         │
├─────────────────────────────────────────────────────────────────┤
│  DstHead pair: ETA-lead 坐标 (orbiting 目标按到达时刻预测位置)      │
│  其余 head 同 v15；zero_sum value head 保留                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     奖励 (Frog curriculum)                         │
├─────────────────────────────────────────────────────────────────┤
│  终端: match win/loss ±1 (BO3, wins_needed=2) — 主信号            │
│  过程: capture 0.03 → anneal → 0 (8000u)                        │
│        × hist 稳定度缩放 (garr_adv 近 10 帧均值)                   │
│        defense_empty 罚高 prod 星球 garrison 低于 reserve          │
│  保留: anti-hoard 0.03 (仅 0-emit 囤兵)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 与 Lux Frog Parade 对齐

| Lux (100 step match) | OrbitWar v17 (500 step match) | 说明 |
|----------------------|-------------------------------|------|
| frame_stack_len=10 | **hist_len=50** | 10/100=10% ≈ 50/500=10% 时间覆盖 |
| temporal global 4d×10 | temporal global **8d×50** | 更丰富的 macro 信号 |
| spatial temporal stack | P1 待加 (planet 4ch×hist) | v17 先只做 global hist |
| ent≈1e-4, clip=0.2 | 同 | 低 entropy 收敛 |
| sparse ±1 终态 | BO3 match ±1 + capture curriculum | dense→sparse |

---

## 3. 特征工程详解

### 3.1 Global History Stack (hist=50)

**模块**：`orbit_wars_rl/features/history.py`

每步 env step 结束后，对双方 POV 各 append 一个 **8 维 temporal slice**，存入环形缓冲：

| Index | 字段 | 含义 |
|-------|------|------|
| TGF 0 | ship_mine_share | 我方 ship 占比 |
| TGF 1 | ship_foe_share | 敌方 ship 占比 |
| TGF 2 | prod_mine_share | 我方 prod 占比 |
| TGF 3 | prod_foe_share | 敌方 prod 占比 |
| TGF 4 | prod_advantage | prod 优势 [-1,1] |
| TGF 5 | garr_advantage | garrison 优势 [-1,1] |
| TGF 6 | fleet_mass_ratio | 在途舰队 / 总兵力 |
| TGF 7 | threat_pressure | 敌舰队威胁 / 我方 garrison |

**状态字段**：`EnvState.global_hist` shape `[NUM_PLAYERS, 50, 8]`

- match / series reset → 清零
- encode 时 flatten 拼到 base global 27d 后面

**维度变化**：

```
BASE_GLOBAL_FEAT_DIM = 27   (v15: step, shares, phase, proxies, BO3 context)
HIST_LEN = 50
TEMPORAL_GLOBAL_DIM = 8
GLOBAL_FEAT_DIM = 27 + 50×8 = 427
```

**保留 dims 18–23**：当前帧 temporal proxy（garr_to_prod, fleet_mass, prod/garr adv 等）与 hist stack **互补** —— proxy 是瞬时导数，hist 是轨迹。

### 3.2 Planet Safe Emit (dims 39–40)

**动机**：让 src/pct head 看见「能安全派出多少兵」，抑制掏空基地。

| Dim | 名称 | 公式 (mine planets) |
|-----|------|---------------------|
| 39 | `safe_emit_margin` | clip((garr − foe_soft_inbound − min_reserve) / garr, 0, 1) |
| 40 | `hold_value` | (prod / my_prod_max) × safe_emit_margin |

`min_reserve = max(8, prod × 2.5)` —— 高 prod 星球需留更多底。

```
PLANET_FEAT_DIM: 39 → 41
```

### 3.3 ETA-Lead Dst Pair

**模块**：`orbit_wars_rl/features/pair.py`

对 **orbiting** 目标：

1. 用当前坐标算初始 `dist`，得 `ETA = dist / fleet_speed`
2. `_predict_planet_pos(phase, radius, ω, ETA)` 得到达时刻 lead 坐标
3. `dist_norm`、sun_risk 基于 lead 坐标而非当前坐标

训练与 rollout 均传入 `planet_orbit_phase/radius/is_orbiting`、`angular_velocity`。

**预期效果**：fleet 打移动 comet/orbit 星球时不再系统性追旧位置。

---

## 4. 奖励：Frog Curriculum

**模块**：`orbit_wars_rl/env/rewards.py`

### 4.1 终态（不变）

- BO3：`wins_needed=2`，每 match 结束 ±1 terminal
- 无其他 shaping（`SHAPING_SCALE=0`）

### 4.2 过程 shaping（curriculum）

| 项 | 初值 | Anneal | 作用 |
|----|------|--------|------|
| `CAPTURE` | 0.03 | → 0 @ 8000u | 教 dst/占点，后期让 sparse 主导 |
| `DEFENSE_EMPTY` | 0.02 | → 0.01 @ 8000u | 罚高 prod 星球 garrison 低于 reserve |
| `ANTI_HOARD` | 0.03 | 固定 | 仅罚 0-emit 囤兵 |

**`set_curriculum(update)`**：runner 每 update 调用，动态更新 Python 侧 coef。

### 4.3 History-Aware Capture

`capture_hist_balance_reward`：

```python
base = CAPTURE * (新占 planet prod / total_prod)
stability = clip(mean(hist[-10:, garr_advantage]) + 0.5, 0.5, 1.5)
reward = base * stability
```

当 hist 显示 garr 优势在崩塌（游牧进攻）时，**降低 capture 奖励**，让 agent 更倾向守家换稳定 sparse win。

### 4.4 Defense Empty Penalty

`defense_empty_penalty_reward`：对我方 **prod≥2** 且 **garrison < min_reserve** 的星球计数，归一化后给负奖励。

---

## 5. 超参与训练配置

**配置**：`orbit_wars_rl/configs/multi_action_v17_frog_hist50.yaml`  
**启动**：`scripts/v17_frog_hist50.sh`

### vs v15 / v16a

| 参数 | v15 | v16a | **v17** |
|------|-----|------|---------|
| GLOBAL_FEAT_DIM | 27 | 27 | **427** |
| PLANET_FEAT_DIM | 39 | 39 | **41** |
| hist_len | 0 | 0 | **50** |
| lr_peak | 5e-5 | 2e-5 | **1e-4** |
| ent_coef_* | 1e-3~5e-3 | 同 v15 | **1e-4** |
| clip_eps | 0.10 | 0.10 | **0.20** |
| gae_lambda | 0.95 | 0.95 | **0.85** |
| CAPTURE | 0 | **0.05 常驻** | **0.03→0 curriculum** |
| eval_every | 100 | 100 | **200** |
| num_updates | 10000 | 5000 | **15000** |
| resume | scratch | v15 u9999 | **v15 u9999 shape-adapt** |

### Resume / Shape Adapt

v17 网络 input 维变大，无法直接 load v15 ckpt。`runner._adapt_strong_params` 对 `planet_proj` / `global_proj` 等 **zero-pad 新维度**，旧权重保留、新特征维初始无贡献，再 fine-tune。

---

## 6. 文件清单

| 文件 | 变更 |
|------|------|
| `orbit_wars_rl/features/history.py` | **新增** hist buffer + temporal slice |
| `orbit_wars_rl/features/encode.py` | planet +2d, global +400d hist flatten |
| `orbit_wars_rl/features/pair.py` | ETA-lead dst pair |
| `orbit_wars_rl/env/state.py` | +`global_hist` |
| `orbit_wars_rl/env/env.py` | step 后 update hist; 新 reward 项 |
| `orbit_wars_rl/env/init.py` | reset/match_reset 清零 hist |
| `orbit_wars_rl/env/rewards.py` | curriculum + capture_hist + defense_empty |
| `orbit_wars_rl/env/constants.py` | HIST_LEN=50, TEMPORAL_GLOBAL_DIM=8 |
| `orbit_wars_rl/net/model.py` | dst pair 传入 orbit 参数 |
| `orbit_wars_rl/ppo/rollout*.py` | rollout 存 orbit raw + model.apply kwargs |
| `orbit_wars_rl/ppo/runner.py` | shape-adapt resume + set_curriculum |
| `orbit_wars_rl/configs/multi_action_v17_frog_hist50.yaml` | **新增** |
| `scripts/v17_frog_hist50.sh` | **新增**（自动 resume 最新 ckpt + 训至 06:00） |
| `scripts/v17_frog_hist50_extend.sh` | **新增**（sparse 续训） |
| `scripts/v17_peak_ckpt_eval.sh` | **新增**（peak ckpt offline eval + HTML） |
| `submission_rl_v17.py` | **新增**（export parity ✅） |
| `orbit_wars_rl/inference/numpy_forward.py` | ETA-lead dst parity ✅ |
| `scripts/quick_replay.sh` | GLOBAL_DIM=427 → v17 template |

---

## 7. 执行进展

### 7.1 时间线

| 时间 | 事件 |
|------|------|
| 6/11 23:15 | Run#1 开跑，resume v15 u9999 shape-adapt |
| 6/12 ~00:03 | Run#1 ~u3600；submission parity **未修** → eval 全 WARN |
| 6/12 ~00:07 | **submission/numpy_forward parity 修复**；Run#2 重启，resume `ckpt_003599.pkl`，目标 06:00（25383u） |
| 6/12 ~00:30+ | eval_vs_v20 gate **恢复**（u3199 起正常） |
| 6/12 ~08:00 | Run#2 跑完 **u25382**；peak ckpt offline eval + HTML 完成 |

**日志**：

- Run#1（已停）：`logs/v17_frog_hist50_20260611_231528/train.log`
- Run#2（主 overnight）：`logs/v17_frog_hist50_20260612_000727/train.log`
- Peak eval：`logs/v17_peak_eval.log`

**Ckpt 目录**：`ckpt_multi_action_v17_frog_hist50/`（最新 `ckpt_025399.pkl` 附近）

### 7.2 基础设施

| 项 | 状态 |
|----|------|
| v17 代码落地 | **✅** |
| submission + numpy_forward parity | **✅**（export 16/16，smoke 通过） |
| eval_vs_v20 inline gate | **✅**（u3199 起） |
| overnight 长训 | **✅** Run#2 完成 ~25383u |

### 7.3 vs-v20：训练 inline gate（3 局，噪声大）

**Run#1 末期（parity 修好后）**

| update | spf | flip | z0 | WLD |
|--------|-----|------|-----|-----|
| u3199 | 44 | 40% | 73% | 0/3 |
| u3399 | 37 | 30% | 65% | 0/3 |

**Run#2 — capture 期 peak（本 run 计数）**

| update | spf | flip | z0 | WLD |
|--------|-----|------|-----|-----|
| u599 | **54** | **50%** | 59% | 0/3 |
| u999 | 49 | **56%** | 63% | 0/3 |
| u1399 | 40 | **58%** | 68% | 0/3 |
| u2199 | 39 | **59%** | 57% | 0/3 |

**Run#2 — capture anneal 后 / sparse 长训（u4400+）**

| update | spf | flip | z0 | WLD |
|--------|-----|------|-----|-----|
| u10199 | 35 | 39% | 51% | 0/3 |
| u22999 | 30 | 33% | 54% | 0/3 |
| u23799 | 29 | 23% | 54% | 0/3 |

**全程 inline gate：0 胜**（含 u2199 flip≈59% 时仍 0/3）。

### 7.4 vs-v20：peak ckpt offline eval（10 局 × 3 seed）

脚本：`bash scripts/v17_peak_ckpt_eval.sh`

| ckpt | seed_base | spf | flip | garr | WLD |
|------|-----------|-----|------|------|-----|
| **u1399** | 0 | 35 | 44% | 130 | **0/10** |
| u1399 | 1 | 37 | 38% | 135 | 0/10 |
| u1399 | 42 | 30 | 46% | 127 | 0/10 |
| **u2199** | 0 | 33 | 45% | 143 | **0/10** |
| u2199 | 1 | 36 | 42% | 150 | 0/10 |
| u2199 | 42 | 30 | **54%** | 166 | 0/10 |

**合计 60 局，0 胜。** flip 40–54% 仍不足以赢 v20；e2+ 全程 ≈0%。

### 7.5 HTML Replay（5090 路径）

| 对局 | 路径 |
|------|------|
| u1399 seed=0 | `logs/replay_html/v17_u1399_vs_v20_seed0/replay.html` |
| u1399 seed=1 | `logs/replay_html/v17_u1399_vs_v20_seed1/replay.html` |
| u2199 seed=0 | `logs/replay_html/v17_u2199_vs_v20_seed0/replay.html` |
| u2199 seed=1 | `logs/replay_html/v17_u2199_vs_v20_seed1/replay.html` |

（无胜局，未生成 `*_win` replay。）

### 7.6 自博弈（Run#2 末期 u~25382）

| 指标 | 值 | vs v15 u9999 |
|------|-----|--------------|
| WRr | 峰值 **0.94** | 0.88 |
| emits | 0.43–0.62 | 0.60 |
| spf | 40–79 | 55 |
| z0 | 0.66–0.83（末期 garr 回潮 450+） | 0.72 |
| e2+ | ~0.04 | ~0 |
| sps | ~1776（长训+eval 后） | ~2834 |

### 7.7 结论

1. **hist=50 + ETA-lead + capture curriculum 方向正确**：中期 flip **50–59%**，高于 v15（22–32%）和 v16a 常见水平。
2. **pure sparse 长训有害**：capture→0 后 flip/spf **回落到 v15 档**，中期 gain 被洗掉。
3. **仍未赢 v20**：即使 peak flip ckpt，offline **0/60**；无 v16a 式零星胜局。
4. **e2+ / spf 仍是短板**：spf peak ~54 vs v16a **69**；multi-emit 几乎为 0。
5. **最佳 ckpt 候选**：`ckpt_002199.pkl`（u2199，flip gate peak）或 `ckpt_001399.pkl` — **勿用末期 ckpt**。

---

## 8. Lux 提升空间路线图（与 v17 兼容）

| 方向 | v17 档位 | 状态 |
|------|----------|------|
| Rich global temporal (hist=50) | **P0 已实现** | ✅ |
| ETA-lead dst | **P0 已实现** | ✅ |
| safe_emit / hold_value | **P0 已实现** | ✅ |
| Frog curriculum capture→sparse | **P0 已实现** | ✅ |
| Lux 超参 (ent/clip/gae) | **P0 已实现** | ✅ |
| Per-tile can_flip (soft) | P1 | 配合 flip_hard_mask=false |
| Planet spatial hist 4ch | P1 | owner/garr/inbound_foe/was_captured |
| Symmetry aug @ test | P1 | Lux test-time 三变换平均 |
| Inferred opponent EMA | P1 | 从 hist buffer 派生 |
| Future frames (动态 ETA) | 部分 | planet lead t+15/30 已有 |

**策略不冲突**：上述 P1 项均为 v17 架构的自然延伸，不需改 reward 终态。

---

## 9. 待办 (P1)

1. ~~**submission_rl_v17.py 完整 parity**~~ **✅ 已完成**
2. ~~**numpy_forward.py ETA-lead**~~ **✅ 已完成**
3. ~~**eval_vs_v20 恢复**~~ **✅ 已完成**
4. **策略分支（优先级）**
   - 从 **u2199 / u1399** 短 fine-tune，**不要 anneal capture 到 0 再长 sparse 训**
   - 或 v16a u3299 权重 + v17 特征 adapter fine-tune
   - 略提 `ent_coef_emit` / 验证 multi-emit
5. **P1 架构**：planet spatial hist 4ch、symmetry aug、`flip_hard_mask=false`
6. **checkpoint 保留 `_eval_u*.pkl`**

---

## 10. 监控命令

```bash
# Run#2 日志（overnight 主 run）
ssh charlie@www.ultrapp.online \
  "tail -5 /home/charlie/project/OrbitWarRL/logs/v17_frog_hist50_20260612_000727/train.log"

# vs-v20 inline gate
ssh charlie@www.ultrapp.online \
  "grep eval_vs_v20 /home/charlie/project/OrbitWarRL/logs/v17_frog_hist50_20260612_000727/train.log | grep -v WARN | tail -10"

# peak ckpt offline eval + HTML
ssh charlie@www.ultrapp.online \
  "cat /home/charlie/project/OrbitWarRL/logs/v17_peak_eval.log | tail -30"

# 重启训练（自动 latest ckpt + 训至 06:00）
ssh charlie@www.ultrapp.online \
  "cd /home/charlie/project/OrbitWarRL && nohup bash scripts/v17_frog_hist50.sh > logs/v17_launch.log 2>&1 &"

# peak ckpt 10 局 eval + HTML
ssh charlie@www.ultrapp.online \
  "cd /home/charlie/project/OrbitWarRL && PYTHON=/home/charlie/anaconda3/bin/python bash scripts/v17_peak_ckpt_eval.sh"

# ckpt 列表
ssh charlie@www.ultrapp.online \
  "ls -lt /home/charlie/project/OrbitWarRL/ckpt_multi_action_v17_frog_hist50/*.pkl | head -5"
```

---

## 11. 训练动态（实测 vs 预期）

| 阶段 | updates | 预期 | **实测** |
|------|---------|------|----------|
| Adapt | 0–500 | 策略扰动 | ✅ loss 稳定 |
| Curriculum dense | 500–4400 | flip↑ | ✅ flip **50–59%**，spf ~40–54 |
| Sparse 长训 | 4400–25383 | vs-v20 WLD | ❌ flip **20–33%** 回落，**0 胜** |

**修正方向**：capture 不应完全 anneal 到 0 后再长 sparse；或 mid-ckpt 短训而非 end-ckpt。

---

## 12. 与 v15/v16a 的关系

- **v15 u9999**：v17 的 **唯一 resume 源**（避免 v16a capture 游牧偏置）
- **v16a**：验证 capture 方向正确（首胜 u3299/u3499），但不宜继续 extend
- **v17**：在 v15 策略基础上加 **时序 + 瞄准 + curriculum**，目标是 **稳定 vs-v20 胜局**

详见 [DAY16_PROGRESS.zh.md](./DAY16_PROGRESS.zh.md) 中 v15/v16a 完整实验记录。
