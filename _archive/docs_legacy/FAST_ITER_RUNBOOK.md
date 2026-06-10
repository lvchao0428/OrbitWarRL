# FAST Iteration Runbook — 500 Update Gate

> Day5 快速迭代操作手册。战术背景见 [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md)；现状与基线数字见 [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md)。
>
> **原则**：2–4 小时筛 1 个 hypothesis → replay×3 vs v20 → PROMOTE / KILL / RETRY。
> 4000-update overnight **只给 PROMOTE 的 config**。

---

## 0. 两条 Lane

| Lane | updates | 时长 (5090) | 用途 |
|---|---|---|---|
| **FAST** | 500 | ~2.5–3h train + ~15min replay | 筛 hypothesis |
| **OVERNIGHT** | 4000 | ~10–12h | PROMOTE winner → gauntlet |

---

## 1. FAST 配置模板

从 frozen base yaml（预期 `multi_action_v9c.yaml`）复制，**只改一项**：

```yaml
train:
  num_updates: 500          # FAST
  ckpt_every: 100
  eval_every: 50
  ckpt_dir: ./ckpt_fast_A1  # 独立目录
  log_every: 1
  # episode_steps: 350      # 不变
  # rollout_length: 256     # 不变

# selfplay / ppo / shaping env vars: 同 frozen base
```

**Shaping env vars（v9c base，launch 前 export）：**

```bash
export ORBITWARS_SHAPING_PROD_SHARE=0.01
export ORBITWARS_SHAPING_PLANET_SHARE=0.005
export ORBITWARS_SHAPING_FLEET_LOG=0.002
```

**Resume 规则：**

| Delta 类型 | 起始权重 |
|---|---|
| C1 curriculum / B1–B3 reward | resume frozen base `@3999` |
| A1' 加 2–3 维 planet feat | resume frozen base `@3999` |
| A2 改坐标系 / A4 改 backbone | **from scratch**，同 seed |

---

## 2. 启动 FAST Run

```bash
cd /workspace/OrbitWarRL  # 5090

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # 单 job FAST

# 示例：FAST-C1（实现后替换 config 路径）
nohup python -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_fast_c1.yaml \
  --log-dir logs/multi_action_fast_c1 \
  > logs/multi_action_fast_c1.log 2>&1 &

# T+30s：确认 banner
sleep 30
grep -E '\[reward\]|episode_steps' logs/multi_action_fast_c1.log | head -5
```

**并行 2 路 FAST（5090）：**

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
# 分别 nohup 两个 config；或用 N_CONCURRENT=2 的 launcher（待写 scripts/run_fast.sh）
```

**Monitor：**

```bash
python -m orbit_wars_rl.scripts.monitor_train --once logs/multi_action_fast_*.log
python -m orbit_wars_rl.scripts.monitor_train --interval 60 logs/multi_action_fast_*.log
```

---

## 3. 训练内 Gate（不等 replay）

| 检查点 | 通过 | KILL |
|---|---|---|
| T+30min ~u50 | ev > 0.4, clip < 0.30 | ev < 0.2 @u100 |
| T+1.5h ~u200 | ev > 0.5, clip < 0.35 | ev < 0.30 持续 @u200 |
| T+2.5h ~u500 | ev > 0.70, clip < 0.20 | clip > 0.40 持续 50 upd |

训练 KILL → `pkill -f multi_action_fast_XX`，**不 export**。

---

## 4. Export + Replay Gate（FAST 主决策）

### 4.1 Export @499

```bash
python -m orbit_wars_rl.scripts.export_submission \
  --ckpt ckpt_fast_A1/ckpt_000499.pkl \
  --template submission_rl_v4.py \
  --out submission_rl_fast_a1_u499.py
```

A1' 改 obs 维度的 run：export 前跑 parity：

```bash
python -m orbit_wars_rl.scripts.parity_check --num-states 128
```

### 4.2 Replay ×3 vs v20（first-80 窗口）

```bash
python -m orbit_wars_rl.scripts.replay_analyze \
  --agent-a submission_rl_fast_a1_u499.py \
  --agent-b submission_v20_0513.py \
  --num-games 3 --seed-base 0 \
  --out logs/replay_analyze/fast_a1_vs_v20.json
```

从 JSON `aggregate_by_window.first_80turns.player_0` 读取：

| metric | JSON 字段 | 算法 |
|---|---|---|
| spf | `mean_ships_per_fleet` | 直接读 |
| garr | `mean_garrison_my` | 直接读 |
| flip_proxy | `fleet_arrival_rate` | `fleets_arrived / fleets_launched` |
| emit≥2 | `emit_count_distribution` | `1 - dist[0] - dist[1]` |

### 4.3 基线（Phase 0 完成前 = v9b@3999）

```
first-80 vs v20 (5局均值，player_0):
  spf=4.76   garr=31.85   flip_proxy=3.88%   emit2=2.8%
```

Phase 0 完成后替换为 **frozen base@499 或 @3999** 的同脚本 replay 数字。

---

## 5. PROMOTE / KILL / RETRY

### PROMOTE → Overnight（需全部满足）

1. 训练：ev@500 **> 0.70**，clip@500 **< 0.20**
2. Replay first-80 vs v20：**4 项中至少 2 项** 过线：

| metric | gate | v9b 基线 |
|---|---|---|
| spf | **> 10** | 4.76 |
| garr | **> 60** | 31.85 |
| flip_proxy | **> 6%** | 3.88% |
| emit≥2 | **> 5%** | 2.8% |

3. **不能 4/4 全劣于基线**（同 seed 0–2）

### KILL（replay 阶段）

- 4/4 metric **均差于** v9b 基线（方向错）
- 或训练 KILL 条件已触发

### RETRY（每 hypothesis 最多 1 次）

- 1–2 项 metric 方向对但未过线 → 系数 ±50% 或 `lr_peak×0.5`
- 记录到 [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md) §6 台账

---

## 6. Overnight Lane（仅 PROMOTE）

```bash
# 从 FAST winner config，num_updates: 4000，resume FAST@499 或 frozen base@3999
nohup python -m orbit_wars_rl.scripts.train \
  --config orbit_wars_rl/configs/multi_action_overnight_a1.yaml \
  --log-dir logs/multi_action_overnight_a1 \
  > logs/multi_action_overnight_a1.log 2>&1 &
```

**完成后：**

```bash
# export @3999
python -m orbit_wars_rl.scripts.export_submission \
  --ckpt ckpt_overnight_a1/ckpt_003999.pkl \
  --template submission_rl_v4.py \
  --out submission_rl_overnight_a1_u3999.py

# gauntlet 双色
python -m orbit_wars_rl.scripts.h2h_gauntlet \
  --agent submission_rl_overnight_a1_u3999.py \
  --opponents submission_rl_v9c_u3999.py submission_v20_0513.py \
  --num-games 20 \
  > logs/h2h_overnight_a1_gauntlet.log 2>&1

# replay ×5 vs v20
python -m orbit_wars_rl.scripts.replay_analyze \
  --agent-a submission_rl_overnight_a1_u3999.py \
  --agent-b submission_v20_0513.py \
  --num-games 5 --seed-base 0 \
  --out logs/replay_analyze/overnight_a1_vs_v20.json
```

**Overnight 成功标准：**

| 级别 | 条件 |
|---|---|
| Sign of life | vs v20 gauntlet **≥ 1/40**（20×2色） |
| Meaningful | replay spf **> 12**, garr **> 80** |
| Production | vs v20 **≥ 5/40** 或 vs frozen base **≥ 18/40** → 上 10k |

---

## 7. 推荐 FAST 顺序

| 优先级 | ID | Delta | 并行 |
|---|---|---|---|
| 1 | **C1** | frozen pool + v20 替身 ~25% | 可与 A1' 并行 |
| 1 | **A1'** | threat_ratio + eta（planet feat +3） | 可与 C1 并行 |
| 2 | **B1** | multi_emit bonus | A1'/C1 均 KILL 时 |
| 3 | **B3** | fleet_log gated（garr>θ） | z0 训练仍≈0 时 |
| 4 | **A2** | relative coords | A1' PROMOTE 但 asymmetry 仍存 |

**不要**在同一 run 混 A + B + C。

---

## 8. 典型一天排期

```
上午  启动 FAST-C1 + FAST-A1'（N_CONCURRENT=2）
下午  export@499 → replay×3 → PROMOTE/KILL
      若 1 PROMOTE → 启动 overnight；另开 1 条 FAST-B1
晚上  monitor overnight
次日  gauntlet + replay×5 → 更新 DAY5_PROGRESS §6
```

**吞吐目标**：3–4 hypothesis / 天（FAST），1 overnight / 天。

---

## 9. 速查卡片

```
FAST:     500 upd · 1 delta · export@499 · replay×3 vs v20 (seed 0-2)
PROMOTE:  ev@500>0.7 AND clip<0.2 AND (2/4): spf>10 garr>60 flip>6% emit2>5%
KILL:     ev@200<0.3 OR clip@200>0.4 sustained OR 4/4 metrics worse than baseline
RETRY:    once per hypothesis, ±50% coef or lr×0.5
BASELINE: v9b replay first-80 (until Phase 0 done) → then frozen base
```
