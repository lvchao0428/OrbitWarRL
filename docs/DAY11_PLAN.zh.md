# DAY11 计划 — f40 主线：BC 过关 + Buffer PPO 长训

> **2026-06-03 计划**
> 接续 [`DAY10_PROGRESS.zh.md`](DAY10_PROGRESS.zh.md)。
> **f38 全系已关闭**；Day11 全部算力投入 **f40 Expert-Seeded Gated League**。

---

## 0. TL;DR

| 维度 | 计划 |
|---|---|
| **Day10 结论** | f33–f38s1b 全线 WLD 0/5；f38s1b 未优于 s1；f40 基建 smoke 通过 |
| **Day11 主线** | 扩 BC 数据 → BC replay gate → f40 buffer PPO 500–1000 upd |
| **老线 f38** | **不再投算力**；仅保留 s1@199 ckpt 作可选 anchor |
| **提交基线** | 仍为 **f29 @599** |
| **唯一 promote 口径** | **replay vs v20**（first-80，5 seeds；关键 ckpt 20 局复测） |
| **算力分配** | ~20% BC 采集/训练；~80% f40 PPO 长训 + replay gate |

---

## 1. Day10 遗留状态（起点）

| 项 | 状态 | 路径 |
|----|------|------|
| f38s1b | ✅ 训满 500 upd，replay 完成，**归档** | `ckpt_multi_action_v11_f38s1b/` |
| BC 数据 | 🟡 仅 20 局 | `data/bc_f40_v20_self.npz` |
| BC seed | 🟡 未过 gate | `ckpt_bc_f40/ckpt_final.pkl` |
| mixed buffer | ✅ 就绪 | `data/f40_mixed_states.npz` |
| PPO smoke | ✅ 5 upd 验证链路 | `ckpt_multi_action_v11_f40_buffer_smoke/` |
| gated pool 自动化 | ❌ 未实现 replay 驱动入池 | 见 Day11 P2 |

---

## 2. Gate 协议（Day11 统一使用）

### 2.1 三层指标

| 层级 | 用途 | 指标 | 阈值 |
|------|------|------|------|
| **L0 健康** | 训练中 early kill | clip_frac, KL, explained_variance | clip>0.35 或 ev<0.30 → kill |
| **L1 alive** | 是否继续长训 | captures, flip, e2+, z0 | 四项中 **≥3 项** 过线 |
| **L2 promote** | 提交/入池 | WLD 或 行为簇 | WLD≥1/5 **或** captures>50 + flip>6% + e2+>10% |

**L1 alive 具体阈值（first-80）**：
- captures > 20
- flip > 3%
- e2+ > 5%
- z0 > 3%

**BC 专用 gate（进 PPO 前）**：
- captures > 40
- e2+ > 10%
- z0 > 5%

### 2.2 明确不信的指标

- 训练 log 的 spf / garr / e2 / WR vs random
- @499 以后 ckpt 的「规模指标」without replay 验证

### 2.3 关键观测窗口

- **@149–249**：历史最佳信号区（f37/f38 s1 @199）
- **@499**：collapse 探测器
- replay 每 **100 upd** 一次（@99/@199/@299/…）

---

## 3. Day11 任务清单

### P0 — 上午：BC 数据与 seed（阻塞 PPO 长训）

| # | 任务 | 完成标准 |
|---|------|----------|
| 1 | 采集 **200 局** v20 BC 数据 | `bc_f40_v20_self.npz` ≥ 40k samples |
| 2 | 重训 BC seed（无 hard mask 训练） | val acc_emit > 0.70 |
| 3 | BC replay gate | captures>40, e2+>10%, z0>5% **至少 2/3** |
| 4 | 不过 gate → 查 action_inverse / emit 分布 | 记录到 DAY11 进展 |

### P1 — 下午：f40 Buffer PPO 长训

| # | 任务 | 完成标准 |
|---|------|----------|
| 5 | 从 BC ckpt resume，跑 **500 upd**（可先 500，过关再加 500） | `logs/v11_f40_buffer.log` |
| 6 | @199 replay gate | alive ≥3/4 或 e2+>10% |
| 7 | @499 replay | 若 e2+<1% 且 captures<15 → **kill 并回退 @199** |

### P2 — 有余力：gated pool 自动化（可延到 Day12）

| # | 任务 |
|---|------|
| 8 | replay 通过后手动 `pool_seed_paths` 加入 learner ckpt |
| 9 | 实现 @100 自动 replay + 仅 alive ckpt 入池（代码） |
| 10 | buffer step 20–120 分桶 filter |

---

## 4. 执行命令（远程 5090）

> **一键全流程**：见 [`scripts/run_day11_f40_pipeline.sh`](../scripts/run_day11_f40_pipeline.sh)

```bash
# 远程后台跑 Day11 全套（BC 200局 + BC训练 + PPO 500upd + replay）
cd /home/charlie/project/OrbitWarRL
nohup bash scripts/run_day11_f40_pipeline.sh \
  > logs/day11_f40_pipeline.nohup 2>&1 &

# 看总进度
tail -f logs/day11_f40_pipeline.log

# 快速 smoke（~30min 验链路）
DAY11_SMOKE=1 bash scripts/run_day11_f40_pipeline.sh
```

> 默认：`cd /home/charlie/project/OrbitWarRL`，`PYTHON=/home/charlie/anaconda3/bin/python`

### 4.1 同步代码（本地改完后）

```bash
rsync -avz orbit_wars_rl/bc/ orbit_wars_rl/ppo/runner.py \
  orbit_wars_rl/configs/multi_action_v11_f40_buffer.yaml \
  scripts/collect_f40_expert_data.sh scripts/run_f40_bc.sh \
  scripts/run_v11_f40_buffer.sh scripts/run_f40_eval.sh \
  charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/
```

### 4.2 P0：采集 BC 200 局

```bash
# ~45–90 分钟
BC_GAMES=200 STATE_GAMES=0 SEED=4200 \
  bash scripts/collect_f40_expert_data.sh
```

仅补 BC、跳过 state buffer（已有 mixed buffer）：

```bash
/home/charlie/anaconda3/bin/python -m orbit_wars_rl.bc.collect_data \
  --num-games 200 \
  --agent submission_v20_0513.py \
  --opponent submission_v20_0513.py \
  --seed 4200 \
  --out data/bc_f40_v20_self.npz
```

### 4.3 P0：训练 BC seed

```bash
EPOCHS=20 BATCH_SIZE=256 EMIT_POS_WEIGHT=4.0 \
  DATA=data/bc_f40_v20_self.npz \
  OUT=ckpt_bc_f40/ckpt_final.pkl \
  bash scripts/run_f40_bc.sh
```

`run_f40_bc.sh` 末尾会自动跑 replay；BC 导出时 **关闭 hard mask**（脚本内已配置 `f40_bc` tag）。

### 4.4 P0：查看 BC 结果

```bash
# 训练 loss / acc
tail -30 logs/f40_bc.log

# replay 一行 gate
cat logs/replay_analyze/v11_f40_bc_seed_vs_v20.summary.txt

# 详细 first-80 + full-game
less logs/replay_analyze/v11_f40_bc_seed_vs_v20.summary.txt.full
```

**BC 关键指标**：

| 指标 | 目标 | 含义 |
|------|------|------|
| e2+ | > 10% | 多路 emit |
| z0 | > 5% | 会停手 |
| captures | > 40 | 会翻转 |
| flip | > 3% | 舰队有效到达 |
| one_ship_rate | < 50% | 非 trickle |
| bin7 | 不要 > 85% | 非全量 bin7 spam |

### 4.5 P1：启动 f40 Buffer PPO（BC 过 gate 后）

```bash
# 后台长训，默认 1000 upd
bash scripts/run_v11_f40_buffer.sh

# 或先试 500 upd
NUM_UPDATES=500 bash scripts/run_v11_f40_buffer.sh
```

环境变量（脚本内已设，可 override）：

```bash
ORBITWARS_SHAPING_CAPTURE=0.02 \
ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.005 \
bash scripts/run_v11_f40_buffer.sh
```

### 4.6 P1：监控训练

```bash
# 实时 log
tail -f logs/v11_f40_buffer.log

# 健康检查（clip/ev）
bash scripts/check_training_health.sh logs/v11_f40_buffer.log

# 末行行为 proxy（仅参考，不作 promote）
grep '^upd' logs/v11_f40_buffer.log | tail -5
```

**训练 log 关键字段（健康 vs 参考）**：

| 字段 | 健康范围 | 仅参考（不信 promote） |
|------|----------|------------------------|
| clip | < 0.20 | — |
| kl | \|kl\| < 0.05 | — |
| ev | > 0.30 | — |
| emits | — | 2.0–3.0 较好 |
| e2 | — | > 0.30 较好 |
| spf | — | 10–25（replay 才准） |
| z0 | — | > 0.05 较好 |
| opp | — | buf/strn/frzn 比例 |

### 4.7 P1：Replay gate（每 100 upd）

```bash
# 单 ckpt
PYTHON=/home/charlie/anaconda3/bin/python \
  bash scripts/quick_replay.sh \
  ckpt_multi_action_v11_f40_buffer/ckpt_000199.pkl \
  v11_f40_buffer_u199

# 批量 eval
bash scripts/run_f40_eval.sh
```

拉结果到本地：

```bash
rsync -avz charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/logs/replay_analyze/v11_f40_* \
  logs/replay_analyze/
```

### 4.8 P1：查看 replay 结果

```bash
# 一行 summary（gate 决策用）
head -20 logs/replay_analyze/v11_f40_buffer_u199_vs_v20.summary.txt

# JSON 精确数值
python3 - <<'PY'
import json
p = "logs/replay_analyze/v11_f40_buffer_u199_vs_v20.json"
d = json.load(open(p))
fb = d["aggregate_by_window"]["first_80turns"]["player_0"]
print("WLD", fb["outcome"])
print("captures", fb.get("captures"))
print("flip%", fb.get("fleet_arrival_rate",0)*100)
print("z0%", fb.get("zero_emit_rate",0)*100)
ed = fb.get("emit_count_distribution") or []
e2 = sum(ed[i] for i in range(2, len(ed))) * 100
print("e2+%", e2)
PY
```

**Replay 关键指标（first-80，vs v20）**：

| 指标 | alive | promote | kill 信号 |
|------|-------|---------|-----------|
| WLD | — | ≥ 1/5/0 | 0/5/0 可接受若 L1 过 |
| captures | > 20 | > 50 | < 15 连续两 ckpt |
| flip | > 3% | > 6% | < 2% |
| e2+ | > 5% | > 10% | < 1% 连续两 ckpt |
| z0 | > 3% | > 8% | < 1% |
| spf | > 10（Day5） | > 13（f29 级） | min_game_spf < 5 |
| one_ship_rate | < 50% | < 35% | > 40% |
| bin0 | < 15% | < 10% | — |
| bin7 | < 80% | < 50% | > 85% |

### 4.9 Kill / 回退决策

```bash
# 若 @499 replay fail，从 @199 resume（示例）
NUM_UPDATES=300 \
RESUME=ckpt_multi_action_v11_f40_buffer/ckpt_000199.pkl \
bash scripts/run_v11_f40_buffer.sh
```

---

## 5. 配置速查（f40）

| 项 | 值 | 文件 |
|----|-----|------|
| config | `orbit_wars_rl/configs/multi_action_v11_f40_buffer.yaml` | |
| resume | `ckpt_bc_f40/ckpt_final.pkl` | |
| buffer | `data/f40_mixed_states.npz` | reset_ratio=0.80 |
| strong anchor | f29 @599 | strong_ratio=0.20 |
| pool seed | f29 @599 | snapshot_current=false |
| shaping | CAPTURE=0.02, PROD_SHARE_DELTA=0.005 | env var |
| ckpt_every | 100 | @99/@199/… eval |

**可选 anchor 扩展**（BC 仍弱时）：

```yaml
# multi_action_v11_f40_buffer.yaml
pool_seed_paths:
  - ./ckpt_multi_action_v11_f29/ckpt_000599.pkl
  - ./ckpt_multi_action_v11_f38/ckpt_000199.pkl   # f38 s1 @199
```

---

## 6. Day11 成功标准

| 级别 | 标准 |
|------|------|
| **最低** | BC 200 局 + replay；PPO 跑到 @199 有 replay 记录 |
| **合格** | @199 alive ≥3/4（captures/flip/e2+/z0） |
| **优秀** | @199 e2+>10% 且 captures>30；@499 未 collapse（e2+>5%） |
| **突破** | 任一 ckpt WLD≥1/5 或 captures>50+flip>6%+e2+>10% |

---

## 7. 时间线建议

| 时段 | 任务 |
|------|------|
| 上午 | BC 200 局采集 + 训练 + replay gate |
| 下午 | 启动 f40 PPO 500 upd |
| 晚间 | @199 replay；决策是否续到 @499 或 kill |
| 次日 | @499 replay；20 局复测若 @199 接近 promote |

---

## 8. 与 f38 老线的关系

| 项 | Day11 动作 |
|----|------------|
| f38s1b | **不再运行**；结果已记入 DAY10 §14 |
| f38 s1 @199 | 可选加入 f40 `pool_seed_paths` |
| f38 Stage 2/3 | **取消** |
| f38s1b @499 | 不作为 resume 点（e2+ 高但 flip/captures fail） |
