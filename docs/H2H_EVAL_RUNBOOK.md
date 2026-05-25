# H2H Eval & Replay Analyze Runbook

> 适用阶段：v9 ablation (v9a/b/c/d) 跑完后，以及任何需要「真实 kaggle env 500-turn 行为」数据的场合。
>
> **前提**：所有命令在 5090（`/workspace/OrbitWarRL`）执行。本地 CPU 也能跑（无 GPU 依赖），但速度慢约 2-3x。

---

## 0. 工具总览

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `export_submission.py` | 把 ckpt `.pkl` 打包成 kaggle submission `.py` | JAX |
| `h2h_local.py` | 单对单，N 局，逐局打印 | kaggle_environments |
| `h2h_gauntlet.py` | 一个 agent 打多个对手，汇总表 | kaggle_environments |
| `replay_analyze.py` | 跑 N 局并聚合 6 个行为 metric，first-80 / full-500 两个窗口 | kaggle_environments |
| `monitor_train.py` | 实时 tail 训练 log | stdlib only |

---

## 1. 第一步：从 ckpt 生成 submission 文件

H2H 脚本吃的是 kaggle submission `.py`，不是 ckpt `.pkl`。训练产出 ckpt 在 `ckpt_multi_action_v9X/`（5090 上），需要先 export。

```bash
# 在 5090 上执行
# --ckpt 填 4000 upds 最后一个；ckpt_every=100，最后一个是 ckpt_003999.pkl（或类似名称）

python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_v9b/ckpt_003999.pkl \
    --template submission_rl_v4.py \
    --out submission_rl_v9b_u3999.py

# v9a (control)
python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_v9a/ckpt_003999.pkl \
    --template submission_rl_v4.py \
    --out submission_rl_v9a_u3999.py

# 同理 v9c、v9d（等它们跑完）
python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_v9c/ckpt_003999.pkl \
    --template submission_rl_v4.py \
    --out submission_rl_v9c_u3999.py

python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_v9d/ckpt_003999.pkl \
    --template submission_rl_v4.py \
    --out submission_rl_v9d_u3999.py
```

export 成功会跑一次 parity check，应看到 `parity OK` 字样。如果 fail 说明 template 和 ckpt 架构不匹配。

---

## 2. H2H 单对单：`h2h_local.py`

适合快速验证一对新老模型的单个实验结果，每局打印结果，最后汇总。

**格式**
```
python -m orbit_wars_rl.scripts.h2h_local \
    --agent-a <submission_A.py> \
    --agent-b <submission_B.py> \
    --num-games <N> \
    --seeds <seed1> <seed2> ...
```

**注意**：这里 `--num-games N` 只打 N 局（A=player0, B=player1），**不自动翻色**。要翻色需要手动对调再跑一次，或直接用 `h2h_gauntlet.py`。

**示例：v9b vs v20，20 局，确定 seed**
```bash
python -m orbit_wars_rl.scripts.h2h_local \
    --agent-a submission_rl_v9b_u3999.py \
    --agent-b submission_v20_0513.py \
    --num-games 20 \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
```

**示例：翻色跑（B 做 player0）**
```bash
python -m orbit_wars_rl.scripts.h2h_local \
    --agent-a submission_v20_0513.py \
    --agent-b submission_rl_v9b_u3999.py \
    --num-games 20 \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
```

**输出格式**
```
  game  0  seed=  0  steps=274  a=1.0  b=-1.0  status=[...]  A   (8.3s)
  game  1  seed=  1  steps=500  a=-1.0 b=1.0   status=[...]  B   (14.1s)
...
agent A: submission_rl_v9b_u3999.py
agent B: submission_v20_0513.py
  A wins: 3/20  (0.15)
  B wins: 17/20 (0.85)
  draws : 0/20
  errors: 0/20
```

---

## 3. H2H Gauntlet（一对多）：`h2h_gauntlet.py`

**自动双色对称**（每个对手各打 2×N 局，A 色 N + B 色 N），最适合 v9 ablation 的最终评估。

**格式**
```
python -m orbit_wars_rl.scripts.h2h_gauntlet \
    --agent <candidate.py> \
    --opponents <opp1.py> <opp2.py> ... \
    --num-games <N 每色> \
    --seeds <seeds>
```

`--num-games N` = 每个对手每色 N 局，总局数 = N × 2 × 对手数。

**示例：v9b 打 3 个对手，每色 10 局（共 60 局），seeds 0-9**
```bash
python -m orbit_wars_rl.scripts.h2h_gauntlet \
    --agent submission_rl_v9b_u3999.py \
    --opponents \
        submission_rl_v9a_u3999.py \
        submission_v20_0513.py \
    --num-games 10 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    2>&1 | tee logs/h2h_v9b_gauntlet.log
```

**预期输出（参考 v4p2 gauntlet 的格式）**
```
agent: submission_rl_v9b_u3999.py
games per opponent: 2 * 10 (both colors)
seeds: [0, 1, ..., 9]

vs submission_rl_v9a_u3999.py   W=14/20  L=6  D=0  asA=7  asB=7  WR=0.70  avg_steps=NNN  (XXs)
vs submission_v20_0513.py        W= 2/20  L=18 D=0  asA=1  asB=1  WR=0.10  avg_steps=NNN  (XXs)

======================================================================
summary
overall: 16/40 = 0.400 WR across 2 opponents
```

**§13.7 决策阈值（每色 10 局 = 总 20 局）**

| 比较 | 结果含义 |
|---|---|
| v9b vs v9a > 12/20 | prod_share 有实质效果 |
| v9c vs v9b > 12/20 | fleet_log 有边际贡献 → 上 v9c |
| v9c vs v9b ≤ 10/20 | fleet_log 可省 → 上 v9d 或 v9b |
| 任一 vs v20 ≥ 1/20 | **DAY4 §6.3「决定性指标」**已触发（v9 系列历史最高） |

---

## 4. Replay Analyze（行为诊断）：`replay_analyze.py`

**与 H2H 不同**：replay_analyze 不只看胜负，还输出 6 个行为 metric，并且按 first-350-turn（我们的训练窗口）和 full-500-turn 两个窗口分别聚合，用来发现 train-eval gap。

**格式**
```
python -m orbit_wars_rl.scripts.replay_analyze \
    --agent-a <submission_A.py> \
    --agent-b <submission_B.py> \
    --num-games <N> \
    --seed-base <int> \
    --out <output.json>
```

seed 是 `seed_base + i`（i=0..N-1），不用手动列举。

**推荐跑的 4 组对比**（等 v9c/v9d 也出 submission 后）

```bash
# 1. v9b vs v9b（self-play baseline，看训练对局里真实行为）
python -m orbit_wars_rl.scripts.replay_analyze \
    --agent-a submission_rl_v9b_u3999.py \
    --agent-b submission_rl_v9b_u3999.py \
    --num-games 5 --seed-base 0 \
    --out logs/replay_analyze/v9b_vs_v9b.json

# 2. v9b vs v20（核心：看 v9b 面对强敌时的行为，对比 DAY4 §2.1 的 v7-vs-v20 崩盘数据）
python -m orbit_wars_rl.scripts.replay_analyze \
    --agent-a submission_rl_v9b_u3999.py \
    --agent-b submission_v20_0513.py \
    --num-games 5 --seed-base 0 \
    --out logs/replay_analyze/v9b_vs_v20.json

# 3. v9c vs v20（看 fleet_log 是否把 spf 推到更高）
python -m orbit_wars_rl.scripts.replay_analyze \
    --agent-a submission_rl_v9c_u3999.py \
    --agent-b submission_v20_0513.py \
    --num-games 5 --seed-base 0 \
    --out logs/replay_analyze/v9c_vs_v20.json

# 4. v9a vs v20（空对照，对比 §13.7 基线）
python -m orbit_wars_rl.scripts.replay_analyze \
    --agent-a submission_rl_v9a_u3999.py \
    --agent-b submission_v20_0513.py \
    --num-games 5 --seed-base 0 \
    --out logs/replay_analyze/v9a_vs_v20.json
```

**输出关键字段对比（v9b vs v20 实测 + Day5 FAST gate）**

| metric | v7 vs v20（DAY4） | v9b vs v20 first-80 | Day5 FAST gate |
|---|---|---|---|
| `mean_ships_per_fleet` | 1.98 | **4.76** | > 10 |
| `mean_garrison_my` | 9.3 | **31.85** | > 60 |
| `fleet_arrival_rate` (flip_proxy) | — | **3.88%** | > 6% |
| emit≥2 turn 占比 | — | **2.8%** | > 5% |
| outcome (5 局) | 0/5/0 | 0/5/0 | ≥ 1/5/0 (overnight) |

---

## 5. Monitor 快照（随时可跑）

v9c/v9d 在跑期间，用这个看 4 个 run 的实时状态：

```bash
# 单次快照（任何时候都能跑，无副作用）
python -m orbit_wars_rl.scripts.monitor_train --once logs/multi_action_v9*.log

# 后台持续监控，60s 刷新一次
nohup python -m orbit_wars_rl.scripts.monitor_train \
    --interval 60 \
    logs/multi_action_v9*.log \
    > logs/monitor_v9_full.log 2>&1 &
```

输出表格格式（2026-05-25 示例）：
```
name                      upd   steps    ev  clip     kl   spf   z0  garr   pS  ptS fLog  WRr   status
---------------------------------------------------------------------------------------------------------
multi_action_v9a         3999  131072K  0.97  0.09 +0.008 11.4 0.04  58.5 0.29 0.28 0.28    -     DONE
multi_action_v9b         3999  131072K  0.98  0.09 +0.006 21.2 0.01  49.9 0.41 0.39 0.36    -     DONE
multi_action_v9c         2620   85885K  0.97  0.08 +0.005 27.2 0.01  38.6 0.39 0.38 0.43    -     live
multi_action_v9d         2619   85852K  0.97  0.08 +0.004 20.9 0.01  56.9 0.41 0.39 0.37    -     live
```

**v9c/d 早期告警解读**：u0–u50 ev<0.5、u136–342 clip>0.35 在 v9b 上同样出现过；
u500 后 clip→0.08 即视为正常 shaping 冲击，无需 kill。

---

## 6. 完整 v9 eval 执行顺序（等 v9c/v9d 跑完后）

```
阶段 1：Export（5090）
  export v9a/b/c/d 各 last ckpt → submission_rl_v9X_u3999.py

阶段 2：H2H Gauntlet（5090 或本地，kaggle_env 即可，无 GPU 依赖）
  v9b vs [v9a, v20]          → logs/h2h_v9b_gauntlet.log
  v9c vs [v9a, v9b, v20]     → logs/h2h_v9c_gauntlet.log
  v9d vs [v9a, v9b, v20]     → logs/h2h_v9d_gauntlet.log

阶段 3：Replay Analyze（本地 CPU 可跑）
  v9b vs v9b, v9b vs v20     → logs/replay_analyze/v9b_*.json
  v9c vs v20, v9d vs v20     → logs/replay_analyze/v9c/d_*.json

阶段 4：决策（参照 §13.7 + 上面的阈值表）
  → 选出 winner → 继续训练至 10k updates，或进入 Day5 架构 sweep
```

---

## 7. 快速排错

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `parity FAILED` in export | template 版本与 ckpt 架构不匹配 | 确认用的 `submission_rl_v4.py` 是当前架构对应的 template |
| `file not found: submission_*.py` | ckpt 没 export，或路径错 | 先跑 export_submission |
| H2H 每局 steps=1 | env 初始化失败 | 确认 `kaggle_environments` 版本 OK，`parity_check` 先跑一下 |
| replay_analyze JSON 里 `mean_ships_per_fleet=0` | agent 全程不动（返回空 action） | submission 文件可能 export 失败，重新 export |
| gauntlet WR 异常低（< 0.5 vs v9a control） | 评估用了错的 ckpt（比如 upd 0） | 确认 ckpt path 末尾数字是最大的（3999） |
| monitor clip ALERT @u136–342 | shaping 梯度冲击（v9c/d 常见） | u500 后 clip<0.15 则正常；持续 >0.35 才调查 |

---

## 8. FAST Iteration Gate（Day5）

500-update 实验的 replay gate 与 PROMOTE/KILL 规则见 **[`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md)**。

**Metric 命名对齐**（replay JSON）：

| 文档用语 | JSON 字段 | 算法 |
|---|---|---|
| flip_proxy / fleet_flip_rate | `fleet_arrival_rate` | `fleets_arrived / fleets_launched` |
| emit≥2 占比 | `emit_count_distribution` | `1 - dist[0] - dist[1]` |

**当前 FAST 基线（v9b@3999 vs v20，first-80，5 局）**：

```
spf=4.76  garr=31.85  flip_proxy=3.88%  emit2=2.8%
```

frozen base 定稿后替换为 v9c@3999 同脚本 replay 数字（见 [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md) §4）。
