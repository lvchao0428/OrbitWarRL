# DAY16 — v15：Frog Parade 风格 Multi-Match + Pure Terminal Reward

> **策略转型**：放弃 v14 系列的密集 shaping + 短 probe 搜参路线，改为 Frog Parade (Lux AI S3 #2, Orbit #1)
> 风格：BO3 多 match 结构 + 纯终端 ±1 reward + anti-hoard 单项防坍塌 + 10000u 单次长训。

---

## 核心变更总结

### 为什么切换策略

| 问题 | v14 系列表现 | v15 解决思路 |
|------|-------------|-------------|
| 早期囤兵坍塌 | v14d 全部坍塌，v14e anti-hoard 有效但 Gate A 过严 | Multi-match 提供 2-3x 更密终端信号 + anti-hoard |
| Reward shaping 复杂 | 15+ 个 shaping 项，难以调参 | 仅保留 anti-hoard，其余全部归零 |
| 搜参效率低 | 二分搜 24 trials 全失败 | 不搜参，固定 v13c 验证过的超参 + Frog Parade 洞见 |
| 训练信号稀疏 | 每 500 步一个 ±1 终端奖励 | 每 match (~500步) 一个 ±1，每 series 2-3 个 match |

### Frog Parade 关键洞见 (来自 `lux-top2-player.txt`)

- **Sparse reward wins**: 简化 reward（match win/loss ±1）优于复杂 shaping
- **Long training**: 300M game steps (~8天)，agent 需要足够时间发现策略
- **Low entropy**: 1e-4 级别 entropy，加速策略收敛
- **High gamma**: 0.9999，长期规划
- **Champion tracking**: 当前 vs 历史最佳，周期性评估（未来可加）

---

## 架构变更

### 1. 环境：Multi-Match BO3 结构

```
Series (同一张地图)
├── Match 1 (500 steps) → done=True, reward=±1 → GAE episode 1
├── Match 2 (500 steps) → done=True, reward=±1 → GAE episode 2
└── Match 3 (if needed) → done=True, reward=±1 → GAE episode 3
                           ↓
                      series_done → init.reset() 新地图
```

- **RL episode = 一个 match**（不是一个 series），每个 match 结束触发 GAE 截断
- **同地图 rematch**：match 结束时保留行星位置/生产力/轨道参数，仅重置驻军/舰队/所有权
- **`wins_needed=2` (BO3)**：先赢 2 场的赢得 series；平局双方各得 1 分防止无限 series
- **向后兼容**：`wins_needed=1`（默认）= 旧版单 match 行为

### 2. 状态扩展 (`state.py`)

新增 4 个字段到 `EnvState`:
- `match_score: [NUM_PLAYERS] int32` — 当前 series 各玩家赢的 match 数
- `match_idx: int32` — 当前 series 中的 match 序号 (0-based)
- `init_planet_ships: [MAX_PLANETS] int32` — series 开始时行星初始驻军（match reset 用）
- `init_planet_owner: [MAX_PLANETS] int8` — series 开始时行星初始所有权

### 3. Match Reset (`init.py`)

新增 `reset_match_same_map(state, winner)`:
- **保留**: 行星位置、生产力、轨道半径/相位、角速度、home_planet_idx
- **重置**: 驻军→初始值、所有权→初始值、舰队全清、step=0、done=False
- **更新**: match_score += winner, match_idx += 1

### 4. 步进逻辑 (`env.py`)

`step_and_autoreset()` 三分支逻辑:
- `match_done & series_done` → `init.reset()` 全新地图
- `match_done & ~series_done` → `reset_match_same_map()` 同地图 rematch
- `~match_done` → 正常步进

### 5. 特征编码 (`encode.py`)

`GLOBAL_FEAT_DIM`: 24 → **27** (+3 多 match 上下文)

| 维度 | 特征 | 范围 |
|------|------|------|
| 24 | `match_score_me / wins_needed` | [0, 1] |
| 25 | `match_score_opp / wins_needed` | [0, 1] |
| 26 | `match_idx / max_matches` | [0, 1] |

让 agent 感知 series 状态（例如 "我 0-1 落后，这局必须赢"）。

### 6. 提交模板 (`submission_rl_v15.py`)

基于 v13c 模板扩展，`GLOBAL_FEAT_DIM = 27`。Kaggle 推理时多 match 维度恒为 0（单 episode）。

---

## 文件清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `orbit_wars_rl/env/state.py` | 修改 | +4 fields: match_score, match_idx, init_planet_ships, init_planet_owner |
| `orbit_wars_rl/env/init.py` | 修改 | reset() 存初始状态; +reset_match_same_map() |
| `orbit_wars_rl/env/env.py` | 修改 | +wins_needed 参数; step_and_autoreset() 三分支 |
| `orbit_wars_rl/features/encode.py` | 修改 | GLOBAL_FEAT_DIM 24→27; +3 多match特征 |
| `orbit_wars_rl/ppo/rollout_symmetric.py` | 修改 | 传递 wins_needed 到 encode() |
| `orbit_wars_rl/ppo/runner.py` | 修改 | TrainConfig +wins_needed; 传递到 env/encode |
| `orbit_wars_rl/configs/multi_action_v15_frog.yaml` | 新增 | BO3, 10000u, ent_emit=0.005, gamma=0.9999 |
| `scripts/v15_frog_train.sh` | 新增 | 启动脚本，所有 shaping=0 except anti-hoard=0.03 |
| `submission_rl_v15.py` | 新增 | Kaggle 提交模板 (GLOBAL_FEAT_DIM=27) |

---

## 训练配置 vs v13c 对比

| 参数 | v13c | v15 | 差异原因 |
|------|------|-----|----------|
| `wins_needed` | 1 (单match) | **2 (BO3)** | 2-3x 更密终端信号 |
| `ent_coef_emit` | 0.01 | **0.005** | 更接近 Frog Parade 低 entropy |
| `min_pct_bin` | 2 | **5** | 禁止微型舰队 |
| `ANTI_HOARD` | 0 | **0.03** | 结构性防坍塌 |
| `SHAPING_SCALE` | 0 | 0 | 均为纯终端 |
| gamma | 0.9999 | 0.9999 | 相同 |
| num_updates | 10000 | 10000 | 相同 |
| d_model/n_layers | 256/4 | 256/4 | 相同 |

---

## 执行进展

| 阶段 | 状态 | 备注 |
|------|------|------|
| v14f A/B 二分 | **已停止** | 为 v15 让出 GPU |
| **v15 Frog BO3 长训** | **运行中 (~u5780/10000, 58%)** | `logs/v15_frog_20260611_165324/train.log`, 5090 GPU, ~2820 sps |

### 训练里程碑指标

| upd | z0 | garr | emits | spf | tR | ev | pS | WRr | sps |
|-----|-----|------|-------|-----|-----|-----|-----|-----|-----|
| 100 | 0.61 | 65.8 | 0.82 | 31.5 | +1.00 | 0.96 | 0.29 | — | 1568 |
| 500 | 0.72 | 201.1 | 0.65 | 35.8 | +0.20 | 0.96 | 0.31 | — | 2453 |
| 1000 | 0.75 | 204.8 | 0.57 | 32.6 | -0.20 | 0.94 | 0.29 | — | 2640 |
| 1799 | **0.85** | **264** | **0.27** | 46.0 | -0.50 | 0.95 | 0.31 | 0.50 | 2746 |
| 2100 | 0.64 | 148.9 | **1.54** | 22.0 | -0.33 | 0.98 | 0.38 | — | 2743 |
| 3000 | 0.70 | 177.6 | 0.99 | 24.1 | -0.14 | 0.96 | 0.32 | — | 2776 |
| 4000 | 0.73 | 167.5 | 0.70 | 36.6 | +0.67 | 0.96 | 0.31 | — | 2796 |
| 5000 | 0.76 | 210.5 | 0.58 | 43.2 | +0.20 | 0.92 | 0.36 | — | 2808 |
| 5500 | 0.72 | 97.1 | 0.60 | 42.9 | +0.50 | 0.96 | 0.31 | — | 2813 |
| 5599 | 0.73 | 149.2 | 0.56 | 36.1 | -0.60 | 0.95 | 0.33 | **0.84** | 2818 |
| ~5780 | 0.65-0.72 | 127-171 | 0.50-0.69 | 47-82 | ±0.50 | 0.95 | 0.31-0.35 | — | 2819 |

**观察**:
- **u0-500**: 快速 warmup + 首次囤兵期 (z0→0.72, garr→200)
- **u1700-1800**: 局部囤兵回潮 (z0=0.85, emits=0.25) — **u1799 replay 快照即落在此区间**
- **u1900-2100**: 快速恢复 (emits 1.15→1.54, z0 0.64)
- **u3000-5500**: 中期稳定，WRr 爬升至 0.84 (u5599)；vs v20 仍 0 胜
- **u5500+**: garr 回落 (97-171)，entropy 继续下降 (ent_emit≈0.42-0.50)，策略在收敛

与 v13c 对比：v13c 在 u200-300 经历更严重的囤兵 (z0=0.91, garr=356)，u400 后恢复，最终 WRr=0.91。
v15 囤兵峰值更轻 (z0=0.85 vs 0.91)，且 u5599 时 WRr=0.84 已接近 v13c 最终水平。

### HTML Replay

| 版本 | 路径 | 结果 | 步数 | 备注 |
|------|------|------|------|------|
| u1799 | `logs/replay_html/v15_frog_u1799_vs_v20_seed42/replay.html` | v20 胜 | 121 | 囤兵高峰快照 |
| **u5599** | `logs/replay_html/v15_frog_u5599_vs_v20_seed42/replay.html` | v20 胜 | 142 | 最新 ckpt，存活略长 |

**u1799 replay 行为分析** (seed=42, 符合当时训练指标 emits=0.27, z0=0.85):
- Step 24/45: 发兵频次极低，v20 已大量扩张，v15 仅零星出舰队
- Step 117: v20 全图控制，v15 最后一支大舰队 (57 ships) 飞向边缘无效方向
- **结论**: 属于 u1700-1800 局部囤兵回潮的正常表现，非结构性 bug

**u5599 replay** (seed=42, emits=0.56, z0=0.73, WRr=0.84):
- 仍负于 v20，但步数 121→142，略有改善
- export smoke test 已能正常发兵 (1 launch, 42 ships)
- vs v20 战斗力仍不足，需继续训练至 u8000+ 再评估

---

## 监控命令

```bash
# 查看最新训练日志 (实时)
ssh charlie@www.ultrapp.online "tail -10 /home/charlie/project/OrbitWarRL/logs/v15_frog_20260611_165324/train.log"

# 查看关键里程碑
ssh charlie@www.ultrapp.online "grep -E 'upd  (500|1000|1500|2000|2500|3000|4000|5000|7000|10000) ' /home/charlie/project/OrbitWarRL/logs/v15_frog_20260611_165324/train.log"

# 查看进程状态
ssh charlie@www.ultrapp.online "ps aux | grep v15 | grep -v grep"

# 查看 checkpoint 列表
ssh charlie@www.ultrapp.online "ls -lt /home/charlie/project/OrbitWarRL/ckpt_multi_action_v15_frog/*.pkl | head -5"

# 导出指定/最新 ckpt 并生成 HTML replay (seed=42)
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/OrbitWarRL && \
  CKPT=ckpt_multi_action_v15_frog/ckpt_005599.pkl && \
  OUT=logs/replay_html/v15_frog_u5599_vs_v20_seed42 && \
  /home/charlie/anaconda3/bin/python -m orbit_wars_rl.scripts.export_submission \
    --ckpt "$CKPT" --template submission_rl_v15.py \
    --out submission_rl_v15_latest.py \
    --allow-hold 1 --force-emit-worth-it 0 --min-pct-bin 5 \
    --emit-hard-stop 1 --flip-hard-mask 1 --emit-hard-stop-min-step 0 && \
  /home/charlie/anaconda3/bin/python -m orbit_wars_rl.scripts.replay_html \
    --agent-a submission_rl_v15_latest.py \
    --agent-b submission_v20_0513.py \
    --seed 42 --out-dir "$OUT"'

# 拉取 replay 到本地并打开
scp charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/logs/replay_html/v15_frog_u5599_vs_v20_seed42/replay.html \
  logs/replay_html/v15_frog_u5599_vs_v20_seed42/replay.html
open logs/replay_html/v15_frog_u5599_vs_v20_seed42/replay.html
```

---

## 预期训练动态

| 阶段 | 更新范围 | 预期行为 | 关注指标 |
|------|---------|----------|---------|
| Warmup | 0-50 | 随机策略，学习基本操作 | loss 下降 |
| 早期 | 50-300 | 可能出现囤兵倾向 (z0 上升) | z0 < 0.85 |
| **中期** | **300-2000** | **agent 发现进攻/占领策略; emits 增加, garr 稳定** | **emits > 0.5** |
| 后期 | 2000-10000 | 策略精细化; WRr 应向 0.8-0.9 爬升 | vs v20 胜率 |

## 风险缓解

- **如果 z0 > 0.90 by u50**: anti-hoard 不够强，需要提升到 0.05
- **如果 u5000 后 vs v20 仍然 0 胜**: 当前 u5599 仍 0 胜 (142 steps)；WRr=0.84 说明自博弈已学会，vs v20 需 u8000+ 再判。若仍无胜，考虑轻量 capture shaping
- **如果训练发散 (loss 暴涨)**: 降低 lr_peak 或提高 target_kl

---

## 与 v14 系列的关系

- v14f 搜参已停止（让出 GPU），后续可在 v15 结果基础上决定是否恢复
- v15 是**独立实验**，与 v14 平行路线
- 如果 v15 在 u8000 时 vs v20 胜率 > 0，证明 multi-match + pure terminal 路线可行
- v13c 仍是 baseline 参考 (最终 WRr=0.91 vs random)
