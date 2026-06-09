# DAY13 — Frog Parade 参考架构重构: 执行手册

> **2026-06-09 Day13 启动**  
> 基于 Lux AI S3 第2名 (Frog Parade) 完整代码深度分析后的重构方案。  
> 核心思路: 简化系统 → 规模化训练 → 逐步增强

---

## 0. 背景诊断

经过 12 天的实验，当前 v11 系列效果一般。对比 Frog Parade（Lux S3 #2, 目前 Orbit Wars #1, 领先400分）的方案，识别出三个核心差距:

| 维度 | Frog Parade | 我们 (v11_f44) | 差距 |
|------|-------------|----------------|------|
| 模型规模 | 10M params | 680K params | 15x |
| 训练量 | 300M steps | ~几M/实验 | 10-100x |
| Self-play | 纯对称同模型 | frozen/strong/buffer/random 四重混合 | 过度复杂 |
| Reward | sparse win/loss | 7-8种 shaping | 策略畸形 |
| Value Head | Zero-sum softmax | 单方标准 value | 缺零和约束 |
| gamma | 0.9999 | 0.99 | 太低 |

---

## 1. 架构改动总览

### 1.1 新增文件

| 文件 | 用途 |
|------|------|
| `orbit_wars_rl/ppo/rollout_symmetric.py` | 对称 self-play rollout (同 params 双方) |
| `orbit_wars_rl/net/heads.py :: ZeroSumValueHead` | 零和 value head (训练时看双方 obs) |
| `orbit_wars_rl/inference/augmentation.py` | 推理时 4-fold 空间增强 |
| `orbit_wars_rl/configs/multi_action_v12_lux.yaml` | v12 训练配置 |

### 1.2 修改文件

| 文件 | 改动 |
|------|------|
| `orbit_wars_rl/net/model.py` | ActorCritic 新增 `zero_sum_value` 参数 + opp_obs 传递 |
| `orbit_wars_rl/ppo/rollout.py` | Rollout dataclass 新增 opp obs 字段 |
| `orbit_wars_rl/ppo/update.py` | PPO loss 传递 opp_obs 到 evaluate |
| `orbit_wars_rl/ppo/runner.py` | 新增 symmetric_selfplay / zero_sum_value 支持 |

### 1.3 新增脚本

| 脚本 | 用途 | 阶段 |
|------|------|------|
| `scripts/v12_smoke.sh` | 快速验证全链路 (~60s) | 前置 |
| `scripts/run_v12_lux.sh` | 阶段一正式训练 | 1 |
| `scripts/watch_v12_lux.sh` | 训练仪表板 | 1-3 |
| `scripts/v12_eval_gate.sh` | 单 checkpoint vs v20 评估 | 1-3 |
| `scripts/v12_milestone_eval.sh` | 批量 checkpoint 评估 | 1-3 |
| `scripts/v12_stage2_shaping.sh` | 阶段二微量 shaping | 2 |

---

## 2. 分阶段执行计划

### 阶段一: 纯 sparse + 对称 self-play (Day13 立即启动)

**目标**: 用大模型 + 纯 ±1 reward + 对称 self-play 建立基线

**配置要点**:
```
d=256, 4 layers, 8 heads, ff=1024 (~3-5M params)
gamma=0.9999, sparse ±1 only, 全部 shaping=0
纯对称 self-play (同 params 双方)
zero-sum value head (训练时看双方)
10000 updates × 4096 steps = ~41M steps
```

**执行**:
```bash
# Step 1: Smoke test (CPU 可跑, ~60s)
bash scripts/v12_smoke.sh

# Step 2: 正式训练 (需要 GPU)
bash scripts/run_v12_lux.sh

# Step 3: 监控 (另一个终端)
bash scripts/watch_v12_lux.sh
# 或者直接 tail
tail -f logs/v12_lux.log
```

**观测指标**:

| 指标 | 健康范围 | 危险信号 | 说明 |
|------|---------|---------|------|
| `ev` (explained_variance) | > 0.5 | < 0.3 | value head 学习能力 |
| `tR` (terminal_reward) | ≈ 0 | \|tR\| > 0.5 | 对称 self-play 中应接近 0 |
| `clip_frac` | < 0.20 | > 0.25 | PPO 更新幅度 |
| `spf` | 5-30 | < 2 or > 100 | 舰队规模 |
| `z0` | 0.1-0.5 | < 0.02 or > 0.9 | 跳过率 |
| `e2` | > 0.05 | 0.0 | 多舰队率 |
| `garr` | > 10 | < 3 | 囤兵量 |
| `emits` | 1-3 | < 1 or > 6 | 每回合发射数 |

**里程碑检查点**:

| Update | 检查内容 | 决策 |
|--------|---------|------|
| 100 | JIT 完成, ev > 0.3, loss 稳定下降 | 确认无 crash |
| 500 | ev > 0.5, WR vs random > 60% | 确认学习 |
| 1000 | spf > 5, garr > 10, WR random > 70% | 策略开始成型 |
| 2000 | 运行 `v12_eval_gate.sh` vs v20 | 关键决策点 |
| 3000 | 如 WR vs v20 > 0 → 继续; = 0 → 进入阶段二 | |
| 5000 | `v12_milestone_eval.sh` 批量评估 | 选择最佳 ckpt |
| 10000 | 最终评估 | 阶段一结果 |

**远程监控命令**:
```bash
# 同步日志到本地
rsync -avz charlie@server:/home/charlie/project/OrbitWarRL/logs/v12_lux.log logs/

# 查看仪表板
bash scripts/watch_v12_lux.sh

# 健康检查 (已有工具)
bash scripts/check_training_health.sh logs/v12_lux.log

# 通用监控器
python3 -m orbit_wars_rl.scripts.monitor_train logs/v12_lux.log --once
```

---

### 阶段二: 微量 Shaping 精调 (阶段一 ~3000 updates 后, 视情况)

**触发条件**: 
- WR vs v20 = 0, 但 ev > 0.5, spf/garr 有改善 → 加微量 shaping
- WR vs v20 > 0 → 跳过, 继续纯 sparse 训练

**执行**:
```bash
# 从阶段一最佳 ckpt resume, 加入微量 CAPTURE shaping
bash scripts/v12_stage2_shaping.sh ckpt_multi_action_v12_lux/ckpt_002000.pkl

# 监控
bash scripts/watch_v12_lux.sh logs/v12_stage2_shaping.log
```

**Shaping 策略**:
- 只用 `CAPTURE=0.02` (prod-weighted flip bonus)
- 系数极小, 不主导 ±1 terminal reward
- 提供 early-game gradient (鼓励占领高产星球)

---

### 阶段三: 推理增强 + 提交 (有可用模型后)

**触发条件**: 有 WR vs v20 > 30% 的 checkpoint

**执行**:
```bash
# 1. 评估最佳 checkpoint
bash scripts/v12_eval_gate.sh <best_ckpt> v12_best

# 2. 导出 submission (需要先创建 v12 submission 模板)
# 注意: v12 模板需要适配 d=256/4 layers/8 heads 的 numpy forward
python3 -m orbit_wars_rl.scripts.export_submission \
  --ckpt <best_ckpt> \
  --template submission_rl_v12_lux.py \
  --out submission_rl_v12_lux_filled.py

# 3. 本地 h2h 测试
python3 -m orbit_wars_rl.scripts.h2h_local \
  --agent-a submission_rl_v12_lux_filled.py \
  --agent-b submission_v20_0513.py \
  --num-games 20
```

**数据增强推理** (在 submission 模板中):
- 4-fold 空间对称增强 (identity / rot180 / mirror_x / mirror_y)
- 4 次 forward pass → logits 取平均 → 采样
- 使用 `orbit_wars_rl/inference/augmentation.py` 中的变换

---

## 3. 脚本速查表

```bash
# ============= 前置验证 =============
bash scripts/v12_smoke.sh                          # 全链路 smoke test (~60s)

# ============= 阶段一: 纯 sparse 训练 =============
bash scripts/run_v12_lux.sh                         # 启动训练 (后台)
FOREGROUND=1 bash scripts/run_v12_lux.sh             # 前台模式
NUM_UPDATES=100 bash scripts/run_v12_lux.sh          # 快速测试

# ============= 监控 =============
bash scripts/watch_v12_lux.sh                        # 训练仪表板
bash scripts/check_training_health.sh logs/v12_lux.log  # 健康检查
python3 -m orbit_wars_rl.scripts.monitor_train logs/v12_lux.log --once  # 通用监控

# ============= 评估 =============
bash scripts/v12_eval_gate.sh <ckpt> <tag>           # 单 ckpt vs v20
bash scripts/v12_milestone_eval.sh                    # 批量评估所有 ckpt
NUM_GAMES=10 bash scripts/v12_eval_gate.sh <ckpt> v12_u2000  # 更多局数

# ============= 阶段二: 微量 shaping =============
bash scripts/v12_stage2_shaping.sh <resume_ckpt>     # 从 ckpt resume + shaping
FOREGROUND=1 bash scripts/v12_stage2_shaping.sh <ckpt>

# ============= 远程操作 =============
# 同步日志
rsync -avz charlie@server:~/project/OrbitWarRL/logs/v12_lux.log logs/
# 同步 checkpoint
rsync -avz charlie@server:~/project/OrbitWarRL/ckpt_multi_action_v12_lux/ ckpt_multi_action_v12_lux/
```

---

## 4. 决策树

```
Day13 AM: v12_smoke.sh → run_v12_lux.sh
          │
Day13 PM: watch_v12_lux.sh (确认 JIT 完成, 训练开始)
          │
Day14 AM: upd ~1000-2000, 检查 ev/spf/garr
          │
          ├─ ev > 0.5, WR random > 70% → 正常, 继续
          │
          ├─ ev < 0.3 → value head 问题, 检查 zero-sum 是否生效
          │
          └─ clip > 0.25 → lr 过高, 考虑降低 lr_peak
          │
Day14 PM: upd ~3000, v12_eval_gate.sh vs v20
          │
          ├─ WR vs v20 > 30% → 继续纯 sparse 到 10000
          │   │
          │   └─ WR vs v20 > 50% → 阶段三: 导出 submission + 数据增强
          │
          ├─ WR vs v20 = 0 但指标改善 → 阶段二: v12_stage2_shaping.sh
          │
          └─ 全线 fail → 回退分析:
              ├─ 模型太大 JIT 太慢? → 尝试 d=192/3 layers 折中
              ├─ gamma=0.9999 导致发散? → 试 0.999
              └─ symmetric selfplay 太难? → 加回 warmup vs random
```

---

## 5. 离线评测 & Replay 可视化 操作手册

> 以下命令可在任意电脑执行，只需能 ssh 到训练服务器。  
> 服务器地址: `charlie@www.ultrapp.online`  
> 项目目录: `~/project/OrbitWarRL`

### 5.1 同步数据到本地

```bash
# 训练日志 (v12_lux_b 是当前活跃的 run)
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/logs/v12_lux_b.log logs/

# checkpoint 目录
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/ckpt_multi_action_v12_lux_b/ ckpt_multi_action_v12_lux_b/

# 评测结果 + replay HTML (如果已跑过)
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/logs/replay_analyze/ logs/replay_analyze/
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/logs/replay_html/ logs/replay_html/
```

### 5.2 训练仪表板 (本地看日志)

```bash
# 同步后在本地看
bash scripts/watch_v12_lux.sh logs/v12_lux_b.log

# 健康检查
bash scripts/check_training_health.sh logs/v12_lux_b.log
```

### 5.3 在服务器上跑评测 (SSH 进去执行)

```bash
ssh charlie@www.ultrapp.online
cd ~/project/OrbitWarRL

# ---- 自动选最新 checkpoint ----
LATEST=$(ls -1 ckpt_multi_action_v12_lux_b/ckpt_*.pkl | sort | tail -1)
echo "最新 checkpoint: $LATEST"

# ---- 评测 vs v20 (5 局, ~3-5 分钟, CPU 跑不占 GPU) ----
NUM_GAMES=5 bash scripts/v12_eval_gate.sh "$LATEST" v12b_latest

# ---- 批量评估所有 checkpoint (每个 5 局) ----
bash scripts/v12_milestone_eval.sh ckpt_multi_action_v12_lux_b

# ---- 查看评估结果 ----
cat logs/v12_milestone_eval.tsv
```

### 5.4 在服务器上生成 HTML Replay

```bash
ssh charlie@www.ultrapp.online
cd ~/project/OrbitWarRL

LATEST=$(ls -1 ckpt_multi_action_v12_lux_b/ckpt_*.pkl | sort | tail -1)

# 跑 3 个不同地图的 replay (vs v20)
bash scripts/v12_replay_html.sh "$LATEST" 0
bash scripts/v12_replay_html.sh "$LATEST" 42
bash scripts/v12_replay_html.sh "$LATEST" 123
```

### 5.5 同步 Replay 到本地浏览器查看

```bash
# 同步 replay
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/logs/replay_html/ logs/replay_html/

# 打开所有 replay (macOS)
open logs/replay_html/v12_u*/replay.html

# Linux
xdg-open logs/replay_html/v12_u*/replay.html
```

### 5.6 一键全套 (服务器上评测+replay, 本地同步+查看)

**服务器上** (SSH 进去一次性执行):
```bash
cd ~/project/OrbitWarRL
LATEST=$(ls -1 ckpt_multi_action_v12_lux_b/ckpt_*.pkl | sort | tail -1)
echo "=== 评估 $LATEST ==="
NUM_GAMES=5 bash scripts/v12_eval_gate.sh "$LATEST" v12b_latest
for SEED in 0 42 123; do
  bash scripts/v12_replay_html.sh "$LATEST" $SEED
done
echo "=== 完成 ==="
```

**本地** (任意电脑):
```bash
cd ~/project/OrbitWarRL   # 或你的本地项目目录

# 一次同步所有结果
rsync -avz charlie@www.ultrapp.online:~/project/OrbitWarRL/logs/ logs/

# 查看训练进度
bash scripts/watch_v12_lux.sh logs/v12_lux_b.log

# 查看评测结果
cat logs/replay_analyze/v12b_latest_vs_v20.summary.txt

# 打开 replay
open logs/replay_html/v12_u*/replay.html
```

### 5.7 训练中途运行状况速查

```bash
# 直接 SSH 看最后几行 (不用同步)
ssh charlie@www.ultrapp.online "tail -5 ~/project/OrbitWarRL/logs/v12_lux_b.log"

# 看进度 (已跑多少 update)
ssh charlie@www.ultrapp.online "grep -c '^upd ' ~/project/OrbitWarRL/logs/v12_lux_b.log"

# 看最近的 eval 结果
ssh charlie@www.ultrapp.online "grep 'WRr\|eval_vs_v20' ~/project/OrbitWarRL/logs/v12_lux_b.log | tail -5"

# 看 checkpoint 列表
ssh charlie@www.ultrapp.online "ls -lh ~/project/OrbitWarRL/ckpt_multi_action_v12_lux_b/*.pkl"
```

---

## 6. 关键超参 (v12_lux_b 当前活跃)

> v12_lux 初版 clip_frac=0.47 过高, v12_lux_b 修正了 lr 和 epochs。

| 参数 | v12_lux (初版) | v12_lux_b (当前) | 说明 |
|------|---------------|-----------------|------|
| d_model | 256 | 256 | Frog Parade |
| n_layers | 4 | 4 | |
| n_heads | 8 | 8 | |
| ff_dim | 1024 | 1024 | |
| gamma | 0.9999 | 0.9999 | Frog Parade |
| gae_lambda | 0.95 | 0.95 | |
| **lr_peak** | 3e-4 | **1e-4** | 修: 降低防 clip 过高 |
| lr_floor | 3e-5 | 2e-5 | |
| **update_epochs** | 4 | **2** | 修: 减少样本复用 |
| num_minibatches | 8 | 8 | |
| num_envs | 64 | 64 | |
| **rollout_length** | 64 | **32** | 修: 适配 32GB 显存 |
| steps/update | 4096 | 2048 | |
| episode_steps | 500 | 500 | Kaggle 官方 |
| symmetric_selfplay | true | true | Frog Parade |
| zero_sum_value | true | true | Frog Parade |
| emit_hard_stop | false | false | |
| flip_hard_mask | false | false | |
| 全部 shaping | 0.0 | 0.0 | 阶段一 pure sparse |

### v12_lux_b 修正效果

| 指标 | v12_lux @ upd 429 | v12_lux_b @ upd 2555 | 目标 |
|------|-------------------|---------------------|------|
| clip_frac | 0.47 ❌ | 0.18 ✓ | < 0.20 |
| kl | 0.106 ❌ | 0.028 ✓ | < 0.05 |
| ev | 0.93 | 0.93 | > 0.5 |
| spf | 18.9 | 23.1 | 5-30 |
| garr | 22.3 | 39.6 ↑↑ | > 10 |
