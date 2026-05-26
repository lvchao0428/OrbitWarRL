# DAY6 进展 — v11 失败复盘 + Ablation 执行计划

> 写于 2026-05-26。承接 [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md) 与 [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md)。
> Day5 结论：**v11 全栈 G1 在 self-play 极亮、vs v20 极暗**；Day6 不再堆全栈，改为 **拆解 ablation + curriculum sanity**。

---

## 0. TL;DR

| 维度 | 状态 |
|---|---|
| **v11 G1** | ✅ 800 upd；replay vs v20 **0/5**，spf **1.50**（v9b **4.76**） |
| **export** | ✅ `submission_rl_v11.py` + parity；**必须用 v11 template** |
| **G2/G3** | ⏳ launcher 已修，待 5090 续跑 |
| **Day6 主线** | Ablation：`k8_no_emit` / `k8_full` / `k16_no_emit` |
| **不做** | 新的「5 信号全栈一次堆」；BC/IL 注入 v20 权重 |

---

## 1. 术语：IL 启动是什么？

**IL = Imitation Learning（模仿学习）**。在 RL 比赛里常说的「IL 启动」，指：

> 先用**专家数据**（人类 replay、v20 对局、Top10 日志）训练一个**行为克隆（BC）** 策略，再用 **RL fine-tune** 把策略推到 self-play / 对手分布里泛化。

典型流程：

```
专家 replay / 强 bot 对局
    → 收集 (obs, action) 对
    → BC 训练（交叉熵，模仿 src/dst/pct/emit）
    → 得到 ckpt_0（已会「大致像高手」）
    → PPO / self-play 继续训（RL fine-tune）
```

**和我们要做的 C1 / v20 curriculum 的区别：**

| 方式 | v20 角色 | 梯度 | 上限 |
|---|---|---|---|
| **IL 启动（BC）** | 专家 **标签** | 反向传播 v20 的 **动作** | 很难超过 v20 |
| **C1 curriculum（strong opp）** | **对手** | 只学 P0；v20 动作 **不进 loss** | 可以超过 v20 |
| **v20 state buffer reset**（Plan B） | **局面分布** | 从 v20 常见 state 开局 self-play | 不模仿动作，只改分布 |

本项目 **Day5 已排除 BC/IL 注入权重**（见 `DAY5_PLAN` §2.3）。Lux/Halite 冠军用 IL 是因为 **没有现成 JIT 强 bot 当对手**；我们有 v20 submission，更划算的是 **当对手或当 state 分布**，不是当 teacher 抄作业。

**若 ablation 全失败**，可选 IL 作为 **最后手段**（collect v20 states → BC 1 epoch → RL），但优先级低于 K=8 / 关 R4 / C1。

---

## 2. Day5 收尾数字（决策依据）

### 2.1 v11 G1 vs v20 replay（first-80，5 局）

| metric | v11 G1 | v9b | v20 (p1) | Day5 gate |
|---|---|---|---|---|
| WR | **0/5** | 0/5 | 5/0 | — |
| `spf` | **1.50** ↓ | 4.76 | 16.10 | >10 |
| `garr` | **7.86** ↓ | 31.85 | 155 | >60 |
| `flip_proxy` | **0.66%** ↓ | 3.88% | 13.5% | >6% |
| `emit≥2` | **~43%** ✓ | 2.8% | — | >5% |
| `mean_emits/turn` | 3.53 | ~2 | 1.11 | — |
| `bin0+bin1` pct | **62.5%** | — | 8.8% | — |

### 2.2 失败机制（一句话）

**R4 `EMIT_LOG` + K=16** 在 self-play 高 garr 下奖励「多 emit + 小 pct」；对 v20 early garr≈8，同一策略 → **1 舰/launch**，比 v9b 更差。

### 2.3 训练 vs eval gap（再次确认）

| | G1 训练 @799 | G1 replay first-80 |
|---|---|---|
| spf | 79.2 | 1.50 |
| garr (per-planet mean) | 291 | 7.86 |
| e2 | 85% | ~43% |

---

## 3. Day6 战略（激进但拆解）

**激进** = 坚持 Top10 方向（R1 delta、R2 release、A1' obs、ep500、multi-emit **行为**），但 **one-delta-at-a-time** 验证，不再全栈打包。

**三条并行轨道（GPU 串行，时间错开）：**

| 轨道 | 内容 | 时长 | 目的 |
|---|---|---|---|
| **A** | G2→G3 续跑（C1：25% strong=G1@399） | ~8h | curriculum 框架是否给信号 |
| **B** | Ablation 三路 scratch | ~12h |  isolate K vs R4 |
| **C** | 每条 winner 候选 → `quick_replay.sh` | CPU ~10min/条 | vs v20 gate，唯一真理 |

---

## 4. 今日执行计划（5090）

### Phase 0 — 同步代码（5 min）

```bash
cd ~/project/OrbitWarRL
git pull   # 或 rsync 本地已改文件
# 确认存在：
#   submission_rl_v11.py
#   scripts/run_v11_validation.sh  (SKIP_PHASE1, stderr fix)
#   scripts/run_v11_ablation.sh
#   scripts/quick_replay.sh
```

### Phase 1 — G2/G3 续跑（优先，~8h）

G1 strong anchor = G1@399（同架构 spam 策略）；**期望有限**，但验证 curriculum 管道。

```bash
nohup env SKIP_PHASE1=1 \
  CKPT_G1=ckpt_multi_action_v11_g1_scratch/ckpt_000799.pkl \
  bash scripts/run_v11_validation.sh \
  > logs/v11_g2g3.launcher.log 2>&1 &
echo "pid=$!"
```

**监控：**

```bash
tail -f logs/v11_g2g3.launcher.log
python -m orbit_wars_rl.scripts.monitor_train --once \
  logs/multi_action_v11_g2_curriculum.log \
  logs/multi_action_v11_g3_continue.log
```

**G2 完成后立即 eval（CPU，不占 GPU）：**

```bash
bash scripts/quick_replay.sh \
  ckpt_multi_action_v11_g2_curriculum/ckpt_000799.pkl \
  v11_g2_u799
```

**G2 sign-of-life（vs v20 first-80）：**

| metric | 比 G1 好就算 | gate |
|---|---|---|
| spf | > **3** | >10 |
| garr | > **20** | >60 |
| flip | > **2%** | >6% |

若 G2 ≈ G1 → **跳过 G3 意义不大**，直接进 Phase 2 ablation。

### Phase 2 — Ablation 三路（G2 跑完或 GPU 空档启动，~12h）

```bash
nohup bash scripts/run_v11_ablation.sh \
  > logs/v11_ablation.launcher.log 2>&1 &
```

| tag | K | R4 emit | 假设 |
|---|---|---|---|
| `k8_no_emit` | 8 | **OFF** | 最可能改善 spf/garr |
| `k8_full` | 8 | ON gated | R4 在 K=8 下是否仍有害 |
| `k16_no_emit` | 16 | OFF | 关 R4 是否足够（K 不变） |

单路调试：

```bash
ONLY=k8_no_emit bash scripts/run_v11_ablation.sh
```

**汇总：**

```bash
column -t -s $'\t' logs/v11_ablation_summary.tsv
```

### Phase 3 — 每条 ablation 完 → replay gate（CPU）

```bash
for v in k8_no_emit k8_full k16_no_emit; do
  bash scripts/quick_replay.sh \
    "ckpt_multi_action_v11_${v}/ckpt_000799.pkl" \
    "v11_${v}"
done
```

输出：`logs/replay_analyze/v11_<tag>_vs_v20.summary.txt`

### Phase 4 — 决策（明晚）

| 结果 | 动作 |
|---|---|
| **`k8_no_emit` ≥ 2/4 gate** | 开 **overnight 4000 upd**（同配方）；G2 可加 C1 strong=该 ckpt@399 |
| **`k8_full` 过、`k8_no_emit` 不过** | R4 仍有害 → 永久关 EMIT_LOG，调 RELEASE |
| **`k16_no_emit` 过、k8 不过** | K 不是主因 → 查 pct head / R2 系数 |
| **三路全 FAIL** | 启动 **v20 state-buffer curriculum**（Plan B 工程） |
| **G2  alone 过 gate** | resume G2 + 4000，strong_ratio 提到 0.35 |

**Overnight 模板（winner 定稿后）：**

```bash
# 示例：k8_no_emit promote
UPD_PER_PHASE=4000 ONLY=k8_no_emit bash scripts/run_v11_ablation.sh
# 或改 yaml num_updates=4000 手动 train
```

---

## 5. 工具与 export 备忘

### 5.1 Export（必须 v11 template）

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  python -m orbit_wars_rl.scripts.export_submission \
  --ckpt ckpt_multi_action_v11_<tag>/ckpt_000799.pkl \
  --template submission_rl_v11.py \
  --out submission_rl_v11_<tag>_u799.py
```

- 勿用 `submission_rl_v4.py`（19-d vs 22-d → agent 静默返回 `[]`）
- parity 默认 CPU，避免与训练抢 VRAM

### 5.2 新增脚本（Day6）

| 文件 | 用途 |
|---|---|
| [`scripts/run_v11_ablation.sh`](../scripts/run_v11_ablation.sh) | 三路 ablation 串行 + TSV |
| [`scripts/quick_replay.sh`](../scripts/quick_replay.sh) | export + replay + gate 一行 |
| [`orbit_wars_rl/configs/multi_action_v11_k8.yaml`](../orbit_wars_rl/configs/multi_action_v11_k8.yaml) | K=8 训练配置 |
| [`submission_rl_v11.py`](../submission_rl_v11.py) | 22/14-d + K=16 inference |

---

## 6. Strong teacher / 外部对手（Day6 立场）

- **需要「强压力」**，不必须 **IL**。
- v20 启发式当 **对手**（C1）或 **局面分布**（buffer reset）≈ AlphaStar league exploiter / Halite rule-bot pool。
- **不做**：BC 把 v20 权重注入 policy（上限锁死）。
- **defer**：MCTS（算力/工程不划算于单卡 5090）。

---

## 7. 成功定义（Day6 末）

**最低：**

- 完成 G2 eval + ablation 三路 replay
- 至少 1 条 config **spf > v9b（4.76）** 或 **garr > 31**

**目标：**

- 某条 ablation **2/4 first-80 gate** → 进 overnight 4000
- 明确 **R4 vs K** 谁主责（文档写进 §8）

**未达成：**

- 不回到 v9 macro 扫参
- 排期 v20 buffer curriculum（Day7 工程）

---

## 8. 日志路径速查

| 用途 | 路径 |
|---|---|
| G2/G3 launcher | `logs/v11_g2g3.launcher.log` |
| Ablation launcher | `logs/v11_ablation.launcher.log` |
| Ablation 汇总 | `logs/v11_ablation_summary.tsv` |
| Replay JSON | `logs/replay_analyze/v11_<tag>_vs_v20.json` |
| Replay gate 一行 | `logs/replay_analyze/v11_<tag>_vs_v20.summary.txt` |
| G1 基线 replay | `logs/replay_analyze/v11_g1_vs_v20.json` |

---

---

## 10. Plan B：v20 State-Buffer Curriculum（兜底方案）

> 已实现，ablation 全 fail 时直接启动。不需要 IL / BC 权重注入。

### 10.1 原理

```
v20 自对局（200 games）
    → 采集每步 EnvState（前 80% 局面）
    → 存 data/v20_states_200g.npz  (~50k states)
    ↓
PPO rollout autoreset 时：
    以 buffer_reset_ratio=0.30 概率
    从 buffer 中随机抽一个 state 代替随机地图
    → learner 从 v20 真实中盘局面开始 self-play
```

v20 不提供动作 label（≠ BC），只提供**局面分布**。Learner 从这些局面继续 self-play，奖励来自 RL，无 teacher 梯度。

### 10.2 新增代码文件

| 文件 | 用途 |
|---|---|
| [`orbit_wars_rl/bc/collect_states.py`](../orbit_wars_rl/bc/collect_states.py) | 采集 v20 自对局 EnvState → npz |
| [`orbit_wars_rl/ppo/rollout.py`](../orbit_wars_rl/ppo/rollout.py) | 新增 `make_rollout_fn_with_buffer_reset()` |
| [`orbit_wars_rl/ppo/runner.py`](../orbit_wars_rl/ppo/runner.py) | `SelfPlayConfig.buffer_path/ratio`；`load_state_buffer()`；训练循环 `opp_tag="buf"` |
| [`orbit_wars_rl/configs/multi_action_v11_buf.yaml`](../orbit_wars_rl/configs/multi_action_v11_buf.yaml) | K=8 + R4 off + buffer curriculum 训练配置 |

### 10.3 执行命令（5090，ablation fail 后）

**Step 1 — 采集 state buffer（在 CPU 上跑，~40 min）**

```bash
# 在 5090 上（用 CPU，不占训练 GPU）
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  python -m orbit_wars_rl.bc.collect_states \
    --agent submission_v20_0513.py \
    --num-games 200 \
    --episode-steps 500 \
    --skip-tail-frac 0.20 \
    --out data/v20_states_200g.npz \
  > logs/collect_states.log 2>&1 &
echo "pid=$!"

# 快速 smoke test（先跑 2 局验证无报错）
python -m orbit_wars_rl.bc.collect_states \
    --agent submission_v20_0513.py \
    --num-games 2 \
    --out /tmp/v20_states_smoke.npz \
    --debug
```

**Step 2 — 验证 npz 尺寸**

```bash
python -c "
import numpy as np
d = np.load('data/v20_states_200g.npz')
n = d['planet_ships'].shape[0]
print(f'states={n}  step_range=[{d[\"step\"].min()},{d[\"step\"].max()}]')
print(f'size: {sum(v.nbytes for v in d.values())/1e6:.1f} MB')
"
```

预期：`states ≈ 40k-60k，step_range=[0, ~400]，size < 100 MB`

**Step 3 — 训练（K=8 + R4 off + buffer 30%）**

```bash
SHAPING_EMIT_LOG=0.0 \
nohup python -m orbit_wars_rl.scripts.train \
    --config orbit_wars_rl/configs/multi_action_v11_buf.yaml \
    --log-dir logs/v11_buf_run1 \
  > logs/v11_buf_run1.log 2>&1 &
echo "pid=$!"
```

监控 `opp_tag` 是否出现 `buf`（每 50 upd 后进 warmup 期后开始）：

```bash
grep "opp buf" logs/v11_buf_run1.log | tail -5
```

**Step 4 — 800 upd 后 replay gate**

```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_buf/ckpt_000799.pkl \
    v11_buf
```

**Step 5 — 决策**

| 结果 | 动作 |
|---|---|
| `spf > 4.76` 或 `garr > 31` | overnight 4000 upd |
| 不过 | buffer_reset_ratio → 0.50 或换 K=16 |
| 全 fail | 联系赛事，换更大模型 / 更多 GPU |

### 10.4 参数调节

`buffer_reset_ratio` 是最关键旋钮：

| 值 | 效果 |
|---|---|
| 0.0 | 退化为普通 self-play（不用 buffer） |
| 0.30 | **推荐起点**：70% 随机地图，30% v20 局面 |
| 0.50 | 更多 v20 场景，但 garr/spf 分布更窄 |
| 1.0 | 全部来自 buffer，policy 可能对 buffer 外泛化差 |

可在 YAML 里直接改 `selfplay.buffer_reset_ratio`，无需重新采集 buffer。

---

## 9. 文档索引

| 文档 | 用途 |
|---|---|
| [`DAY5_PROGRESS.zh.md`](DAY5_PROGRESS.zh.md) | Day5 SSOT、v11 G1 数字 |
| [`DAY5_PLAN.zh.md`](DAY5_PLAN.zh.md) | Track A/B/C 原始规划 |
| [`DAY5_TRAINING_ACTIONS.zh.md`](DAY5_TRAINING_ACTIONS.zh.md) | R1/R2/R4 公式与系数 |
| [`FAST_ITER_RUNBOOK.md`](FAST_ITER_RUNBOOK.md) | FAST gate 操作 |
| [`H2H_EVAL_RUNBOOK.md`](H2H_EVAL_RUNBOOK.md) | gauntlet / replay 全流程 |
| **本文** | Day6 执行计划 + IL 术语 + Plan B 执行命令 |
