# DAY21 — v26 ROI 收束 + v27 frontier 快训

> **动机**：v25 u9599 后 vs-v20 仍难稳定开张；v26 加 ROI capture 特征/reward，跑一夜与 v25 同级。
> 中后盘 **friendly shuffle** 未改善 → v27 加 frontier / anti-shuffle / capture_need + top10 状态 buffer。
> 比赛冻结前采用 **快训版**（4000u，~3.5h 全流程）。

---

## 1. v26 结论（已停，u7999）

| 项 | 值 |
|----|-----|
| 初始化 | v25 `ckpt_009599.pkl` |
| 训练量 | ~u7999 / 14200（~56%，手动停跑转 v27） |
| 特征 | dim32 `capture_roi_norm`；global dim15 `local_roi_targets_norm` |
| Reward | `ORBITWARS_SHAPING_CAPTURE_ROI=0.025`（flip-gated，legacy capture=0） |
| ckpt | `ckpt_multi_action_v26_roi/ckpt_007999.pkl` |

### vs-v20 inline eval（末段）

| u | flip | e2+ | z0 | garr | WLD |
|---|------|-----|-----|------|-----|
| u6599 | 24.3% | 12.4% | 51.1% | 79.6 | 0/20 |
| u7199 | 9.3% | 11.2% | 46.8% | 92.7 | **1/19** |
| u7999 | 24.1% | 14.4% | 45.8% | 82.0 | 0/20 |

**结论**：flip/e2+ 与 v25 同量级，**WLD 仍几乎不开张**；u7199 胜局 replay 可见 mid-game 己方互调、远点占星。

### v26 ROI 诊断（为何「加了特征仍≈v25」）

| 问题 | 说明 |
|------|------|
| dim32 己方恒 0 | ROI **不会**让己方 dst 分数变高；shuffle 来自 pair `[2]`、`hold_value`、v25 权重惯性 |
| Reward 稀疏 | 仅 **flip 成功** 给 ROI shaping；shuffle **无惩罚** |
| 空间 horizon | `early_term=max(0,20−η)`，远点 dim32→0，中后盘边线外目标「看不见 ROI」 |
| 热启动 | dim32 语义从 `v20_target_score` 替换，KL 拴 v25，8000u 不够重学 dst |

---

## 2. v27 方案（快训）

### 2.1 特征（63/427 不变，语义替换）

| 维度 | v27 信号 |
|------|----------|
| planet 26 | `frontier_score_norm` |
| planet 30 | `capture_need_exact_norm` |
| planet 41 | `shuffle_dst_risk_norm` |
| planet 42 | `interior_planet_bin` |
| global 17 | `frontier_owned_norm` |
| global 22 | `home_under_threat` |
| dim32 ROI | × frontier × 距离衰减 |

### 2.2 Reward / 数据

| 项 | 值 |
|----|-----|
| capture_roi | 0.025 + frontier 加权 + flip 精准度 |
| friendly_shuffle | **0.01**（v26 无） |
| BC | v20 **200g**（4×50）→ `ckpt_bc_v27_epw30` |
| top10 buffer | `data/top10_winner_states.npz`，reset **20%**（仅状态，无动作 BC） |
| PPO | **4000u**，eval 10g / 400u |
| Resume | v26 **`ckpt_007999.pkl`** |
| Anchor KL | 0.04 → 0.015 / 4000u |

### 2.3 文件

| 文件 | 说明 |
|------|------|
| `orbit_wars_rl/features/frontier_util.py` | frontier / shuffle / capture_need 共享 |
| `orbit_wars_rl/configs/multi_action_v27_frontier.yaml` | 快训配置 |
| `scripts/v27_pipeline.sh` | Gate0–4 一键 |
| `scripts/v27_one_click.sh` | sync → smoke → pipeline |
| `scripts/v27_remote.sh` | 远程 status/tail/eval/wait |
| `TOP10_BEHAVIOR_REFERENCE.zh.md` | Top10 vs v26 行为对照 |

---

## 3. 本地 Gate0（已通过）

```bash
bash scripts/v27_smoke.sh          # test_v27_frontier + capture_roi + smoke_v27
CHECK_REMOTE=0 bash scripts/test_v27_preflight.sh
```

---

## 4. 远程 v27 流水线

**状态**：2026-06-14 已启动；v26 进程已 kill。

```bash
bash scripts/v27_remote.sh status
bash scripts/v27_remote.sh log
bash scripts/v27_remote.sh tail      # Gate3 后看 train.log
bash scripts/v27_remote.sh eval
bash scripts/v27_remote.sh wait
```

### 预计耗时（快训）

| 阶段 | ~时间 |
|------|--------|
| Gate1 BC 200g + top10 并行 | 30–45 min |
| Gate2 BC train 5ep | 15 min |
| Gate3 PPO 4000u | **2.5 h** |
| Gate4 post h2h | 30 min |
| **合计** | **~3.5 h** |

### 验收（同 v26 目标）

| 指标 | 目标 |
|------|------|
| 20 局 h2h WLD | ≥ 4/16/0 |
| flip + e2+ | flip≥8% 且 e2+≥12% |
| replay 目测 | friendly shuffle ↓，边线 neutral 占掉 |
| vs-random | ≥ 95% |

---

## 5. 版本路线简表

| 版本 | 核心 | vs-v20 WLD | 备注 |
|------|------|------------|------|
| v25 u9599 | extend + strong BC | 偶发 1/4~1/19 | 当前最强基线 |
| v26 u7999 | + ROI feature/reward | ~0/20 | 已停，shuffle 未解 |
| **v27** | frontier + anti-shuffle + top10 buffer | 训练中 | 冻结前主版本 |

---

## 6. 提交路径（v27 跑完后）

1. `bash scripts/v27_post_train.sh` → h2h + 胜局 HTML  
2. 选 eval WLD 最高 ckpt：`ckpt_multi_action_v27_frontier/eval/`  
3. `python -m orbit_wars_rl.scripts.export_submission --ckpt ... --template submission_rl_v21.py`  
4. 胜局 replay 查 shuffle / frontier 占点
