# DAY20 — v23 停跑 + BC 蒸馏 v20 + v24 KL 锚定微调

> **动机**：v22/v23 从头 PPO 仍无法同时稳住 flip 与 e2+，且 vs-v20 全程 0 胜。
> v20 启发式本身 flip 与 multi-emit 双高 → 改走 **BC 克隆 v20 行为 → PPO 在 BC 基础上微调** 路线。
> 你的 reward / 特征在 P0 **特征全程生效**（BC 用同一套 `encode()`）；reward 在 P0 不参与，**P1 v24 全部回归**。

---

## 1. 背景：v23 Route A 结论（已停）

| 项 | v23 Route A |
|----|-------------|
| 训练 | 从头，v21 特征（planet 63 / global 427） |
| 对手 | strong v16a 25% + frozen 25% + rand 50% |
| Reward | flip-gated capture 0.02 + fleet_scale 0.01 + multi_emit 0.02 + anti_hoard 0.04 |
| 状态 | **u4000+ 停跑**，WLD 持续 0/5，flip/e2+ 反相关震荡 |

**结论**：5090 算力不再投入 v23；继续 PPO 探索局部最优，不如换 BC 初始化。

---

## 2. 路线总览

```
┌─────────────────────────────────────────────────────────────────┐
│  P0  BC 蒸馏 v20（今晚跑通）                                      │
│    v20 自对弈 → (state, action) → 监督训练 ActorCritic           │
│    特征：与 v23 同款 encode()（hist50 + planet_hist + ETA-lead）  │
│    Reward：不参与（纯模仿 v20 动作分布）                            │
├─────────────────────────────────────────────────────────────────┤
│  P1  v24 PPO 微调（BC ckpt 热启动 + KL 锚）                       │
│    resume_ckpt = BC ckpt（同 arch，零 shape 适配）                  │
│    KL(pi || pi_BC) 防塌回 turtling / multi-spam                  │
│    对手：BC 网 strong 40% + frozen 30% + rand 30%                 │
│    Reward：v23 全套 shaping 回归                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 启发式 v20 如何变成 RL 初始参数？

BC 不是把 v20 的 Python 代码塞进网络，而是：

1. 跑 v20 vs v20，每 turn 记录 **encode(state)** + v20 的 moves
2. `action_inverse.py` 把 moves 反解为 K-step 头标签（src/dst/pct/emit）
3. 监督训练 **同一个 ActorCritic**（与 PPO 完全同 arch）
4. 保存的 `.pkl` 与 PPO ckpt 格式一致 → `resume_ckpt` 直接热启动

---

## 3. P0：BC 模块

### 3.1 新增 / 恢复文件

| 文件 | 说明 |
|------|------|
| `orbit_wars_rl/bc/collect_data.py` | v20 自对弈采集；跨 turn 维护 global_hist / planet_hist；home_idx + 轨道字段 |
| `orbit_wars_rl/bc/train_bc.py` | 监督训练；v23 网络规格 d256/L4；BC 阶段关闭推理掩码 |
| `orbit_wars_rl/bc/action_inverse.py` | kaggle moves → K-step 标签；emit_free 对齐 allow_hold |
| `orbit_wars_rl/bc/test_action_inverse.py` | 单测（16 pct bins） |
| `scripts/bc_collect_parallel.sh` | 8 进程并行采集 + merge |

### 3.2 采集数据

| 项 | 值 |
|----|-----|
| 规模 | 8 × 50 局 = **400 局** |
| 样本 | **113,900** turn 样本（双视角） |
| 维度 | planet_feats (40, 63)，global_feats (427) |
| 分布 | hold 60.6%，e2+ 20.5%（与 v20 本体一致） |
| 输出 | `data/bc_v20_self_400g.npz`（200 MB） |

### 3.3 BC 训练

| 版本 | emit_pos_weight | val acc (src/dst/pct/emit) | h2h vs v20 (20 局) |
|------|-----------------|---------------------------|-------------------|
| 默认 | 1.0 | 0.37 / 0.47 / 0.39 / **0.775** | flip 7.0%，z0 **70%**，e2+ **2.2%** — 太保守 |
| **epw25** | **2.5** | 0.37 / 0.47 / 0.38 / 0.745 | flip **12.1%**，z0 **43.6%**，e2+ **9.7%** — **选用** |

**选用 epw25**：val emit acc 略降，但行为分布更接近 v20（z0 从 70% 降到 44%）。

Ckpt：`ckpt_bc_v20_epw25/ckpt_final.pkl`（export flags: allow_hold=1, emit_hard_stop=0, min_pct_bin=0）

---

## 4. P1：v24 配置

| 项 | v24 |
|----|-----|
| Init | `resume_ckpt = ckpt_bc_v20_epw25/ckpt_final.pkl` |
| KL 锚 | `bc_anchor_ckpt` 同上，`kl_ref_coef = 0.05` |
| 对手 | strong(BC) **40%** + frozen **30%** + rand **30%** |
| lr | peak **3e-5**（BC 基础上微调，非从头探索） |
| Masks | 忠实 BC：no emit_hard_stop, min_pct_bin=0, allow_hold=true |
| Reward | 同 v23：flip-gated capture 0.02 + fleet_scale 0.01 + multi_emit 0.02 + anti_hoard 0.04 |
| num_updates | **4000**（已跑完） |

### 新增代码

| 文件 | 改动 |
|------|------|
| `orbit_wars_rl/ppo/update.py` | `kl_ref_coef` + per-head masked KL(pi \|\| pi_ref) |
| `orbit_wars_rl/ppo/runner.py` | `bc_anchor_ckpt` 加载冻结参考策略；日志 `klR` |
| `orbit_wars_rl/configs/multi_action_v24_bc_ft.yaml` | v24 配置 |
| `scripts/v24_bc_ft.sh` | 启动脚本 |
| `scripts/v24_remote.sh` | 远程 collect / train-bc / h2h / start / eval |

---

## 5. v24 训练结果（u0–u3999，已完成）

日志：`logs/v24_bc_ft_20260613_010928/train.log`  
Ckpt：`ckpt_multi_action_v24_bc_ft/`

### 5.1 vs-v20 inline eval 全表（每 100u × 5 局）

| u | flip | e2+ | z0 | spf | WLD |
|---|------|-----|-----|-----|-----|
| 99 | 7.7% | 5.8% | 45.5% | 37.2 | 0/5/0 |
| 199 | 9.3% | 2.2% | 55.7% | 35.9 | 0/5/0 |
| 299 | 11.7% | 2.8% | 49.8% | 36.9 | 0/5/0 |
| 399 | 11.3% | 6.8% | 35.2% | 27.4 | 0/5/0 |
| **499** | 10.6% | **12.8%** | 25.1% | 21.4 | 0/5/0 |
| 599 | 13.7% | 5.8% | 40.1% | 21.7 | 0/5/0 |
| 699 | 10.5% | **14.3%** | 31.2% | 18.2 | 0/5/0 |
| 799 | 10.5% | 10.2% | 42.5% | 24.6 | 0/5/0 |
| **999** | **17.1%** | 4.2% | 43.3% | 18.2 | 0/5/0 |
| 1099 | 5.7% | 9.0% | 37.5% | 21.5 | **1/4/0** |
| 1199 | 14.6% | 5.2% | 42.3% | 25.1 | 0/5/0 |
| 1499 | 4.5% | **20.2%** | 22.1% | 21.6 | **1/4/0** |
| 1699 | 2.8% | **21.3%** | 22.6% | 18.4 | **1/4/0** |
| 1999 | 15.0% | 13.5% | 32.4% | 20.5 | 0/5/0 |
| **2399** | 10.1% | **29.2%** | 29.5% | 16.6 | 0/5/0 |
| 2699 | 7.9% | 17.5% | 27.9% | 17.5 | 0/5/0 |
| 2899–3499 | ~3–4% | **21–28%** | ~22% | ~20 | **1/4/0 ×6** |
| 3599 | **15.9%** | 12.5% | 40.3% | 22.9 | 0/5/0 |
| 3699 | **16.1%** | 11.5% | 42.7% | 23.6 | 0/5/0 |
| **3999** | 9.4% | 14.0% | 46.8% | 23.6 | **1/4/0** |

### 5.2 关键指标

| 信号 | 峰值 | 终态 u3999 | vs v23 / v22 |
|------|------|-----------|--------------|
| flip | **17.1%** @u999 | 9.4% | 历史首次 eval 里 flip 与 e2+ **可同时为正** |
| e2+ | **29.2%** @u2399 | 14.0% | 远超 v23 全程 ~0–18% 震荡 |
| z0 | 最低 20.6% @u2299 | 46.8% | 后期有所回升（囤兵） |
| WLD | **13 次 1/4**（u1099–u3999） | 1/4/0 | **首次 vs-v20 开张**（5 局噪声下） |
| vs-random | — | **97%** | 自博弈侧正常 |
| klR | — | ~0.35 | KL 锚稳定，未断 |

训练侧终态（u3999）：e2=0.58，z0=0.33，prod_share=0.48，spf=27.8 — 比 v23 终态健康得多。

### 5.3 u500 验收门

| 门 | 目标 | u499 实际 | 通过？ |
|----|------|----------|--------|
| flip | ≥30% | 10.6% | ❌ |
| e2+ | ≥15% | 12.8% | ⚠️ 接近 |
| WLD 开张 | ≥1/5 | 0/5 | ❌（u1099 后多次 1/4） |

**结论**：BC→v24 路线 **显著优于 v23**（e2+ 与 WLD 突破），但 flip↔e2+ tradeoff **仍未根治**——高 e2+ 段（u2299–u3499）flip 塌到 3–4%，高 flip 段（u999/u3599）e2+ 仅 4–12%。

---

## 6. 诊断与下一步

### 6.1 flip↔e2+ 为何还在？

| 因素 | 说明 |
|------|------|
| BC 上限 | epw25 clone flip 仅 12%，e2+ 10% — PPO 起点就不及 v20 本体 |
| 5 局 eval 噪声 | WLD 1/4 与 0/5 交替，信号不可靠 |
| Reward 仍多吸引子 | multi_emit 与 flip-gated capture 在 PPO 阶段仍可能拉扯 |
| KL 锚偏保守 | klR≈0.35 持续，策略被拉回 BC 附近，难超越 v20 |

### 6.2 v25 已启动（u3999 续训 ~8h）

HTML replay 确认 **u3999 seed=0** 为最佳胜局（210 步快赢，flip~9% e2+~14%）。v25 从该 ckpt 续训：

| 项 | v25 |
|----|-----|
| Init / Anchor | `ckpt_003999`（KL 锚向 u3999 自身） |
| KL | 0.04 → 0.02 线性衰减，8000 updates |
| LR | 2e-5（低于 v24 3e-5） |
| Reward | v24 + multi_emit 需 home_garr≥30、min_ships=10 |
| Opp | strong(BC) 45% + frozen 30% + rand 25% |
| Updates | 11000（~8h） |

```bash
bash scripts/v25_remote.sh sync && bash scripts/v25_remote.sh start 11000
bash scripts/v25_remote.sh tail
bash scripts/v25_remote.sh eval
```

### 6.3 后续方案（v25 跑完再试）

| 优先级 | 方案 | 说明 |
|--------|------|------|
| B1 | BC 加强 | 800 局采集 / emit_pos_weight=3.0 重训 |
| B2 | u999 flip peak resume | flip=17% ckpt 续训，补 e2+ |
| B3 | u3399 消耗型 | 高 e2+ 低 flip，仅对照 replay |
| B4 | Eval 20 局 | 最终 h2h 验收 |

---

## 7. 远程操作速查

```bash
# BC 管线
bash scripts/v24_remote.sh collect          # 并行采集 v20 自对弈
bash scripts/v24_remote.sh train-bc 10       # 训 BC（epochs）
bash scripts/v24_remote.sh h2h 20          # BC clone vs v20 replay

# v24 PPO
bash scripts/v24_remote.sh sync
bash scripts/v24_remote.sh start 4000
bash scripts/v24_remote.sh tail
bash scripts/v24_remote.sh eval

# v23（已停，仅查历史）
bash scripts/v23_remote.sh stop-v23
bash scripts/v23_remote.sh eval
```

---

## 8. 执行时间线

| 时间 | 事件 |
|------|------|
| 6/12 晚 | v23 Route A u4000+ 停跑；BC 模块恢复 + 适配 |
| 6/13 00:00 | 400 局 v20 数据采集完成（113k 样本） |
| 6/13 00:35 | BC 10 epochs 训练完成；epw25 版 h2h flip=12% e2+=10% |
| 6/13 01:09 | **v24 开跑**（BC init + KL 锚） |
| 6/13 ~05:30 | **v24 u3999 跑完**；vs-random 97%；13 次 vs-v20 1/4 |
| 6/13 | 日志 `logs/v24_bc_ft_20260613_010928/train.log` |

---

## 9. 版本对比（决策参考）

| 版本 | 初始化 | flip peak | e2+ peak | vs-v20 WLD | 核心教训 |
|------|--------|-----------|----------|------------|----------|
| v16a | scratch + capture 0.05 | — | — | **1/3** | 裸 capture 能赢但缺 multi-emit |
| v18 | v17 resume | 42% | ~0% | 0/5 | flip 强、e2+ 结构性为 0 |
| v23 | scratch v21 feat | ~17% | ~19% | 0/5 | mixed opp 仍塌；flip↔e2+ 震荡 |
| BC epw25 | v20 克隆 | 12% | 10% | 0/20 | 行为对齐 v20，但不会赢 |
| **v24** | **BC + KL PPO** | **17%** | **29%** | **1/4 ×13** | **路线正确，需继续调 BC 上限 + eval 样本** |
