# DAY15 — v14d：A/B 备料 + Phase C 长训

> **策略**：Phase A/B 用二分搜参 + confirm **为 C 长训准备 ckpt**；Phase C **单独 10k updates 长跑**（不在二分里打断）。

---

## 流水线

```
Phase A 二分搜 (8维) → confirm (1200u) ──gate A──┐
                                                  ▼
Phase B 二分搜 (9维) → confirm (1500u) ──gate B──┘
                                                  │
                     logs/v14d_c_longtrain.ready.json
                                                  ▼
Phase C 长训: 10000u (+ extend×3) resume B ckpt
  force_emit_worth_it + 低 lr/entropy
  目标: Gate C + beats v13c_final
```

### Gate 继承

| 阶段 | 作用 | 过 gate 才继承 |
|------|------|----------------|
| **A** | 自博弈节奏：z0∈25–65%, garr∈35–120, emits∈0.35–0.80, vs v20 spf≥3 | → B |
| **B** | vs v20 flip≥8% & spf≥15 & garr≥18，或 ≥1 胜 | → C 长训 |
| **C 长训** | WLD≥1/4 或 flip≥10%+spf≥18+z0∈20–45%；对标 v13c | 最终提交 |

### v2 搜参区间（对齐 v13c）

Phase A 关键改动 vs v1：
- `HOLD`: **0.0–0.008**（v13c=0，禁止高 HOLD 囤兵）
- `CAPTURE`: **0.02–0.06**（不再从 0 探）
- `PROD_DELTA`: **0.01–0.04**（v13c=0.02）
- `RELEASE_K`: **14–20**（v13c=15）

---

## 启动命令

```bash
# 1. v14e A/B 二分（anti-hoard 修复版）
bash scripts/v14e_binary_search.sh

# 2. B confirm 通过后 — Phase C 长训
bash scripts/v14e_phase_c_longtrain.sh

# 监控
tail -f logs/v14e_search.log
python scripts/v14d_sensitivity_report.py
```

---

## 文件

| 文件 | 说明 |
|------|------|
| `scripts/v14e_search_space.yaml` | v14e: anti-hoard + 高 entropy 搜参 |
| `scripts/v14e_binary_search.sh` | v14e 启动脚本 |
| `scripts/v14e_phase_c_longtrain.sh` | v14e C 长训（10k u） |
| `scripts/v14d_curriculum_search.py` | 二分主控（v14d/v14e 共用） |
| `orbit_wars_rl/configs/multi_action_v14e_phase_*.yaml` | v14e A/B/C 配置 |
| `orbit_wars_rl/env/rewards.py` | 新增 `anti_hoard_penalty_reward` |
| `logs/v14e_search.state.json` | v14e A/B 搜索状态 |
| `logs/v14e_c_longtrain.ready.json` | v14e B 通过后生成 |

---

## 执行进展

| 阶段 | 状态 | 备注 |
|------|------|------|
| v1 二分 (binary_v1) | 已弃 | CAPTURE 0/0.02/0.04 全 abort，HOLD 中点 0.012 |
| v2 A/B 二分 | **已弃** | 6 trials 全部囤兵坍塌 z0>85% emits<0.15 |
| **v14e A/B 二分** | **运行中** | 三修复: anti-hoard + 高 entropy + 早 abort |
| **C 长训** | 待 B | 10k u，目标 ≥ v13c |

### v14d → v14e 诊断与修复

**v14d 失败根因**：6 个 trial 全部在 ~update 16 出现囤兵坍塌——agent 发现
"不发射=不被抢=garr 高" 的局部最优，z0 从 0.7 暴涨到 0.93+，emits 从 0.45
骤降到 0.10 以下，此后不再恢复。early abort 阈值太松 (min_updates=80, z0≥0.92)
未能及时杀掉坏 trial。

**v14e 三项修复**：

1. **Anti-hoard 惩罚** (结构性修复)：新增 `SHAPING_ANTI_HOARD` env var，当
   garrison 占比 > 0.6 且不发射时施加负 reward，`-coef * excess`。搜参范围
   0.005–0.03。
2. **提高 emit entropy**：`ent_coef_emit` 从 0.002–0.006 提升到 0.010–0.025，
   防止 emit head 过早收敛到 "always hold"。同步提升 src/dst/pct entropy。
3. **收紧 early abort**：`min_updates` 从 80 降到 30；hoard 检测阈值从
   `z0≥0.92 & garr≥150 & emits≤0.08` 收紧到 `z0≥0.85 & emits≤0.12`；
   garr 上限从 200 降到 150。

---

## 与 v13c 的关系

- A/B：**从头训**，但 shaping 区间 **贴近 v13c_final**
- C 长训：在 B 已能占点的基础上 **抛光**；v13c 仍是 replay / WLD 参考上界
