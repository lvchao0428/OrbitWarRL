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
# 1. A/B 二分（~106 trials，fresh search_id=v14d_binary_v2）
bash scripts/v14d_binary_search.sh

# 2. B confirm 通过后 — Phase C 长训
bash scripts/v14d_phase_c_longtrain.sh
# 或 nohup bash scripts/v14d_phase_c_longtrain.sh >> logs/v14d_c_long.log 2>&1 &

# 监控
bash scripts/monitor_v14d.sh
python scripts/v14d_sensitivity_report.py
```

---

## 文件

| 文件 | 说明 |
|------|------|
| `scripts/v14d_search_space.yaml` | v2: 只搜 A/B + `phase_c_longtrain` 块 |
| `scripts/v14d_curriculum_search.py` | 二分主控 |
| `scripts/v14d_phase_c_longtrain.sh` | C 长训（10k u） |
| `orbit_wars_rl/configs/multi_action_v14d_phase_c_long.yaml` | C 长训 YAML |
| `logs/v14d_search.state.json` | A/B 状态（search_id 变则 fresh） |
| `logs/v14d_c_longtrain.ready.json` | B 通过后生成 |

---

## 执行进展

| 阶段 | 状态 | 备注 |
|------|------|------|
| v1 二分 (binary_v1) | 已弃 | CAPTURE 0/0.02/0.04 全 abort，HOLD 中点 0.012 |
| **v2 A/B 二分** | 待启动 | HOLD 0–0.008，CAPTURE≥0.02 |
| **C 长训** | 待 B | 10k u，目标 ≥ v13c |

---

## 与 v13c 的关系

- A/B：**从头训**，但 shaping 区间 **贴近 v13c_final**
- C 长训：在 B 已能占点的基础上 **抛光**；v13c 仍是 replay / WLD 参考上界
