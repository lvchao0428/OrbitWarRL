# DAY15 — v14d/e/f：A/B 备料 + Phase C 长训

> **策略**：Phase A/B 用二分搜参 + confirm **为 C 长训准备 ckpt**；Phase C **单独 10k updates 长跑**（不在二分里打断）。

---

## 流水线

```
Phase A 二分搜 (8维, 500u probe) → confirm (1200u) ──gate A──┐
                                                              ▼
Phase B 二分搜 (9维, 500u probe) → confirm (1500u) ──gate B──┘
                                                              │
                     logs/v14f_c_longtrain.ready.json
                                                              ▼
Phase C 长训: 10000u (+ extend×3) resume B ckpt
  force_emit_worth_it + 低 lr/entropy
  目标: Gate C + beats v13c_final
```

### Gate 继承 (v14f 放宽版)

| 阶段 | 作用 | 过 gate 才继承 |
|------|------|----------------|
| **A** | 自博弈节奏：z0∈25–**78**%, garr∈35–**200**, emits∈**0.25**–0.80, vs v20 spf≥3 | → B |
| **B** | vs v20 flip≥8% & spf≥15 & garr≥18，或 ≥1 胜 | → C 长训 |
| **C 长训** | WLD≥1/4 或 flip≥10%+spf≥18+z0∈20–45%；对标 v13c | 最终提交 |

### v14f 搜参区间（基于 v14e 最佳值收紧）

Phase A 改动 vs v14e：
- `ANTI_HOARD`: **0.02–0.05**（v14e winner=0.03，向上搜更强惩罚）
- `CAPTURE`: **0.04–0.08**（v14e winner=0.06）
- `ent_coef_emit`: **0.015–0.030**（v14e winner=0.025）
- `probe_updates`: **500**（v14e=300，更多时间让 agent 稳定）

---

## 启动命令

```bash
# 1. v14f A/B 二分（放宽 Gate A + 长 probe）
bash scripts/v14f_binary_search.sh

# 2. B confirm 通过后 — Phase C 长训
bash scripts/v14f_phase_c_longtrain.sh

# 监控（非阻塞）
ssh charlie@www.ultrapp.online "tail -30 /home/charlie/project/OrbitWarRL/logs/v14f_search.log"
ssh charlie@www.ultrapp.online "ps aux | grep v14f | grep -v grep"
```

---

## 文件

| 文件 | 说明 |
|------|------|
| `scripts/v14f_search_space.yaml` | v14f: 放宽 Gate A + 长 probe 搜参 |
| `scripts/v14f_binary_search.sh` | v14f 启动脚本 |
| `scripts/v14f_phase_c_longtrain.sh` | v14f C 长训（10k u） |
| `scripts/v14d_curriculum_search.py` | 二分主控（v14d/e/f 共用） |
| `scripts/v14d_gate_check.py` | Gate A/B/C 判定（已放宽 z0/garr） |
| `scripts/v14d_early_abort.py` | 早期终止（garr 上限 200） |
| `scripts/v14d_scoring.py` | 评分（z0 理想值 0.55, garr 理想值 90） |
| `orbit_wars_rl/configs/multi_action_v14e_phase_*.yaml` | v14e/f A/B/C 配置 |
| `orbit_wars_rl/env/rewards.py` | `anti_hoard_penalty_reward` |
| `logs/v14f_search.state.json` | v14f A/B 搜索状态 |
| `logs/v14f_c_longtrain.ready.json` | v14f B 通过后生成 |

---

## 执行进展

| 阶段 | 状态 | 备注 |
|------|------|------|
| v14d v1 二分 | 已弃 | CAPTURE 0/0.02/0.04 全 abort |
| v14d v2 二分 | **已弃** | 6 trials 全部囤兵坍塌 z0>85% emits<0.15 |
| v14e A/B 二分 | **已弃** | 24 trials 全部 abort/fail，Gate A z0 上界 0.65 太紧 |
| **v14f A/B 二分** | **运行中** | 放宽 Gate A + 长 probe (500u) + garr abort 200 |
| **C 长训** | 待 B | 10k u，目标 ≥ v13c |

### 监控命令

```bash
# 查看搜索主日志
ssh charlie@www.ultrapp.online "tail -30 /home/charlie/project/OrbitWarRL/logs/v14e_search.log"

# 查看搜索状态 JSON
ssh charlie@www.ultrapp.online "python3 -c \"import json; st=json.load(open('/home/charlie/project/OrbitWarRL/logs/v14e_search.state.json')); a=st['binary']['a']; print('Phase A:', 'done' if a['done'] else 'pass=%d idx=%d' % (a['pass_idx'],a['param_idx'])); print('Trials:', len(st['trials']), '| Best:', a['best']['score'] if a['best'] else 'none')\""

# 查看当前训练进程
ssh charlie@www.ultrapp.online "ps aux | grep 'v14e\|orbit_wars' | grep -v grep"

# 查看最新 trial 日志
ssh charlie@www.ultrapp.online "ls -t /home/charlie/project/OrbitWarRL/logs/search/v14e_binary_v1_a_*.log | head -1 | xargs tail -3"
```

---

### v14e Phase A 搜索结果 (2026-06-11 04:51 UTC)

**状态**: Pass 0 全部 8 参数扫完，24 trials，**0 过 Gate A**。搜索在 pass 1 卡死
（trial ID 去重导致无法调度新 probe）。

| 排名 | trial (简称) | score | z0 | garr | emits | spf | v20_spf | status |
|------|-------------|-------|----|------|-------|-----|---------|--------|
| 1 | lr_peak=4.5e-05 | 55.1 | 0.72 | 70.7 | 0.53 | 48.8 | 0.0 | aborted |
| 2 | PROD_SHARE_DELTA=0.01 | 54.2 | 0.65 | 59.9 | 0.67 | 46.5 | 0.0 | aborted |
| 3 | ONE_SHIP_PENALTY=0.01 | 48.7 | 0.71 | 73.8 | 0.52 | 62.2 | 0.0 | aborted |
| 4 | RELEASE_K=14 | 48.6 | 0.70 | 80.3 | 0.50 | 57.7 | 0.0 | aborted |
| 5 | ONE_SHIP_PENALTY=0.04 | 48.1 | 0.75 | 95.9 | 0.41 | 70.1 | 14.8 | probed |
| 6 | ANTI_HOARD=0.03 | 46.2 | 0.76 | 114.2 | 0.39 | 68.5 | 40.2 | probed |

**关键发现**:

- Gate A 要求 `z0 ∈ [0.25, 0.65]`，但最佳 trial z0=0.65~0.72，全部偏高
- Anti-hoard 惩罚有效地防止了 v14d 的极端坍塌 (z0>0.90)，但不足以把 z0 压到 0.65 以下
- 最佳 trial (score 55.1) 在 upd 100~200 时 z0 在 0.59-0.70 范围震荡，接近 Gate A 边界
- 没有 trial 产生有效的 vs-v20 战斗力 (v20_spf=0.0)

---

### v14e vs v13c 同期对比 (upd 100~300)

| 指标 | v14e best (upd 200) | v13c (upd 199) | v13c (upd 299) | v13c (upd 9999) |
|------|---------------------|----------------|----------------|-----------------|
| z0 | 0.66~0.72 | 0.85 | 0.91 | 0.75 |
| garr | 60~80 | 270.1 | 356.1 | 197.3 |
| emits | 0.53~0.67 | 0.21 | 0.12 | 0.58 |
| spf | 45~55 | 47.8 | 49.9 | 79.0 |
| entropy(e) | 0.48~0.52 | 0.33 | 0.24 | 0.42 |
| WRr | - | 0.69 | 0.16 | 0.91 |
| vs v20 | 未产生 | (arch mismatch) | (arch mismatch) | WRr=0.91 |

**对比分析**:

1. **v14e 的 anti-hoard 有效**: v13c 在 upd 200 时 z0=0.85, garr=270，典型囤兵；v14e z0=0.66, garr=60，节奏更健康
2. **v13c 早期也是囤兵**: v13c 在 upd 100~300 经历了严重囤兵 (z0=0.91, emits=0.12)，但在 upd 400 后自然恢复 (z0=0.74, emits=0.46)
3. **v14e 被过早 abort**: Phase A probe 只跑 300u，最佳 trial 被 `garr≥150` 早期 abort 杀掉。v13c 在 300u 时也不会过 Gate A
4. **v14e 节奏好但还没学会战斗**: emits/garr/z0 都接近理想范围，但训练时间不够无法产生 vs-v20 战斗力

---

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

### v14e 失败分析

**核心问题**: Gate A 门槛和 early abort 配合不当。

1. **Gate A z0 上界太紧 (0.65)**: v13c 在同期 (upd 200~300) z0=0.85~0.91，
   远超这个阈值。v14e 的 z0=0.65~0.72 其实已经比 v13c 同期好得多，但仍过不了 Gate
2. **Early abort garr≥150 太激进**: v14e best trial 在 upd 180 时 garr=102，
   upd 250 时短暂升到 garr=130，被 abort。v13c 在 upd 200 时 garr=270，
   如果用这个 abort 规则也会被杀掉
3. **Probe 只有 300u 太短**: 需要足够多的 updates 让 agent 度过早期震荡，
   稳定到健康节奏。300u 对于判断一个 trial 是否能过 Gate A 信息量不够

---

## 与 v13c 的关系

- A/B：**从头训**，但 shaping 区间 **贴近 v13c_final**
- C 长训：在 B 已能占点的基础上 **抛光**；v13c 仍是 replay / WLD 参考上界
- v13c 的 vs-v20 eval 因 template/ckpt architecture mismatch **全部失败**，
  只有 WRr (vs random) 作为参考: 最终 WRr=0.91
