# DAY18 — v18：Multi-Emit 解锁 + Mixed Opponent + 常数 Capture

> **核心突破**：针对 v15-v17b 全程 e2+≈0% 的结构性瓶颈，一次性落地 **EmitHead 放松 + 高 emit entropy + multi-emit bonus + flip soft + mixed opponent**。
> 目标是突破「单路大舰队」局部最优，学会 Top10 式多线进攻。

---

## 1. 动机：v17b 告诉我们什么

### v17b 终态（u3199, capture=0.025 常数, fine-tune from v17 u2199）

| 维度 | 自博弈 | vs v20 | 结论 |
|------|--------|--------|------|
| flip | — | 48%→23% (下降) | capture 常数有效但趋势回落 |
| spf | 90–130 | 23–29 | 实战仍保守 |
| e2+ | ~0.04 | **0.0%** | **结构性瓶颈**：multi-emit 完全没学到 |
| WLD | — | **0/全程** | 无胜局 |

### 根因分析

1. **EmitHead 链被硬切**：`emit_hard_stop_min_step=0` + `emit_hard_stop=true` → t≥0 无 worth-it pair 就 `-inf` continue logit → 第二路几乎无法出
2. **低 emit entropy**：ent_emit=1e-4 → continue 采样极保守
3. **纯对称自博弈**：单路大舰队就能赢 random/自己，无梯度逼 multi-emit
4. **flip_hard_mask=true**：只能打「有把握 flip 的目标」→ 鼓励单路碾压而非多路施压
5. **Capture 退火 (v17)**：anneal 到 0 后 flip/占点被洗掉

### Lux Frog Parade 对照

| | Lux | OrbitWar v17 | v18 |
|---|-----|-------------|-----|
| Dense→Sparse | 手动换 reward 定义（match→series） | 同 run 内线性退火 capture coef | **capture 常数 0.02** |
| Multi-action | 16 unit 并行独立 | K=8 串行 stop/continue | **t=1 自由 + ent 10x** |
| Mixed opponent | 同参数自博弈 | 同 | **+ v16a strong + frozen pool** |

---

## 2. v18 方案总览

```
┌──────────────────────────────────────────────────────────┐
│                    v18 改动栈                              │
├──────────────────────────────────────────────────────────┤
│  P0 EmitHead 放松                                         │
│    emit_hard_stop_min_step: 0 → 2 (t=0,1 自由出舰队)       │
│    ent_coef_emit: 1e-4 → 1e-3 (10x, 促 multi-route)       │
│    MULTI_EMIT=0.02 bonus (gated: ≥1 fleet with ≥8 ships)   │
│    flip_hard_mask=false (soft dst exploration)              │
│                                                            │
│  P1 Mixed Opponent                                         │
│    selfplay.enabled=true                                   │
│    strong_ckpt = v16a u3199 (首胜附近, shape-adapted)       │
│    strong_ratio=0.2  frozen_pool=0.3  self=0.5             │
│                                                            │
│  保留                                                      │
│    hist=50 / ETA-lead / safe_emit / BO3 / zero_sum_value   │
│    capture 0.02 常数 / anti_hoard 0.03 / defense_empty 0.01│
└──────────────────────────────────────────────────────────┘
```

### Resume

**v17 u2199**（flip peak ckpt）→ shape 完全匹配（同 GLOBAL=427, PLANET=41），无需 adapt。

### vs v17b 对比

| 参数 | v17b | **v18** |
|------|------|---------|
| emit_hard_stop_min_step | 0 | **2** |
| ent_coef_emit | 5e-4 | **1e-3** |
| MULTI_EMIT | 0 | **0.02** |
| flip_hard_mask | true | **false** |
| selfplay | symmetric only | **symmetric 50% + strong 20% + frozen 30%** |
| strong_ckpt | — | **v16a u3199** |
| capture | 0.025 常数 | **0.02 常数** |
| clip_eps | 0.15 | 0.15 |
| num_updates | 5000 | **8000** |

---

## 3. 文件清单

| 文件 | 变更 |
|------|------|
| `orbit_wars_rl/configs/multi_action_v18_multi_emit.yaml` | **新增** |
| `scripts/v18_multi_emit.sh` | **新增** |

---

## 4. 执行进展

### 4.1 时间线

| 时间 | 事件 |
|------|------|
| 6/12 11:46 | v18 开跑，resume v17 u2199 + strong v16a u3199 (shape-adapted) |

### 4.2 早期指标（u75）—— **e2+ 突破**

| 指标 | v17b u75 | **v18 u75** | 解读 |
|------|---------|-------------|------|
| emits/turn | 0.10 | **3.51** | **35x**——多路出舰队已被激活 |
| e2 rate | 0.01 | **0.49** | **49% 回合发 ≥2 路**——历史首次 |
| spf | 93 | **31** | 多路 → 舰队更小（预期内） |
| z0 | 0.92 | **0.24** | 极低零 emit 率——agent 积极出舰队 |
| nF (舰队数) | 0.8 | **42** | 场上舰队爆发性增加 |

**关键**：e2 rate **从 0.01 跳到 0.49**，这是 v11 以来整个项目首次在持续训练中看到稳定的高 multi-emit。

### 4.3 需要观察

1. **自博弈 → selfplay 后 e2 是否保持**（当前还在 warmup vs rand）
2. **spf 是否太低（multi-spam？）**——如果全是 1–3 ship 小舰队，flip 和 WLD 不会好
3. **vs v20 首次 eval gate**（u99）
4. **strong opp u3199 是否有效推高策略天花板**

---

## 5. 监控命令

```bash
# v18 训练日志
ssh charlie@www.ultrapp.online \
  "tail -5 /home/charlie/project/OrbitWarRL/logs/v18_multi_emit_*/train.log"

# vs-v20 inline gate
ssh charlie@www.ultrapp.online \
  "grep eval_vs_v20 /home/charlie/project/OrbitWarRL/logs/v18_multi_emit_*/train.log | grep -v WARN | tail -10"

# e2 和 emits 趋势
ssh charlie@www.ultrapp.online \
  "grep -oP 'emits [\d.]+.*e2 [\d.]+' /home/charlie/project/OrbitWarRL/logs/v18_multi_emit_*/train.log | tail -10"

# ckpts
ssh charlie@www.ultrapp.online \
  "ls -lt /home/charlie/project/OrbitWarRL/ckpt_multi_action_v18_multi_emit/*.pkl 2>/dev/null | head -5"
```

---

## 6. 风险与应对

| 风险 | 信号 | 应对 |
|------|------|------|
| Multi-spam（大量 1-ship） | spf < 5, bin0 > 30% | 提高 min_pct_bin 或加 ONE_SHIP_PENALTY |
| e2 collapse（强对手后回落） | e2 < 0.05 after warmup | 提高 ent_emit 或 MULTI_EMIT bonus |
| 囤兵回潮 | z0 > 0.85, garr > 300 | 提高 anti_hoard 到 0.05 |
| vs v20 仍 0 胜 by u2000 | flip < 20%, WLD 0/全 | 考虑 from v16a u3299 resume |

---

## 7. 下一步

| 优先级 | 方案 | 状态 |
|--------|------|------|
| **P0 v18** | multi-emit + mixed opp + constant capture | **🔄 训练中** |
| P1 | planet spatial hist 4ch | v18b 备选 |
| P1 | test-time symmetry aug | 推理侧，不影响训练 |
| P2 | parallel emit 架构改造（Lux 式） | v19 方向 |
