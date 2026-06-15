# Top10 高手 Replay 行为参考 + 模型上限对照

> **用途**：校正 RL/BC 策略的行为分布，避免只模仿 v20 启发式而卡在更低上限。  
> **数据源**：[`_archive/top10_episodes_2026-05-04/`](_archive/top10_episodes_2026-05-04/)（2631 局 4P JSON，Top 10% rating 对局）  
> **聚合结果**：[`logs/top10_aggregate_2026-05-04.json`](logs/top10_aggregate_2026-05-04.json)  
> **详细 SSOT**：[`_archive/docs_legacy/TOP10_REPLAY_METRICS.zh.md`](_archive/docs_legacy/TOP10_REPLAY_METRICS.zh.md)  
> **更新**：2026-06-14（结合 v26 u7199 胜局 replay 肉眼诊断）

---

## 1. 数据集速查

| 项 | 值 |
|----|-----|
| 路径 | `_archive/top10_episodes_2026-05-04/episodes/episodes/*.json` |
| manifest | `_archive/top10_episodes_2026-05-04/manifest.csv`（按 `sum_score` 降序） |
| 局数 | 2630 局成功解析（4 玩家） |
| 典型长度 | p50 **222** turn，p95 **500** turn |
| 选取 | 当日对局 rating 总和 Top 10% |

**复用命令**：

```bash
# 聚合指标（已跑过，结果见 logs/top10_aggregate_2026-05-04.json）
python -m orbit_wars_rl.scripts.aggregate_top10_replays \
  --replay-dir _archive/top10_episodes_2026-05-04/episodes/episodes \
  --out logs/top10_aggregate_2026-05-04.json

# 从 JSON 抽 BC 状态（赢家视角）
python -m orbit_wars_rl.bc.collect_states_from_json \
  --replay-dir _archive/top10_episodes_2026-05-04/episodes/episodes \
  --winners-only --out data/top10_winner_states.npz
```

---

## 2. 关键指标：Top10 赢家 vs 输家 vs v20 vs 我方

> first-80 = 前 80 turn 窗口（与训练 inline eval 对齐）；全场 = 整局。  
> **garr** = 该选手所有占有行星驻防舰**总和**（不是每星平均）。

### 2.1 First-80（早期节奏）

| 指标 | Top10 **赢家** | Top10 输家 | **v20**（5 局样本） | **v26 u7199** inline | 解读 |
|------|----------------|------------|---------------------|----------------------|------|
| spf（均舰队规模） | **28.7** | 23.2 | ~15.9 | 24.7 | v26 接近高手，v20 偏小 |
| garr（总驻防） | **152** | 98 | ~132 | 92.7 | v26 偏薄，early 易失守 |
| emit≥2 占比 | **14%** | 10% | ~**55%** | 11.2% | v20 多线远高于高手；v26 接近高手下限 |
| z0（本 turn 不发） | **62%** | 67% | ~24% | 46.8% | 高手大量囤兵 turn；v20/v26 更「手痒」 |
| planet_count | **5.3** | 3.8 | — | — | 扩张先行 |
| max_launch p50 | **181** | 77 | — | — | 决战气魄差数量级 |
| fleets_in_flight | **4.3** | — | — | — | 多线调度 |

### 2.2 Full game（终局格局）

| 指标 | Top10 **赢家** | Top10 输家 | 启示 |
|------|----------------|------------|------|
| prod_share | **0.48** (p50 0.47) | 0.12 | **最强胜负指标**；≥0.35 时 95% 赢家 |
| planet_count | **13.5** | 3.4 | 控图 = 控产 |
| emit≥2 | **27%** | 9% | 中后期 multi 更频繁 |
| z0 | **51%** | 75% | 赢家也会大量 hold（主动节奏） |
| garr（总驻防） | **1717** (p50 716) | 239 | 终局囤放规模 |
| max_launch p50 | **181** | 77 | 单次决战 100–1000+ 舰 |

### 2.3 与 v20 的本质差距

v20 是**可打的启发式**，但不是 Top10 行为上界：

| 维度 | Top10 赢家 | v20 | 我方常见问题 |
|------|------------|-----|--------------|
| 节奏 | 高 z0 囤 → 大 launch 决战 → 再囤 | 中 z0、较高 multi | BC 后 z0 偏低、乱动 |
| 扩张 | 持续吃边、prod_share 滚到 0.45+ | 有 outer hoard 但不控图 | **换兵不拓边**（u7199 replay） |
| 发兵精度 | 按 need 一次性 flip | 有 batch 启发式 | **过量派兵、己方互调** |
| 距离感 | 先近后远、边线推进 | 会 skip 中间 neutral | **远点占星** |
| 多线 | mean_fleets_in_flight ~11（全场） | 中等 | 有 multi 但 dst 选错 |

**结论**：上限应朝 Top10 分布校正；v20 仅作**对手/暖启动**，不应作行为克隆的唯一标准。

---

## 3. v26 u7199 胜局 replay 诊断（step ~280）

肉眼特征（与用户标注一致）：

1. **己方集群内频繁换兵**（红框：多颗蓝星之间舰队来回，不向外推进）
2. **远点占星**：ROI 有距离项但仍会打远处 neutral，忽视近处 orbit 未占星
3. **占星发兵不精准**：安全己方星仍在收兵；无威胁 turn 仍向已有驻防星球增援
4. **对比 v25 胜局**：同样存在「中间 4 颗 grey 未占」— 赢靠 outer hoard + 压强，非 map control

这些是 **dst/src 决策与节奏** 问题，不是 flip/e2+ 绝对值 alone 能解决的。

---

## 4. 现有特征里有什么、缺什么

### 已有（encode 63 维）

| 已有信号 | 维度/位置 | 为何不够 |
|----------|-----------|----------|
| `capture_roi_norm` | planet dim32 | 只解决「打哪颗 ROI 高」，不禁止 friendly shuffle |
| `threat_ratio`, `net_inbound`, `enemy_eta_*` | planet 19–21, 35–36 | 有威胁感知，**未约束「无威胁时勿向己方 dst 派兵」** |
| `friendly_surplus`, `safe_surplus_norm` | 23, 29 | 鼓励从 surplus 发兵，**未区分 dst=友军 vs 敌军** |
| `friendly_eta_w1/w2` | 33–34 | 看见友军在路上，**无「友军→友军无效调兵」惩罚特征** |
| `safe_emit_margin`, `hold_value` | 39–40 | 偏 hold，未覆盖 mid-game shuffle |
| `local_roi_targets_norm` | global 15 | 全局可打目标数，无 frontier/border 概念 |

### 建议新增信号（v27 候选，保持 63 维需替换或 hist 外挂）

| 优先级 | 信号名 | 含义 | 作用 |
|--------|--------|------|------|
| **P0** | `dst_is_friendly_shuffle` | 若 dst 是己方且该星 `threat_ratio≈0` 且 `friendly_inbound>0` → 高惩罚特征 | 直接压制换兵 |
| **P0** | `frontier_score_norm` | 该星邻接 neutral/enemy 边界数 / 度数；内部安全星低、边线星高 | 推 expansion 不推内调 |
| **P0** | `capture_need_exact_norm` | `need_garr - already_inbound_friendly` 归一化（占星刚好够） | 精准占星发兵 |
| **P1** | `dist_to_frontier_norm` | 到最近「可扩张边界」的 ETA，替代纯几何近距 | 抑制远点占星 |
| **P1** | `home_under_threat` | 己方 home / 高 prod 星 `enemy_eta_w1>0` 全局 flag | 防换家；有威胁才留兵 |
| **P1** | `interior_planet_bin` | 完全被己方包围、无邻 neutral 的星 = 1 | dst 不应再收兵 |
| **P2** | `friendly_loop_penalty`（reward） | 一步内 launch 从 A→B 且 B 无威胁 | shaping 补刀 |

### 建议 reward 调整（与特征配套）

| 项 | 做法 |
|----|------|
| 换兵 | `ORBITWARS_SHAPING_FRIENDLY_SHUFFLE=-0.01` × (ships sent to safe owned dst) |
| 拓边 | 现有 `capture_roi` 乘 `frontier_score`；占 **border neutral** 加成 |
| 精准占 | flip 奖励按 `\|ships_sent - need\|/need` 衰减，过量/不足都扣 |
| 远占 | ROI 乘 `exp(-dist/frontier_dist)` 或 hard mask dist > θ 除非 home 安全 |

---

## 5. Top10 教我们的「上限形状」（非 v20 形状）

```
高手典型节奏（2630 局统计）：

  early (0–80):   z0↑↑ 囤 garr~152，planet~5，emit2~14%
       ↓
  mid:            边线推进 + 多线 in_flight↑，仍保持 z0~50%
       ↓
  late:           max_launch 100–1000+，prod_share→0.45+，planet~13+
```

**我方 v26 常见偏差**：

```
  early:          还能看（spf/garr 尚可）
  mid–late:       己方互调 ↑↑，frontier 不动，远点零星占星
  win condition:    靠体量/压强碰运气，非 map/prod 滚雪球
```

**训练分布建议**：

1. BC 混合 **top10 赢家状态**（`collect_states_from_json --winners-only`），不只 v20 自对弈
2. 对手池：v20 + strong BC + **top10 风格 bot**（若可蒸馏）
3. 验收除 WLD 外加：**frontier_capture_rate**、**friendly_shuffle_rate**、**orbit_neutral@200**

---

## 6. 指标验收模板（对齐 Top10）

| 指标 | Top10 赢家目标带 | v26 u7199 | v25 best |
|------|------------------|-----------|----------|
| first-80 garr | 120–200 | 92.7 | ~80 |
| first-80 emit2+ | 10–25% | 11.2% | ~14% |
| first-80 z0 | 45–65% | 46.8% | ~44% |
| full prod_share | ≥0.35 才像赢 | 待测 | 待测 |
| full planet | ≥10 | 待测 | 待测 |
| friendly_shuffle / turn | **低**（待量化） | 高（replay 目测） | 类似 |
| orbit_neutral@200 | ≤1（胜局） | 多颗未占 | 4 颗未占 |

---

## 7. 相关文件索引

| 文件 | 说明 |
|------|------|
| `_archive/top10_episodes_2026-05-04/` | 原始 JSON + manifest |
| `logs/top10_aggregate_2026-05-04.json` | 2630 局数值聚合 |
| `_archive/docs_legacy/TOP10_REPLAY_METRICS.zh.md` | Day5 完整分析报告 |
| `orbit_wars_rl/scripts/aggregate_top10_replays.py` | 聚合脚本 |
| `orbit_wars_rl/bc/collect_states_from_json.py` | top10 → BC 状态 buffer |
| `logs/replay_html/v26_u7199_win_s3/v26_u7199_win.html` | v26 胜局 replay（换兵明显） |
| `docs/H2H_EVAL_RUNBOOK.md` | 正式 h2h 评测流程 |

---

## 8. 下一步（v27 方向摘要）

1. **特征**：frontier + anti-shuffle + capture_need_exact（替换或叠 hist）
2. **Reward**：friendly_shuffle 惩罚 + frontier-weighted capture_roi
3. **数据**：BC 800g 改为 **v20 400g + top10 赢家 400g** 混合
4. **验收**：replay 自动统计 `friendly_dst_rate`、`border_capture_rate`

> 本文档为根目录速查；数值以 `logs/top10_aggregate_2026-05-04.json` 为准，更新聚合后请同步修订 §2 表格。
