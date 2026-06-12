# DAY19 — v19：v20 分布对齐 + v16a 强对手 + v20 状态缓冲

> **动机**：v18 自博弈 multi-emit 已解锁（e2≈0.4），但 vs-v20 仍 0 胜、z0 50–66%、train/eval 严重脱节。
> v19 不改 hist=50 / emit 链，改 **对手分布 + episode 起点**。

---

## 1. v19 相对 v18

| 项 | v18 | **v19** |
|----|-----|---------|
| Resume | v17 u2199 | **v18 u1799**（flip peak ckpt） |
| 对手 | symm 50% + v16a u3199 20% + pool 30% | **strong v16a u3399 25% + gated pool 30% + buf 20% + rand 25%** |
| Pool | snapshot_current=true（自快照污染） | **snapshot_current=false**，仅 v16a u3399/u3199 seed |
| v20 对齐 | 仅 eval gate | **mixed_v20_top10 buffer reset 50%** |
| anti_hoard | 0.03 | **0.05** |
| prod_share_delta | 0 | **0.005** |
| num_updates | 8000 | **6000** |

---

## 2. 文件

| 文件 | 说明 |
|------|------|
| `orbit_wars_rl/configs/multi_action_v19_v20_align.yaml` | v19 配置 |
| `scripts/v19_v20_align.sh` | 本地/远程启动 |
| `scripts/v19_remote.sh` | 远程 sync / stop / start / tail |

---

## 3. 远程启停

```bash
# 状态
bash scripts/v19_remote.sh status

# 同步配置并 v18→v19 切换（默认 resume v18 u1799）
bash scripts/v19_remote.sh switch

# 或分步
bash scripts/v19_remote.sh sync
bash scripts/v19_remote.sh stop-v18
bash scripts/v19_remote.sh start-v19 ./ckpt_multi_action_v18_multi_emit/ckpt_002599.pkl 6000

# 盯盘
bash scripts/v19_remote.sh tail-v19
bash scripts/v19_remote.sh eval-v19
```

---

## 4. 观察指标（同 v18 + 重点看 buf/strn/frzn）

| 信号 | 健康 | 告警 |
|------|------|------|
| vs-v20 flip | >25% 且稳定 | <15% 连续 5 次 eval |
| vs-v20 z0 | <45% | >55% |
| vs-v20 e2+ | 5–15% | 0% 或 >20% |
| vs-v20 WLD | ≥1/3 | 全程 0/3 @u2000 |
| opp=buf 行 spf/z0 | 接近 v20 行 | 仍 spf>60 z0<0.2 |

---

## 5. 执行记录

| 时间 | 事件 |
|------|------|
| 6/12 13:43 | v18 停跑（u4798 附近） |
| 6/12 13:44 | v19 开跑，resume `ckpt_001799.pkl`，buffer `mixed_v20_top10.npz` |
| 日志 | `logs/v19_v20_align_20260612_134437/train.log` |
| Ckpt | `ckpt_multi_action_v19_v20_align/` |
| 6/12 16:30 | v19 停跑（切 v21），结论：flip↔e2+ tradeoff 未解 |

---

# v21 — Lux-Aligned：从头训练 + 极简 reward + 全面特征增强

> **动机**：v18/v19 所有版本均无法同时满足「win + flip + e2+ + spf + z0」五项指标。
> 根因诊断：1）shaping 太杂导致梯度冲突；2）ETA-lead 信息未传递给 EmitHead/PctHead 导致发兵打不准；
> 3）缺少跨星球协同和未来局势预测特征。
>
> v21 采纳 Lux Top2 方法论：极简 reward、纯对称 self-play、从头训练、4h 预算。

## 6. v21 特征工程改动（6 项）

### P0: ETA-lead → 全部 heads

| head | 之前 | v21 |
|------|------|-----|
| DstHead | ✅ pair_feats 中 dist_norm 已经是 lead 距离 | 不变 |
| PctHead | ❌ 不知道 lead 距离 | ✅ `lead_dist_norm` 加入 pct_pair_features (dim 7) |
| EmitHead | ❌ 不知道最短 lead 距离 | ✅ `best_lead_dist_norm` 加入 emit_pair_globals (dim [6]) |

### P0: 全局多星球协同观测

`emit_pair_globals` 扩展为 11 维（原 7 维 + 4 维新增）：

| dim | 名称 | 含义 |
|-----|------|------|
| [7] | `active_emitter_ratio` | 本 turn 已出兵星球占己方总数比例 |
| [8] | `top2_src_remain_norm` | 前 2 强源星球剩余兵力均值（log1p/8） |
| [9] | `concentration_ratio` | 兵力集中度 max(rem)/total |
| [10] | `coord_coverage` | 出兵星球覆盖率 |

### P1: 未来局势预测

| planet dim | 名称 | 公式 |
|------------|------|------|
| [41] | `future_owner_flip_risk` | enemy_w2/(garr+friendly_w2) 或反向 |
| [42] | `future_garrison_growth` | clip(prod×15/garr, 0, 2)/2 |

### P1: Planet Spatial Hist

5 帧 × 4 维 = 20 维 per planet（dim [43..62]）:
- `is_mine`, `garr_norm`, `inbound_foe_norm`, `was_flipped`

### Reward 极简化

| 项 | v19 | v21 |
|----|-----|-----|
| terminal | ±1 | ±1 |
| capture | 0.02 | 0.02 |
| anti_hoard | 0.05 | 0.02（安全网） |
| defense_empty | 0.015 | **0** |
| prod_share_delta | 0.005 | **0** |
| multi_emit | 0.02 | **0** |
| 其他 | 0 | 0 |

### Self-play：纯对称

`symmetric_selfplay: true`, `selfplay.enabled: false`（不用 strong/frozen/buffer 混合对手）。

## 7. v21 文件

| 文件 | 说明 |
|------|------|
| `orbit_wars_rl/configs/multi_action_v21_lux_align.yaml` | v21 配置 |
| `scripts/v21_lux_align.sh` | 本地/远程启动 |
| `scripts/v21_remote.sh` | 远程 sync / stop-v19 / start / tail / eval |

## 8. v21 远程操作

```bash
# 一键切换：同步代码 → 停 v19 → 启 v21
bash scripts/v21_remote.sh switch

# 或分步
bash scripts/v21_remote.sh sync     # rsync 全量代码到远程
bash scripts/v21_remote.sh stop-v19 # 停掉 v19
bash scripts/v21_remote.sh start    # 从头启 v21（15000 updates ≈ 4h）
bash scripts/v21_remote.sh tail     # 看日志
bash scripts/v21_remote.sh eval     # 查 eval_vs_v20 结果
```

## 9. v21 观察重点

| 信号 | 健康 | 告警 |
|------|------|------|
| vs-v20 flip | 稳升 >20% @u3000 | <10% @u5000 |
| vs-v20 WLD | ≥1/5 @u5000 | 0/5 @u8000 |
| symm entropy | 平稳下降 | emit_entropy 骤降（early commit） |
| loss/pg | 正常波动 | pg_loss→0（collapse） |

## 10. v21 执行记录

| 时间 | 事件 |
|------|------|
| 6/12 17:11 | v21 开跑（从头训，15000 updates） |
| 日志 | `logs/v21_lux_align_20260612_171146/train.log` |
| Ckpt | `ckpt_multi_action_v21_lux_align/` |
| 备注 | v19 已在 u5999 自然结束；此前仅写好脚本未远程启动 |
| 6/12 ~21:00 | **v21 停跑 u4926** — eval export 维度 bug 导致 inline eval 全程失败；u3999 手动 eval flip=3.3% WLD=0/5 |
| 结论 | 100% symm + 常数 capture → e2=0 坍缩、vs-v20 全败；**不可 resume** |

---

# v22 — 结构性修复（mixed opp + flip-gated capture + 可观测 eval）

> v21 教训：继续训只会强化 symm 里的错误均衡；eval 失败导致 u100–u4900 盲训。

| 项 | v21（失败） | **v22** |
|----|-------------|---------|
| Self-play | 100% symm | **50% frozen v16a + 50% rand** |
| Capture | 常数 0.02，弱 flip 也给分 | **flip proxy 门控**（garr/prod on capture） |
| Emit | ent=0.002 → e2=0 坍缩 | **ent_emit=0.004, min_pct_bin=3** |
| Resume | 从头 | **从头（禁 resume v21）** |
| Inline eval | export 失败 | **weights v21 维 + SKIP_PARITY** |

```bash
bash scripts/v22_remote.sh switch   # sync + 启 v22
bash scripts/v22_remote.sh eval     # 每 100u 看 flip
```
