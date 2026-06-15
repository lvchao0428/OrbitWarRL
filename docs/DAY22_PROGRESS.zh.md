# DAY22 — v29 验收 + v30 经济独立头

> **动机**：v29 在 pair_roi dim6 + dst dedup 上训满 4000u，dedup 生效但 **opening aim 仍错**（seed=0 首攻 id=26 而非 id=20）。
> 根因不是 ROI 公式排错，而是 **DstHead cross-attn 淹没经济信号** + **flip 硬挡把 id=20 踢出候选集**。
> v30 用 **经济学独立头 + 公式统一 + flip 修正 + ROI aux loss**（不做推理提权），从 v29 u3999 再训 4000u。

---

## 1. v29 训满验收（u3999）

| 项 | 结果 |
|----|------|
| 训练 | 4000u 完成，`ckpt_multi_action_v29_aim/ckpt_003999.pkl` |
| dedup | ✅ 同回合无重复 `(src,dst)` |
| flip（inline @ u3999） | 24.3%（u3199 15.9% → 有提升） |
| vs-v20 | 0/10 胜；quick_replay 5g flip 27.7% |
| seed=0 replay | ❌ 首攻 t=14 → **id=26**（prod=1），非 id=20；整局 10 发、负 |

### v29 为何「特征对了、行为仍错」

| 现象 | 诊断 |
|------|------|
| pair_roi 排名 | seed=0 决策点 **id=20 始终 #1**（roi≈0.35） |
| 实际 argmax | id=26（roi≈0.04），cross-attn 旧习惯 |
| dim6 权重 | u3999 时 pair_roi 列 norm≈0.78，dist 列≈5.78，网络几乎不读 ROI |
| flip 与训练 | v29 训练 `flip_hard_mask=false`；若开启则 id=20 被挡、只能在弱 prod=1 mite 里选 |

**结论**：v29 验证了 dedup 与 flip 趋势，**未解决 opening aim**；继续 zero-pad dim6 边际收益低。

---

## 2. v30 方案为何能生效（设计逻辑）

核心原则：**不用 `logit += α×roi` 等推理提权**（长训会被 value/policy 梯度冲掉），而是让「算准 ROI → 结构上用 ROI 决策 → loss 里持续监督 ROI」。

### 2.1 统一 ROI 公式（算准）

| 层级 | v30 做法 |
|------|----------|
| 公式 | 全局 `capture_roi_util`：`prod / (payback + eta)`，`ROI_NORM=0.35` |
| planet dim32 | `min_dist_to_owned` 作 eta（观测侧「最近己方距离」） |
| pair dim5 | **src→dst** 距离作 eta（与 dst 决策一致） |
| reward | 同一 payback 公式 + `ROI_NORM` |
| need | 统一 `garrison + pad`（neutral +2 / enemy +8） |

**为何有效**：特征、pair、reward 对「高 prod 近厂」给出同一排序；PPO 梯度与 aux teacher 不打架。

### 2.2 DstEconomicsHead（结构上用 ROI）

| 项 | 说明 |
|----|------|
| 输入 | 仅经济 7 维：pair×6 + `prod_norm`（无 embedding、无 cross-attn） |
| 输出 | `econ_logit` **加** 到原 DstHead logits |
| init | 从 v29 resume 时 **新头随机 init**（不 zero-pad 进大 MLP 第 6 列） |

**为何有效**：v29 把 ROI  buried 在 512 维 MLP 的一列里，梯度弱、易被 attn 覆盖；独立小头输入**就是** ROI/距离/flip/margin，网络无法「忽略经济」。

### 2.3 flip 策略修正（候选集含 id=20）

| v29 问题 | v30 修正 |
|----------|----------|
| `floor(rem×0.7) > garr` 硬挡 | 改为 **max pct(1.0)** 估计可发兵量 |
| 高 ROI 目标被挡 → 只能在 id=16/26 等弱星里选 | **roi ≥ 0.12 豁免 flip_block** |
| 训练未开 flip mask | config **`flip_hard_mask: true`**（与 submission 一致） |

**为何有效**：seed=0 校准确认 id=20 **flip_block=False、roi_teacher=20**；合法集不再被压成「只能打 prod=1 mite」。

### 2.4 ROI auxiliary CE（长训不消散）

| 项 | 值 |
|----|-----|
| Teacher | 每步 `argmax(pair_roi)` among 合法 capturable dst |
| Loss | `roi_aux_coef=0.08` × CE on **DstEconomicsHead** logits |
| 与 PPO | 写进 `total_loss`，非 inference hack |

**为何有效**：即使 RL 回报稀疏，每 emit 步都有「跟 ROI 排序对齐」的梯度；aux 与 capture_roi reward 同向，不会训着训着被 KL 单独拉回去打弱星。

### 2.5 保留 v29 已验证项

- same-turn **`used_dst` dedup**（mask + sample 后 mark）
- selfplay / KL anchor / reward 与 v28–v29 同档

---

## 3. 本地 Gate0（已通过）

```bash
python -m orbit_wars_rl.features.test_v30_econ   # id=20 ROI #1 + flip 豁免 + teacher=20
python -m orbit_wars_rl.scripts.smoke_v30        # v29 u3999 resume → 1 update OK
```

---

## 4. v30 训练（进行中）

| 项 | 值 |
|----|-----|
| Config | `orbit_wars_rl/configs/multi_action_v30_econ_dst.yaml` |
| Init | v29 `ckpt_003999.pkl` + 新 `dst_economics_head` |
| 预算 | **4000u**（~3.5h） |
| Log | `logs/v30_extend_20260614_205232/train.log` |
| Ckpt dir | `ckpt_multi_action_v30_econ/` |
| Remote | `bash scripts/v30_remote.sh {status,tail,eval}` |

### 中期行为（~u380，未最终）

- emits ≈2.5–2.7/turn，spf ≈25，e2+ ≈0.84，garr ≈50（比 v29 seed=0 replay 的 10 发/局 **更积极**）
- 待 u3999：`eval_vs_v20` + seed=0 HTML replay 验收

---

## 5. u3999 验收标准

| # | 标准 |
|---|------|
| 1 | seed=0 首攻 **id=20**（t≤16） |
| 2 | 无 dup `(src,dst)` |
| 3 | quick_replay flip ≥35% |
| 4 | `test_v30_econ` 通过 |

---

## 6. 关键文件

| 文件 | 说明 |
|------|------|
| `orbit_wars_rl/features/capture_roi_util.py` | 统一 ROI + max-pct flip |
| `orbit_wars_rl/features/pair.py` | pair_roi、flip 豁免、`econ_dst_features`、`roi_teacher_dst` |
| `orbit_wars_rl/net/heads.py` | `DstEconomicsHead` |
| `orbit_wars_rl/net/model.py` | `dst = attn + econ` |
| `orbit_wars_rl/ppo/update.py` | `roi_aux_coef` |
| `orbit_wars_rl/ppo/runner.py` | `_merge_ckpt_params`（v29→v30 新头） |
| `submission_rl_v21.py` | 推理 parity（检测 econ head 权重） |
| `scripts/v30_*.sh` | smoke / extend / pipeline / remote |

---

## 7. 与 v29「提权调参」路线的区别

| 路线 | 问题 |
|------|------|
| `logit += α×pair_roi` | 固定系数，PPO 可学反向权重抵消，长训消散 |
| 仅放大 pair dim6 | zero-pad 列梯度弱，4000u 不够 |
| **v30** | 独立头 + aux loss + flip 修正 → **可学习、可监督、候选集正确** |

---

*生成：2026-06-14；v30 pipeline 已 remote 启动。*
