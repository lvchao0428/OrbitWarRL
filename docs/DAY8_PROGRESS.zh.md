# DAY8 进展 — f32 结果分析 + 顶层选手对标

> **2026-05-29 更新**
> **2026-06-01 补记**：`f35` 训练 + replay 已完成，结论见 §13；Day9 转入「动作机制 + 训练分布」路线规划。
> 接续 [`DAY7_PROGRESS.zh.md`](DAY7_PROGRESS.zh.md)。
> **项目综述（架构图 + 全流程）**：[`OVERVIEW.zh.md`](OVERVIEW.zh.md)

---

## 0. TL;DR — 当前状态

| 维度 | 状态 |
|---|---|
| **f32** | 🔴 **@599 不 promote**；WLD 0/5；bin7≈64% + z0=0.5% + spf≈6 |
| **f33** | 🔴 **replay 完成 — 全 ckpt WLD 0/5，不 promote**；训练/replay **非单调** |
| **f33 replay 最优** | **@1999**：spf=23.9 garr=78 flip=5.53%（flip 最接近 gate）；仍 **0 胜** |
| **f33 replay 最差** | **@599**：spf=**4.27**（训练 spf=404 → replay 崩塌，验证膨胀不可信） |
| **f33 新病理** | **bin7≈77–81%**（高于 f29 66%）；e2+≈0% @399（非 f32 式 spam，而是 **bin7 拉满 + 几乎单路 emit**） |
| **f32 clip_frac** | ✅ max **0.39**；f33 max **0.10**（超参轨有效，但挡不住 self-play 膨胀） |
| **f35** | 🔴 **replay 完成 — @199/@299 均 WLD 0/5，不 promote**；garr 上来了，但 spf/flip/z0/e2+ 全 fail |
| **f35 replay 结论** | **单路小舰队病仍在**：bin7≈60–63%，e2+=0%，z0<1%，one_ship_rate≈35% |
| **当前主线** | **Day9 路线三规划**：不再加特征，转向「动作机制 + 训练分布」 |
| **提交候选** | **f29 @599**；f32/f33/f35 均不提交 |
| **决策标准** | replay vs v20 only |

---

## 1. 每日顶层选手回顾（Lin Myat Ko, 5th place）

### 1.1 超参对比

| 参数 | 顶层选手 | 我们 f32 | 差距 | 风险 |
|---|---|---|---|---|
| **ent_coef** | 0.05（单一） | 0.003/0.001/0.008/0.002（分 head） | **10-25x 偏低** | 策略探索不足 |
| **clip_eps** | 0.20 | 0.15 | 25% 偏低 | 更新更保守 |
| **lr** | 3e-5 | 1.5e-4 | **5x 偏高** | 训练不稳定风险 |
| **ppo_epochs** | 1 | 4 | 4x | 过拟合 rollout |
| **num_envs** | 512 | 128 | 4x 偏少 | 多样性不足 |
| **rollout_steps** | 32 | 256 | 8x 更长 | 优势估计过时 |
| **vf_coef** | 0.5 | 0.2 | 60% 偏低 | 价值学习弱 |
| **模型大小** | ~600K 参数 | ~2-3M（d_model=128） | 4-5x 更大 | 更难训练 |
| **训练量** | 600M steps | ~19M steps（600upd） | **30x 不足** | 严重欠训练 |

### 1.2 顶层选手关键建议（我们可能违反的）

1. **"Add one architecture delta at a time"**
   - f32 同时改了 4 件事（速度特征、阈值放宽、shaping）
   - 无法隔离哪个变更有效
   - **教训**：未来每次只改一个变量

2. **"The limitations of a working baseline might be doing free regularization work nobody notices"**
   - 我们的小 ent_coef 可能恰好让 PPO 稳定
   - 提高 ent_coef（如 f32 计划）可能破坏训练
   - **教训**：谨慎调整 entropy

3. **"clip_frac trajectory is your most reliable warning sign"**
   - clip_frac 从 0.10 爬到 0.30+ = 优化器失控
   - 我们**从未监控过 clip_frac**
   - **教训**：每个 checkpoint 都要检查 clip_frac

4. **"RL training is not monotonic"**
   - 150M steps 模型可能比 300M steps 模型更好
   - 更多训练 ≠ 更好
   - **教训**：定期 replay 评估，不要盲目续训

5. **"ent_coef: 0.05"**
   - 单一 entropy 系数 = 0.05
   - 我们的分 head ent_coef 总和 ≈ 0.014
   - **差距 3.5x** → 策略探索空间远小于顶层选手

### 1.3 根本差距：训练量

| 维度 | 顶层选手 | 我们 | 差距 |
|---|---|---|---|
| 总 steps | 600M | ~19M | **30x** |
| SPS | ~10K | ~5K | 2x |
| 训练时间 | ~3 天 | ~1 小时 | 72x |
| 环境多样性 | 512 envs | 128 envs | 4x |

**结论**：即使架构和特征完美，我们也可能只是训练量不足。

---

## 2. 现状总结

### 2.1 已验证有效（f29-f31）

- ✅ hard masks（emit_hard_stop + flip_hard_mask）防止 spam
- ✅ pair 特征（margin, worth_it）提供有用的归纳偏置
- ✅ bin0 可控（~9-10%）
- ✅ pct hard mask（min_bin）防止过少发兵

### 2.2 未解决

- ❌ 模型过度保守（z0=2%, emit=1.08, flip=1%）
- ❌ WLD = 0/5 vs v20（600 upd 后）
- ❌ 缺少速度/ETA 信号
- ❌ hard mask 阈值（70%）过于激进

### 2.3 f32 结论（@299 + @599 replay，已关闭）

- ❌ **未修复保守**：目标 z0>8%，实际 @599 **z0=0.5%**（比 f31 @599 的 ~2% 更差）
- ❌ **未 beat v20**：WLD **0/5**（@299/@599 相同）
- ❌ **spf/garr/flip** 仍 fail（spf≈6，garr≈32，flip≈2.4% < 3% promote 线）
- ⚠️ **e8 回弹**：@599 **e8=8.0%**（与 f31 @599 同级），e2+=34%（多路 emit spam）
- ✅ **bin0** 仍可控（~5–6% < 15%）
- 📌 **新病理**：**bin7 占 64–72%** — 小守军星球上选 100% pct，舰队仍极小（spf≈6），flip 极低

### 2.4 仍待确认

- ✅ f32 `logs/v11_f32.log` **clip_frac**：max **0.39**（>0.20 告警，f32 lr/ent 变更导致优化器失控）
- ✅ f33 replay 已跑：`logs/replay_analyze/v11_f33_u{399,599,1199,1999}_vs_v20.*`
- ✅ **@399 vs @1999**：replay **非单调** — @1999 综合优于 @399；@599 最差（见 §11）

---

## 8. f33 训练完成（2000 upd）

### 8.1 实现

| 项 | 值 |
|---|---|
| 轨道 | **f33c**（DAY8 §7.3 #3）：f31 arch + 顶层选手 PPO |
| 超参 | ent=0.05×4, lr=3e-5, ppo_epochs=1, clip=0.20, vf=0.5 |
| 步数 | 2000 upd ≈ **65M steps**（f31 的 3.4×） |
| 脚本 | `scripts/run_v11_f33.sh`, `orbit_wars_rl/configs/multi_action_v11_f33.yaml` |
| ckpt | `ckpt_multi_action_v11_f33/ckpt_*.pkl` |
| replay | `logs/replay_analyze/v11_f33_u*_vs_v20.json` |
| HTML | `logs/replay_html/v11_f33_u399_seed0/`, `..._u1999_seed0/` |
| final eval | WR vs random **0.844**；WR vs frozen **0.94**（不可作 promote 依据） |

### 8.2 训练健康 vs f32

| 指标 | f32 @599 | f33 @1999 |
|---|---|---|
| clip_frac max | **0.39** ❌ | **0.10** ✅ |
| clip_frac last | ~0.31 | 0.04 |
| KL last | ~0.003 | 0.004 |
| SPS | ~5K | ~16.6K |

顶层选手 PPO 配方 **稳定了优化器**；f32 的高 clip_frac 与 replay 失败相关。

### 8.3 Self-play 膨胀时间线（训练 log）

| upd | emits | spf | z0 | garr | 备注 |
|---|---|---|---|---|---|
| 199 | 1.18 | 38.7 | 3% | 54 | 正常 |
| **399** | 1.16 | **55.1** | **3%** | **65** | **collapse 前 sweet spot** |
| **449** | 1.44 | **231.6** | 3% | 173 | ⚠️ spf 4× 跳变（frozen self-play） |
| 599 | 1.55 | 404.5 | 2% | 316 | 已膨胀 |
| 1199 | 1.71 | 475.1 | 1% | 346 | 平台期 |
| 1999 | 3.27 | 435.7 | **0%** | 294 | 晚期 hyper-emit |

**解读**：与 f32 相同模式 — 训练 spf/garr 虚高，z0 归零；**@449 后指标不可信**。Promote 必须 replay；**优先评 @399**，勿默认 @1999。

### 8.4 决策（replay 后）

| 动作 | 说明 |
|---|---|
| **不 promote f33** | 全部 ckpt WLD **0/5** |
| **保留 f29 @599 提交** | 仍是最稳 replay 基线 |
| **启动 f34** | f29 strong opponent，治 self-play 膨胀 |
| 备选 f33b | 若 f34 仍 WLD=0：去 flip_hard_mask + ETA |

---

## 11. f33 replay 实测（vs v20，first-80，5 seeds）

```bash
# 已跑
bash scripts/run_f33_eval.sh
```

| 指标 | f29 @599 | f33 @399 | f33 @599 | f33 @1199 | f33 @1999 |
|---|---|---|---|---|---|
| bin0 | 9.7% | 9.1% | 11.8% | 6.0% | 6.9% |
| **bin7** | 66.0% | **77.0%** | **78.8%** | 78.4% | **81.0%** |
| spf | 13.15 | **18.02** | **4.27** | 14.84 | **23.91** |
| garr | 44.60 | **86.48** | 45.30 | 56.76 | **78.37** |
| flip | 3.03% | 3.71% | 2.22% | 3.38% | **5.53%** |
| z0 | 1.0% | 0.7% | 1.1% | 1.1% | 0.7% |
| e8 | 0.2% | 0.0% | 0.0% | 0.2% | 0.0% |
| e2+ | 30.2% | **0.0%** | **0.0%** | 1.2% | 10.5% |
| WLD | 0/5 | **0/5** | **0/5** | **0/5** | **0/5** |
| train spf† | — | 55 | **404** | 475 | 436 |

†同 upd 训练 log spf，对比用。

### 11.1 关键发现

1. **训练 log 不能选 ckpt**：@599 训练 spf=404，replay spf=**4.27**（全场最差）— 膨胀期 ckpt 必须丢弃。
2. **replay 非单调**（顶层选手警告应验）：@1999 **优于** @399（spf/garr/flip）；不能假设「越早越好」。
3. **相对 f29 的 trade-off**：
   - ✅ spf/garr 提升（@1999 spf=24 garr=78 vs f29 13/45）
   - ❌ bin7 更高（81% vs 66%）— hard mask 止 spam 但没学会 v20 式 pct 分布
   - ❌ e2+ 极低 @399/@599（0%）— 几乎从不多路 emit，战术多样性不足
   - ❌ **WLD 仍 0/5** — 指标改善不等于能赢 v20
4. **共同 fail**：z0≈0.7–1.1%（全程 hyper-emit）；flip 最高 5.53% < 6% promote 线；min_game_spf≈1.13（仍有 1-ship 局）。

### 11.2 场景判定

| 场景 | 条件 | f33 实际 | 结论 |
|---|---|---|---|
| Promote | WLD≥1/5 | 全 0/5 | ❌ |
| 优于 f29 部分 gate | spf/garr/flip | @1999 三项更好 | ⚠️ 无胜场，不提交 |
| @399 sweet spot | 训练 log 指向 @399 | @1999 replay 更好 | ❌ 训练 log 选 ckpt 规则需修正 |
| f34 强对手 | WLD=0 但 spf/garr↑ | **匹配** | ✅ **启动 f34** |

---

## 12. 根因深挖 — 「永远 1 艘 1 艘发」（f33 replay 解剖）

### 12.1 发兵规模分布（first-80, vs v20）

| | f33 @399 | f29 @599 | **v20（对手）** |
|---|---|---|---|
| 平均每队舰船 | 18 | 13 | **19.2** |
| 发兵中位数 | — | — | **12** |
| **==1 艘** | **31%** | 25% | **0%** |
| **源星驻军 ≤3 艘** | **55%** | 34% | — |
| 源星驻军中位数 | **3** | 10 | — |
| bin7(pct=100%) | 77% | 66% | — |
| 每回合 emit 次数 | n1=**99%** | n1=68% | — |

### 12.2 三连锁根因

1. **选错源星（核心）**：55% 发兵来自驻军≤3 的星球。`SrcHead` 只吃 `log1p(remaining)/8`，把 50 艘 vs 5 艘压成 0.49 vs 0.22，**分不清兵库与空壳**。
2. **pct=100% 是虚胖**：bin7 占 77%，但 `100% × 3 艘 = 1~2 艘`（floor）。pct head 再激进也救不了小源星。
3. **emit 发一发就停**：`emit_worth_it` 闸门 → n1=99%。

→ **结论**：模型学到的是「从随便一个小星球把剩余 1~3 艘全发掉然后停」，而非 v20 的「攒大军一波推」。

---

## 13. f35 — 三管齐下大改（已完成训练 + replay，已关闭）

> 用户决策：**三个都改，0→1 也值得，先求能看**。不走小步快跑。

### 13.1 三个 delta

**① src-quality 特征（治「选错源星」）** — `encode.py` planet 特征 28→33：

| dim | 特征 | 对应 v20 启发式 |
|---|---|---|
| [28] | `garrison_rank` = 我驻军 / 我最大驻军 | 「我是不是兵库」（不被 log 压缩） |
| [29] | `safe_surplus_norm` = (驻军−敌软到达)/我总兵 | `calculate_safe_surplus` |
| [30] | `is_strong_source` = 我方且驻军>我均驻军 | 蓄力枢纽 |

**② v20 target_score 蒸馏（治「打错星球」）**：

| dim | 特征 | 对应 v20 |
|---|---|---|
| [31] | `prod_per_need` = prod²/need/8 | `neutral_mfg`（高产优先） |
| [32] | `v20_target_score` = (prod·20+weak·30)·dist_decay − need·0.5 | `target_score`（近处高产主导） |

**③ reward shaping（治「1-ship trickle」）** — `rewards.py` 新增：

- `ONE_SHIP_PENALTY` = −0.01 / 每个 ≤3 艘的发兵（flat，不 log，直击 trickle）
- `HIGH_PROD_CAPTURE` = +0.05 × 占领 prod≥3 工厂的 prod 占比

### 13.2 实现校验

| 检查 | 结果 |
|---|---|
| JAX encode 33 维 | ✅ 无 NaN/Inf |
| **submission encode_obs 对拍 JAX** | ✅ **新维 diff=0**，整体 max diff=6e-8 |
| test_parity（训练 numpy_forward） | ✅ **16/16** whole-turn match |
| reward smoke（2 tiny launch×0.01） | ✅ −0.02 |
| **2-upd 训练 smoke**（CPU） | ✅ loss/value/entropy 正常，无 shape 错 |
| export 模板路由（planet=33→f35） | ✅ 已加 quick_replay + export_submission |

### 13.3 PPO / 训练设置（= f33 配方 + 防膨胀）

| 项 | 值 |
|---|---|
| arch | f33（5+4 pair, bin5 hard masks） |
| PPO | ent=0.05×4, lr=3e-5, ppo_epochs=1, clip=0.20, vf=0.5 |
| frozen_ratio | **0.25**（f33 是 0.40，防 self-play 膨胀） |
| num_updates | **800**（在 f33 式 @449 collapse 前停） |
| strong opponent | ❌ 不用（f29 是 28 维，与 f35 33 维 shape 不兼容） |

### 13.4 训练完成（800 upd）

| upd | emits | spf | z0 | garr | clip | 判定 |
|---|---|---|---|---|---|---|
| 199 | 1.15 | 38.4 | 1% | 45.3 | 0.02 | 仍可评 replay |
| 299 | 1.18 | 45.4 | 2% | 51.1 | 0.02 | 仍可评 replay |
| 349 | 1.23 | 58.5 | 2% | 61.2 | 0.04 | 边缘，接近 collapse |
| 399 | 1.46 | 213.5 | 2% | 181.1 | 0.02 | **collapse 开始** |
| 799 | 1.68 | 461.6 | 1% | 364.9 | 0.03 | 晚期膨胀，不可信 |

`f35` 的 PPO/优化器侧保持稳定（`clip_frac` 未失控），但 **self-play 膨胀仍在 `@350-400` 左右出现**；因此决策只看早期 checkpoint replay，不看晚期训练指标。

### 13.5 replay 实测（vs v20，first-80，5 seeds）

> 先评最有希望的早期 ckpt：`@199`、`@299`。两者均已足够完成 go/no-go 判断。

| 指标 | f35 @199 | f35 @299 | f35 期望 | 结论 |
|---|---|---|---|---|
| bin0 | 7.6% | 13.9% | <15% | ✅ |
| spf | 5.94 | 6.81 | >10 | ❌ |
| garr | 68.36 | 79.56 | >60 | ✅ |
| flip | 4.30% | 3.17% | >6% | ❌ |
| z0 | 0.6% | 0.7% | >8% | ❌ |
| e8 | 0.0% | 0.0% | <5% | ✅ |
| e2+ | 0.0% | 0.0% | >5% | ❌ |
| one_ship_rate | 35.4% | 35.9% | <15% | ❌ |
| min_game_spf | 1.23 | 1.20 | >5 | ❌ |
| WLD | 0/5 | 0/5 | >=1/5 | ❌ |

**关键观察**：

1. **f35 修到的是“表象”，不是“机制”**：`garr` 上来了，`bin0`/`e8` 也健康，但 `spf` 仍只有 6-7，远低于 v20 的 16+。
2. **单路 emit 病完全没治好**：两次 replay 都是 `n1=98.8%`、`e2+=0%`，说明策略仍然几乎每回合只发一次。
3. **1-ship / 小源星 trickle 仍在**：`one_ship_rate≈35%`，而且 `bin7≈60%+` 仍是在把小守军星球“拉满 pct”。
4. **早期最佳窗口也不接近 promote**：即使只看最有希望的 `@199/@299`，`WLD` 仍是 `0/5`，且 `flip/z0/e2+` 三个核心指标都远未过线。

### 13.6 结论

| 动作 | 说明 |
|---|---|
| **不 promote f35** | `@199/@299` 已足够证明无胜场且离 gate 较远 |
| **关闭 f35 路线** | `src-quality + target_score 蒸馏 + anti-1-ship reward` 这组思路不再作为主线继续迭代 |
| **保留 f29 @599** | 仍是当前唯一稳定提交候选 |
| **Day9 转路线三** | 不再优先加特征，转向 **动作机制 + 训练分布**（见 `DAY9_PLAN.zh.md`） |

### 13.7 执行 / 路径备忘

```bash
bash scripts/run_v11_f35.sh          # 训练（含 parity smoke gate）
tail -f logs/v11_f35.log
bash scripts/check_training_health.sh logs/v11_f35.log

# 已跑 replay
bash scripts/quick_replay.sh ckpt_multi_action_v11_f35/ckpt_000199.pkl v11_f35_u199
bash scripts/quick_replay.sh ckpt_multi_action_v11_f35/ckpt_000299.pkl v11_f35_u299
```

| 用途 | 路径 |
|---|---|
| f35 训练 log | `logs/v11_f35.log` |
| f35 replay | `logs/replay_analyze/v11_f35_u{199,299}_vs_v20.json` |
| f35 特征 | `orbit_wars_rl/features/encode.py`（dims 28-32） |
| f35 reward | `orbit_wars_rl/env/rewards.py`（`one_ship_penalty_reward`, `high_prod_capture_reward`） |
| f35 config | `orbit_wars_rl/configs/multi_action_v11_f35.yaml` |
| f35 submission | `submission_rl_v11_f35.py`（PLANET_FEAT_DIM=33, F35_SRC_FEATS=1） |
| f35 脚本 | `scripts/run_v11_f35.sh`, `scripts/run_f35_eval.sh` |

---

## 9. f34 — f29 强对手锚定（下一轨）

**动机**：f33 @449 collapse = 纯 frozen pool 螺旋；f29 是 replay 最优 arch，作 fixed strong opponent 拉回 v20 分布。

**单 delta vs f33**：

| 参数 | f33 | f34 |
|---|---|---|
| strong_ckpt | — | f29 @599 |
| strong_ratio | 0 | **0.35** |
| frozen_ratio | 0.40 | **0.25** |
| num_updates | 2000 | **800**（在 collapse 前停） |

```bash
# 需 f29 ckpt 存在
bash scripts/run_v11_f34.sh
bash scripts/run_f34_eval.sh   # 训完后
```

配置：`orbit_wars_rl/configs/multi_action_v11_f34.yaml`

---

## 10. f32 clip_frac 确认

```
f32 max clip_frac=0.39  [CRITICAL]
f33 max clip_frac=0.10  (none)
```

f32 同时改 4 件事 + 原 f31 低 ent/lr 组合 → clip 失控；f33 对齐顶层选手后 clip 正常，但 **self-play 膨胀仍发生**（问题不在 clip，在对手分布）。

---

## 3. Day8 计划（基于 f32 结果的场景）

### 场景 A：f32 @299 进展良好（bin0 <15%, z0 >5%, flip >2%）

**动作**：继续训练到 @599。若 flip >3% 且 WLD >=1/5，promote。

**后续**：f33 用 mixed buffer（v20 + f32 top10）续训 800 upd。

### 场景 B：f32 修复保守但仍输（z0 >5%, flip >2%, WLD=0/5）

**根因**：模型容量或训练量不足。

**动作**：f33 超参对齐顶层选手：
- 缩小模型：d_model=64, ff_dim=128（~600K 参数）
- 对齐 PPO：ent_coef=0.05（单一）, lr=3e-5, ppo_epochs=1, clip_eps=0.2
- 增加训练：2000+ upd（30M+ steps）
- 每 50 upd 监控 clip_frac

### 场景 C：f32 仍过度保守（z0 <5%, emit <1.2）

**根因**：hard masks 仍过于限制，或速度特征不足。

**动作**：f33 更柔和的方法：
- 完全移除 flip_hard_mask，仅保留 emit_hard_stop
- 增加 ETA 特征："敌方舰队到达我方星球的回合数"
- 提高 ent_coef_pct 到 0.015（推动 pct head 选更大 bin）

### 场景 D：f32 训练不稳定（loss 发散, KL 飙升）

**根因**：lr 过高或 ent_coef 变更破坏稳定性。

**动作**：f33 用顶层选手的保守设置：
- lr=3e-5, clip_eps=0.2, ppo_epochs=1
- 单一 ent_coef=0.05
- 更小模型：d_model=64

---

## 4. Day8 执行步骤

### 早晨：检查 f32 训练状态

```bash
# 5090 上执行
bash scripts/check_training_health.sh logs/v11_f32.log
tail -20 logs/v11_f32.log
```

### 若 f32 仍在训练

1. 监控 clip_frac 趋势（>0.20 告警）
2. 若 @099 checkpoint 存在，跑早期 replay
3. 对比 f31 同等 update 数的训练指标

### 若 f32 训练完成

1. 跑 replay gate：@299, @599
2. 生成 seed0 HTML replay
3. 对比 vs f31 和 f29
4. 决策：promote / 续训 / pivot

### 无论 f32 结果如何

1. ✅ `scripts/check_training_health.sh` 已创建
2. ✅ clip_frac 已确认在训练循环中记录
3. 创建 `docs/DAY8_PROGRESS.zh.md`（本文档）

---

## 5. 关键决策标准

| Gate | f32 @299 | f32 @599 | 动作 |
|---|---|---|---|
| bin0 <15%, z0 >8%, flip >3%, WLD>=1/5 | FAIL | FAIL | ~~Promote~~ **否决** |
| bin0 <15%, z0 >5%, flip >2% | FAIL | FAIL | ~~继续@599~~ **已训完，无收益** |
| z0 <5%, emit <1.2 | @299 近似 | @599 **emit=2.25** | **Pivot**（见 §7.3） |
| clip_frac >0.30 | — | 待查 log | 若超则 f33 降 lr |
| loss 发散 | — | 未见 | — |

---

## 7. f32 replay 实测（vs v20，first-80，5 seeds）

| 指标 | f31 @599 | f32 @299 | f32 @599 | f32 目标 | @599 |
|---|---|---|---|---|---|
| bin0 | ~9% | 5.3% | **5.9%** | <15% | ✅ |
| z0 | ~2% | 1.3% | **0.5%** | >8% | ❌ |
| emit | ~1.08 | 1.27 | **2.25** | 1.2–2.0 | ⚠️ 偏高且无效 |
| spf | ~41 | 5.96 | **5.88** | >5 / >10 | ❌ |
| garr | ~48 | 28.84 | **31.65** | >60 | ❌ |
| flip | ~1% | 3.36% | **2.38%** | >3% | ❌ |
| e8 | ~8% | 1.8% | **8.0%** | <5% | ❌ |
| e2+ | — | 6.8% | **34.0%** | — | 多路 spam |
| WLD | 0/5 | 0/5 | **0/5** | ≥1/5 | ❌ |
| bin7 pct | — | 72.5% | **64.0%** | — | 小守军「拉满 pct」 |

**@299→@599 趋势**：emit↑、e8↑、e2+↑（更吵），z0↓、flip↓、spf 不动 — **训练越久 vs v20 越差**，非单调变好。

**训练侧**：`ckpt_000599.pkl` final eval **WR vs random 0.688**（可用但不可作 promote 依据）。

```bash
# 已跑
bash scripts/quick_replay.sh ckpt_multi_action_v11_f32/ckpt_000299.pkl v11_f32_u299
bash scripts/quick_replay.sh ckpt_multi_action_v11_f32/ckpt_000599.pkl v11_f32_u599
# 建议对照
bash scripts/quick_replay.sh ckpt_multi_action_v11_f29/ckpt_000599.pkl v11_f29_u599_baseline
bash scripts/check_training_health.sh logs/v11_f32.log
```

### 7.1 场景判定

| 场景 | 条件 | f32 实际 | 结论 |
|---|---|---|---|
| A promote | z0>8%, flip>3%, WLD≥1 | 全 fail | ❌ |
| B 容量/步数 | z0>5%, flip>2%, WLD=0 | z0 fail | ❌ |
| C 过度保守 | z0<5%, emit<1.2 | @599 emit=2.25 | ❌ 不完全是 C |
| **C+** | z0<5% 但 emit↑、bin7↑、spf 仍极低 | **匹配** | ✅ **采用** |

**C+ 解读**：放宽 bin3 + 速度特征没有带来「有意义的大舰队进攻」，而是 **高频多路小舰队 + bin7 表面拉满**；hard mask 止住了 bin0，但没学会 v20 式蓄力/flip。

### 7.2 决策

| 动作 | 说明 |
|---|---|
| **不 promote f32** | 不提交 `submission_rl_v11_f32_u599.py` |
| **保留 f29 @599** | 当前最强 replay 基线（bin0~10%, spf~13） |
| **f33 优先** | 从 **f31**（或 f29）ckpt 出发，**一次一个 delta** |

### 7.3 f33 候选（按优先级）

1. **f33a — 仅回退 flip 阈值**：f31 arch + `FLIP_BIN_FRAC=0.70`，保留 f32 速度 pair（隔离「bin3 放宽」害处）
2. **f33b — 去掉 flip_hard_mask**：仅 `emit_hard_stop` + f29 信号栈；加 **ETA-to-home** 全局特征（1 个 delta）
3. **f33c — 超参轨（场景 B）**：d_model=64、ent_coef=0.05 单一、lr=3e-5、ppo_epochs=1；从 f29 @599 resume
4. **禁止**：在 f32 @599 上续训（已验证 replay 退化）

---

## 6. 路径备忘

| 用途 | 路径 |
|---|---|
| f32 config / log | `orbit_wars_rl/configs/multi_action_v11_f32.yaml`, `logs/v11_f32.log` |
| f32 ckpt | `ckpt_multi_action_v11_f32/ckpt_XXXXXX.pkl` |
| f32 submission | `submission_rl_v11_f32.py` |
| f32 replay JSON | `logs/replay_analyze/v11_f32_u299_vs_v20.json`, `..._u599_...` |
| f32 eval 脚本 | `scripts/run_f32_eval.sh` |
| 训练健康检查 | `scripts/check_training_health.sh` |
| Day7 进展 | `docs/DAY7_PROGRESS.zh.md` |
| 顶层选手经验 | `top_players_rl.txt` |
| f33 config / log | `orbit_wars_rl/configs/multi_action_v11_f33.yaml`, `logs/v11_f33.log` |
| f33 ckpt | `ckpt_multi_action_v11_f33/ckpt_XXXXXX.pkl` |
| f33 eval | `scripts/run_f33_eval.sh`（✅ 已跑） |
| f33 replay | `logs/replay_analyze/v11_f33_u399_vs_v20.summary.txt` 等 |
| f33 HTML | `logs/replay_html/v11_f33_u399_seed0/replay.html` |
| f34 config / script | `orbit_wars_rl/configs/multi_action_v11_f34.yaml`, `scripts/run_v11_f34.sh` |
| f34 eval | `scripts/run_f34_eval.sh` |
