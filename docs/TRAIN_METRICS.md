# 训练日志指标速查表

> 本文档对应 `orbit_wars_rl/ppo/runner.py` 中每条 `upd` 行打印的所有字段。
> 写于 v6 训练期间(2026-05-23),针对 multi-action transformer PPO。

## 0. 一条样本日志

```
upd  191  steps 3145728  sps 5471  opp rand  loss -0.012  pg -0.0068  v 0.018
ev +0.82  adv_std 0.198  tR +0.44  ent[s/d/p/e] 0.66/2.67/1.77/0.48
emits 2.62  clip 0.11  kl +0.005
```

**v8 update**: We now print `ev` (explained_variance) right after `v`
(value_loss). This is the single best value-head health metric —
top_players_rl.txt §307 says "should hit at least 0.8 in 100 iters".
If `ev < 0.5` at upd 100, obs representation or architecture is
broken.

---

## 1. 训练进度类(只告诉跑了多少,不解释 RL 状态)

| 字段 | 含义 | 计算 |
|---|---|---|
| `upd` | 第几次 PPO 更新 | 单调增。每次 = 收集 rollout + GAE + 跑 `update_epochs` 轮 SGD |
| `steps` | 累计 env step | `upd × num_envs × rollout_length` |
| `sps` | steps per second | 累计平均;v6 实测 5400,v5p3 实测 6800(d_model=64→128 慢约 25%) |
| `opp` | 当前对手 | `rand` = 随机,`frzn` = self-play frozen pool 中的旧 ckpt |

---

## 2. Loss 类(衡量 PPO 在干什么)

### `loss` (total loss)
- 计算: `pg + value_coef × v + entropy_loss`
- 健康范围: 略负或缓慢下降
- 红线: 突然跳到 `+0.1` 或 `-0.5` 说明训练崩了

### `pg` (policy gradient loss)
- 计算: `−mean(advantage × ratio)`,clip 后取 min
- **物理意义**: policy 真正"在被推动"的力度
- 健康: `< -0.001`,**越负越强**
- 红线: 接近 0 → policy 卡死(v5p2 现象)

### `v` (value loss, MSE/Huber 风格)
- 计算: `(value_pred − returns)²` 的均值
- 健康: 训练初期 0.05+;收敛后 0.01-0.03
- ⚠️ **注意**: `v` 接近 0 **不是好事**,意味着 value head 完美 fit return → advantage ≈ 0 → policy 没东西学(参见 v5p2 失败)
- 这就是为什么我们在 v5p3 把 `value_coef` 从 0.5 砍到 0.20——让 value head 学慢点

### ⭐ `ev` (explained_variance) — v8 起新增

- 计算: `1 − Var(returns − value_pred) / Var(returns)`
- **物理意义**: value head 解释了 returns 多少方差。范围 [−∞, 1.0]。
  - **`ev = 1.0`** = value 完美预测 returns (理论上限)
  - **`ev = 0.0`** = value 等于 mean(returns)(基线,什么都没学)
  - **`ev < 0`** = value head 比"常数 mean"还差(罕见,表示在 random 之下)

| `ev` 区间 | 含义 | 行动 |
|---|---|---|
| `> 0.9` | value head 收敛到信号,advantage 几乎全来自策略层面的 noise | 健康 |
| `0.6–0.9` | value head 学得 OK | 健康 |
| `0.4–0.6` | value 刚 sane 但弱 | watch — 几十个 update 内应升 |
| `0.2–0.4` | value 严重欠拟合,advantage 信号噪音大 | 检查 value_coef 是否太低 |
| `< 0.2` | obs 表征或 architecture 问题 | **kill,debug obs/encoder** |
| `< 0` | value head 在退化 | **立即 kill** |

- **Top1 (top_players_rl.txt §307)**: "should go up to at least **0.8 in 100
  iters**. 0.9 in 20 iters. If `ev` never gets past 0.5, you should check
  your obs representation or architecture. I would suspect obs representations."
- 跟 `v` 的区别: `v` 是 raw MSE,**跟 returns 的 scale 强相关**(reward shaping
  改了 returns 范围 v 也跟着变);`ev` 是归一化的,**跨配置可比**。
- 跟 `adv_std` 的区别: `adv_std` 衡量"policy 收到多少信号";`ev` 衡量"value
  head 在多准确地 baseline 这些信号"。两者**互补**,不重复。
- **v8 阶段 kill 准则**:
  - `ev < 0.5` at upd 100 → 重检 obs 和 encoder。可能是 v7 加的 reserved-aware
    输入 scale 不对(也可能是 reserved 信号根本没被 encoder 看到)。
  - `ev` 在 upd 500+ 突然下跌 0.2+ → 值头 saturate / advantage 信号 vanish
    的早期警告。看 `pg` 是不是也跟着塌(配合 v5p2 复盘)。

### `kl` (mean KL divergence)
- 计算: `mean(old_logp − new_logp)`,跨 update_epochs 平均
- 健康: **0.001-0.02 = 活跃在更新**
- 红线: `kl = 0.000` → policy 没更新(信号死),`kl > 0.05` → 步长太大

---

## 3. Advantage / Clip 类 ⭐ **关键质量指标**

### `adv_std` (advantage standard deviation)
- 计算: rollout 全部 (s,a) 的 advantage 的 std
- **物理意义**: 学习信号的"强度"
- 严格区间:

| `adv_std` | 含义 |
|---|---|
| 0.30-0.60 | 最佳状态,学得快 |
| 0.15-0.30 | 次优但在学(需结合 `pg_loss` 判断) |
| < 0.15 | 信号过弱,policy 几乎不动 |
| > 1.0 | 不稳定,advantage 爆炸 |

- ⚠️ 单看 `adv_std` 不够,**必须配合 `pg_loss` 判断**:
  - `adv_std 0.18 + pg_loss -0.007` = "信号弱但 policy 仍持续被推" = **健康**
  - `adv_std 0.18 + pg_loss -0.0005` = "信号弱且 policy 不动" = **死局**

### `clip` (clip fraction)
- 计算: `mean(abs(ratio - 1) > clip_eps)`,即被 ε-clip 截掉的样本比例
- 健康: **0.05-0.15 = policy 在持续刷新**
- 红线:
  - `clip < 0.02` → policy 没动,要么死局要么 lr 太低
  - `clip > 0.30` → policy 步长太大,top_players_rl.txt §75 的明确警告

### `tR` (mean terminal reward)
- 计算: rollout 中已结束 episode 的 reward 均值
- 物理意义: 当前 rollout 的胜率(`tR = +1` 全赢, `tR = -1` 全输)
- v6 健康: `tR +0.3 to +0.6` vs rand opp

---

## 4. Entropy 类(衡量探索)

### `ent[s/d/p/e]` 4 个 head 的 entropy

每个 head 的 max entropy 是 `ln(N_options)`,**用百分比看更直观**:

| Head | N_options | max H | 健康区间(% of max) | 说明 |
|---|---|---|---|---|
| `s` (src) | 40 planets | 3.69 | 10-30% (0.4-1.1) | src 通常只能选少数 own planets,所以 entropy 必然低 |
| `d` (dst) | 40 planets | 3.69 | 50-80% (1.8-3.0) | dst 选项多,前期高,后期收敛到 1.5-2.5 |
| `p` (pct) | **8 bins** | 2.08 | 40-70% (0.8-1.45) | v6 起步 1.9 (90%),目标降到 < 1.0 |
| `e` (emit) | 2 | 0.69 | 35-70% (0.24-0.48) | < 0.30 = 太确定地停发,> 0.55 = 还在乱发 |

**注**: 旧 v3-v5p3 的 `p` 是 4 bins,max=1.39。**v6 的 1.7-1.9 不能直接和 v5p3 的 1.0-1.2 比**,要换算成百分比。

---

## 5. 行为/胜率类(策略涌现的可见信号)

### `emits` (mean fleets per turn)
- 计算: rollout 中每个 turn 实际发射的 fleet 数均值,范围 [0, K=8]
- **领域知识**: Kaggle orbit_wars 中 fleet 速度 = log(ships) (1 ship → 1.0 unit/turn, 500 ships → 5.0 unit/turn)。**小编队物理上到不了战场**
- 健康路标(vs v20 这种 26 fleet/turn 的强 heuristic):
  - 1.0-1.5 = 死锁(v3-v5p3 全是这状态)
  - 1.5-2.5 = **policy 在学多发** (v6 当前)
  - 2.5-4.0 = 接近能赢的水平
  - 4.0+ = 接近 v20 / F12 水平

### `WRr` / `WRf` (win rates from eval)
- `WRr`: vs random opponent 的胜率(`eval_every` updates 跑一次)
- `WRf`: vs frozen self-play pool 的胜率;0.5 = 跟自己平,> 0.55 = 持续进步,< 0.45 = 在变差

---

## 6. 经典失败模式速识

| 现象 | 病名 | 根因 | 修复 |
|---|---|---|---|
| `pg ≈ 0` + `adv_std < 0.15` + `clip 0.00` | **死局** | reward signal sparse (v5p2) | 缩短 episode_steps 让 terminal reward 变密 |
| `pg ≈ 0` + `adv_std < 0.15` + `v` 突然跌到 0 | **value outran policy** | value_coef 太大 (v5p2 末段) | `value_coef 0.5 → 0.20` |
| `clip` 单调爬升 0.10 → 0.30+ | **clip creep** (§75) | optimizer 输给 value head | 降 lr 一半 |
| `kl 0.000` 持续多个 update | **policy 不更新** | lr 太低 / 信号死 | 检查上面两条 |
| `emits` 锁死 1.0-1.5 | **emit 信用分配死锁** | reward 太稀疏,emit head 学不到 | 调 `ent_coef_emit` 或上 BC bootstrap |
| `loss = NaN` | grad 爆 | lr 太高 / `max_grad_norm` 太松 | 降 lr |

---

## 7. v6 当前(upd 191)真实诊断模板

```
adv_std 0.18  → 次优区间
pg_loss -0.0068 → 健康强信号 ✓
clip 0.11 → 健康 ✓
kl 0.005 → 健康 ✓
v 0.018 → 没崩到 0 ✓
emits 2.6 → 终于动起来 ✓
ent_emit 0.50 → emit 开始 commit ✓
WRr 0.72 / WRf 0.72 ✓
```

**整体判断**: "在学但偏慢"模式。可下迭代用 `value_coef 0.20 → 0.10` 增强 advantage。

---

## 8. 何时该 kill 训练

按严重度排序:

1. **立即 kill**: `loss = NaN`,或 `pg / v / adv_std` 任何一个产生 NaN/Inf
2. **upd 30 内 kill**: `pg_loss > -0.0005` 且 `clip < 0.02` (死局)
3. **upd 50 内 kill**: `clip` 持续爬到 > 0.30 (clip creep)
4. **upd 200 看趋势**: `mean_emits` 持续 < 1.5 (策略卡死)
5. **upd 500 看 H2H**: vs v4.2 < 10/20 (无进步)

---

## 9. 阅读训练曲线的"5 秒检查清单"

每次看新 log,按这顺序 5 秒读完:
1. `loss` 不是 NaN? ✓
2. `pg < -0.001`? policy 在学
3. `clip 0.05-0.15`? 步长合理
4. `kl > 0.0005`? policy 在更新
5. `emits` 在涨 / 没在跌? 策略涌现
6. `WRr` / `WRf` 趋势?

任何一条不对,立刻深挖。
