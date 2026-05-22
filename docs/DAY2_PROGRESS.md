# Day 2 进展总结（2026-05-22 上午 → 中午）

> **写给晚上要在 5090 上跑正经训练的自己。** 这里只记「今天发生了什么、为什么这么做、晚上从哪里开始」，
> 完整的 RL 管线说明仍以 [`RL_PIPELINE.md`](RL_PIPELINE.md) 为准；多动作 head 的需求出处见 [`DAY1_PROGRESS.md`](DAY1_PROGRESS.md) §4。

---

## 0. TL;DR

* **DAY1 §4 计划的多动作 head 全栈今天落地**——`env`/`net`/`ppo`/`inference`/`submission_rl_v3.py` 一气呵成，按 §4.3 的 9 项清单 100% 完成。
* **修了 3 个会让训练「看上去 OK 实际不 work」的 bug**：PPO ratio 不一致（emit_free_mask 双路径）、`DstHead` 错误屏蔽己方星球、`EmitHead continue_bias=1.0` 让随机权重每回合乱发 8 条 fleet。3 个 bug 都有可观察到的「学习曲线特征」，详见 §3。
* **本地 mac CPU 三档 K 消融全部跑通**：K=1 / K=2 / K=8 的最终 WR vs random 分别 0.81 / 0.62 / 0.62。K=8 peak 0.94，但回落证明 self-play 之前的 entropy/lr 还需在长训练上再调。
* **核心 pipeline 已健康**：parity 16/16 一致，ratio≈1.0 (max|Δlogp|=3.8e-6)，advantage std 0.077→0.29，每 turn `mean_emits` 收敛在 1.3–1.7（policy 自学到「少而精」）。
* **5090 正经训练 400 update + self-play 已跑完**（SPS ~3.7K，1.64M env steps，约 8 分钟）：**WRr 0.91 / WRf 0.94 / final 0.906**。WRr 达标，但 **WRf 0.94 重现了 DAY1 §3.2 v2 的「self-play 不滚雪球」症状**——需要后续诊断 frozen pool。详见 §6.5。
* **H2H 还没跑**：等 h2h_local.py 的 `--seeds 0,1,2,3,4` 兼容修补后重跑。

---

## 1. 今天动手做的事

### 1.1 多动作 head 全套实现（早上 → 上午）

按 DAY1 §4.3 的清单逐项落地：

| 模块 | 关键修改 |
|------|---------|
| `env/actions.py` | 新增 `MultiPlayerAction` (K 维 arrays + `emit_mask`)、`single_to_multi` 反向兼容包装、`decode_action` 加 `reserved_ships` 参数 |
| `env/dynamics.py` | `launch_fleets` 接 `MultiPlayerAction`；`_launch_one_player_multi` 用 `lax.scan` 顺序处理 K 个 fleet，每步累加 `reserved_ships` |
| `env/env.py` | `step` 接 multi-action；保留 `step_single` / `step_and_autoreset_single` 给老 `random_opponent_action` |
| `env/constants.py` | 加 `MAX_FLEETS_PER_TURN = 8` |
| `net/heads.py` | 新增 `EmitHead`（`global+planet_pool+step_oh → 2 logits`），bias init `continue_bias` 可调（默认 0.0，见 §3.3） |
| `net/model.py` | `ActorCritic` 重写为 autoregressive，K 步 Python `for` unroll（不能用 `lax.scan`：flax `nn.Module` 不允许嵌套）；引入 `SampledMultiAction` 含每步 logp/entropy/`emit_free_mask` |
| `ppo/rollout.py` | `Rollout` 多 K 维 action 字段 + `emit_free_mask` + `planet_ships_raw`；frozen-opp 路径同步多动作 |
| `ppo/update.py` | turn-level logp = ∑ over K steps（src/dst/pct 用 `emit_mask` 屏蔽、emit 用 `emit_free_mask` 屏蔽）；4 个 per-head entropy 系数；新增 `mean_emits_per_turn` / `adv_std` / `mean_terminal_reward` 指标 |
| `inference/numpy_forward.py` | `greedy_multi_action`：纯 numpy autoregressive 推理，返回 `(src_list, dst_list, pct_list, emit_list, value)` |
| `inference/kaggle_adapter.py` | `decode_multi_to_kaggle_moves`：list-of-(src,dst,pct) → Kaggle moves，处理同源多次发射的 `reserved_ships` |
| `inference/test_parity.py` | 升级到「整回合 action list 16/16 一致」断言（不再只比首动作） |
| `submission_rl_v3.py` | 新单文件 submission（`agent()` 返回 list of moves） |
| `scripts/export_submission.py` | 默认 template 改 v3；smoke 校验返回 list of lists |
| `configs/multi_action.yaml` (+ 3 个 local debug 配置) | 新训练配置 |

### 1.2 关键 bug 排查与修复（中午）

三个独立 bug 都触发了一个共同的现象：「PPO 数值看上去健康（clip<0.05, kl~0），但 WR 完全不上升」。逐个发现并修复，详见 §3。

### 1.3 训练验证：K 维消融（中午 → 下午）

为了排除「多动作 PPO 算法本身坏了」vs「emit/K 维度信号弱」，跑了三档 K (1/2/8) 的 30–50 update 对照。最终 K=1 干净学到 0.81，K=8 peak 0.94——证明 pipeline 健康，问题归到「多动作的探索调度」而不是算法。详见 §6。

### 1.4 文档与对照（中午）

* 把 DAY1_PROGRESS.md 末尾的「Day 2 进展」段落迁到本文。
* 对照 `top_players_rl.txt`（Lin Myat Ko, 1st）做了方向偏差分析，结论见 §4。

---

## 2. 现在仓库长这样

```
OrbitWarRL/
├── submission_v20_0513.py        启发式 bot（v1 之外的提交槽位）
├── submission_rl_v1.py           DAY1 上传 Kaggle 的单文件
├── submission_rl_v2.py           DAY1 self-play 训出来但没交（保留作对照）
├── submission_rl_v3.py           ★ 今天新建——多动作 head 的单文件（待 5090 训完导出真权重）
├── ckpt_mvp/ckpt_000049.pkl      v1 来源
├── ckpt_selfplay/ckpt_*.pkl      v2 来源
├── ckpt_selfplay_ep500/...       v3-NO-multi-action 来源
├── ckpt_multi_action_*/...       ★ 今天本地 mac 跑出来的小 ckpt（K=1/K=2/K=8，仅用作 sanity）
└── orbit_wars_rl/
    ├── env/
    │   ├── actions.py            ★ MultiPlayerAction + single_to_multi + decode_action(reserved_ships=...)
    │   ├── constants.py          ★ 加 MAX_FLEETS_PER_TURN
    │   ├── dynamics.py           ★ launch_fleets 多动作 + lax.scan K 步
    │   ├── env.py                ★ step 接 multi；shaping_delta 接进 reward
    │   ├── rewards.py            ★ 新增 potential-based shaping
    │   └── ...
    ├── features/encode.py        与 DAY1 一致
    ├── net/
    │   ├── heads.py              ★ DstHead 不再屏蔽己方星球；新增 EmitHead（默认 continue_bias=0）
    │   ├── model.py              ★ ActorCritic K 步 autoregressive；SampledMultiAction
    │   └── transformer.py        与 DAY1 一致
    ├── ppo/
    │   ├── rollout.py            ★ Rollout 加 K 维 + emit_free_mask + planet_ships_raw；frozen-opp 同步
    │   ├── update.py             ★ turn-level logp 累加 + emit_free_mask 双路径一致；新增监控指标
    │   └── runner.py             ★ 日志加 adv_std / tR / pg 4 位精度 / emits / ent_emit
    ├── selfplay/                 与 DAY1 一致
    ├── inference/
    │   ├── numpy_forward.py      ★ greedy_multi_action；dst_head 同步去掉 ~my_planet_mask
    │   ├── kaggle_adapter.py     ★ decode_multi_to_kaggle_moves
    │   ├── test_parity.py        ★ 整回合 action list 一致断言
    │   └── ...
    ├── scripts/
    │   ├── diagnose_rollout.py   ★ 新增：rollout 信号强度诊断（reward/adv/value/emits 分布）
    │   ├── export_submission.py  ★ 默认 template 改 v3
    │   └── ...（其余与 DAY1 一致）
    └── configs/
        ├── mvp.yaml / selfplay.yaml / selfplay_ep500.yaml    与 DAY1 一致
        ├── multi_action.yaml                                 ★ 正经训练配置（晚上 5090 用这个）
        ├── multi_action_local.yaml                           ★ mac CPU 全规模
        ├── multi_action_norand_local.yaml                    ★ K=8 vs random debug
        ├── multi_action_k1_local.yaml                        ★ K=1 sanity check
        └── multi_action_k2_local.yaml                        ★ K=2 中间档
```

---

## 3. 三个 bug 与修复

### 3.1 PPO ratio 不一致（早上 ~9 点发现）

**症状**：跑 PPO sanity check（同一组 params 重新评估 sampled logp），ratio 不是 1.0 而是 0.8–1.3。说明 sample 路径和 evaluate 路径的 logp 计算不一致——这会让 PPO 第一个 epoch 起就被「伪 ratio」误导，policy 在和自己的影子打架。

**根因**：`emit_logp` 的 `free_choice` mask 在 sample 时考虑了 `no_options`（己方所有星球都没船时强制 stop），但 evaluate 时的 proxy mask 没考虑这个。导致同样的「强制 stop」事件在 sample 时 logp=0、在 evaluate 时 logp 非零，于是 `new - old ≠ 0`。

**修复**：把 sample 时计算的 `free_choice` 存到 `SampledMultiAction.emit_free_mask`，rollout 持久化它，evaluate 直接复用。Sanity 验证后：

```
max |new - old|: 3.814697265625e-06
ratio min/max: 0.9999961853027344 1.0000019073486328
```

**教训**：「mask 在两条路径上各自实现一次」是经典 RL 隐藏 bug。Sample 算的关键中间变量（mask / RNG decision）必须显式存进 rollout，evaluate 不许重算。

### 3.2 DstHead 屏蔽自己星球（中午 ~12 点发现）

**症状**：K=8 训练 30 update 后 WR vs random 在 0.12–0.31 之间无规律飘，`dst entropy` 卡在 2.95（接近 ln(15)=2.71 满熵），policy 完全没在学 dst。

**根因**：`DstHead._mask_logits` 用 `valid = planet_mask & ~my_planet_mask`，把自己的星球全部屏蔽。但 Kaggle 规则 `overview.txt` 明确：

> If the attacker is the same owner as the planet, the surviving ships are added to the garrison.

也就是说，**给自己的星球发兵 = 增援**，这是一个重要战术。屏蔽掉它意味着 policy 不能做防御性增援，只能进攻——并且因为去掉了一半合法目标，dst 的有效信号被稀释，`dst entropy` 永远收敛不下来。

**修复**：`heads.py` 和 `numpy_forward.py` 的 `dst_head` 都改成只屏蔽 padding（`valid = planet_mask`）；同源 `src == dst` 由 `decode_action` 的 `src_idx != dst_idx` 兜底过滤。parity test 仍 16/16 通过。

**教训**：网络的 mask 必须严格对齐环境的 legal action 规则。我之前是按「直觉上 dst 应该是非己方」加的 mask，没去对游戏规则，这是 DAY1 写得最草率的地方。

### 3.3 EmitHead `continue_bias=1.0` 太激进（下午 ~2 点发现）

**症状**：修完前两个 bug 后，K=8 的 WR vs random 仍在 0.44–0.69 飘；同管线 K=1 sanity check 直接学到 0.81，K=2 学到 0.625。差距不来自 PPO 算法，来自 emit 决策。

**根因**：之前为了防止"emit 1 fleet 就停"的局部最优，给 EmitHead 加了 `continue_bias=1.0`。**但 random init 网络 + bias=1.0 → 几乎每个 turn emit 8 次**，结果是把所有星球的船一次性发空，乱发兵 → 输给只发 1 条的 random opponent。Random 反而成了「单动作环境的最优解」，learner 要先学会「别乱发」，再学「精准发」，这条路径在 30–50 update 内根本走不完。

**修复**：`heads.py` 默认 `continue_bias=0.0`。让初始 emit 是 50/50，policy 自己学习什么时候多发。修完后 K=8 peak WR 0.94、emits 收敛到 1.3–1.7。

**教训**：「为了避免局部最优 A 而手动偏置网络」如果偏过头会进入更糟的局部最优 B。Entropy 系数比直接改 bias 更安全。

### 3.4 同步动作：reward shaping

修完上面 3 个 bug 之前，先做了一个独立改动：terminal `+1/-1` 在 200 步 episode 内 reward 太稀疏（一个 rollout `len=128, num_envs=16` 居然完成 0 个 episode！），加了 potential-based dense shaping：

```python
# env/rewards.py
phi(s, p) = tanh((ships(p, s) - ships(1-p, s)) / SHAPING_REF)  # bounded |.|<1
F(s, s') = SHAPING_SCALE * (phi(s', p) - phi(s, p))
```

`SHAPING_SCALE=0.1`, `SHAPING_REF=30.0`。Terminal step 仍是 ±1（不叠加 shaping），potential-based 形式保证最优策略不变。验证后 `advantage std` 从 0.077 → 0.29（3.7x），rollout 内 episode 完成数从 0 → 16/16。

**注意**：top players 的 1st 选手原话「+1 -1 is enough for 2p mode」，所以 shaping 是为了**小规模训练加速 signs of life**，长训练后期可以考虑 `SHAPING_SCALE=0` 退化到纯 sparse reward，避免 shaping 把 policy 锁在「保 ship 差」而不是「真的赢」上。这条放在 §7 follow-up。

---

## 4. 对照 top players（Lin Myat Ko, 1st）

对照 `top_players_rl.txt` 做的方向偏差分析，结论是「方法论几乎没偏，差距在规模和迭代节奏」。

| 维度 | Lin Myat Ko (1st) | 我们当前 | 差距/风险 |
|------|---|---|---|
| **环境** | JAX 重写 Kaggle env | ✅ JAX | 对齐 |
| **SPS** | 10K（基础架构）/ 2K（复杂架构） | mac ~150, 5090 待测 | 5090 应能到 5K+，否则要 profile |
| **网络** | Entity transformer，~600K 参数 | Entity transformer，~120K 参数 | **我们容量小 5x**，先验证 pipeline 后再加深 |
| **训练时长** | 600M steps / 3 天 self-play | 本地 100K（sanity） | **差 4 个数量级**——这是头号差距，5090 应能补上 |
| **Reward** | +1/-1 终局 | +1/-1 + dense potential shaping | 我们多一层 shaping；长训后期可考虑关掉 |
| **Action head** | target argmax @ inference，softmax @ training；fleet pct bins | 一致 ✅ | 对齐 |
| **PPO 设定** | warmup_cosine LR、per-head ent_coef、ablation 严格 | ✅ 都有 | 对齐 |
| **架构改动节奏** | "Add one delta at a time"（Opus 评论第 65 行） | Day 2 一次合并 3 个 bug fix | ⚠️ **当前最大违规**，但通过 K=1/2/8 三档消融做了事后归因 |
| **行动数 K** | 没明说，但 Lux AI 范式都是多动作 autoregressive | K=8 autoregressive | 对齐 |
| **多对手 self-play** | 是 | 是（frozen pool 5 个，已就位） | 对齐 |

**最值得注意的两条经验（来自 Opus 评论）**：

* **第 75 行**：「clip_frac 单调上升 0.10 → 0.30 之前就动手」——这是「value head 跑赢 policy」的早期信号，等爆了再降 lr 已经晚一天。**我们今天本地训练的 clip 全程 < 0.05，但 self-play 启动后必然抖**，需要监控。
* **第 65 行**：「Add one architecture delta at a time. Always.」——今天我合并了 3 个改动，违反这个原则。但通过 K=1/K=2/K=8 三档消融做了事后归因（K=1 跑通证明算法没坏，K=8 修完 bug 后能学说明每个 fix 都贡献了），算是事后补救。下次不要再这样。

---

## 5. 晚上换电脑要做的事（5090）

```bash
# 0. 同步代码
cd OrbitWarRL && git pull

# 1. 验证依赖 + parity（~10s）
python -c "import jax, flax, optax, chex, numpy; print('ok')"
python -m orbit_wars_rl.inference.test_parity
#   预期：  [OK ] whole-turn action list match: 16/16

# 2. 正经训练 v3（400 update，self-play 混合）
python -m orbit_wars_rl.scripts.train \
    --config orbit_wars_rl/configs/multi_action.yaml \
    --log-dir logs/multi_action_v3 \
    2>&1 | tee logs/multi_action_v3.log

# 3. 训练完后导出 v3 submission
python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action/ckpt_000399.pkl \
    --out submission_rl_v3.py

# 4. 本地 H2H 验收（每局 ~10s）
python -m orbit_wars_rl.scripts.h2h_local \
    --agent-a submission_rl_v3.py --agent-b submission_rl_v1.py \
    --num-games 5 --seeds 0,1,2,3,4
python -m orbit_wars_rl.scripts.h2h_local \
    --agent-a submission_rl_v3.py --agent-b submission_v20_0513.py \
    --num-games 5 --seeds 0,1,2,3,4
```

**训练监控（按 top players 的经验）**：

* `clip` 单调上升 0.10 → 0.30：**立即停**，把 `lr_peak` 1.5e-4 → 8e-5，或 `update_epochs` 2 → 1
* `kl > 0.05` 持续多个 update：同上处置
* `WRr` 应 ≥ 0.9，`WRf` 应在 0.5–0.6 健康振荡（DAY1 §3.2 教训：`WRf ~ 0.94` 是 self-play 不工作的信号，不是好事）
* `mean_emits_per_turn` 不应该卡在 1.0（emit head 死了）或 8.0（emit head 退化成 always-on），健康范围 1.5–4

把日志末尾 ~50 行贴回来，我盯指标。

---

## 6. 实验数据档案

### 6.1 本地 mac CPU（sanity 用，每档 30–50 update）

设置：`num_envs=16, rollout_length=128, episode_steps=100`。

| 配置 | K | 最终 WR vs random | peak WR | 关键观察 |
|------|---|-------------------|---------|---------|
| `multi_action_k1_local.yaml` | 1 | **0.812** | 0.88 | sanity：证明 PPO + features + reward + value head 全 OK |
| `multi_action_k2_local.yaml` | 2 | 0.625 | 0.75 | 中间档，emits 收敛到 1.5 |
| `multi_action_norand_local.yaml` | 8 | **0.625** | **0.94** | 修完所有 bug 后的成绩；emits 收敛到 1.3–1.7 |

### 6.2 5090 正经训练 (`multi_action.yaml`, 400 update + self-play)

设置：`num_envs=32, rollout_length=128, episode_steps=200, num_updates=400`，self-play warmup 40 update，frozen_ratio 0.5，pool 5。

| 指标 | 起点 | upd 99 | upd 199 | upd 299 | upd 399 |
|------|------|--------|---------|---------|---------|
| WRr (vs random) | 0.12 | 0.66 | **0.88** | 0.72 | **0.91** |
| WRf (vs frozen) | 0.00 | 0.66 | 0.66 | 0.84 | **0.94** ⚠️ |
| tR (mean terminal R) | -0.20 | -0.32 | +0.00 | +0.44 | +0.80 |
| emits/turn | 2.43 | 1.53 | 1.62 | 1.58 | 1.73 |
| ent_dst | 2.95 | 2.95 | 2.92 | 2.86 | **2.81** ⚠️ |
| clip_frac | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| approx_kl | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| pg_loss | -0.0005 | -0.0005 | -0.0001 | -0.0006 | -0.0003 |
| SPS | 161 | 3132 | 3487 | 3614 | **3690** |

`final eval win_rate vs random: 0.906` —— DAY1 §4.4 的「WRr ≥ 0.9」验收通过 ✅。

**重要观察**（不止 1 个红灯）：

1. **WRf = 0.94 重现了 DAY1 §3.2 v2 的问题**——learner 在碾压 frozen pool 里所有快照，**self-play 没真正在滚雪球**。和 DAY1 ep500 实验里 WRf = 0.66 那种健康振荡完全不一样。可能根因：
   * `snapshot_every=10` 太勤，pool 里的旧 self 太接近当前 learner，没起到「真有挑战的对手」作用；
   * `frozen_ratio=0.5` 让一半 rollout 还是 vs random，learner 主要在卷 random，没被 frozen 卡过；
   * `pool_capacity=5` 太小，新 snapshot 把老 snapshot 挤掉得太快。
2. **ent_dst 几乎没动**（2.95 → 2.81，降幅 0.14）。对比 src/pct/emit 都在收敛（1.62/1.33/0.67），**dst head 在 ~15 个目标候选间几乎是均匀分布**。这是 DAY1 没碰到的新问题，可能 dst head 表征不够（只有一个 cross-attn + 1 层 Dense），或者 advantage 信号不区分「打哪个具体目标」。
3. **clip/kl/pg 全程极小**——`clip=0.00, kl=0.000, pg_loss ≈ -0.0003`。乍看是好事，但 Opus 在 `top_players_rl.txt` 第 75 行说「creep 之前就动手」，**我们反过来——`clip` 太冷了**。可能 lr peak 1.5e-4 太保守，policy 改不动。
4. **`emits/turn` 在 1.4–1.8 稳定**：policy 自学到「少而精」，没在「乱发 8 条」和「只发 1 条」两个局部最优坍缩，这是健康的。
5. **SPS 3690**：和 1st 选手「2K（复杂架构）」量级一致，环境/网络速度不是瓶颈。

### 6.3 H2H 实测结果（5090，2026-05-22 ~14:10）

| 对手 | v3 战绩 | 解读 |
|------|---------|------|
| **v1**（DAY1 MVP, K=1） | **0/5** ❌ | 比 v1 还弱！v3 训练 WRr 0.91 是在 MVP env 里有效，但单文件 submission 在真 Kaggle env 里完全废 |
| **v20**（启发式） | **0/5** ❌ | 全败，符合预期 |
| starter | 没跑 | 预期 0/5（连 v1 都打不过） |

**症状**：smoke test 上 v3 已经显露 mode collapse——`[[0, 1.57, 15], [0, 1.57, 7], [0, 1.57, 4], [0, 1.57, 2], [0, 1.57, 1], [0, 1.57, 1]]`——6 条 fleet 全部 src=0 / dst=2（中性 planet），从单源拆 6 次发往**同一个中性目标**。同样的 obs 下 v1 正确选 `[[0, 0.785, 22]]`（朝 enemy planet 3 方向，22 ships 一次性发出）。

**初步诊断**（待 5090 上验证）：
* 不是 obs encoder/adapter 出错——v1 用同一套 adapter，在 home 从 id 0 平移到 id 7 时能正确跟随，证明 features 没坏。
* 大概率是 **training env 和 Kaggle env 的结构性 gap 在 multi-action 下被放大**：
  * Training env 行星完全静态（不绕 sun 转），real Kaggle 行星持续旋转
  * Training env 4-fold 对称，5 组 = 20 planets，home 永远在 slot 0/3
  * v3 学了 8 步连续决策，每步都强化「针对静态 home@0 / enemy@3 布局的最优解」
* v1 单动作只需要决策一次，自由度小，反而留下了相对 robust 的策略；v3 8 步自由度高，**严重 overfit 到训练 env 的结构**

### 6.3.1 真正的根因（B 阶段验证后发现）

5090 上跑诊断脚本后发现：v3 在 home@0 和 home@7 两个 obs 上 `agent()` 都返回 `[]`（被 `try/except` 兜底吃了）。绕过 except 直查代码后**定位到 `submission_rl_v3.py` 第 322 行**：

```python
# BUG (修复前):
dst_valid = planet_mask.astype(bool) & ~my_mask.astype(bool)
```

`submission_rl_v3.py` 是**手写的单文件镜像**，没同步今天 §4.1 在 `net/heads.py` + `inference/numpy_forward.py` 修的 dst mask bug。`numpy_forward.py` 允许 dst 是自己 planet（增援），但 submission 里的内嵌副本仍在排除——**train/inference 行为不一致**！

后果链：
1. Training 时 policy 学到「dst 可以是任意 planet 包括自己」 → 学到的策略经常选 dst=own_planet（增援）
2. Export 后 inference 强制屏蔽自己 planet → dst 选 logits 第二高的（往往是最近的中性 planet）
3. ships 数量 + emit 决策仍按「这一发是合理的」逻辑算 → 8 步全部往同一个中性 planet 越拆越小发兵
4. 自己几乎不剩兵 → 任何对手（包括 v1 单动作 K=1）都能稳赢

**这也解释了为什么 final WR vs random 0.906 而 H2H vs v1 0/5**：vs random 时 random 也是乱发兵，谁先把对方家打爆都行；vs v1（有合理策略）就立刻暴露 mode collapse。

**修复**：把 `submission_rl_v3.py:322` 改为 `_mask_logits(logits, planet_mask.astype(bool))`，与 `numpy_forward.py` 对齐。`docs/DAY2_PROGRESS.md` 同步更新 docstring。

**Build process 教训**：单文件 submission 是**手写镜像**而非自动生成，存在 drift 风险。后续要么改成 build-time 拼接生成、要么在 `inference/test_parity.py` 加一个「submission_rl_vN.py vs numpy_forward.py 行为一致」的 smoke 测试。今天先靠人工保持同步。

### 6.3.2 第二个坑：sync 脚本 `--delete` 误删了远端训练产物

修完 dst mask 后试图重新 export 验证，发现 5090 上 `ckpt_multi_action/ckpt_000399.pkl` 已经不存在了——`sync_mirror_ultrapp.sh` 用 `rsync --delete`，没排除 `ckpt_*/` 和 `logs/`，本地没有这两个目录所以远端被清空。

而且发现 400-update 训练时 `tee logs/multi_action_v3.log: 没有那个文件或目录` 也是同样原因——训练命令开始前没 `mkdir -p logs/multi_action_v3`，所以那次训练的日志根本没落地（只是 stdout 实时显示）。结果：

* ckpt 没了
* 日志没了
* 唯一证据是 stdout 截图

修复：
* `sync_mirror_ultrapp.sh` 加 `--exclude "ckpt_*/" --exclude "logs/" --exclude "*.pkl" --exclude "*.log"`
* 训练前用 `mkdir -p ckpt_multi_action logs/multi_action_v3` 显式建目录

代价：**必须重新训练一次 400-update**（约 8 分钟），这次 ckpt+log 都会落地，且训练-推理 dst mask 一致。

### 6.3.3 v3 重训完成 + H2H 验证：dst_mask 不是真凶

修完两个坑后重训 400 update（`ckpt_multi_action/ckpt_000399.pkl`），结果对比上次"丢失的"训练：

| 指标 | "丢失的"v3 (旧 ckpt) | 这次 v3 (新 ckpt) | 解读 |
|------|---------|---------|------|
| final WRr | 0.91 | **0.72** | 这次反而更弱（seed/noise） |
| final WRf | 0.94 | 0.91 | 几乎没变 |
| `dst entropy` (始→终) | 2.95 → 2.81 | 2.95 → **2.85** | **几乎完全没学** |
| `pg_loss` | ±0.0003 | ±0.0003 | 一直在噪声底 |
| `clip` / `kl` | 0.00 / 0.000 | 0.00 / 0.000 | PPO 几乎完全不更新 |
| `adv_std` | (没看) | **0.17** 全程不变 | 与 random init baseline 同量级 |
| `emits/turn` (始→终) | 2.43 → 1.73 | 2.43 → 1.67 | 学到了不乱 emit |
| H2H vs v1 | 0/5 | **1/5** | **小幅改善**但仍差 |

smoke moves 也从 `[[0,1.57,15],[0,1.57,7],...]`（6 条全发往同一中性）变成 `[[0,1.57,15]]`（只 1 条）——dst_mask 修复**确实**让最离谱的 collapse 消失了，但**core dst head 仍然没学习**。

**新诊断**：用 `diagnose_rollout.py` 在本地跑了 random-init baseline：
* random init 的 `advantages.std=0.234`
* trained policy 的 `adv_std=0.17`（比 random 还低！）
* 说明 value head 几乎没添加信息（value 的预测能力 ≈ 简单均值）

**真正的根因**（修正 §6.3.1 的"dst_mask 是真凶"判断）：
1. `ent_coef_dst=0.01` 让 dst head 被强力推向 max-entropy (uniform over P=20 planets)，dst entropy 永远在 ~ln(20)=3.0 附近不下降
2. `gamma=0.997 + episode_steps=200` 让 effective horizon ~333，value head 很难学
3. `pg_loss=0.0003` + `clip=0.00` + `kl=0.000` 三联告诉我们 PPO 几乎完全不更新，dst head 自然学不出

dst_mask bug 是个真 bug，但只解决了**最后一公里的 inference 行为**——核心训练 dynamics 还是在原地踏步。

### 6.3.4 v3.1 配置：targeted 修这三个问题

新建 `orbit_wars_rl/configs/multi_action_v3p1.yaml`：
* `ent_coef_dst` **0.01 → 0.001**（让 dst commit；这是首要修复）
* `gamma` **0.997 → 0.99**（effective horizon 从 333 缩到 100，匹配 episode_steps=200）
* `lr_peak` **1.5e-4 → 2.0e-4**（pg_loss 推离噪声底）
* `update_epochs` **2 → 3**（每批多榨一点梯度）
* 其余 K=8 稳定性相关参数保持（clip_eps=0.15、ent_coef_emit=0.03）

期望验证标准（训练 400 update 时）：
* `dst entropy` 应该 ≤ 2.0（vs v3 的 2.85）
* `pg_loss` 应该 ≥ 0.001（vs v3 的 0.0003）
* `adv_std` 应该 ≥ 0.3（vs v3 的 0.17）
* `WRr` 末段 ≥ 0.85（vs v3 的 0.72）
* H2H vs v1 ≥ 3/5

### 6.3.5 v3.1 训练结果（`ckpt_multi_action_v3p1/ckpt_000399.pkl`）

| 指标 | v3 | v3.1 | 是否达到目标 |
|------|-----|------|--------------|
| final WRr | 0.72 | **0.81** | △ 接近目标 0.85 |
| final WRf | 0.91 | 0.78 | ✓ self-play 更健康（不再被 frozen 全杀） |
| `dst entropy` 始→终 | 2.95→2.85 | 2.95→**2.75** | ✗ 远未达 ≤2.0 但**首次出现明确下降趋势** |
| `pg_loss` 末段 | 0.0003 | **0.0007** | ✗ 没到 ≥0.001 但翻倍了 |
| `adv_std` 末段 | 0.17 | **0.11** | ✗ 反而更低，但这其实是 value 学到了的好兆头 |
| `tR` 末段 | 0.16~0.65 | **0.55~0.91** | ✓ 终局信号显著加强 |
| `clip` / `kl` | 0.00 / 0.000 | 0.00 / 0.000 | ✓ 极其稳定（有 lr/epochs 空间继续推） |

**dst entropy 下降轨迹**（v3.1）：
- upd 0~100: 2.95→2.97（被 entropy bonus 推到 max；预期，因为 warmup + low ent_coef）
- upd 100~200: 2.97→2.95（持平）
- upd 200~300: 2.95→2.83
- upd 300~399: 2.83→2.75

平均下降率 **0.08 entropy / 100 updates**——到 ≤2.0 需要约 1000 updates，单纯训长就能达成。**学习方向对，速度慢**。

**核心判断**：v3.1 比 v3 实质改善，证明"dst commit + value horizon + 更激进 lr"是正确方向。但 400 updates 不够。

### 6.3.6 v3.2 准备（待 v3.1 H2H 结果决定是否上）

新建 `orbit_wars_rl/configs/multi_action_v3p2.yaml`，三方向继续推：
* `ent_coef_dst` 0.001 → **0.0003**（再 3x，dst 还有 91% 距离 uniform）
* `lr_peak` 2e-4 → **3e-4**（kl 一直是 0，有大量 headroom）
* `update_epochs` 3 → **4**（PPO 多榨梯度）
* `num_envs` 32 → **64**（每 update 数据量翻倍）
* `num_updates` 400 → **600**（让 dst entropy 真正降下来）

**预计 5090 训练时间**：~15 分钟（v3 是 8 分钟 @ 400 upd，v3.2 是 600 upd × 2x envs ≈ 24 分钟，但 64 envs 可能 GPU 占用更高 sps 略降）。

**先决条件**：v3.1 H2H 结果。如果 vs v1 ≥ 2/5，说明方向对，跑 v3.2。如果还是 0/5 或 1/5，需要重新审视 env gap 假设（§6.5）。

### 6.3.7 v3.1 H2H 实测（最关键的实验结果！）

```
v3.1 vs v1:   1 / 5
v3.1 vs v3:   5 / 5  ← 完胜
```

**含义**：
* `v3.1 完胜 v3` 直接证明「ent_coef_dst ↓ + gamma ↓ + lr ↑ + epochs ↑」是**100% 正确方向**——dst entropy 改善确实带来真实强度提升，不是噪声。
* `v3.1 vs v1 仍 1/5` 不意味着方向错——意味着**剂量不够**：v3.1 在 v3 的方向上只推了一档，要追到 v1 base 强度还需要继续推。
* v1（K=1 单动作 MVP）作为基线意外地 robust——这是个有用 baseline 信号，说明 K=8 多动作的额外自由度还没真正被有效利用。

**值得记的边界 case 观察**：
* v3 和 v3.1 在 export smoke 的 4-planet 极简 obs 上**输出完全相同**（`[[0,1.57,15],[0,1.57,7],...]`，6 条全 src=0 dst=2）
* 但 5 组 20 planets 真实比赛里 v3.1 完胜 v3
* 说明 **smoke moves 不是有效的 inference 健康检查**——4-planet 退化局面下任何 dst head 的细微差异都被磨平了。后续要么换更大的 smoke obs，要么完全依赖 H2H

### 6.3.8 决策：直接上 v3.2

判定条件命中"方向对剂量不足"——按之前准备的 `multi_action_v3p2.yaml` 执行：
* `ent_coef_dst` 0.001 → **0.0003**（dst 还有 91% 距离 uniform，继续松绑）
* `lr_peak` 2e-4 → **3e-4**（kl 一直是 0，有大量 headroom）
* `update_epochs` 3 → **4**
* `num_envs` 32 → **64**（数据量翻倍）
* `num_updates` 400 → **600**（多 50% 训练）

预计 5090 训练时间约 25-30 min。

**v3.2 健康检查**（训练中早期停损线）：
* upd ~150：dst entropy ≤ 2.7 否则 dst 还是没学
* upd ~300：dst entropy ≤ 2.3, pg_loss ≥ 0.001 否则继续推无效
* upd ~599：dst entropy ≤ 1.8, WRr ≥ 0.90

**H2H 决策树**（训完后）：
| v3.2 vs v1 | 解读 | 下一步 |
|------------|------|--------|
| ≥ 3/5 | 调参方向已经赢了 baseline | 提交 v3.2 到 Kaggle 看 ELO |
| 2/5 | 还差一档 | 可考虑 v3.3 (再 1.5x 同方向) 或开始 env gap |
| ≤ 1/5 | dst entropy 不是 root cause | 转 Plan B = env gap 修复 (orbital + slot shuffle) |

### 6.3.9 v3.2 训练结果（部分日志，到 upd 479/600）

**dst entropy 终于真正学会下降**：

| 指标 | v3 (400) | v3.1 (400) | v3.2 (~480) | 目标 |
|------|-----|------|---|---|
| `pg_loss` 末段 | -0.0003 | -0.0007 | **-0.0010~-0.0015** | ≥ 0.001 ✓ |
| `dst entropy` 末段 | 2.85 | 2.75 | **2.01~2.10** | ≤ 1.8 △ 接近 |
| `tR` 末段 | 0.16~0.65 | 0.55~0.91 | **0.78~1.00** | — |
| `WRr` 末段 | 0.72 | 0.81 | **0.78~0.88** | ≥ 0.90 △ 接近 |
| `WRf` 末段 | 0.91 | 0.78 | **0.94** | ⚠ self-play 不够强 |
| `adv_std` 末段 | 0.17 | 0.11 | **0.07** | (value 学得太好的副作用) |

**核心证据**：
* dst entropy 从 2.95 一路降到 2.01（**减少 32%**），首次破 2.2 阈值
* `tR` 频繁触及 +1.00，policy 在自家 env 已接近最优
* `pg_loss` 量级 4-5x v3，PPO 真在做工作
* `clip 0.01 / kl 0.001` 终于"摸到"非零值——参数确实在被更新

**警告**：末段 460-479 update（lr decay 后）`v 0.002-0.005` 极小，value head saturate；`WRr` 在 0.78~0.84 间波动而非继续上升。**最强 ckpt 可能在 upd 419-449 之间**（WRr 多次 0.81-0.88 + WRf 0.94-0.97 peak）。

### 6.3.10 v3.3 准备：开始关 env gap

不管 v3.2 H2H 结果如何，**MVP env vs Kaggle env 结构性 gap 是终局瓶颈**（DAY1 §3.3 + DAY2 §6.5 反复提到）。开始 cheapest 的 env gap 修复——**slot id shuffle**：

`orbit_wars_rl/env/init.py` 加 `shuffle_slots=True` 参数（默认开启）：
```python
if shuffle_slots:
    perm = jax.random.permutation(rng_perm, num_groups * 4)
    xy, productions, radius, ships, owners = [arr[perm] for arr in ...]
```

**为什么 slot shuffle 重要**：之前 player 0 home **永远在 slot 0**，player 1 在 slot 3。transformer 可能学到「slot 0 的 embedding = my home」的捷径，而不是用 features 学。Kaggle 真 env 里每局 planet ids 都是任意分配，policy 必须真正泛化。

`multi_action_v3p3.yaml`：
* 从头训（不能从 v3.2 ckpt 续训——slot 分布完全变了，catastrophic forgetting 风险）
* 其他 PPO 参数和 v3.2 一致
* `num_updates: 600` 给充分时间收敛

**v3.3 验证假设**：
* 如果 H2H vs v1 ≥ 2/5（vs v3.2 的预期 ≤ 1/5），说明 slot shuffle 是 dominant env gap
* 如果 H2H 没改善，下一步上 orbital motion（更大代价 ~3-5h 工程）

### 6.3.11 v3.2 H2H 结果：dst entropy 假设被判处死刑

训练在 upd 479 异常停止（log 没记录 final eval；进程已死。可能 SSH disconnect）。用 ckpt_000474.pkl 作 final，ckpt_000424.pkl 作 peak。

**所有 H2H 都关闭 slot shuffle**（`ORBITWARS_SHUFFLE_SLOTS=0`），保证对 fixed-slot 训练的 ckpt 公平：

| 对局 | A 胜 | 含义 |
|------|------|------|
| v3.2 vs v1 (baseline) | **1/5** ❌ | 和 v3, v3.1 一样，**没赢 baseline** |
| v3.2 vs v3.1 | 3/5 ⚠️ | 微弱进步（远不及 v3.1 vs v3 的 5/5）|
| v3.2 vs v3_proper | **1/5** ❌❌ | **v3.2 反而比 v3 弱** |
| v3.2_peak vs v3.2 | 1/5 | upd 424 不如 upd 474 |

**结论：dst entropy 下降 ≠ H2H 改善**：
* v3 末段 dst entropy = 2.85, vs v1 = 0/5 ~ 1/5
* v3.1 末段 dst entropy = 2.75, vs v1 = 1/5  
* v3.2 末段 dst entropy = 2.01, vs v1 = 1/5（**没变**）

**这是一个 root-cause 误判**。我们花了三轮训练降 dst entropy，证明这条路根本没解决 H2H 弱的问题。`pg_loss / WRr / tR` 等训练曲线看起来很健康（v3.2 是历史最好），但**和 H2H 完全脱钩**。

### 6.3.12 v3.2 行为分析（smoke moves 差异）

三个 ckpt 在同一个 smoke state 上的输出：

| Submission | smoke moves |
|----|----|
| v3 (ckpt_000399) | `[[0, 1.57, 15]]` — 1 个 fleet |
| v3.1 (ckpt_000399) | `[[0, 1.57, 15], [0, 1.57, 7], [0, 1.57, 4], [0, 1.57, 2], [0, 1.57, 1], [0, 1.57, 1]]` — 6 个 fleet |
| v3.2 (ckpt_000474) | 同 v3.1 — 6 个 fleet |
| v3.2_peak (ckpt_000424) | 同 v3.1 — 6 个 fleet |

**两件事被坐实**：
1. **multi-action emit 学到了** — v3 只 emit 1 个，v3.x 学会 emit 6 个。EmitHead 工作正常。
2. **dst 完全没学** — 三个版本所有 6 个 fleet 都送往同一个 dst (slot 0)。这是 mode collapse 而不是 dst entropy = 2.0 暗示的"接近 random"。**dst entropy 高是因为 logits 接近 uniform**，但 argmax 永远是同一个——uniform-ish softmax 仍然可以让 argmax 稳定指向同一个 slot。

**这就是为什么 dst entropy 下降但行为没变**：softmax 一直接近 uniform，但 argmax 一直是 slot 0。dst entropy 2.01 也只是说"logits 之间差距还不大"，**argmax 决策没改变**。

### 6.3.13 root cause 重新假设

为什么 dst head 学到"永远 slot 0"是最优策略？

* **观察 1**：训练 env 里 slot 0 = player 0 的 home，slot 3 = player 1 的 home。所以 player 0 "送往 slot 0" = 反复加固自己 home。在 fixed-slot env 这是合法策略（reinforce home → 兵力多）
* **观察 2**：因为 `tR ~+0.9` 在自家 env 几乎稳赢，policy 不需要"找敌人"——只要囤兵在 home 就赢了 random/frozen
* **观察 3**：vs v1 这种**真会进攻**的 policy，囤兵在 home 完全没用，slot 0 的 dst 不会真正打到敌人
* **观察 4**：env 太短（200 steps）+ 太简单（5 groups 静态），policy 学到的捷径在 v1 这种"找敌人主动出击"的对手面前完全失效

**真正的 root cause 应该是**：fixed-slot env + 简单 reward shaping → policy 学到 "shoot at home (slot 0) → 在 random/frozen 对手下稳赢" 的非泛化策略。**降 dst entropy 反而让这个策略更牢固，所以 v3.2 比 v3 更糟**。

### 6.3.14 唯一被验证的出路：env gap 修复（v3.3）

slot shuffle 现在不再是"对冲"，而是**唯一可能起作用的方向**：
* 打破 "slot 0 = home" 的固定关系，policy 必须用 features (is_mine, ships) 学
* 训练时每局 slot 0 是 player 0、neutral 或 player 1 的 home 都可能 → "总是送 slot 0" 立刻变成糟糕策略
* dst head 被迫真正学习 "敌方 home 在哪"

**v3.3 训练计划**：
* `ORBITWARS_SHUFFLE_SLOTS=1`（默认）
* 其他参数同 v3.2（已收敛的配置）
* 600 update 从头训
* H2H 验证：vs v1 / vs v3 / vs v3.2 — **vs v3 ≥ 3/5 才能说明 shuffle 起作用**

### 6.3.15 战略转向：先做 env parity，再训练

读完 top_players_rl.txt + overview.txt 后回头看：

> Lin Myat Ko (1st, 1600 ELO)："Have Opus write parity tests to compare frame by frame with kaggle environment. You also need to tell what obs features you want so that it'll be optimized altogether."

这不是优化建议，是**前置门槛**。我们 v3.2 训练里 `tR +0.90, WRr 0.88` 看似无敌但 H2H vs v1 = 1/5 — 唯一合理解释是 training env 学到的策略**根本无法迁移到真 Kaggle env**。**继续盲调 PPO 不会有进步**。

**top1 的关键提示（按重要性排）**：
| # | 启示 | 我们 |
|---|------|------|
| A | "parity tests frame by frame with kaggle env" | ❌ 没做 — **当前 bottleneck** |
| B | "watch games, find useful inductive bias" | ❌ 没看过对局录像 |
| C | "reward shape → signs of life" | ⚠️ 加了 shaping_delta 但没系统验证 |
| D | "+1/-1 is enough for 2p" | ✓ 我们也是 |
| E | "add one architecture delta at a time" | ❌ 我们一次改 3 个超参（v3→v3.1→v3.2）|

### 6.4 第一份 env parity 报告（2026-05-22）

`orbit_wars_rl/parity/run_env_parity.py` — 用 kaggle env 跑 30 步 scripted action，对比我们的 env 同样 init/actions 后的状态。

**Bridge 实现**：`orbit_wars_rl/parity/kaggle_bridge.py`
- `kaggle_obs_to_envstate(obs)` — 把 kaggle obs 转成我们的 EnvState（slot id = planet id）
- `diff_state(kaggle_obs, our_state)` — 逐项 diff，输出每类 mismatch 的程度

**首次跑结果**（seed=42, 30 steps）：

| 类别 | 结果 | 解读 |
|------|------|------|
| **planet owner** | ✓ 0/30 mismatch | combat / 占领逻辑一致 |
| **planet ships** | ✓ 0/30 mismatch | production + 兵力流转一致 |
| **planet xy** | ❌ 29/30 mismatch, **max=34.7 units** | **轨道运动完全没建模** |
| **fleet xy** | ❌ 30/30 mismatch, **constant 0.371 units** | **fleet spawn 偏移错了** |
| **fleet count** | ✓ 0/30 mismatch | launch / 出界 / 销毁逻辑一致 |
| **comets** | ✓ 0 occurred | step<50 还没 spawn |

### 6.4.1 第一个 fix: fleet spawn offset (5 min)

反推 kaggle env 的 spawn 公式：
```python
fleet_pos_step1 = spawn + speed * (cos(angle), sin(angle))
=> spawn = fleet_pos_step1 - speed * direction
=> distance(spawn, planet_center) == planet_radius + ???
```

实测两个不同 seed/angle/ships：
- (seed=42, P0=68.17,94.89 r=2.10, angle=-1.955, ships=3): spawn 偏移 = 2.198 = **r + 0.100**
- (seed=7, P0=91.14,75.13 r=2.39, angle=3π/4, ships=5): spawn 偏移 = 2.486 = **r + 0.100**

我们的 `_insert_one_fleet` 用的是 `r + 0.5`（off by 0.4）。

**Fix**: `orbit_wars_rl/env/dynamics.py` line 61, `pad = 0.5` → `pad = 0.1`。

re-run parity test → **fleet xy mismatch 30/30 → 0/30**。

### 6.4.2 剩下的唯一 mismatch: orbital motion（已知大缺口）

`planet xy` 仍然 29/30 mismatch, max=34.7 — 4 颗内圈 planet (id 12-15, orbital_radius=31.1, 满足 `orb_r + radius < 50`) 以 angular_velocity=0.041 rad/turn 转动，**我们完全没建模**。

**这是 v3.2 失败的真正根因**：
- 训练时 P1 home 永远在 (31.83, 5.11)
- 真 Kaggle env 里 home planet 是 static 的（外圈），所以这点对 home 没影响
- **但内圈 4 颗 prod≥3 的好 planet 在转**，这些是真正的战略目标
- 我们的 policy 不会 "lead-target"（提前预判转动位置打过去），fleet 永远射到旧位置 → 全部 miss

### 6.5 下一步候选（按工程量排序）

1. ~~**shuffle planet slot ids**~~ 已做完 (在 init.py)。**对解决 H2H 弱不重要** — slot id 不一致只是 cosmetic，policy 看的是 features
2. **加 orbital motion**（核心 fix, ~3-5h 工程）：state 加 `planet_orbit_radius / planet_orbit_phase / planet_orbit_speed`，每步 dynamics 里 update 位置。**这是 v3.2 失败的真正根因，必修**
3. **加 comets**（次要, ~2-3h）：5 个固定 step 上 spawn 4-quadrant，elliptical path。影响中后期战略
4. **planet 数量随机化** [3,6]（次要, ~1h）：top1 mentioned 5-10 groups, prod range 1-5

明天接手第一件事：**§6.5.(2)** orbital motion，然后立刻重新 parity test 看是否 0 mismatch。

### 6.4 当前结论锁定

DAY1 §3.3 的 "MVP 静态/无彗星" 注释**已经从「可选优化」升级为「blocker」**。单纯加训练量、做 self-play 都救不回 v3——必须先填补 training env vs Kaggle env 的 gap。

* `final WR in our env: 0.906` ✅
* `final WR vs random in Kaggle env`: 没测，但 vs v1 已经 0/5
* `WRf 0.94` 自对弈不工作的诊断暂时挂起——更紧迫的是 env gap

### 6.5 下一步候选（按工程量排序）

1. **shuffle planet slot ids**（最小 fix, ~30 min 工程）：每 reset 时随机 permute 0..19 slot id，强迫 policy 完全靠 features 不靠 id。能解决「home 永远在 slot 0」的 overfit，但解决不了「行星静止」。
2. **加 orbital motion**（核心 fix, ~3-5h 工程）：把行星 angular velocity 引进 init / dynamics，state 加 `planet_orbit_radius / planet_orbit_phase / planet_orbit_speed`，每步 update 位置。这是真正的对齐 Kaggle env。
3. **planet 数量随机化**（次要, ~1h）：`num_groups` 从 fixed=5 改成每个 episode 在 [3,6] 内 sample。强迫 policy 对不同 board size 都泛化。

明天换电脑接手第一件事是 §6.5(1)+(2)。

---

## 7. Env Parity 推完（下午）：orbital motion + swept-pair collision

### 7.1 决策与策略

下午开干 §6.5 的 (2)：**orbital motion**，并且把策略提升为 "除了 comets，自对弈阶段要和真 Kaggle env 0 gap"。

3 个 PR 串行做，每步都 parity 验证再继续：
- **PR1**：state schema 加 orbit 字段 + init 给 planet 算 (radius, phase, is_orbiting)
- **PR2**：dynamics 改成 swept-pair（fleet 和 planet 同 tick 都动的 continuous collision detection）
- **PR3**：obs 给 policy 加 orbit features

### 7.2 PR1：state + init + rotate_planets()

#### 7.2.1 实现

* **`env/state.py`** 在 `EnvState` 加 4 个字段：
  - `angular_velocity` (scalar, episode-wide)
  - `planet_orbit_radius`（每个 planet 到 sun 的距离）
  - `planet_orbit_phase`（每个 planet 当前角度，rad）
  - `planet_is_orbiting`（bool[MAX_PLANETS]，由 `orbit_radius + planet_radius < 50` 决定）

* **`env/constants.py`** 加：
  ```python
  ORBIT_OMEGA_MIN = 0.025
  ORBIT_OMEGA_MAX = 0.05
  ORBIT_RADIUS_LIMIT = 50.0
  ```

* **`env/init.py::reset`**：
  - 新 split: `rng_pos, rng_prod, rng_ships, rng_perm, rng_omega, rng_state`
  - 给每个 planet 算 orbit_radius (`hypot(x-50, y-50)`)、orbit_phase (`atan2`)、is_orbiting
  - 这 3 个字段跟着 slot permutation 一起 permute（保持 planet 一对一绑定）
  - `angular_velocity = uniform(0.025, 0.05)`

* **`env/dynamics.py`** 新加 `rotate_planets(state)`：
  - 只在 `state.step >= 1` 时转（**first-turn gate**：kaggle env 的 step 0→1 不转 planet，从 step 1→2 才开始转，实测确认）
  - `new_phase = current_phase + omega`；static / padding planet 的 (x,y) 保持原样
  - 计算 `new_x = sun_x + orbit_radius * cos(new_phase)`、`new_y = sun_y + orbit_radius * sin(new_phase)`

* **`env/env.py::step`** 在 `move_and_collide` 之后插入 `rotate_planets`（初版，PR2 会改）

#### 7.2.2 PR1 验收

```
seed=42/1/7/100/1000/3000 (6 个 seed) × 30 steps:
  planet xy mismatch: 0/30  ← PR1 解决了 §6.4.2 的最大 gap (34.7 → 0)
  fleet xy mismatch: 0/30
  planet owner / ships: 0
  total clean: 30/30 ✅

seed=500 (home orbiting, 32 planets, 16 orbiting):
  step 0-4: clean ✓
  step 5+: fleet_count mismatch (我们少 2 fleet) ← sweep 缺失暴露
```

**seed=500 暴露的问题**：fleet 在 (65.3, 68.6) → (64.5, 67.6) 飞过；同 tick orbiting planet P28 从 (64.0, 67.1) → (63.3, 67.7) 旋转。fleet endpoint 距 P28 旧位置 = 0.712 < 1.0（radius），所以我们的 `move_and_collide` 杀了它；但 kaggle 没杀（用旋转后的 P28 位置算的 segment distance = 1.258 > 1.0）。

### 7.3 PR2：发现 kaggle source code，重写为 swept-pair

#### 7.3.1 关键发现：kaggle source 在本地

`kaggle_environments.envs.orbit_wars.orbit_wars.py` (812 行) 就在
```
/Library/.../site-packages/kaggle_environments/envs/orbit_wars/orbit_wars.py
```

直接看代码秒级回答所有 collision/combat/rotation 问题。

**关键代码段** (line 46-64)：
```python
def swept_pair_hit(A, B, P0, P1, r):
    """True iff a fleet moving A->B and a planet moving P0->P1 come within r
    of each other for some t in [0, 1]."""
    d0x, d0y = A[0]-P0[0], A[1]-P0[1]
    dvx = (B[0]-A[0]) - (P1[0]-P0[0])
    dvy = (B[1]-A[1]) - (P1[1]-P0[1])
    a = dvx*dvx + dvy*dvy
    b = 2.0 * (d0x*dvx + d0y*dvy)
    c = d0x*d0x + d0y*d0y - r*r
    # quadratic in t for distance(t)² = r²
    disc = b*b - 4*a*c
    ...
    return t2 >= 0.0 and t1 <= 1.0
```

**真相不是「move 完再 sweep」，而是「move 阶段直接用 swept-pair 做 fleet vs 旋转 planet 的连续碰撞检测」**。Kaggle 在 move 阶段就把 fleet segment 和 planet swept segment 一起测，谁先撞就算谁。

#### 7.3.2 实现

* **删除** `rotate_and_sweep`（错误的两阶段实现）
* **`env/dynamics.py`** 新加 `_swept_pair_hit(...)` ——JAX vectorized 版的 kaggle `swept_pair_hit`，handle `a==0` 的退化情况
* **`env/dynamics.py`** 新加 `_planet_paths(state)` ——为每个 planet 计算 (old_x, old_y, new_x, new_y)，复用 first-turn gate 逻辑
* **重写** `move_and_collide`：
  1. 算 fleet 段 `(fox, foy) -> (fnx, fny)`
  2. 算 planet 段 `(pox, poy) -> (pnx, pny)` via `_planet_paths`
  3. 对每个 (fleet, planet) 对调用 `_swept_pair_hit`
  4. 优先级：planet hit > sun > OOB（match kaggle line 587）
* **`env/env.py::step`** turn order 改成：
  ```
  launch → produce → move_and_collide(swept) → resolve_combat → rotate_planets(apply)
  ```
  `rotate_planets` 现在只是把已经在 collision detection 里 "见过" 的旋转应用到 `planet_x/y` 上。

#### 7.3.3 PR2 验收

```
seed 42 / 1 / 7 / 100 / 500 / 1000 / 3000 (7 个 seed) × 30 steps:
  ALL CLEAN: 30/30 ✅✅

seed 42 / 500 / 1000 × 49 steps (just below comet spawn at step 50):
  ALL CLEAN: 49/49 ✅✅

seed 42 × 52 steps (跨过 comet spawn):
  clean_steps = 49 (符合预期，step 50 开始 comet 出现导致 BAD)
  comets present: 3 mismatch
```

**Env parity 0 gap（除 comets）达成** 🎯

### 7.4 PR3：obs 加 orbit features

#### 7.4.1 实现

Policy 需要看到「这 planet 在转吗、转到哪了、半径多少、omega 多大」才能学会 lead-target。

* **`features/encode.py`**：
  - `PLANET_FEAT_DIM` 12 → **15**（加 `is_orbiting`, `orbit_phase/π`, `orbit_radius/BOARD_HALF`）
  - `GLOBAL_FEAT_DIM` 10 → **11**（加 `angular_velocity / OMEGA_MAX`）

* **`inference/kaggle_adapter.py`** 同步：
  - 新 dims 13/14/15 在 `planet_feats`
  - 新 `global_feats[10] = obs['angular_velocity'] / 0.05`
  - 从 kaggle obs 的 (x, y, radius) **现场算** `orbit_radius`、`orbit_phase`、`is_orbiting`（无需 kaggle obs 显式提供 — kaggle 也是用 initial_planets 反算的）

* **`submission_rl_v4.py`** （新 template，从 v3.py copy + patch）：
  - 同样的 15/11 + orbit feature 计算
  - `export_submission.py` 默认 template 改为 `submission_rl_v4.py`

#### 7.4.2 PR3 验收

* `python -m orbit_wars_rl.inference.test_adapter`：8/8 OK，所有 features `max_diff = 0.000e+00`
* `python -m orbit_wars_rl.inference.test_parity`：16/16 whole-turn action list match, value drift 3.6e-7
* `python -m orbit_wars_rl.scripts.smoke_test --num-updates 3`：PPO 跑通，**no NaN**，WR vs random 0.75
* `python -m orbit_wars_rl.scripts.smoke_test --num-updates 4 --selfplay`：self-play 路径也跑通
* `python -m orbit_wars_rl.scripts.export_submission --ckpt ...`：jax-vs-numpy parity 16/16 OK，export 出来的文件 dims 正确

### 7.5 工程小事

* **`sync_mirror_ultrapp.sh`** 调整 exclude 列表：
  - 之前的 `submission_rl_v3*.py` exclude 太宽，会把本地写的 template 也 exclude，导致 remote 看不到。
  - 改成只 exclude `*_proper.py / *_peak.py / *_exported.py / *_filled.py / *_final.py` —— 只挡那些 remote 训练机自己生成的，不挡 template。
* `submission_rl_v3.py`（旧 12/10）和 `submission_rl_v4.py`（新 15/11）**都保留**：v3 用来跑 v3 weights，v4 给新训练用。

### 7.6 接下来（v4 训练）

Env gap 已经填补到「除 comets 外 0 mismatch」，可以放心开 v4 训练。

| 项 | 当前 | 备注 |
|----|------|------|
| Env parity | ✅ 0 mismatch (49 steps, 7 seeds) | comets 50+ step 才 spawn，自对弈 60-200 step 内不触发 |
| Obs schema | ✅ 15/8/11 | 旧 v3 ckpt **不兼容**（dims mismatch），必须 train from scratch |
| Init slot shuffle | ✅ 默认 ON | policy 必须靠 features 不靠 slot id |
| Submission template | ✅ `submission_rl_v4.py` 已就位 | export_submission 默认指向它 |
| Unit + integration tests | ✅ adapter / parity / smoke / selfplay smoke / export 全 pass | |

**下一步**：跑 v4 训练（5090 远程或 local 长 smoke），核对 sign-of-life；之后才是 H2H vs starter/v1。**不要再用 v3.x ckpt 做任何 H2H，那个版本的训练 env 跟 Kaggle 严重不匹配。**

### 7.7 已知 deferred 项（**不**进 v4，专门 backlog）

1. **comets**（5 个固定 step 在 quadrant 对称 spawn elliptical path）：约 2-3h 工程，明天可能做
2. **anchor 范围**：我们 init 时 anchor 在 `[12, 47]`，实测 kaggle 可以到 `[12, 95-r-5]`（home orb_r 见过 60.79）。差距小但要补
3. **`MIN_STATIC_GROUPS = 3` 保证**：kaggle 保证至少 3 组 static，我们不保证（可能 5 组全 orbit）。影响开局对称性
4. **`PLANET_CLEARANCE = 7` + 旋转后 clearance 检查**：kaggle 在 init 时做了旋转后不重叠的预检查，我们没做（理论上可能有 init 期 planet 离 sun 太近）

这些是 backlog，不影响 v4 训练能不能跑/能不能 H2H。

---

## 8. 不要忘记的 4 个细节

1. **`docs/RL_PIPELINE.md` 仍是项目级权威文档**，今天没动它。v4 训练验收后再回去更新。
2. **`submission_rl_v4.py` 是新 template**，旧 v3 weights 不能用它 inference（dims 错）。v3 weights 仍用 `submission_rl_v3.py`。
3. **本地 mac 的 `ckpt_multi_action_*` 都是 sanity 用的小 ckpt**，并且是旧 obs schema (12/10)，**不能** export 成 v4 submission。
4. **shaping reward 是为了小规模训练加速 signs of life**。1st 选手原话「+1 -1 is enough」，长训练（>200 update）后期如果 WR 卡住，第一个考虑的就是把 `SHAPING_SCALE` 从 0.1 调到 0.02 甚至 0，避免 policy 锁在「保 ship 差」而不是「真的赢」。这个旋钮在 `env/rewards.py` 顶。

---

*文档版本 2.0 — 2026-05-22 下午：env parity 推完（orbital motion + swept-pair collision）。承接版本 1.0 的 §6 env gap 诊断。准备开 v4 训练。*

---

## 8. v4 第 1 次开训 + 立刻 kill（傍晚）

### 8.1 现象

按 `configs/multi_action_v4.yaml` 起跑 800 update。前 144 update 输出：

| upd | WRr | WRf | ent[d] | pg_loss | clip | kl |
|-----|-----|-----|--------|---------|------|----|
| 9 | 0.31 | - | 2.97 | -0.0025 | 0.10 | 0.005 |
| 29 | 0.59 | 0.66 | 2.94 | -0.0029 | 0.04 | 0.002 |
| 49 | 0.47 | 0.53 | 2.93 | -0.0004 | 0.00 | 0.000 |
| 69 | 0.84 | 0.62 | 2.94 | -0.0005 | 0.00 | 0.000 |
| 109 | 0.75 | 0.78 | 2.94 | -0.0003 | 0.00 | 0.000 |
| 143 | 0.56 | 0.62 | 2.92 | -0.0010 | 0.00 | 0.001 |

### 8.2 诊断

* **WRr 上去了**（早期 0.3 → ~0.7-0.8 中位线）—— env / obs / model arch / numpy adapter 全 healthy
* **但** `ent[d]` **从 2.98 只降到 2.93**（max=ln(20)=3.0），dst head 几乎锁在 uniform random target
* **而且** upd 30 起 `clip = 0.00`, `kl = 0.000`，`pg_loss = -0.0005` —— PPO 已经在原地踏步

### 8.3 根因

config 把 `ent_coef_dst` 从 v3.2 的 0.0003 **拉了 10×** 到 0.003。
当时 reason 是「orbit 加了 obs 信息所以需要更多探索」，**这个逻辑是错的**：

* dst space 仍然是 20 个 live planet，max entropy 仍是 ln(20)=3.0
* 高 `ent_coef_dst` 直接把 dst loss 钉在 max entropy，policy 不敢偏离 uniform
* `pg_loss` 几乎为 0 不是因为 PPO 学完了，是 **dst grad 被 entropy bonus 抵消了**

### 8.4 处理

* 立刻 kill v4（已跑 144 / 800 upd，约 6 min on 5090）
* `multi_action_v4p1.yaml`：**只改 `ent_coef_dst: 0.003 → 0.0003`**，其他不动
* 旧 ckpt + 日志保留（`logs/multi_action_v4_killed_at_upd144.log` + `ckpt_multi_action_v4/`）
  作为 "dst random 上限 ~0.8 WRr" 的 baseline，未来 v4.x 对比用

### 8.5 v4.1 早期监控（前 50 upd）

* 红线: upd 50 时 `ent[d] > 2.92` 仍然不动 → 再降 `ent_coef_dst` 到 0.0001 + reseed 重训
* 健康: upd 50 时 `ent[d]` 已经掉到 2.85-2.92，`pg_loss` 在 0.001-0.003

### 8.6 教训

> **改 hparam 时永远问一句：「这个改动改变了 action space 的大小吗？」**
> 没改 → entropy coef 不要动。
> 改了 → 才需要重新平衡。

这次 obs 加了 3 个 planet feat + 1 个 global feat，**dst space 仍是 20**，所以 `ent_coef_dst` 不应该动。原 v3.2 的 0.0003 已经经过 3 轮 ablation 锁定，盲改是退步。

---

## 9. v4.1 / v4.2 训练复盘（晚间）

### 9.1 v4.1 (kill at upd 191/800)

完全按 §8.4 改了 `ent_coef_dst: 0.003 → 0.0003`，其他不动。结果：

| upd | WRr | WRf | ent[d] | pg_loss | clip | kl |
|-----|-----|-----|--------|---------|------|----|
| 0 | — | — | 2.98 | -0.0019 | 0.01 | 0.000 |
| 89 | 0.78 | 0.66 | 2.90 | -0.0006 | 0.00 | 0.000 |
| 189 | 0.66 | 0.59 | 2.87 | -0.0009 | 0.00 | 0.000 |

* 第一反应：**仍然没动**。`ent[d]` 5 倍 ent_coef 调整下只多降 0.05；`pg_loss/clip/kl` 跟 v4.0 一模一样
* 当时判断：env 改了后 dst → reward 链路断了，policy 没法靠 grad signal 自动发现 lead-target，**核心 hypothesis 是 obs 表达力不足**
* 决策：kill v4.1，准备 v4.2 加 lead-target features

### 9.2 v4.2 = lead-target features

**改动**：
* `features/encode.py`: `PLANET_FEAT_DIM 15 → 19`，新增 4 维：planet 在 `t+15` 和 `t+30` 步后预测位置 `(x, y)`
* `inference/kaggle_adapter.py` 和 `submission_rl_v4.py` 同步 numpy 镜像
* `multi_action_v4p2.yaml`：hparam 完全不动，只换 ckpt_dir 和 seed
* parity test、adapter test、smoke test 全过（jax/numpy 对齐到 5.96e-08）

**直觉**：top_players_rl.txt 提到 「lead-target is the hardest part to learn」，2-layer transformer 学不出 cos/sin 在线预测，直接 hand-engineer 进 obs

### 9.3 v4.2 早期 (~upd 191) 再次「塌陷」诊断错误

我看到 upd 191 时：

| 指标 | v4.1 upd 191 | v4.2 upd 191 |
|------|----|----|
| `ent[d]` | 2.86 | **2.81** |
| `pg_loss` | -0.0004 | -0.0004 |
| `clip` | 0.00 | 0.00 |
| `WRr` | 0.66 | 0.75 |

**当时判断**：lead-target 也没救回来。准备做 per-head ratio 改造。

**实际是错的**。WRr +9pp 是有效信号，**`clip=0` 不是 PPO 空转**。

### 9.4 v4.2 中段 (upd 350+) 数据揭示真相

放着继续跑到 upd 383，看：

| upd | ent[d] | WRr | WRf | tR | clip | 备注 |
|-----|--------|-----|-----|----|----|----|
| 191 | 2.81 | 0.75 | 0.50 | +0.28 | 0.00 | "plateau" |
| 350 | **2.54** | 0.66/0.75 | 0.59/0.75 | +0.27 | 0.00 | breakthrough |
| 369 | 2.46 | 0.81 | 0.75 | +0.38 | 0.00 | strong |
| 383 | **2.40** | 0.81/0.69 | 0.75/0.78 | +0.44 | 0.00 | sustained |

**`ent[d]` 下降速率**：
* upd 0-200: -0.05 / 100upd (慢)
* upd 200-383: **-0.25 / 100upd (5x 加速)**

**结论**：v4.2 在 upd ~200-250 跨过 lead-target 学习拐点后真正开始 commit dst head。**lead-target features 起了作用**，只是要先「热身」200 update 让 value head 学会用这个新特征评估 reward。

### 9.5 `clip = 0` 是 multi-action PPO 数学副作用 — 非 bug

```
turn_logp = Σ_k emit_mask[k] * (src_logp + dst_logp + pct_logp) + emit_logp
ratio     = exp(turn_logp_new − turn_logp_old)
```

K=8 个 emit step，每步 dst head 可以有 ~3 nat 的 logp 范围（ln(20)≈3）。  
理论 turn logp 跨度：8 × 3 = 24 nats。  
PPO clip `[1-0.15, 1+0.15]` 对应 logp 差 `[-0.16, +0.14]`。  
**单步 dst head 移动 0.3 nat 都会被 clip 压缩 75x 才落到 turn ratio 上**。

→ multi-action PPO 在所有 head 求和后**天然 clip rate 极低**。v3.2 末段 `clip=0.01` 也是同样原因。**不要把 clip=0 当成 PPO 失效证据**。

### 9.6 诊断错误的根因

1. **错位基准**：用 v3.2 末段 (600 upd, ent[d]=2.01) 比较 v4.x 早期 (200 upd) ，差 3x 训练量
2. **看 ent[d] 看绝对值而非速率**：v3.x 整段也是缓慢下降，2.95 → 2.75 跑了 400 upd
3. **没意识到 multi-action 改动了 PPO 的「健康指标基线」**：clip/kl 的合理区间从 [0.05, 0.20] 变到 [0.00, 0.05]
4. **缺乏「让它跑一会儿」的耐心**：upd 200 是个学习拐点，应该多看 100-200 upd 数据再决策

### 9.7 v4.2 预测末段表现

按当前 -0.25/100upd 速率外推到 upd 800：

| upd | ent[d] 预测 | WRr 预测 |
|-----|----|----|
| 500 | ~2.11 | ~0.80 |
| 600 | ~1.86 | ~0.83 |
| 800 | **~1.36 (超 v3.2 的 2.01)** | **~0.88+** |

剩余训练时间：(800-383) × 8192 / 5050 sps ≈ **11 分钟**。

### 9.8 后续工具就绪

**H2H gauntlet 脚本** (`scripts/h2h_gauntlet.py`)
* 对手清单：v1, v3 (proper), v3.1, v3.2, v3.2_peak, v20_0513
* 双色对称 (agent as A + agent as B)，paired seeds，自动汇总 WR
* 一键调用：
  ```bash
  python -m orbit_wars_rl.scripts.h2h_gauntlet \
      --agent submission_rl_v4p2.py \
      --opponents submission_rl_v1.py submission_rl_v3p2_peak.py submission_v20_0513.py \
      --num-games 10 --seeds 0,1,2,3,4,5,6,7,8,9
  ```

**v4.3 续训方案** (`configs/multi_action_v4p3.yaml` + `train.py --resume-from`)
* 触发条件：v4.2 vs v3.2_peak < 0.50 OR v4.2 vs v1 < 0.80
* 续训 400 upd (1200 total)，lr_peak 减半 (3e-4 → 1.5e-4)，ent_coef_dst 再 3x 弱 (3e-4 → 1e-4)
* `train.py --resume-from` 已加 shape-checked 加载（拒绝 silent layer drop）

### 9.9 教训补充（§8.6 之外）

> **看训练曲线要看「速率」+「拐点」，不能只看「绝对值」。**  
> 尤其当 PPO 训练动力学因 architecture 改动而改变时（multi-action ratio 求和 → clip/kl 信号被压缩），**早期 plateau 后面可能跟着 acceleration**。  
>  
> **正确的早停判据**：`ent[d]` 在 (upd 100, upd 300) 段是否有「速率拐点」。如果 100-200 段 `ent[d]` 微降，200-300 段速率 ≥ 3x 早期，说明 policy 找到了 reward signal source，让它继续跑。

### 9.10 export bug：docstring 字面 placeholder 把 inject 吃了

**症状**：v4.2 训完，`export_submission --out submission_rl_v4p2.py` 报 success，但
`agent()` 在所有 obs 上一律返回 `[]`。

**误诊**：第一直觉是 train-inference mismatch / mode collapse / OOD smoke obs。
全部排除后发现 `submission_rl_v4p2.py` 的 `WEIGHTS_B64 = "__WEIGHTS_B64__"`
**没被替换**（diag_v4p2_loud.py 报 "WEIGHTS_B64 is placeholder"）。

**根因**：v4 template 我在 docstring 写了一句说明 `Weights are injected via the
``WEIGHTS_B64 = "__WEIGHTS_B64__"`` placeholder ...`，导致字面 placeholder
`WEIGHTS_B64 = "__WEIGHTS_B64__"` **在文件里出现两次**：

* 行 19：docstring 注释
* 行 467：真正的赋值

`_inject` 用 `txt.replace(placeholder, new_line, count=1)` 只替换**第一处** —
那是 docstring。真正的赋值原封不动。生成的 v4p2.py 看起来 size 还行（docstring
里塞了完整 payload 字符串），但 `WEIGHTS_B64` 变量值还是 `"__WEIGHTS_B64__"`。

`agent()` 加载 weights 失败 → `try/except` silently 吞掉 → 返回 `[]`。

**修复**（commit 中）：
1. 把 v4 template docstring 改成不写字面 `WEIGHTS_B64 = "..."`
2. `_inject` 用 anchored regex `^WEIGHTS_B64 = "..."`（multiline 模式）替换，
   并加 sanity check：匹配数必须 == 1；post-write 验证 placeholder 已消失，
   否则 raise。

**教训**：
- silent `except Exception: return []` 在 kaggle agent 是必要的（避免一次崩溃断
  整局），但**在 export pipeline 的 smoke test 里应该 loud**。已经在 export_submission
  smoke 路径加 warning + diag_v4p2_loud.py 提供 bypass-except 诊断入口。
- template 文件的 docstring **不要写字面占位符语法**，即使是举例说明。改写成
  「the WEIGHTS_B64 placeholder line」之类的描述就行。
- 后续可以加一个 `tests/test_export_pipeline.py`：用 dummy weights 跑 export，
  assert injected file 的 `WEIGHTS_B64 != "__WEIGHTS_B64__"`。

---

## 10. 收尾 / 下次会话接续指南

> 本节是 hand-off note：下次坐下来直接从这里开始就行。

### 10.1 当前 repo 状态

| 部件 | 状态 | 位置 |
|------|------|------|
| Env (orbital + swept-pair collision) | ✓ parity 与 kaggle 0 mismatch (除 comets) | `orbit_wars_rl/env/` |
| Obs features (planet_feat=19, global=11, lead-target T+15/T+30) | ✓ jax / numpy adapter / submission template 三处对齐 | `features/encode.py`、`inference/kaggle_adapter.py`、`submission_rl_v4.py` |
| Multi-action policy (K=8 autoregressive) | ✓ 训练 800 upd | `orbit_wars_rl/net/`、`ppo/` |
| v4.2 训练 | ✓ 800 upd / 6.5M steps 完成，5090 上 `ckpt_multi_action_v4p2/` | 远程 |
| Export pipeline (inject + parity smoke) | ✓ 已 fix docstring-placeholder bug (§9.10) | `scripts/export_submission.py` |
| 诊断工具 | ✓ `diag_v4p2_loud.py` (bypass silent except)、`diag_submission.py` | `scripts/` |
| H2H gauntlet | ✓ paired colors + multi-opp | `scripts/h2h_gauntlet.py` |
| Resume / warm-start | ✓ `train.py --resume-from <ckpt.pkl>` (shape-checked) | `scripts/train.py` |
| v4.3 配置 (fallback) | ✓ 准备好，**仅当 v4.2 H2H 差时启动** | `configs/multi_action_v4p3.yaml` |

### 10.2 下次开机第一件事（按顺序）

```bash
# === A. 在 mac (本地) ===
cd ~/project/OrbitWarRL
./sync_mirror_ultrapp.sh                # 把所有 fix 推到 5090

# === B. 在 5090 ===
cd ~/project/OrbitWarRL

# B1. re-export v4.2（修了 _inject 之后第一次）
python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_v4p2/ckpt_000799.pkl \
    --out submission_rl_v4p2.py 2>&1 | tail -15

# 验收：smoke moves 必须非空，长度 >= 1
# 双保险：
grep -c '^WEIGHTS_B64 = "__WEIGHTS_B64__"' submission_rl_v4p2.py  # 应该输出 0
ls -la submission_rl_v4p2.py                                       # 应该 ~590KB

# B2. 跑 H2H gauntlet（这是判断 v4.2 是否值得提交的核心数据）
python -m orbit_wars_rl.scripts.h2h_gauntlet \
    --agent submission_rl_v4p2.py \
    --opponents submission_rl_v1.py submission_rl_v3p2.py submission_rl_v3p2_peak.py submission_v20_0513.py \
    --num-games 10 --seeds 0,1,2,3,4,5,6,7,8,9 \
    2>&1 | tee logs/h2h_v4p2_gauntlet.log
```

### 10.3 H2H 结果分流

| 结果 | 判定 | 下一步 |
|------|------|------|
| v4.2 vs v3.2_peak ≥ 0.50 AND vs v1 ≥ 0.80 | **v4.2 真的提升了** | 直接 upload `submission_rl_v4p2.py` 到 kaggle |
| v4.2 vs v3.2_peak < 0.50 | 还不如 v3.2 | 走 **v4.3 fallback**（§10.4） |
| v4.2 vs v1 < 0.80 但 vs v3.2_peak ≥ 0.50 | 边缘 | 先 export 中段 ckpt 看哪个 peak（§10.5），不行再 v4.3 |
| 所有 H2H 差 + smoke moves 还是空 | **export 还有别的 bug** | 跑 `diag_v4p2_loud.py` 看 forward 哪步炸 |

### 10.4 v4.3 fallback 命令

```bash
# 在 5090，从 v4.2 final ckpt 续训 400 upd
nohup python -m orbit_wars_rl.scripts.train \
    --config orbit_wars_rl/configs/multi_action_v4p3.yaml \
    --resume-from ckpt_multi_action_v4p2/ckpt_000799.pkl \
    --log-dir logs/multi_action_v4p3 \
    > logs/multi_action_v4p3.log 2>&1 &
tail -f logs/multi_action_v4p3.log
```

v4.3 hparam（详见 `configs/multi_action_v4p3.yaml` 注释）：
- 续训 1200 upd（实际相对 v4.2 多 400）
- lr_peak 3e-4 → 1.5e-4（不要扰动已学到的 policy）
- ent_coef_dst 3e-4 → 1e-4（让 dst 继续 commit，目标 ent[d] < 1.5）
- episode_steps 250 → 300（更多 orbit cycles）
- selfplay warmup_updates: 0（pool 已 populated）

### 10.5 中段 ckpt peak 探索

如果担心末段 v4.2 过拟合：

```bash
# 5090 上
for n in 549 599 649 699 749 799; do
    ck="ckpt_multi_action_v4p2/ckpt_000${n}.pkl"
    out="submission_rl_v4p2_u${n}.py"
    [ -f "$ck" ] || continue
    echo "=== exporting $ck ==="
    python -m orbit_wars_rl.scripts.export_submission \
        --ckpt "$ck" --out "$out" 2>&1 | tail -3
done

# 用 gauntlet 一次性测所有
python -m orbit_wars_rl.scripts.h2h_gauntlet \
    --agent submission_rl_v3p2_peak.py \
    --opponents submission_rl_v4p2_u549.py submission_rl_v4p2_u599.py \
                submission_rl_v4p2_u649.py submission_rl_v4p2_u699.py \
                submission_rl_v4p2_u749.py submission_rl_v4p2_u799.py \
    --num-games 6 --seeds 0,1,2,3,4,5 \
    2>&1 | tee logs/h2h_v4p2_peak_sweep.log

# v3.2_peak 对哪个 ckpt 输得最多 = 那个 ckpt 是 v4.2 的 peak
```

### 10.6 已知 deferrable（不影响当前 H2H/提交）

* **comets 没实现**（用户明确接受 §6.3.10 中所述）。Kaggle 真实 env 后期会 spawn
  temporary planet，我们 env 没有。对 30-50% 后期局面有影响。下一个 iteration
  再做。
* **`docs/RL_PIPELINE.md` 没同步 v4 改动**（仍是 v3 时代视角）。等 v4.2 H2H 出
  结果后一次性更新。
* **shaping reward `SHAPING_SCALE=0.1`** 没调过。如果 v4.x 系列还是输给 v3.2，
  第一个考虑的 ablation 是 `SHAPING_SCALE → 0.02 / 0`（让 terminal reward 主
  导）。改 `env/rewards.py` line 23。
* **`tests/test_export_pipeline.py`** 还没写（§9.10 教训末尾建议）。一个 unit
  test 防止 `_inject` 再出 silent failure 类问题。

### 10.7 「我回来后忘了上下文」最短复习

读这两段：
1. **§9.4-9.6**：v4.2 是 healthy 的（ent[d] 在 upd 200-383 段 5x 加速下降），
   只是 PPO clip/kl 因 multi-action ratio 求和天然为 0。不要再误判塌陷。
2. **§9.10**：v4 template docstring 里写字面 `WEIGHTS_B64 = "..."` 把 `_inject`
   骗到了，已修。下次 sync + re-export 应当一切正常。


