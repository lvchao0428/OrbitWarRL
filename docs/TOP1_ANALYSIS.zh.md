# Top1 方案分析 — IsaiahP "Scaling RL to the Stars" vs 我们的 OrbitWarRL

> **比赛**：Kaggle Orbit Wars (2026.06)  
> **Top1**：IsaiahP (Tufa Labs)，200M 参数 Transformer，15B 步自博弈 PPO  
> **我们**：v21–v30 系列，~3–5M 参数 Entity Transformer，JAX PPO，单 5090 GPU  
> **最终排名**：Top1 稳居第一；我们 RL 路线未能超越启发式 v20，全场 5000+ 人中下游

---

## 1. Top1 方案核心摘要

### 1.1 哲学
- **Sutton's Bitter Lesson**：不做精巧特征工程，靠模型规模 + 算力暴力学习
- **全程 agentic coding**（Codex 写代码，几乎不手动 review）
- **低层级 obs/action**：尽量不加归纳偏置，让模型自己学物理

### 1.2 架构

| 项 | Top1 | 
|---|---|
| 模型 | 200M 参数 Transformer |
| 隐层 | 768-d |
| Blocks | 38 层 residual self-attention |
| Attention | 16 heads/block |
| MLP | 768 → 1536 → 768 |
| 输入 | Entity tokens（planet/comet/fleet → MLP → 768d）+ 17 special tokens |
| Special tokens | 5 summary（obs 条件化）+ 4 plan + 4 value + 4 scratch（学习的全局 workspace） |
| 输出 | **单次前向 → 所有玩家的动作 + 价值**（2–4× compute savings） |

### 1.3 动作空间

| 项 | 设计 |
|---|---|
| Launch | 每个 source planet 独立 Bernoulli（发/不发） |
| Target | Cross-attn: Q(source)·K(target)/√d |
| Fleet size | Truncated discretized logistic mixture（8 components, [3, num_ships]） |
| 采样 | 所有 source **独立同时采样**（不是自回归） |
| PPO loss | 每个玩家取**联合概率**（所有 eligible source 的乘积） |

### 1.4 Critic
- 4 个 value token → MLP → **softmax over 存活玩家 → 胜率估计**
- gamma = 1.0（无折扣，胜率语义精确）

### 1.5 训练

| 项 | 值 |
|---|---|
| 算法 | PPO + GAE-λ |
| 总步数 | **15B steps** |
| 硬件 | 4 × 8xB200 节点（最终），8192 并行环境 |
| Throughput | ~6.3M steps/GPU·hour（~110K tokens/GPU·s）|
| GPU-hours | **~2400 B200-hours**（≈$10,000+） |
| Rollout | 64 步 |
| Self-play | 纯自博弈，**无 league / 无对手池** |
| Checkpoint | 新 ckpt 70%+ 胜率 → 替换 best；加 KL + CE value loss vs best |
| 环境 | **Rust 重写**（Python 太慢），pinned memory，多线程并行 |
| 2P/4P | 同时训练，最终 90% 2P（误判了 LB 匹配规则） |

### 1.6 推理优化

| 约束 | 解法 |
|---|---|
| 1s/turn + 60s 余量 | int8 量化线性层 + 限制可见 fleet 数 |
| 慢 CPU 降级 | 余量 <1s 时切换到 5M 小模型（100% 赢势转化） |
| 100MiB 提交大小 | **4-bit NormalFloat codebook 量化**（group=128, fp16 scale） |
| 量化损失 | 200M 4-bit vs 200M fp32: ~40% 胜率（可接受） |

### 1.7 关键发现
1. **Action mask 去掉反而更好** — 迫使模型内部建模更多物理（后来微调恢复）
2. **角度直接输出不如选目标** — 试了一周没成功，模型不够大
3. **gamma=1 导致拖延** — 有赢势不结束，浪费训练 compute
4. **batch size↑ → 初期更慢但上限更高**
5. **LayerNorm 防崩** — 之前用 CNN 无 norm，常崩；这次几乎没崩过

---

## 2. 我们的方案摘要

### 2.1 架构

| 项 | 我们（v30 最终版） |
|---|---|
| 模型 | ~3–5M 参数 Entity Transformer |
| 隐层 | 256-d |
| Blocks | 4 层 self-attention |
| Attention | 8 heads |
| MLP | 256 → 1024 → 256 |
| 输入 | planet(63d) + fleet(10d) + global(427d) → type embedding → shared tokens |
| 输出 | 单玩家视角 |

### 2.2 动作空间（自回归 K=8）

| 步骤 | 说明 |
|---|---|
| EmitHead | 继续发射？（Bernoulli） |
| SrcHead | 选源星球 |
| DstHead | Cross-attn 选目标 + DstEconomicsHead（v30 独立经济头） |
| PctHead | 16 bins 离散比例 |
| 自回归 | K=8 steps，每步更新 reserved，下步看扣减后 remaining |

### 2.3 训练

| 项 | 值 |
|---|---|
| 算法 | PPO + GAE-λ（JAX jit） |
| 总步数 | 每版本 ~4000–11000 updates × 64 envs × 32 rollout ≈ **8M–22M steps/版** |
| 硬件 | **单张 5090 GPU** |
| 环境 | Python + JAX（vmap 并行），非 Rust |
| Self-play | v21 纯对称 → v22 mixed opp → v24 BC init + KL 锚 → v25–v30 迭代 |
| 对手 | frozen v16a + random + v20 buffer + BC anchor |
| Reward | terminal ±1 + capture shaping + flip-gated + ROI aux loss（v30） |
| 版本迭代 | v21 → v22 → v23 → v24(BC) → v25 → v26 → v27 → v28 → v29 → v30 |

### 2.4 特征工程（大量手工设计）

- ETA-lead 距离传递给全部 heads
- 多星球协同观测（active_emitter_ratio, concentration_ratio...）
- 未来局势预测（flip_risk, garrison_growth）
- 5 帧 planet spatial hist
- capture_roi 公式 + frontier + anti-shuffle
- DstEconomicsHead 独立经济决策头 + ROI aux CE loss

---

## 3. 核心差异对比

| 维度 | Top1 (IsaiahP) | 我们 (OrbitWarRL) | 评价 |
|------|----------------|-------------------|------|
| **模型规模** | 200M params, 768d, 38L | ~3–5M params, 256d, 4L | **40–60× 差距** |
| **训练算力** | 2400 B200-hours (~$10K+) | 单 5090, 每版 ~3–8h | **>100× 差距** |
| **总训练步数** | 15B | 累计 ~0.1–0.2B | **~100× 差距** |
| **环境速度** | Rust 重写 + pinned memory | Python + JAX vmap | Top1 快 5–10× |
| **特征工程** | 极少（raw entity + 位置/速度/兵力） | 极多（63 维 planet + 427 维 global + hist） | 方向相反 |
| **动作设计** | 所有 source 独立同时采样 | K=8 自回归串行 | 我们更复杂但吞吐低 |
| **Fleet size** | 连续 logistic mixture（8 components） | 离散 16 bins | Top1 更精细 |
| **Critic** | 4 player 胜率 softmax, gamma=1 | ZeroSumValueHead, gamma<1 | Top1 语义更清晰 |
| **Self-play** | 纯自博弈 + best ckpt 替换 | 混合对手池 + BC init + KL 锚 | 我们更复杂 |
| **推理** | 单次前向 → 所有玩家 | 单玩家视角 | Top1 节省 2–4× |
| **量化** | 4-bit NF + fallback 5M | 无 | — |
| **代码方式** | 全 agentic (Codex) | 人+AI 协作 | — |

---

## 4. 为什么 Top1 赢了 — 深层分析

### 4.1 规模碾压：Bitter Lesson 生效

Top1 的核心论点被验证了：**足够大的模型 + 足够多的训练，可以弥补任何特征工程的不足**。

- 200M 参数 + 38 层 attention 有足够容量自己学到轨道物理、ETA lead、协同攻击
- 我们花大量时间手工设计的 `capture_roi`、`frontier_score`、`future_flip_risk` 等特征，Top1 的模型可以从 raw 数据中自行涌现
- Top1 作者自己也说：「每次 model 不够准时，加长训练就好了，不需要改架构」

### 4.2 算力不对等

| 资源 | Top1 | 我们 | 倍率 |
|------|------|------|------|
| GPU 类型 | B200 (最新顶级) | 5090 (消费级) | ~4× 单卡性能 |
| GPU 数量 | 32 (4×8) | 1 | 32× |
| 训练时间 | ~数周 | 每版 3–8h, 迭代 ~10 版 | 训练总量 100×+ |
| 环境吞吐 | 8192 并行 (Rust) | 64 并行 (JAX) | 128× |

**关键**：即使我们架构完美，单张 5090 上 JAX 环境跑 0.1B 步，vs Top1 的 15B 步，**学习信号量差 150 倍**。

### 4.3 环境实现差距

Top1 用 **Rust 重写环境** + pinned memory + 多线程，环境吞吐远高于我们的 Python+JAX vmap。这不仅影响训练速度，还影响了我们能否在有限时间内做更多实验。

### 4.4 我们的架构选择问题

| 我们的选择 | 潜在问题 |
|---|---|
| K=8 自回归 | 串行 8 步，吞吐低；Top1 的并行独立采样 **简单且高效** |
| 离散 16 bins | 精度不如连续 logistic mixture |
| 单玩家前向 | 每步需为每个玩家单独推理；Top1 单次前向出全部 |
| 大量手工特征 | 增加了特征 bug 风险（v21 export 维度 bug 导致盲训），且限制了泛化 |

### 4.5 训练策略差距

| 方面 | Top1 | 我们 | 评价 |
|------|------|------|------|
| 初始化 | 从头训 | v20 BC init → PPO 微调 | BC 有上限瓶颈 |
| 对手 | 纯自博弈 + 70% 门控替换 | 复杂混合（v16a/frozen/rand/buffer） | 我们过度工程 |
| 版本迭代 | 连续训练 15B 步 | 10+ 版本频繁切换重训 | **我们损失了连续学习的收益** |
| Reward | 仅 terminal ±1 | 大量 shaping（capture/flip-gate/ROI/anti-hoard...） | shaping 引入梯度冲突 |

---

## 5. 我们做对了什么

尽管结果不佳，有些设计思路与 Top1 殊途同归：

| 我们的做法 | 与 Top1 的关系 |
|---|---|
| Entity Transformer backbone | 同类架构 ✅ |
| Target-based action（选星球非角度） | Top1 也从角度改成了选目标 ✅ |
| PPO + GAE | 同样选择，Top1 也没用 IMPALA ✅ |
| 多 fleet 发射 | 同样支持 ✅ |
| Cross-attn 选目标 | Top1 用 Q·K/√d，本质同 ✅ |
| gamma=1 + 胜率 critic (ZeroSum) | 类似思路 ✅ |

---

## 6. 我们做错了什么（复盘教训）

### 6.1 过度特征工程
我们在 planet 特征上堆到 63 维、global 427 维，包括 `capture_roi_norm`、`frontier_score`、`shuffle_dst_risk` 等大量手工设计。Top1 证明：**这些可以被大模型自动学到**。在小模型+少 compute 的约束下，这些特征确实有帮助，但它们引入了：
- 维度 bug（v21 export 失败导致盲训）
- 特征间语义冲突
- 开发时间大量消耗在调特征而非训练模型

### 6.2 版本迭代太碎
v21 → v22 → ... → v30，**10 个版本每版 3–8h，频繁重头训练或 resume**。相比 Top1 连续跑 15B 步只做 LR/batch 热重启，我们每次切版本都丢失了已有学习成果。

### 6.3 Reward shaping 过多
从 `capture_roi` 到 `anti_hoard` 到 `multi_emit` 到 `fleet_scale` 到 `flip-gated capture`... 多个 shaping 项互相拉扯，导致 **flip↔e2+ tradeoff 始终无法解决**。Top1 只用 terminal ±1，简单粗暴但有效。

### 6.4 未投资环境加速
Python + JAX vmap 的环境速度远不如 Rust。如果我们在 Day 1–3 就用 Rust/C++ 重写环境，后续迭代速度会好得多。

### 6.5 自回归 K=8 的吞吐代价
K=8 串行 8 步推理，在训练时成为 bottleneck。Top1 所有 source 并行独立采样，简单且高效。我们的自回归设计理论上能更好协调多 fleet，但在算力受限时是负优化。

### 6.6 始终没打过 v20 启发式
从 v21 到 v30，vs-v20 最好成绩也只是偶发 1/4（5 局评估下 20% 胜率），大量时间花在 debug「为什么打不过 v20」。**heuristic v20 本身就是强基线**，在有限算力下 RL 要超越一个精心调参的启发式非常困难。

---

## 7. 如果重来，我们应该怎么做

### 7.1 接受算力约束，走「精致小模型」路线
- 不去模仿 Top1 的 Bitter Lesson 路线（没有 32 张 B200）
- 参考第 5 名 Jake Will（也用 RL，非巨量 compute）、第 2 名 simjeg 的方案
- **模型适度放大**（16–25M），但不追求 200M
- 把有限算力花在更长的连续训练上，而非频繁切版本

### 7.2 Day 1 就投资环境加速
- Rust 或 C++ 环境重写是 **最高 ROI 投资**
- JAX vmap 64 envs vs Rust 8192 envs → 128× 吞吐差距
- 更快环境 = 更多实验 = 更好的超参搜索

### 7.3 简化 reward
- 只用 **terminal ±1**，顶多加一个 capture shaping
- 不要 anti_hoard / multi_emit / fleet_scale / prod_share_delta 等多头 shaping
- Top1 和第 2 名都证明：简单 reward + 长训 > 复杂 shaping + 短训

### 7.4 简化动作空间
- 参考 Top1：每个星球独立 launch/target/size，不用自回归
- Fleet size 用连续分布（logistic mixture 或 Beta）而非离散 bins
- 减少推理延迟，提高训练吞吐

### 7.5 连续训练，不频繁切版本
- 选定一个合理架构后，**坚持训长**（至少 1B+ 步）
- 只做 LR / batch 的 warm restart，不改架构
- Top1 在 8B 步时 plateau → 调大 batch + 降 LR → 继续涨到 15B

---

## 8. 比赛生态观察

### 8.1 算力不对等争议
评论区多人质疑 Top1 的 ~$10K+ B200 算力是否公平。Top1 自己也承认这是「pyrrhic victory」——
> 「成功建立在正是让他爱上 Kaggle 的那种算法创造力的淘汰之上。」

### 8.2 其他高名次方案
- **2nd (simjeg)**：模仿学习初始化 + 最后一周才跑自博弈，也用了 Bitter Lesson 思路但规模更小
- **5th (Jake Will)**：自回归 micro-step with observation updating（每步都更新 obs），创新架构
- **8th (flg)**：中等规模 RL，认为 coding agent 的 RL 设计建议「incredibly bad」
- **9th (Billy Bradley)**：赞叹 Top1 能让每个星球独立决策仍表现出色
- **10th (Fei Wang)**：第一次看到 200M 规模，以为是 typo

### 8.3 启发式仍有竞争力
比赛中大量选手（包括我们）的 RL agent 打不过精心调参的启发式。这说明在 **中低算力**下，RL 在复杂实时策略游戏中并不一定优于 rule-based。

---

## 9. 总结

| 维度 | 结论 |
|------|------|
| **根因** | 算力差距 100×+ 是决定性因素；特征工程在足够算力下可被模型自行学到 |
| **架构方向** | 双方都选了 Entity Transformer + target-based action，方向正确 |
| **训练策略** | 纯自博弈 + terminal reward + 长训 > 混合对手 + 复杂 shaping + 短迭代 |
| **工程投资** | Rust 环境重写的 ROI 远高于特征工程 |
| **心态** | 在单卡约束下应走「精致小模型 + 长训」路线，而非频繁推翻重来 |
| **收获** | 完整走了一遍 RL 竞赛流程（特征/架构/PPO/self-play/eval），积累了端到端经验 |

> **Bitter Lesson 的另一面**：Top1 有 32 张 B200，Bitter Lesson 对他成立。我们只有 1 张 5090，
> Bitter Lesson 在告诉我们「scale wins」的同时，也在提醒我们——**没有 scale 的时候，smart engineering 才是唯一出路**，
> 但 smart engineering 的方向应该是「加速环境 + 简化 reward + 坚持长训」，而非「堆特征 + 频繁切版本」。

---

*生成：2026-06-30；基于 Top1 公开 write-up + 本项目 v21–v30 完整记录。*
