# Analysis: 「LB 1000+ Search + Learned Value Function」notebook

本文档针对仓库中的 [`lb-highest-1000-search-learned-value-function.ipynb`](../lb-highest-1000-search-learned-value-function.ipynb)（据称来自 Kaggle 论坛 / 高分方案分享）做**方法级解构**，不涉及复现 leaderboard 真伪；一切以 notebook 自述与内嵌代码为准。

---

## 1. 核心思想一句话

**用启发式收窄候选出兵，对每个候选做一次「短地平线前向推演」，再把推演终点状态扔进「从对局回放学出的价值模型」里，比较谁更像「最终会赢的格局」——而不是手写终局打分公式。**

这是典型的 **Planning / search + supervise learned evaluation**，与端到端 RL（policy gradient）是正交路线：这里没有可微策略网络，有的是 **离线监督的判别式价值函数 + 在线 rollouts**。

---

## 2. 三大模块（与作者自述一致）

| 模块 | 作用 | notebook 中的具体形态 |
|------|------|------------------------|
| **快速局部模拟器** | 在 ~1s/回合预算内对每个候选多做几次 rollout | `simulate_outcome(...)`：`SIM_LOOKAHEAD = 20` _tick_，作者在文中对比称相对官方环境快若干数量级（自述「~7000 turns/s」量级，需在本地独立基准验证） |
| **学习价值函数 V** | \( \mathbb{R}^{16} \rightarrow \mathbb{R} \)（GBC logit 求和） | `GradientBoostingClassifier` **500 × depth 6**，树结构 dump 成 `GBC_*` 放在独立 Dataset：`value_gbc_trees_big.py`，推理时 **`exec` 加载 + 手写树遍历**，**零 sklearn/numpy**（notebook 顶层仍 `import numpy as np`，但价值推理路径不依赖它） |
| **浅层博弈搜索** | 减轻「一厢情愿」假设 | 「v8: 2-ply minimax」在代码注释里：**每个对手玩家在 sim 的 tick 1 走一步启发式反击**（从兵力最多的己方星抽 45% 攻「优先打我方星球」的目标） |

---

## 3. 价值模型：特征、标签与性能声明

### 3.1 特征 `_value_state_features`（作者称 16 维）

向量在 **整张图聚合层面**刻画局面，不显式建模「哪条舰队正飞向哪颗星」——与真实 Orbit Wars 的动态相比是**重度抽象**：

- **时间**：`step / 500`
- **人数格式**：`n_players / 4`，以及 one-hot：`2P` / `4P`
- **相对规模**：己方 vs「最强敌」（按 `驻军 + 100×行星数` 选一个敌）——在 **行星数占比、驻军占比、总产占比、中心度（距 (50,50) 加权）** 上两两对照
- **差分**：船差、行星差、产差（除以总量归一）
- **固定槽位**：向量里有一处恒为 **0** 的占位（`0.0,`），疑为预留特征或对齐 sklearn 特征索引——若训练脚本未同步，可能影响「树 walk」与离线训练的一致性，需在完整管线中核对

**多敌人处理**：只对「被选中的那一个最强敌」做对比，其它对手的信息被边际化；这在 **四国战** 中可能是系统性盲区。

### 3.2 标签与数据量（notebook 自述）

- **回放**：196 盘高分回放 → 约 **26k** `(state → eventual_winner)` 样本  
- **自弈**：1000 盘自博弈 → 约 **241k** 样本  
- **合计**：作者给出 **267,766** 行量级与 **验证 AUC ≈ 0.97 / Brier ≈ 0.054** 等数字——属于 **离线可完美预测「给定中间状态who赢」** 的范畴；这不等价于线上 **下棋序决策** 一定最优（见第 7 节 covariate shift）。

---

## 4. 前向模拟 `simulate_outcome`：explicit 近似

作者在 docstring 中写明的简化（与你的 [`README.md`](../README.md) 完整规则对比时非常重要）：

1. **行星在 lookahead 内视为静态**：**忽略内侧轨道旋转**与彗星漂移； Meanwhile 主 agents 侧的 `find_angle_to_moving_planet` / `moving_planets` 用于**真实出牌**，两套动力学不一致——价值函数吃的是「近似 rollouts 后面的聚合特征」，存在 **sim–real gap**。
2. **除 tick-1 对手反击外，lookahead 内对方不再发包**。
3. **已有在途舰队**：沿直线以 log-speed 继续前进，并用 `_sim_predict_fleet_target` 预测撞上哪颗星、在第几 tick——**必须与真实几何、太阳障碍物、planet 线段碰撞规则一致**，否则 rollout 打分偏置。notebook 中称与官方 **byte-equivalent** 并已对 `lb1224` baseline 校验；**本地 ipynb 中未包含 `_sim_predict_fleet_target` / `_sim_capture_planet 的定义**，若你 clone 的版本同样缺失，则说明 **提交 `main.py` 不完整或对 helper 另有来源**，使用前需对齐 Kaggle 完整内核或作者仓库。
4. **生产与战斗**：loop 顺序为「每 tick 先全员产兵，再处理该 tick 到达的战斗」并用 `_sim_capture_planet`；需与环境中「生产 / 移动 / 公转 / 交战」的实际顺序逐项对照（[`README.md`](../README.md) 中顺序为 launch → production → fleet movement → rotation → combat 等）。

5. **价值输入的 step**：推演后调用 `_value_state_features(..., step=0)`——即 **不显式编码「推演过了多少回合」对未来的影响**，而是用「推演后的驻军/归属」隐含表达；是否与训练时标注用的 `step` 分布一致要小心。

---

## 5. 决策外壳：仍然是「heavy heuristic bot + sim 破局」

`main.py` 的主体不是「神经网络直接吐动作」，而是 **经典启发式 RTS 管线**：

- **增援**：`get_planets_under_attack` + `get_reinforcement_plans`，带 `reinforcement_trajectories` / `fleet_trajectories` 等跨回合记忆。
- **按兵力排序己方星球**：每个源星对该源上的目标用 `get_custom_score`（距离、产兵、敌方奖励、预估路上敌产、`FORMULA_*`）排序；**彗星**目标硬减 **40** 分偏好。
- **Search 仅在「Top-K tie-break」触发**：预算 `SIM_TIME_BUDGET_S = 0.55`，`SIM_TOP_K = 8`；在时间允许且候选多于 1 时，对 top-K **逐一** `simulate_outcome`，取 **`_value_score` 最高**。
- **v10 多源协调**：在同一回合内，后处理的源星会把先定案的移动建模成 **合成舰队 + 削减源星球兵力**，放进后续源的模拟里，减轻多源重复投资同一目标的错误。

此外还有 **大规模的协同攻占**分支（静态/转动、单方/多方 `calculate_req_ships*`），语义上接近工程化 brute-force 的规划片段。

结论：**学到的价值函数是「在启发式骨架上的第二层排序器」**，不是替代整个 bot；这与 AlphaZero「policy+value+MCTS」里 value 的比重相比，启发式一侧更重。

---

## 6. 工程与评测细节

- **禁止 `kaggle_environments` import**：自用 `namedtuple` 复制 `Planet` / `Fleet` 形状；符合「单文件_submit_」的常见约束。
- **`agent(obs, config)` 安全封装**：捕获异常返回 `[]`，用 `(player, initial_planets 片段)` **检测新局**并重置全局状态。
- **`steps` 前若干回合放空**：`_agent_impl` 在 `steps < 2` 时直接 `[]`，用于与某 baseline 对齐或内部 warm-up——可能损失开局红利，也可能是刻意 parity。
- **Kaggle Dataset**：notebook 体积小，**大树 dump ~1.3MB** 外置；metadata 中出现 `datasetId`/`sourceType: datasetVersion`，说明运行环境需手动挂载。

---

## 7. 优势（为何论坛方案可能有效）

1. **跳出纯手写势能场**：当局面的「好」更接近人类/强 bot 的最终胜率结构时，**短视界 rollout + 习得价值** 往往比单靠 `FORMULA_*` 更贴真实终局比分（驻军 + 舰队总量）。
2. **计算预算友好**：推理侧无深度网络；GBC tree walk 在 CPython 上对 16 维很快，可把预算用在 **多次 simulate**。
3. **可与强启发式共生**：你已投入大量 engineering 的 bot（类似你们的 `submission_v20`）可以 **保留**，只把「最后几个难分轩轾的候选」交给 value + sim——迭代风险可控。
4. **数据闭环清晰**：replay 蒸馏 winner + 自弈扩增；作者路线图里已点出 **distribution mismatch** 的修补方向。

---

## 8. 风险与局限

1. **特征不含在途舰队结构**：同一行星驻军表可能对应完全不同的威胁结构，价值函数可能 **overconfident**。
2. **最强敌近似**：四国时忽略结盟/多方牵制，可能与真实对局动力学不符。
3. **静态行星 sim vs 转动真图**：用你的 README 话术，内陆公转线是核心机制；近似会系统性误判「intercept window」。
4. **标签是「离线终局」，决策是在线因果**：棋盘中间状态与未来赢家相关很高，并不意味着 **marginal improvement of one move** 的排序总是被同一模型校准（离线 RL literature 里的 **off-policy / hindsight** 问题）。
5. **对手一步反击过于粗糙**：.tick 1. 单笔 45% 大军可能既过强（吓退己方投资）或过弱（仍乐观）。
6. **代码完整性**：本节针对**当前仓库内 ipynb**。若 `_sim_*` helpers 缺失，则该 notebook **无法单独生成可运行 `main.py`**——分析基于作者意图与可见片段。

### 8.1 疑似实现问题（建议自行 diff）

`refresh_local_obs` 中使用 `obs.get("player", [])` 作为缺省后与 `p.owner` 比较：若观测缺 key，会得到 **恒 false 的 planet 列表**。正常对局一般有 `player`，但健壮性逊于显式默认值（如 `-1`）。

---

## 9. 与「你们要上的 PPO RL」之间的关系

| 维度 | Search + Learned V | On-policy RL (PPO) |
|------|-------------------|---------------------|
| 信号来源 | 人类/强回放 + 自弈离线标签 | 与环境交互的即时/终局 reward |
| Credit assignment | 弱（状态→胜负） | 可通过 GA 等塑形加强 |
| 上线依赖 | tree dump / 或无 ML | 常为 torch + policy 导出 |
| 与现有 v20 | 最易做「sim 打分替换」 | 需 Gym 封装与动作空间离散化 |

**可组合**：PPO 学 **policy 或 shorter horizon value** 用作 `simulate_outcome` 里的 opponent model 或 rollout 截断估值；或用 RL 只做 **candidate proposal**，仍用 GBC 做 safety filter。

---

## 10. 作者列出的后续（Reproducibility 单元）

Notebook 指向未随附的脚本（若公开可逐一对齐）：

- `value_extract.py` — 从 replay 抽 `(state, winner)`  
- `selfplay_value_data.py` — 快模拟自弈造数据  
- `value_train_gbc_v2.py` — 训练并 dump 树  

**建议调参方向**（原文摘要）：更多 Meta-Kaggle 回放；**2P/4P 分模**；**用 v11 自己打自己** 生成数据以闭合 sim 分布；更长 lookahead；更深 minimax。

---

## 11. 小结

该方案是 **「强启发式 Orbit Wars bot + 20-tick 简化动力学 + 一步对手响应 + GBC 全局局面价值」** 的混合系统。论坛标题中的 **LB 1000+** 需以公开榜与提交 hash 独立核实；就方法而言，它把提升点放在 **「终局打分的可学习性」** 与 **「有限前向搜索」**，而不是扩大 policy 网络容量——对你们已有 `submission_v20` 系工程是 **低侵入、可对照实验** 的一条支线，同时与全 RL 路线 **互补而非重复**。

---

*文档生成自对本地 notebook 的静态阅读；若 Kaggle 上存在更新版内核，请以上游为准。*
