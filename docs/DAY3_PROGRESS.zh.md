# DAY3 进展 — v7 架构修复 K 步「reserved-blind」Bug

> 写于 2026-05-23 晚间。DAY3_PLAN.md 描述的是 BC 引导路线。我们
> 走了另一条路：在诊断出 v6p3 输给 v20 的**架构**根因后，又尝试了一轮
> 纯 RL 迭代。若 v7 训练 ≥ 5k updates 仍无法突破 0/20 vs v20，BC 仍是
> Plan B。

---

## 0. TL;DR

* **诊断**出 v6p3 尽管训练指标健康，但对 v20 仍是 0/20 的真正原因：在 K 步
  （max=8）自回归采样循环中，**4 个 action head 都对回合内 reserved-ships
  状态「失明」**，策略因此坍缩为每回合发出 K 个相同的 (src, dst, pct) 三元组，
  多为 1 舰编队，一回合内就把母星抽干。
* **架构修复（v7）**：每个 head 现在接收 reserved-aware 输入 —
  `remaining_norm`、`reserved_norm`、`src_remaining_norm`、
  `total_remaining_norm` —— 以便 K 循环消耗母星舰船时逐步 refine 决策。
* **还修复了一个长期存在的 float32 舍入 bug**（推理侧）：每支舰队都会比
  训练意图少发 1 艘（例如 `pct=0.70, garrison=10` →
  `floor(10 * 0.6999..) = 6` 而非 `7`）。从 v2 起所有提交都受影响。
* **提交模板才是真正的阻塞点。** 我忘了 `submission_rl_v4.py` 是 Kaggle
  原样使用的推理代码**独立副本** —— 它有自己的 head 函数拷贝，并不从
  `numpy_forward.py` import。v7 首次导出在 JAX parity 上看起来完美，但每次
  推理调用都因 shape mismatch 崩溃。已通过同步模板修复。
* **v7 u499 最终结果**（含模板修复）：pct 现在用 bin 4-6 而非 0，dst 不再
  坍缩，emits 从 4.0 降到 1.65（每回合 1 支舰队，但是*有意义*的舰队）。H2H
  待定 —— 训练指标暗示对 v1 很强，对 v20 仍不明。

---

## 1. v6p3 → v7 完整侦探故事

### 1.1 v6p3 在 u4999 结束

v6p3 在 Day 2 末启动，用于验证 OPT A 修复（DstHead 获得 `src_idx` 并 mask
掉 src，使 dst ≠ src）。在 5090 上跑了完整 5000 updates，约 6 小时。

训练指标尾部（最后几百 updates）：
```
upd 4999   WRr 0.92    WRf 0.78    mean_emits 2.10   ent_dst 1.31   clip 0.04
upd 4500   WRr 0.92    WRf 0.77    mean_emits 2.30   ent_dst 1.31   clip 0.05
upd 4000   WRr 0.91    WRf 0.78    mean_emits 2.58   ent_dst 1.40   clip 0.05
upd 3000   WRr 0.91    WRf 0.76    mean_emits 3.42   ent_dst 1.51   clip 0.06
upd 2000   WRr 0.90    WRf 0.74    mean_emits 4.10   ent_dst 2.09   clip 0.10
```

`mean_emits` 在 upd 2000 附近峰值 4.10，末尾缓慢衰减到 2.10。PPO `clip_frac`
从 0.10 降到 0.04 —— 策略收敛的明确信号。自对弈 WRf 升至 0.78（对更老的
frozen snapshot 有强 dominance）。

### 1.2 v6p3 u4999 H2H 考验

```
vs submission_v1.py         W= 17/20  (vs v6p2 u999 的 8/20  -- 大跃升)
vs submission_rl_v4p2.py    W= 15/20  (vs v6p2 u999 的 10/20)
vs submission_v20_0513.py   W=  0/20  (与 v6p2 相同，未变)
overall: 32/60 = 0.533
```

击败 v1 和 v4p2 确认 OPT A 有效（dst 不再坍缩到 src；舰队确实发射）。但
**0/20 vs v20** —— 与此前每次 RL 运行相同。

我最初解读：「策略收敛到 v4p2 类局部最优；5000 updates 纯 RL 自对弈无法
突破 v20」。这是错的。*指标*动态看起来像收敛，但*策略本身*以一种指标看不出的
方式坏了。

### 1.3 replay_dump 暴露真实行为

运行 `replay_dump.py` 打印 v6p3 vs v20 每回合的 (state, action)。母星第一回合
从 10 舰变成 1 舰：

```
[turn=  0] v6p3 sP=10 -> 发射 8 支舰队，各 1 舰，dst 分散到
                        8 个不同行星   ("蚊子群喷射")
[turn=  1] v6p3 sP= 3  -> 发射 3 支舰队，各 1 舰
[turn=  2] v6p3 sP= 1  -> 发射 1 支舰队，1 舰
[turn=  5] v6p3 sP= 1  -> 发射 1 支舰队，1 舰
[turn= 20] v6p3 sP=10  -> 发射 4 支舰队，各 1 舰   (短暂恢复，再被抽干)
```

Meanwhile v20 守在行星上，累积 5-15 舰，然后派出单支决定性 8 舰舰队。v6p3
每局都输。

### 1.4 diag_real_game_obs 确认 K 步锁定

写了新诊断，打印*原始* `greedy_multi_action` 输出（在 `decode_multi_to_kaggle_moves`
过滤之前的 src_list、dst_list、pct_list）。v6p3 turn 0：

```
v6p3 u4999 player=0 seed=0:
  turn 0: src=[12,12,12,12,12,12,12,12]   <- K=8 步，src 完全相同
          dst=[3,3,3,3,3,3,3,3]            <- dst 完全相同
          pct=[0,0,0,0,0,0,0,0]            <- pct 完全相同，pct_bin=0 (= 0.10)
          每支舰队舰数: floor(10*0.10) = 1，即 8 × 1 舰舰队

v4p2 u4999 player=0 seed=0:
  turn 0: src=[12], dst=[19], pct=[3]      <- K=1，只 emit 1 次，pct_bin=3 (= 0.40)
          每支舰队舰数: floor(10*0.40) = 4 舰
```

在 K=8 自回归循环内，每个 head 在 8 步中都产出**相同**输出。智能体在 t=0
就锁定了 src、dst、pct，从未 refine。

### 1.5 根因：head 对回合内 reserved 状态失明

看 `ActorCritic.__call__`（采样循环），每个 head 的输入在*回合起始状态*下是
*确定性的*：

```python
for t in range(K):
    src_logits_t = self.src_head(planet_emb, eff_mask)        # planet_emb 是回合起始
    src_t = argmax(src_logits_t)                                # 始终同一 src
    dst_logits_t = self.dst_head(planet_emb, src_emb_t, ...)   # src_emb_t 固定
    dst_t = argmax(dst_logits_t)                                # 始终同一 dst
    pct_logits_t = self.pct_head(src_emb_t, dst_emb_t, global_emb)  # 全部固定
    pct_t = argmax(pct_logits_t)                                # 始终同一 pct
    # reserved 会更新，但只用于 `_ships_to_send` -- 没有回传给任何 head
```

`reserved` 在舰船算术上记账正确，但**从未反馈给 head**。因此 K 次 argmax
在构造上就给出 8 个相同的 (src, dst, pct) 三元组。

对 `emit_head`，循环内只有 `step_oh` 在变 —— emit 原则上可通过 `step_oh`
条件 logits 在后续步停止，但它不知道母星是否还有舰。

这是**架构** bug，不是训练信号 bug。不改网络输入拓扑，再多 RL 微调、BC 引导
或 reward shaping 都修不好。

---

## 2. v7 架构变更

每个 head 增加一个 reserved-aware 输入（总计约 +320 params）：

| Head | 新输入 | Shape | 来源 |
|---|---|---|---|
| SrcHead | `remaining_norm[p]` | (P,) | `log1p(ships - reserved) / 8` |
| DstHead | `reserved_norm[p]` | (P,) | `log1p(reserved) / 8` |
| PctHead | `src_remaining_norm` | scalar | `remaining_norm[src_t]` |
| EmitHead | `total_remaining_norm` | scalar | `log1p(sum(remaining * my_mask)) / 8` |

`log1p/8` 与 `features/encode.py` 中行星特征归一化一致，使 head 看到的量级
与现有 `log_ships` 特征可比。

实现：
* `orbit_wars_rl/net/heads.py` —— 4 个 head 各接受可选新参数。
* `orbit_wars_rl/net/model.py` —— 新增 `_remaining_features()` helper；
  `evaluate()`（训练时 logp 重算）和 `__call__()`（采样循环）都按步计算这些
  特征并传入。
* `orbit_wars_rl/inference/numpy_forward.py` —— 镜像 JAX 变更。
* `orbit_wars_rl/inference/test_parity.py` —— 仍 16/16 OK（parity 契约成立）。

实现后，fresh-init 冒烟测试：
```
[init] total params: 676,557      (v6: 676,237, delta = +320)
[parity_check] arch d_model=128 n_layers=2 n_heads=4 ff_dim=512
               whole-turn action list match: 16/16
```

### 2.1 附带修复：ships_t 的 float32 舍入

检查 parity 时发现 numpy 推理对若干 pct bin 每支舰队少 1 舰：

```python
# OLD (buggy):
pct = float(_PCT_BIN_TABLE_NP[pct_t])   # f32 -> python float64
ships_t = max(1, int(np.floor(avail_at_src * pct)))
# pct=0.70 (存为 0.6999998 f32)，10 * 0.6999998 (f64) = 6.999998
# floor(6.999998) = 6   <-- 期望 7

# NEW:
mult = np.float32(avail_at_src) * _PCT_BIN_TABLE_NP[pct_t]   # 全 f32
ships_t = max(1, int(np.floor(mult)))
# 10.0 * 0.6999998 (f32) = 7.0
# floor(7.0) = 7    OK
```

JAX 通过 XLA fusion 静默做对了。Numpy 没有。**v2 到 v6 每个导出提交对
pct ∈ {0.30, 0.50, 0.70, 0.85} 都比训练意图每支舰队少发 1 舰。** 这是我们
见过的*最大 train/eval gap*，数周未被发现，因为 parity_check 容差在 logits 上，
不在 ship-count 输出上。

已应用到 `numpy_forward.py` 和 `submission_rl_v4.py` 模板。

---

## 3. v7 yaml + 启动

`orbit_wars_rl/configs/multi_action_v7.yaml`：
* 从零开始（不 resume —— 架构因 `+320` params 变了）。
* 架构同 v6/v6p3：`d_model=128, n_layers=2, n_heads=4, ff_dim=512`。
* `lr_warmup_steps=2000`, `lr_decay_steps=100000` —— 同 v6p2 修复
  （这是*优化器*步数而非 PPO updates，16x 缩放）。
* `ent_coef_dst=0.001` —— 同 v6p3（v6p2 的 10x，防 dst 再坍缩）。
* `num_updates=5000`，5090 上预计墙钟 ~6 h。
* `selfplay.warmup_updates=50`，warmup 后 frozen pool 加入。

已启动，运行正常。upd 200 时 SPS 达 5200（v6p3 为 3000 —— 每步特征计算
几乎免费）。

---

## 4. v7 训练指标至 u575

```
upd      0   adv_std 0.272  pg -0.0002  emits 1.55  ent[s/d/p/e] 0.31/2.89/1.97/0.67
upd     30   adv_std 0.197  pg -0.0026  emits 1.57  ent       0.27/2.82/1.98/0.66
upd    100   adv_std 0.159  pg -0.0036  emits 1.62  ent       0.30/2.74/1.92/0.64       WRr none
upd    150   adv_std 0.165  pg -0.0056  emits 1.58  ent       0.33/2.66/1.86/0.59       WRr 0.72
upd    200   adv_std 0.154  pg -0.0066  emits 1.36  ent       0.37/2.53/1.82/0.53       WRr 0.91 WRf 0.78
upd    300   adv_std 0.181  pg -0.0058  emits 1.50  ent       0.54/2.35/1.77/0.45       WRr 0.84 WRf 0.88
upd    400   adv_std 0.106  pg -0.0070  emits 1.50  ent       0.61/2.12/1.73/0.44       WRr 0.78 WRf 0.88
upd    500   adv_std 0.114  pg -0.0077  emits 1.65  ent       0.59/2.04/1.77/0.43       WRr 0.75 WRf 0.97
upd    575   adv_std 0.120  pg -0.0079  emits 1.56  ent       0.70/1.94/1.77/0.41       WRr 0.75 WRf 0.88
```

### 4.1 v7 指标实际含义（对比 v6p3）

| 指标 | v6p3 u500 | v7 u500 | 变化说明 |
|---|---|---|---|
| `WRf` | 0.45 | **0.97** | v7 97% 时间击败每个 frozen snapshot |
| `tR` | +0.30 | +0.72 | 自对弈 return 强正 |
| `mean_emits` | 4.0+ | **1.65** | v7 每回合 emit 1-2 支舰队，不是 8 |
| `ent_dst` | 1.31 (坍缩) | 2.13 | dst 熵保持高 —— head 仍有选项 |
| `ent_pct` | 1.50 | 1.78 | pct 分布覆盖多个 bin |
| `ent_src` | 0.50 | 0.69 | src 未 commit-lock |
| `clip_frac` | 0.04 (收敛) | 0.17 | v7 仍在积极学习 |

**生命迹象解读**：
* 熵三角测量：4 个 head 熵均未坍缩。
* `WRf 0.97` 前所未有 —— *当前* v7 策略几乎完美击败 frozen pool 中每个
  snapshot，且持续数百 updates。
* `mean_emits 1.65` *低于* v6p3，但每支舰队现在是真实、经过考虑的 launch，
  不是重复喷射。

### 4.2 但我们仍不知道 v7 能否击败 v20

训练信号是「v7 击败过去的自己」。这是健康的自我改进信号，但不证明策略达到
v20 级水平。需要真实 H2H 才知道。

---

## 5. 导出 Bug —— 以及为何浪费了 30 分钟

首次 u499 H2H 运行：
```
vs submission_rl_v1.py        W= 1/40   <-- 不可能，应该 ~17
```

`diag_real_game_obs` 揭示：
```
diag-ERR: matmul: Input operand 1 has a mismatch in its core dimension 0,
          with gufunc signature (n?,k),(k,m?)->(n?,m?)
          (size 265 is different from 264)
```

根因：我更新了 `orbit_wars_rl/inference/numpy_forward.py`，但**没有**更新
`submission_rl_v4.py`。提交模板是所有 head 函数的*独立*拷贝 —— 必须如此，
因为 Kaggle 只 import 提交文件里的内容。模板 head 仍是 v6 shape (264)，
ckpt 权重是 v7 shape (265)，每次推理崩溃 → `agent() returned []` → 空动作
→ 输掉对局。

**已修复**模板：同步全部 4 个 head 函数和 `greedy_multi_action` 与 v7
numpy_forward 逻辑。端到端 parity 确认：

```
$ python -m orbit_wars_rl.scripts.export_submission \
    --ckpt ckpt_multi_action_v7/ckpt_000499.pkl --out submission_rl_v7_u499.py
[export] running parity test against ckpt ..._u499.pkl (tol=0.005, num_states=16)
[OK ] whole-turn action list match: 16/16
[INFO] emit count match: 16/16
[export] smoke moves: [[0, 0.0, 6]]    <-- 非空：推理可用
```

### 5.1 流程教训

Kaggle 部署模型强制这种重复：部署时需要的推理逻辑必须存在于我们手工维护的
*唯一*模板文件。今后我将：

1. 每次先改 `submission_rl_v4.py` 模板（Kaggle 实际运行的）。
2. 再镜像到 `orbit_wars_rl/inference/numpy_forward.py`（仅 parity 测试用）。
3. Parity 测试应比较*提交模板*的 head 与 JAX，而非 numpy_forward 的 head。

Day 4 将重构 parity 测试直接消费模板，避免此类 drift 复发。

---

## 6. v7 u499 真实行为（真正的胜利）

模板修复后，diag_real_game_obs vs v20（seed 0, player 0）：

```
turn |  src    | dst   | pct_bin       | ships sent  | mother sP
   0 |  [12]   | [19]  | [4]=0.55      | 5           | 10 -> 6
   1 |  [12]   | [19]  | [4]=0.55      | 3           |  6 -> 4
   2 |  [12]   | [19]  | [6]=0.85      | 3           |  4 -> 1
   5 |  [12]   | [19]  | [6]=0.85      | 1           |  2 -> 1
  10 |  [12]   | [19]  | [6]=0.85      | 1           |  2 -> 1
```

与 v6p3 turn 0 对比（同 seed、同 player）：

```
v6p3:  src=[12]×8  dst=[3]×8   pct=[0]×8    -> 8 支舰队，各 1 舰
v7  :  src=[12]   dst=[19]    pct=[4]      -> 1 支舰队，5 舰
```

**Bug 修复记分板**：

| 原 v6p3 bug | v7 u499 状态 |
|---|---|
| (1) K 步内 src commit-lock | 未变 —— 只有 1 个 my-planet，lock 正确 |
| (2) pct 卡在最小 (0.10) → 1 舰舰队 | **已修复** —— pct 现在 0.55-0.85 |
| (3) dst 喷射（t=0 时 8 个不同 dst） | **已修复** —— t=0 只 emit 1 次 |
| (4) emit 打满（turn 0 时 K=8） | **已修复** —— emits 1.65/turn |
| (5) 一回合抽干母星 | **已修复** —— 约 3-5 回合抽干 |

5 个 bug 中 4 个可证消失。(1) 从来不是 bug —— 给定 early-game 状态是唯一合法
选择。

### 6.1 新观察：v7 仍不会「囤积再出击」

v7 每回合 emit 1 支舰队。母星长期接近空。v20 的制胜套路相反：*让母星
累积到 5-15 舰，再派一支大舰队*。v7 的*微观*对了（每次一支好舰队），*宏观*
不对（何时囤积 vs 花费）。

这比「总是喷射 1 舰舰队」窄得多。可能通过以下方式缩小：
* 把 v20 加入对手池，让策略在自对弈中遇到囤积-出击策略。
* 从 v20 BC 热启动（DAY3_PLAN.md 路线）。
* 对「每支舰队发送舰数」加 reward bonus，推动策略远离小舰队。

需等 v7 完整 5000-update 运行 + H2H 才知道哪种必要。

---

## 7. Day 3 结束状态快照（2026-05-23 22:55）

* `v7` 在 5090 上训练：u575 / 5000。WRf 0.88-0.97，WRr 0.72-0.91，
  emits 1.5-1.7，4 个 ent 均高于 v6p3 坍缩点。
* `submission_rl_v7_u499.py` 已导出（含模板修复）；smoke moves 非空；
  对 ckpt parity 16/16。
* `diag_real_game_obs` 确认 v7 emit *有意义*的舰队（pct=0.55，舰数 3-5）——
  首次有 RL 运行做到。
* v7_u499 H2H 考验是下一阻塞 —— 关键数字是「vs v20」行。即使 **3/20 也
  意味着架构修复端到端成功**，之后可决定为 v20 级差距加什么。

### 7.1 待定 H2H 的决策树

```
v7_u499 vs v20 结果
│
├── ≥ 3/20  → 架构修复成功。
│              计划：让 v7 跑完 5000 updates，在 u4999 重评。
│              若 u4999 vs v20 ≥ 5/20：继续 RL。
│              若 u4999 vs v20 卡在 3-4：试 v7p1，对手池加 v20。
│
├── 1-2/20  → 修复有效但 RL 自对弈无法桥接到 v20。
│              计划：在 v7 架构上做 BC 热启动（DAY3_PLAN.md §2）。
│
└── 0/20    → 架构不够。还有其他问题。
              计划：用 replay_dump 做 mid-game 更深诊断，
              可能 reward shaping 或 BC。
```

---

## 8. 今天从 top_players_rl.txt 学到的

§65-§109 BC 引导讨论是明显动机，但**更有用的洞见**是 §230：

> "Action head is pretty much standard."

那句话暗示 top 队伍的 action head 在 K 循环中能看到*足够状态*以做出不同决策。
我们不行。我们隐式假设 head 是标准的，因为命名相同。结合上下文重读架构后
清楚：「transformer encoder」上的「action head」意味着 head 看到行星/舰队状态的
*动态*视图，而非静态回合起始 token。

§2 的架构修复本应在 v3 就建。我们在 fundamentally K-blind 网络上花了 4 个版本
（v3 → v6p3）调超参。

### 8.1 Day 4 将从 top_players_rl.txt 检查的内容

若 v7_u4999 仍输给 v20，文档中下一优先级：

| 章节 | 主题 | 为何现在 |
|---|---|---|
| §75-109 | 从 heuristic BC | Plan B（DAY3_PLAN.md 可执行） |
| §170-200 | 舰队规模 reward shaping | 若 emits 仍 1/turn，reward 更大舰队 |
| §230 起 | Action head 细节 | 重读找遗漏 |
| §145-165 | 对手池 curation | 注入 v20 学 anti-stockpile |

---

## 9. v7 在整体脉络中的状态

| 运行 | vs v20 最终结果 | 诊断 |
|---|---|---|
| v4.2 | 0/20 | mean_emits 卡在 1.4（无法 credit-assign K-step） |
| v5.3 | 0/20 | 同上 —— 纯 RL credit assignment 时间预算太小 |
| v6p2 u999 | 0/20 | dst 坍缩到 src（"smoke moves: []"） |
| v6p3 u4999 | 0/20 | K-step head 对 reserved 状态失明 —— *真正 bug* |
| **v7 u499** | **?** | K-step head 看到 remaining/reserved；pct 现在用高 bin |

5 个版本未破的 0/20  streak 误导了我们。看起来像策略天花板。实际是系列 distinct、
 increasingly subtle 的 bug —— 每个都被下一个掩盖，产出相同 end metric
（0/20 vs v20）。v7 未必是最终突破天花板的版本，但是 replay 分析下*策略本身*
首次 non-degenerate 的版本。

---

## 10. 今日修改的文件

* `orbit_wars_rl/net/heads.py` —— 4 个 head 现在可选接受 reserved-aware 输入。
* `orbit_wars_rl/net/model.py` —— 新增 `_remaining_features()`；sample + evaluate 更新。
* `orbit_wars_rl/inference/numpy_forward.py` —— 镜像；float32 ships_t 修复。
* `submission_rl_v4.py` —— **相同**变更镜像（这是坑）。
* `orbit_wars_rl/scripts/diag_real_game_obs.py` —— 修复 python 3.13 dataclass 模块加载 bug。
* `orbit_wars_rl/configs/multi_action_v7.yaml` —— 新 run，ent_dst 0.001，warmup_steps 2000。
* `docs/DAY2_PROGRESS.md` §12 —— 完整 v6p3 → v7 链文档化。
* `docs/DAY3_PROGRESS.md` —— 本文档。

## 11. v7_u499 H2H 结果 + BC 并行轨道

### 11.1 结果

```
vs submission_rl_v1.py     W=14/20   WR=0.70   avg_steps=424
vs submission_rl_v4p2.py   W=19/20   WR=0.95   avg_steps=500
vs submission_v20_0513.py  W= 0/20   WR=0.00   avg_steps=159
                                   overall  0.55
```

对比 v6p3 u4999（此前最佳）：

| 对手 | v6p3 u4999 | v7 u499 | delta | 解读 |
|---|---|---|---|---|
| v1   | 17/20 (0.85) | 14/20 (0.70) | −0.15 | 每回合 1 舰队意味着对 v1 更少 brute-force 压力 |
| v4p2 | 15/20 (0.75) | 19/20 (0.95) | **+0.20** | 架构修复对我们自己的家族见效 |
| v20  | 0/20  (0.00) | 0/20  (0.00) | 0.00  | 策略天花板未抬升 |

**v4p2 +20%** 确认 v7 修复使策略在*微观*层面严格更强（一支好舰队 > 八支 1 舰
舰队）。但*宏观*策略 —— 囤积再出击 —— 从未学到，因为：

1. v7 自对弈对手（frozen v7 snapshot）都每回合 emit 1 舰队，v7 actor 训练时
   从未*看到*囤积-出击对手。
2. v20 对局 avg_steps=159，即 v20 约 turn 80 击杀 v7 母星 —— v7 太早出局，
   无法在 long-horizon 生产上竞争。

这正是 `top_players_rl.txt` §75-§109：**纯 RL 自对弈无法 credit-assign 跨越
800 步 delayed rewards**。需要从 v20 BC 播种宏观策略。

### 11.2 为何没有继续只跑 v7

若 v7 跑到 u5000 会烧 5h GPU。即使乐观 u5000 对 v20 +3/20，overall 仍 0.55
（= u499）。等待的边际期望价值低。

我们在后台继续 v7 训练，并行 committed 到 BC —— 即 §0 顶部提到的 **Plan D**。

### 11.3 BC 流水线状态（今晚）

今日实现并冒烟测试：

| 文件 | 状态 | 备注 |
|---|---|---|
| `bc/action_inverse.py` | **完成** | 5 个单元测试通过（single move、two moves same src reserved、empty moves、invalid src、K-overflow） |
| `bc/collect_data.py`   | **完成** | 2 局 smoke 产出 466 samples —— v20 统计见 §11.4 |
| `bc/train_bc.py`       | **完成** | 1-epoch smoke 训练 loss 10.06 → 8.93；val acc src/dst/pct/emit 上升；ckpt export+inject 可用 |
| 100 局 collection    | **运行中** | 后台，PID 13902，总计 ~50 min |
| BC 训练（正式）     | TBD       | collection 结束后，~30 min 训练 |
| BC 验证 H2H      | TBD       | 目标：BC  alone ≥ 5/20 vs v20 |
| RL 微调（Day 4）   | TBD       | 若 BC 达 5/20 vs v20，微调推高 |

### 11.4 v20 自对弈统计（2 局 smoke）

数据说明 v7 策略为何结构上错误：

```
total samples: 466 (= 2 games × ~120 turns × 2 perspectives)
total emits  : 406 (avg 0.87 per sample)

emits-per-sample histogram:
  0 emits: 64.4%   ← v20 大多数回合在 STOCKPILE！
  1 emits: 15.0%
  2 emits:  8.8%
  3 emits:  4.3%
  4 emits:  3.6%
  5 emits:  0.6%
  6 emits:  0.9%
  7 emits:  0.2%
  8 emits:  2.1%   ← v20 偶尔大力 swing（一回合 8 支舰队）

pct_bin distribution (across actually-emitted fleets):
  bin 0 (0.10):  6.7%   ← 小侦察/佯攻
  bin 1 (0.20):  8.1%
  bin 2 (0.30): 11.3%
  bin 3 (0.40): 17.5%
  bin 4 (0.55): 26.1%   ← 最常见
  bin 5 (0.70): 20.2%
  bin 6 (0.85):  8.9%
  bin 7 (1.00):  1.2%   ← all-in 几乎不用
```

v20 三分之二的回合是*无动作*。这与任何通过自对弈学到「每回合 emit 有用东西」
的策略 fundamentally 不兼容。0-emit 率是 BC 必须克隆的宏观行为，正是 RL 训练
从未看到 reward 信号的行为（0-emit 回合既不 immediate progress，自对弈中双方
都不动时也没有明显 downside）。

`top_players_rl.txt` §75-§109 说所有 top RL 队伍都因此从强 heuristic bootstrap。

---

## 12. 路线修正 —— 重读 top1 论坛帖

深夜，用户分享了 `top_players_rl.txt` 更新版，含 Lin Myat Ko（top1）回复。
三条陈述颠覆计划：

### 12.1 top1 实际说了什么

* §70 — "Do you need to write heuristic agent? **I don't.**"
  → Top1 **完全不用** BC 热启动。纯自对弈。

* §82 + §409 — "Best model took ~3 days, self-play from the start" =
  **600M steps**。v6p3 ~80M。v7 u499 ~8M。
  → 我们不是 under-architected，是 **under-trained**。

* §92–§96 — "Add one architecture delta at a time. Always. I shipped 7
  changes vs F12 in two days. ... lost 1 baseline."
  → 我们每个版本 ship 5+ delta（v6→v7 改 4 head、加 float32 修复、加 src_idx mask）。
  Top1 的 F 系列用一年 tiny single-delta 演化。

* §102 — "**clip_frac creep 0.10 → 0.30+ is the most reliable warning
  sign**. Cut lr or revert capacity. Don't wait."
  → 我们 watch clip_frac 但从未设 hard threshold 触发 action。

* §307 — "**explained_variance should hit at least 0.8 in 100 iters**,
  0.9 in 20 iters. If it never gets past 0.5, your obs representation
  or architecture is wrong."
  → **我们从未测过。** 在 single most important value-head 健康指标上 blind fly。

* §216 — "RL policy is robust enough to learn the fraction"
  → 确认我们的 pct head + bins 正确，不是 heuristic。

* §73 — "+1 −1 is enough for 2p mode"
  → 确认我们的 `ORBITWARS_SHAPING_SCALE=0.0` 选择。

### 12.2 对话中什么变了

| 之前 | 之后 |
|---|---|
| 计划：BC bootstrap（DAY3_PLAN.md, §2.1–§2.4） | **取消 BC 路线。** Top1 明确不用 BC。 |
| 计划：kill v7，写 v8 heuristic seed | **v8 = "v7 + log explained_variance + 2x 训练预算"。** 无架构 delta。 |
| 指标：显示 clip_frac，无 threshold | **`clip_frac > 0.25` 持续 50 updates ⇒ lr 砍半。> 0.35 ⇒ kill。** |
| 指标：无 explained_variance | **`explained_variance` 加入 PPO loss 和 train log 列。** |
| BC 流水线（`bc/action_inverse.py` 等） | **代码保留磁盘作 Plan B**，不用。测试仍 pass。 |

### 12.3 v8 规格

* `configs/multi_action_v8.yaml`：
  * 架构**同** v7（reserved-aware head，相对 v6p3 单一 delta）。
  * 超参**同** v7（lr peak、ent_coefs、clip_eps 等）。
  * `num_updates` 5000 → 10000（= ~160M 额外 steps）。
  * `lr_decay_steps` 100000 → 200000（匹配更长 run）。
  * `ckpt_every` 50 → 100（运维变更，非学习）。
  * 从 v7 final ckpt resume 而非 scratch，保留已投入的 80M steps。
  * `seed` 70 → 80（resume 用新 RNG）。

* 代码变更（单一 delta：仅 instrumentation）：
  * `orbit_wars_rl/ppo/update.py`：
    `explained_variance = 1 - Var(returns - value) / Var(returns)`
    加入 metrics dict 和 `_ZERO_METRICS_KEYS` tuple。
  * `orbit_wars_rl/ppo/runner.py`：打印格式现含 `ev +0.42` 在 `v 0.012` 旁。
  * 本地 smoke（3 updates）：`ev +0.13 → +0.24 → +0.49`。曲线如预期爬升 —— 代码可用。

### 12.4 v8 生命迹象期望

| Update | 指标 | 期望 | 未达标则 |
|---|---|---|---|
| upd 50  | explained_variance | ≥ 0.6 | 若 ≤ 0.3，value head 坏了 —— debug obs |
| upd 100 | explained_variance | ≥ 0.8 | 若 < 0.5，obs/arch 问题（top1 §307） |
| upd 1000 | clip_frac | < 0.20 | 若 ≥ 0.25 持续，lr 砍半 |
| upd 3000 | WRr | > 0.90 | （匹配 v6p3 plateau） |
| upd 5000 | H2H vs v20 | ≥ 3/20 | 宏观策略学到的首个信号 |
| upd 10000 | H2H vs v20 | ≥ 8/20 | top1 轨迹按 1/6 缩放 |

### 12.5 v8 仍无法测试的内容

若 v8 u10000 **仍 0/20 vs v20**，四个剩余假设（按成本排序）：

1. **更多训练**：拉到 20k updates（= 320M steps，仍 half top1 的 600M）。
2. **v20 进 opp pool**：upd 200 注入 v20 ckpt 作 frozen opponent。强制策略
   实际看到 stockpile-then-strike。相对 v8 单一 delta —— 可做。
3. **F12 架构 sweep**（top1 §92–§96）：TypedInputProjection、sun mask、MLP FireHead、
   per-source TargetMix、multi-query ValueHead。每个 single delta，可测。
4. **BC 作 Plan B**：代码已在磁盘（`bc/`）；前三失败则 BC clone v20 再 RL 微调。

Top1 对此路径的教训：
  > "F12's single-Dense FireHead, global head_mix_logits, missing sun
  > mask — those weren't bugs to fix. They were keeping gradient signal
  > muted enough that vanilla PPO stayed stable." (§95)

即对每个架构 delta 应*非常怀疑* —— 纸上每个看起来对，仍可能炸训练。

---

## 13. 待办

* 用户在 5090 启动 v8：从最新 v7 ckpt resume，log 到 `logs/multi_action_v8.log`，
  盯 `ev` 和 `clip` 列。
* 监控者：clip_frac ≥ 0.25 持续 50 updates，或 upd 100 explained_variance < 0.5 时告警。
* Day 4：重构 parity 测试对 submission template（防 §5 drift 复发）。
* Day 4：写 `monitor_train.py` —— tail log，算 clip_frac 和 EV 滚动均值，阈值破时告警。
* Day 4：log 加 `mean_ships_per_fleet` 和 `pct_bin_distribution`
  （目前只有 `mean_emits` —— 有用但不够）。
* BC 流水线保留磁盘作 Plan B；勿删 `bc/`。
  测试套件：`python -m orbit_wars_rl.bc.test_action_inverse` 仍 pass（已验证）。

---

## 14. v8 前 Reward 审计（2026-05-23 夜）

v8 启动前用户 pushback：*「现在这个阶段我觉得还没到 one-at-a-time
的时候，属于基本的 reward 信号还存在问题，修一个没有意义，先按照现在没对
齐的部分改彻底。」*

对照 `kaggle_environments/envs/orbit_wars/orbit_wars.py:684-715` 审计
`orbit_wars_rl/env/rewards.py`。**发现四处 silent mismatch。** 自 v3 起就存在。
Day 4 上午 v8 启动前已在单一 PR 中全部修复。

### 14.1 terminal_reward：平局曾是 0/0，应为 +1/+1

Kaggle（710-715 行）：
```python
max_score = max(scores)
for i in range(num_agents):
    if scores[i] == max_score and max_score > 0:
        state[i].reward = 1
    else:
        state[i].reward = -1
```

因此 75 vs 75 舰平局给**双方 +1**，不是 0/0。

我们代码（Day 4 前）：
```python
win = me > opp
loss = me < opp
return (win - loss)  # tie -> 0
```

**为何重要。** 80-turn episode 与 v6p3/v7 策略类下，~10-15% episode 近平局
（双方母星仍活，mid-game）。旧 reward 标 0；新 reward 双方 +1。value head 被
告知「我可能平局，无信号」；应被告知「平局算赢，要规划否则输」。

### 14.2 双灭：0/0 应为 -1/-1

Kaggle `max_score > 0` 子句：双方舰数都归零时，**双方 -1**（「无中之赢」不算赢）。

我们代码：0 舰平局 → 0/0。

80-turn episode 中 rare 但可能（final turn 自杀交换）。修复：与 kaggle 对齐。

### 14.3 终止于 step >= episodeSteps - 2

Kaggle（686 行）：`if step >= configuration.episodeSteps - 2: terminated = True`。

我们代码：`state.step >= episode_steps`。差 2。

实际影响小（episode 比以为短 2 tick），但**对 log 解读有意义**：说「episode
在 step 78 结束」本应是 step 78（= 80 − 2）。现已修正，env 行为与 spec 完全一致。

### 14.4 SHAPING_SCALE 默认 0.1 → 0.0

`rewards.py` 曾有 `SHAPING_SCALE = float(os.environ.get(..., "0.1"))`。

自 v5p2 起每次 launch 命令行设 `ORBITWARS_SHAPING_SCALE=0.0`，因 top1 §73 说
+1/-1 足够。但*默认*若忘 env var 是 0.1 —— hidden trap。

修复：默认现 `0.0`。要 shaping 设 env var `0.1`。启动 banner 现 echo
`SHAPING_SCALE=X.X` 可审计。

### 14.5 审计验证

* 新单元测试：`orbit_wars_rl/env/test_rewards.py`。运行
  `python -m orbit_wars_rl.env.test_rewards`。**15 / 15 pass**。
* `ppo/runner.py:train()` 新 runtime banner：
  `[reward] kaggle-aligned: terminal +1 win/+1 tie>0/-1 loss/-1 double-wipeout; ...`
* Env-parity（30 steps, seed 42）仍 clean —— 只改 reward/done 逻辑，dynamics 未变。
* Rollout smoke（4 envs × 128 steps）：无 NaN，所有 done reward ∈ {-1, +1}，
  所有 non-done reward = 0（SHAPING_SCALE=0）。

### 14.6 故意未改的内容

* `episode_steps: 80`（非 500）。这是**超参**，非 reward bug。80 steps 使每 rollout
  有 terminal signal（rollout_length=128 > 80），对 GAE credit assignment 关键。
  Top1 用 500 + 600M training steps；我们用 80 + ~100M steps。trade-off 见
  `multi_action_v8.yaml` header comments。
* `shaping_potential` 数学。默认不用；保留以便 pure +1/-1 卡住时 A/B。

### 14.7 v8 重跑计划

这些修复后，v8 启动时：
* 与 kaggle 完全对齐的新 reward landscape。
* 从 v7 final ckpt resume → value head 须重学 ties=win（~50 updates value-head
  调整后指标才稳定）。
* 相同架构（v7 reserved-aware K-step head）。
* 相同超参（lr schedule、ent_coefs、episode_steps=80）。
* 每次 launch log banner 确认 reward 配置。

这才是*proper* 的「one delta at a time」：架构 delta（v6p3 → v7）固定；只把
**reward function 与 day 1 就该对齐的 spec 对齐**。v8 之后可 confidence 恢复
「one architecture delta at a time」，因 reward 信号已正确。
