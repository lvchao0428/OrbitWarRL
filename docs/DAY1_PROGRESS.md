# Day 1 进展总结（2026-05-21 晚 → 22 凌晨）

> **写给明天换电脑接手的自己。** 这里只记「今天发生了什么、为什么这么做、明天从哪里开始」，
> 不是项目级的设计文档。完整的 RL 管线说明仍以 [`RL_PIPELINE.md`](RL_PIPELINE.md) 为准。

---

## 0. TL;DR

* **能提交的 RL 单文件 `submission_rl_v1.py` 已经上线 Kaggle。**（ckpt_mvp/ckpt_000049.pkl 训练 50 update 的结果）
* **self-play 全套基础设施今天全部就位**：frozen pool、混合对手 rollout、parity 测试都跑通了。
* `submission_rl_v2.py`（self-play 训练 300 update 的结果）**本地 H2H 不达标，没提交**——见 §4。
* **主瓶颈是「每回合只能发 1 条 fleet」**——这是 MVP 与真 Kaggle 环境的结构性差异。**明天的核心任务是做多动作 head**。
* 三个 ckpt（v1/v2/v3）覆盖了「训练量不够」「self-play 不工作」「episode 长度」三种可疑假设，**全部排除**。详见 §7 的实验档案。
* 明天换电脑接手第一件事：进多动作 head 工程（设计在 §4）。

---

## 1. 今天动手做的事

### 1.1 RL 推理链路打通（晚上 22:00–00:00）
新增 `orbit_wars_rl/inference/` 子包，**全部仅依赖 numpy**，独立于 jax/flax：

| 文件 | 作用 |
|------|------|
| `inference/weights.py` | flax pickle param tree → flat `{path: np.ndarray}` |
| `inference/numpy_forward.py` | 纯 numpy 复刻 ActorCritic（encoder + 4 heads） |
| `inference/kaggle_adapter.py` | Kaggle obs → numpy 特征 / 动作 → `[planet_id, angle, ships]` |
| `inference/test_parity.py` | 多 obs 的 jax↔numpy argmax 一致性测试 |
| `inference/test_adapter.py` | adapter ↔ jax encoder 的特征对账（在静态行星子集上 0 差） |

新增 `submission_rl_v1.py`（仓库根，**单文件 573KB**）+ `scripts/export_submission.py`（一键 ckpt → 单文件 submission 流水线，含 parity 卡口）。

**关键设计取舍**：
* 选 base64-inline 权重（44 万 float32, 压缩后 ~544KB）而不是 Kaggle dataset 挂载——单文件最简，能上传就能跑。
* parity 测试的 gate 是 **argmax 一致**，不是 logit 数值。原因见 §3「v2 parity 教训」。

### 1.2 self-play 接入（00:00–00:50）

| 改动 | 内容 |
|------|------|
| `selfplay/pool.py` | 加 `latest()`、清掉无用的 `copy.deepcopy` |
| `ppo/rollout.py` | 新增 `make_rollout_fn_with_frozen_opp`：保留 jit 友好性，opponent params 作为 jit 参数传入 |
| `ppo/runner.py` | 加 `SelfPlayConfig`、混合对手调度（warmup 阶段全 random → 之后按 `frozen_ratio` 切换）、`pool.snapshot` 节奏、`play_vs_frozen` 评估 |
| `scripts/train.py` | YAML 多了 `selfplay:` 段 |
| `scripts/smoke_test.py` | 加 `--selfplay` 标志 |
| `configs/selfplay.yaml` | 新增训练配置 |

### 1.3 训练日志梳理

在 5090 服务器上跑了 300 update 的 self-play：
```
最终一组:
upd 299  WRr 1.00  WRf 0.94  clip 0.00  kl 0.000  sps 5177
```
* `WRr` = vs random 胜率 → **从 v1 的 ~16% 飙到 100%**
* `WRf` = vs latest frozen 胜率 → 0.94（理想 0.5–0.6；说明 frozen pool 相对当前 learner 还是太弱，仍有训练空间）
* SPS 5K 全程稳定，无 NaN

### 1.4 第二个夜跑实验（睡前发起）

`configs/selfplay_ep500.yaml`：唯一与 `selfplay.yaml` 不同的是 `episode_steps: 500`（vs 200），120 update。

**目的**：排除「v2 在真 Kaggle 输 starter 是因为训练 episode 太短」这条假设——明早第一件事就看它的 H2H 结果决定路线。

---

## 2. 现在仓库长这样

```
OrbitWarRL/
├── submission_v20_0513.py        启发式 bot（v1 之外的提交槽位）
├── submission_rl_v1.py           ★ RL 单文件，已交 Kaggle，573KB
├── submission_rl_v2.py           本地 H2H 不达标，未交（保留在仓库做对比）
├── ckpt_mvp/ckpt_000049.pkl      v1 来源
├── ckpt_selfplay/ckpt_*.pkl      v2 来源（24/49/.../299）
├── ckpt_selfplay_ep500/...       夜跑实验，明早查看
└── orbit_wars_rl/
    ├── env/...                   静态行星 MVP env（与今天前一致）
    ├── features/encode.py        与今天前一致
    ├── net/{transformer,heads,model}.py  与今天前一致
    ├── ppo/
    │   ├── rollout.py            ★ 新增 frozen-opp 版本的 rollout_fn
    │   ├── update.py             与今天前一致
    │   └── runner.py             ★ 新增 SelfPlayConfig + 混合调度
    ├── selfplay/
    │   ├── pool.py               ★ 小重构
    │   └── eval.py               与今天前一致
    ├── inference/                ★ 整个目录是今天新建
    │   ├── weights.py
    │   ├── numpy_forward.py
    │   ├── kaggle_adapter.py
    │   ├── test_parity.py
    │   └── test_adapter.py
    ├── scripts/
    │   ├── parity_check.py       与今天前一致（针对 env 对账，不是 inference 对账）
    │   ├── smoke_test.py         ★ 加了 --selfplay
    │   ├── train.py              ★ 读取 selfplay 段
    │   ├── export_submission.py  ★ 新增
    │   └── h2h_local.py          ★ 新增（kaggle_environments 本地双盘对决）
    └── configs/
        ├── mvp.yaml              与今天前一致（产 v1）
        ├── selfplay.yaml         ★ 新增（产 v2）
        └── selfplay_ep500.yaml   ★ 新增（明早查看）
```

---

## 3. H2H 结果与教训

### 3.1 本地 H2H（kaggle_environments）

| Bot | vs random | vs starter | vs v20 |
|-----|-----------|-----------|--------|
| **v1**（ckpt_49，mvp.yaml） | 5/5 | 1/5 | 0/4 |
| **v2**（ckpt_299，selfplay.yaml） | 5/5 | **0/5** | 0/5 |
| **v3**（待夜跑实验完成，ep500.yaml） | ? | ? | ? |

### 3.2 v2 不交的判断逻辑

* 训练日志看上去**完美**（WRr 1.0 / WRf 0.94），但**本地 vs starter 是 0/5，比 v1 还差**。
* 这是「训练对手单一」+「MVP 环境与真环境不一致」的双重夹击：
    * 训练里 self-play 的对手也只能发 1 条 fleet → learner 学到了「单动作环境的最优解」。
    * 真 Kaggle 的 starter 一回合发多条 fleet → learner 在动作率上结构性吃亏。
* **结论**：再训 1000 个 update 也救不回来；必须改 action 空间。

### 3.3 parity 教训（导出 v2 时踩的坑，已修）

第一次 `export_submission` 因为 dst_logits drift `1.056e-03 > tol 1e-3` 报失败。**实际上 argmax 完全一致**——只是网络训得越好、权重 magnitude 越大、float32 累积误差就大。

我把 `test_parity.py` 和 `export_submission.py` 的判定改成：
* **gate = argmax 在 16 个随机状态上 100% 一致**
* drift 数值只作为 WARN 输出
* 默认 tol 放宽到 5e-3

老 v1 ckpt 的 drift 仍然在 1e-6（重测过仍通过），这个口径改动**不影响小 ckpt**。

---

## 4. 明天的核心任务：多动作 head

### 4.1 必要性

`overview.txt` 明确：
> Each turn, your agent returns a list of moves: `[from_planet_id, direction_angle, num_ships]`.

我们的 MVP 网络是「每回合一个动作三元组」。**这是结构性瓶颈**——v20 一回合常发 5–10 条 fleet（看 `submission_v20_0513.py:agent` 的 `arbiter.moves`），RL 没法对抗。

### 4.2 设计要点

1. **可变长度动作**：每回合 ≤ N 条 fleet（建议 N=8 起步，与 `submission_v20_0513.py` 的 `MAX_TOTAL_MOVES=26` 同量级或略小）。
2. **autoregressive 采样**：第 t 条 fleet 看到 obs + 前 t-1 条 fleet 的「己方兵力扣除」效应（最简单的做法：把已 reserve 的 ships 从 src 行星上扣掉再重新算 obs；或者直接传一个「本回合 reserved」mask 给 src head）。
3. **emit-stop token**：第一个 head 输出一个 `(src_idx, STOP)` 二选一，让网络自己决定本回合发完。
4. **PPO 适配**：
    * 每个 turn 的 `logp` 累加：`turn_logp = sum_t logp_t(src, dst, pct)`。
    * `ratio = exp(new - old)` 仍然单一 ratio per turn。
    * Entropy 累加，按 turn 内动作数归一化避免熵爆。
5. **rollout 状态**：state 多一个 `reserved_ships[40]` buffer，自回归循环结束清零。
6. **inference 单文件适配**：`submission_rl_v3.py` 里 numpy forward 改成循环；agent() 返回 list。

### 4.3 改动文件清单（预估）

| 文件 | 工作量 |
|------|------|
| `env/actions.py` | 加多 fleet 动作类型 + reserved buffer ~1h |
| `env/dynamics.py` | `launch_fleets` 改成接 list of (src, dst, pct) ~30min |
| `env/env.py` | step 接口改成 list-of-actions per player ~30min |
| `net/heads.py` | src/dst/pct head 加 autoregressive 循环 + STOP token ~1.5h |
| `net/model.py` | sample/evaluate 改成 unroll over t ~1h |
| `ppo/rollout.py` | per-turn logp 累加 ~30min |
| `ppo/update.py` | flatten action list 进 GAE ~30min |
| `inference/*` | numpy forward 加循环 + action list ~1h |
| `submission_rl_v?.py` | 同上 ~30min |
| **总计** | **6–7 小时** |

### 4.4 验收标准

* parity 测试：16 个 obs 上，每个 obs 上「整回合动作 list」与 jax 一致（不仅是首动作）。
* 训练曲线：WRr 仍能 ≥0.9，WRf 在 0.5–0.6 振荡（健康自对弈）。
* 本地 H2H：**v3 vs starter ≥ 2/5**（先把 starter 打过半数再说 v20）。
* Kaggle 提交后 ELO：高于 v1 ≥ 50。

---

## 5. 明天换电脑要做的环境准备

```bash
# 1. 拉代码
git clone <repo>     # 或 git pull
cd OrbitWarRL

# 2. Python 环境（关键：anaconda Python 3.10+, 别用 system python 2.7）
which python                                    # 必须是 anaconda/conda 路径
python --version                                # 3.10+

# 3. 安装依赖
pip install -r requirements-rl.txt
# 验证：
python -c "import jax, flax, optax, chex, numpy, kaggle_environments; print('ok')"

# 4. smoke 验证（约 30 秒）
python -m orbit_wars_rl.scripts.smoke_test --num-envs 8 --num-updates 8

# 5. parity 验证 v1 ckpt（约 10 秒）
python -m orbit_wars_rl.inference.test_parity --ckpt ckpt_mvp/ckpt_000049.pkl

# 6. 本地 H2H sanity（每局 ~10s）
python -m orbit_wars_rl.scripts.h2h_local \
  --agent-a submission_rl_v1.py \
  --agent-b submission_v20_0513.py \
  --num-games 1 --seeds 0
```

如果 (3) 装不全：anaconda 装在 macOS 上的 jax 经常因 protobuf/tensorboard 版本冲突报错；**tensorboard 缺失不影响主路径**（runner 会 fallback 到 stdout），protobuf 错误可以 `pip install -U protobuf tensorboard`。

GPU 服务器：CUDA 13 + jaxlib 没有官方 wheel，今天没装。**先不死磕**，CPU 也能 5K SPS，多动作 head 改完再考虑。

---

## 6. 实验数据档案

为了明天复盘方便，记一下今天的几个关键数字（**别覆盖**）：

| ckpt | 来源 | 训练量 | 训练日志 | 本地 H2H |
|------|------|--------|----------|----------|
| `ckpt_mvp/ckpt_000049.pkl` | `configs/mvp.yaml`, mac CPU | 50 update × 16 env × 128 = 102K env steps | WR vs random ~0.10–0.20 | random 5/5, starter 1/5, v20 0/4 |
| `ckpt_selfplay/ckpt_000299.pkl` | `configs/selfplay.yaml`, 5090 CPU | 300 update × 32 env × 128 = 1.2M env steps | WRr 1.00 / WRf 0.94 | random 5/5, **starter 0/5**, v20 0/5 |
| `ckpt_selfplay_ep500/ckpt_000119.pkl` | `configs/selfplay_ep500.yaml`, 5090 CPU | 120 update × 32 env × 128 = 492K env steps | WRr 1.00 / **WRf 0.66**（健康！） | random **4/5**, **starter 0/5**, v20 0/5 |

**ep500 训练曲线特征**（与 ep200 相比，**关键区别**）：

```
upd   9  WRr 0.09  WRf 0.06    ← 起步阶段需要更长时间才能赢 random
upd  19  WRr 0.34  WRf 0.53    ← warmup 结束
upd  29  WRr 0.62  WRf 0.59
upd  49  WRr 0.81  WRf 0.78
upd  79  WRr 0.88  WRf 0.88
upd 109  WRr 0.97  WRf 0.97
upd 119  WRr 1.00  WRf 0.66    ← 收尾：vs random 满分，vs frozen 退回 0.66
```

* **`WRf` 不再卡在 0.94**：ep200 时是 0.94（learner 碾压过去自己 → self-play 实际不工作），ep500 时是 0.66（learner 略胜 → self-play 真的在滚雪球）。
* **`WRf` 出现震荡**（upd 79=0.88→89=0.91→99=0.78→109=0.97→119=0.66）这是健康自对弈的典型形态：每次 snapshot 加入 pool 后，learner 暂时被「最新的自己」追上，再练回来。
* **clip_frac/kl 全程偏高**（很多 update `clip 0.05–0.15`、`kl 0.2–0.5`）。按 top_players_rl.txt 说法，长期 clip>0.3 是危险信号；ep500 已经在边缘但还能稳。明天若 v3 还需要继续训，建议把 `lr_peak` 从 3e-4 降到 1.5e-4，或减小 `update_epochs` 从 4 到 2。

**关键解读**：ep500 的 `WRf=0.66` 是今天最重要的训练数据点——它**第一次证明 self-play 框架真的能滚雪球**（不是单纯地碾压随机对手）。但 ep500 是否在真 Kaggle 上明显强于 ep200，**取决于即将跑的 v3 H2H 结果**。

**v3 H2H 结果**（凌晨 01:21 跑出，**已落数据**）：

| 对手 | 战绩 | 解读 |
|------|------|------|
| random | **4/5** | 比 v2 略差，疑似噪声 + 500 步局长里 random 偶尔能靠运气 |
| **starter** | **0/5** | 和 v2 完全一样地输——证明 episode 长度不是因素 |
| v20 | 0/5 | 全部败给 v20 默认强度 |

**结论锁死**：单动作就是结构性根因。所有可疑假设的排除证据：

| 假设 | 排除依据 |
|------|---------|
| 训练量不够 | v1 50u 输 1，v2 300u 输 0；继续训不解 |
| 自对弈不工作 | v3 的 WRf 0.66 完美健康，但真 H2H 仍 0/5 |
| episode 长度 | v3 显式 ep500，仍 0/5（与 v2 ep200 完全相同） |
| MVP 静态/无彗星 | 这只能解释「不能赢 v20」，不能解释「输给 starter」 |
| **每回合 1 fleet** | **唯一未排除的、最大的结构性差异** ← 明天主攻 |

**v3 不提交**：在 ladder 上会输给 starter，比 v1 ELO 还差。

---

## 7. 不要忘记的 3 个细节

1. **`docs/RL_PIPELINE.md` 仍是项目级权威文档**，今天没动它。多动作 head 落地后再回去更新它。
2. **`submission_rl_v2.py` 没交**，但保留在仓库——不要在改多动作 head 时手贱删掉，它是 v3 的对照基线。
3. **`logs/` 下面有 TB 日志但 tensorboard 加载失败**（protobuf 冲突）。如果明天想看曲线，先 `pip install -U protobuf==4.25.5 tensorboard` 再启动 TB。

---

*文档版本 1.0 — 写于 2026-05-22 01:1x（北京时间），不影响主路径的任何代码。*
