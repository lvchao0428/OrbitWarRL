# DAY6 进展

> 2026-05-26 晚更新。Day5 根因：**R4 EMIT_LOG + K=16** → self-play 多 emit 小 pct，vs v20 退化为 1 舰/launch。
> Day6 主线：**k8_no_emit 已验证方向正确 → 续跑 4k**。Ablation 对照 + Plan B buffer 并行就绪。

---

## 1. 当前进展

| 阶段 | 状态 | replay vs v20 (first-80) |
|---|---|---|
| v11 G1 (K=16, R4 ON) | ✅ 完成 | spf **1.50** garr 7.86 flip 0.66% — FAIL |
| v11 G2 (curriculum) | ✅ 完成 | spf **1.09** garr 9.25 flip 0.31% — FAIL，**跳过 G3** |
| **k8_no_emit** (K=8, R4 OFF) | ✅ 800 upd + replay | spf **2.32** garr **27.45** flip 0.85% — **方向对，upd 不够** |
| k8_full (K=8, R4 ON) | ⏳ **5090 训练中** | 对照：R4 在 K=8 下是否仍有害 |
| k16_no_emit (K=16, R4 OFF) | ⏳ ablation 队列 | 对照：只关 R4 是否足够 |
| v20 state buffer | ✅ **已采集** | `data/v20_states_200g.npz` |
| k8_no_emit 4k 续跑 | 🔜 GPU 空出后启动 | resume @799 → +3200 upd |

### replay 数字汇总（first-80，5 局 vs v20）

| metric | v9b | G1 | G2 | **k8_no_emit** | 目标 |
|---|---|---|---|---|---|
| spf | 4.76 | 1.50 | 1.09 | **2.32** ↑ | > **4.76**（最低）> 10（gate） |
| garr | 31.85 | 7.86 | 9.25 | **27.45** ↑ | > **31**（最低）> 60（gate） |
| flip | 3.88% | 0.66% | 0.31% | **0.85%** ↑ | > **3%**（最低）> 6%（gate） |
| bin0+bin1 | — | 62.5% | 75.6% | **52.7%** ↓ | < 40% 为佳 |
| bin7 (100%) | — | ~5% | ~5% | **16.5%** ↑ | 接近 v20 ~26% |
| WR | 0/5 | 0/5 | 0/5 | **0/5** | — |

**结论**：关 R4 + K=8 有效；train spf≈220 vs replay 2.32，gap 仍大但已从 G1 的 53× 收窄。**不加 R4，续跑步数。**

---

## 2. 接下来动作（按顺序）

### Phase A — 等 k8_full / k16_no_emit 跑完（ablation 自动串行，勿 kill）

```bash
# 看 ablation 当前阶段
tail -f logs/v11_ablation.launcher.log

# 看正在跑的 variant
ls -lt logs/multi_action_v11_*.log | head -3
tail -f logs/multi_action_v11_k8_full.log   # 或 k16_no_emit
```

**k8_full 完成后立即 replay：**
```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_k8_full/ckpt_000799.pkl v11_k8_full
```

**k16_no_emit 完成后：**
```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_k16_no_emit/ckpt_000799.pkl v11_k16_no_emit
```

**Ablation 决策（对照用，不阻塞主线）：**

| 结果 | 含义 |
|---|---|
| k8_full spf < k8_no_emit | R4 在 K=8 仍有害 → **永久关 EMIT_LOG** |
| k16_no_emit > k8_no_emit | K 不是主因，800 upd 不够或 pct head 问题 |
| 三路 replay spf 均 < 4.76 | 4k 续跑仍 fail → 启动 Plan B |

---

### Phase B — k8_no_emit 4k 续跑（**GPU 空出后立即启动，主线**）

k8_full 跑完或你主动停掉 ablation 后：

```bash
ORBITWARS_SHAPING_EMIT_LOG=0.0 \
ORBITWARS_SHAPING_EMIT_GATED=0 \
ORBITWARS_SHAPING_PLANET_SHARE=0.005 \
ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0 \
ORBITWARS_SHAPING_RELEASE=0.05 \
ORBITWARS_SHAPING_RELEASE_K=20.0 \
ORBITWARS_SHAPING_CAPTURE=0.02 \
nohup python -m orbit_wars_rl.scripts.train \
    --config orbit_wars_rl/configs/multi_action_v11_k8_4k.yaml \
    --log-dir logs/v11_k8_no_emit_4k \
    --resume-from ckpt_multi_action_v11_k8_no_emit/ckpt_000799.pkl \
  > logs/v11_k8_no_emit_4k.log 2>&1 &
echo "pid=$!"
```

- 配置：[`multi_action_v11_k8_4k.yaml`](../orbit_wars_rl/configs/multi_action_v11_k8_4k.yaml)
- 总量：800 + 3200 = **4000 upd**，约 **14h** @ 6100 sps
- ckpt 每 200 upd → `ckpt_multi_action_v11_k8_4k/ckpt_XXXXXX.pkl`

**中途 checkpoint replay（不必等 4000 完）：**
```bash
# 例如 @1199 / @1999 / @3199
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_k8_4k/ckpt_001999.pkl v11_k8_4k_u1999
```

**4k 完成后最终 gate：**
```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_k8_4k/ckpt_003999.pkl v11_k8_4k_u3999
```

| 4k replay 结果 | 动作 |
|---|---|
| spf > 4.76 或 garr > 31 | **PROMOTE**，可再加 4k 或上 C1 strong opp |
| spf 3~4.76，趋势仍升 | 再加 4k 或 Plan B buffer 30% |
| spf < 3，与 800 upd 差不多 | → Plan B |

---

### Phase C — Plan B（4k 仍 fail 时）

buffer 已就绪，直接训练：

```bash
SHAPING_EMIT_LOG=0.0 \
nohup python -m orbit_wars_rl.scripts.train \
    --config orbit_wars_rl/configs/multi_action_v11_buf.yaml \
    --log-dir logs/v11_buf_run1 \
  > logs/v11_buf_run1.log 2>&1 &
```

800 upd 后：
```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_buf/ckpt_000799.pkl v11_buf
```

---

## 3. 训练途中监控

### 3.1 快速看进度

```bash
# 最新一行训练指标
grep "^upd " logs/v11_k8_no_emit_4k.log | tail -1

# 实时 tail
tail -f logs/v11_k8_no_emit_4k.log

# monitor_train 汇总 + 告警（推荐）
python -m orbit_wars_rl.scripts.monitor_train --once logs/v11_k8_no_emit_4k.log

# 持续刷新（60s）
python -m orbit_wars_rl.scripts.monitor_train --interval 60 logs/v11_k8_no_emit_4k.log
```

### 3.2 训练 log 字段含义

| 字段 | 含义 | 健康范围（k8_no_emit） |
|---|---|---|
| `opp` | rand / frzn / strn / buf | frzn+rand 交替正常 |
| `ev` | explained variance | **> 0.7** 好；**< 0.5** WARN；**< 0.3** 坏 |
| `spf` | self-play 每舰队平均舰数 | **> 150** 正常；**< 50** 异常 |
| `garr` | 己方行星平均驻军 | **> 300** 正常 |
| `tG` | 总驻军 | **> 1500** 正常 |
| `e2` | 每 turn emit≥2 比例 | 0.65~0.75 正常（self-play） |
| `z0` | 零 emit 比例 | **< 0.05** 正常 |
| `pkR` | peak/mean garr | 5~8 正常 |
| `clip` | PPO clip fraction | **< 0.25** 好；**> 0.35** 更新过激 |
| `kl` | approx KL | **< 0.05** 好；**> 0.10** 学习率过大 |
| `WRr` | vs random（eval 行） | **> 0.90** |
| `WRf` | vs frozen pool | **> 0.85** |

### 3.3 训练 vs replay 对照（sign-of-life）

训练 @700 参考 vs k8_no_emit replay @800：

| | 训练 @700 | replay @800 | 期望 4k 后 replay |
|---|---|---|---|
| spf | ~200 | 2.32 | **> 4.76** |
| garr | ~400 | 27.45 | **> 31** |
| flip | — | 0.85% | **> 3%** |

**replay 才是决策依据**；训练 spf 高不代表 vs v20 好（G1 教训）。

### 3.4 replay gate 阈值（`quick_replay.sh` 输出）

| 指标 | sign-of-life | Day5 gate | 说明 |
|---|---|---|---|
| spf | > **3** | > **10** | 每 launch 平均舰数 |
| garr | > **20** | > **60** | 己方平均驻军 |
| flip | > **2%** | > **6%** | 到达翻转率 |
| e2+ | > 5% | > 5% | multi-emit 比例（已通过） |
| bin0+bin1 | < 50% | — | pct 小值占比，越低越好 |
| WR | — | — | 5 局样本小，看行为指标 |

---

## 4. 工具备忘

**Export / replay 数字 gate（K 自动探测）：**
```bash
bash scripts/quick_replay.sh <ckpt.pkl> <tag>
# K=8 → submission_rl_v11_k8.py
# K=16 → submission_rl_v11.py
```

**HTML 可视化回放（Kaggle 原生 player，浏览器逐步播放）：**

先 export submission（`quick_replay.sh` 会自动做，或手动）：

```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_k8_full/ckpt_000799.pkl v11_k8_full
# 产出 submission_rl_v11_k8_full.py
```

生成 JSON + HTML（单局，~30-120s）：

```bash
python -m orbit_wars_rl.scripts.replay_html \
    --agent-a submission_rl_v11_k8_full.py \
    --agent-b submission_v20_0513.py \
    --seed 0 \
    --out-dir logs/replay_html/v11_k8_full_seed0

# 浏览器打开（5090 上把 html 拉到本地，或 scp）
# logs/replay_html/v11_k8_full_seed0/replay.html
```

已有 JSON 只重渲染 HTML（不重新跑对局）：

```bash
python -m orbit_wars_rl.scripts.replay_html \
    --from-json logs/replay_html/v11_k8_full_seed0/replay.json \
    --out-dir logs/replay_html/v11_k8_full_seed0
```

Jupyter 里 inline 播放（可选）：

```python
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 0})
env.run(["submission_rl_v11_k8_full.py", "submission_v20_0513.py"])
env.render(mode="ipython", width=900, height=700)
```

**文本逐步 dump（不看 HTML 时用）：**

```bash
python -m orbit_wars_rl.scripts.replay_dump \
    --agent-a submission_rl_v11_k8_full.py \
    --agent-b submission_v20_0513.py \
    --seed 0 --dump-every 10
```

**Ablation 汇总：**
```bash
column -t -s $'\t' logs/v11_ablation_summary.tsv
```

---

## 5. 日志路径

| 用途 | 路径 |
|---|---|
| Ablation launcher | `logs/v11_ablation.launcher.log` |
| k8_no_emit 800 upd | `logs/multi_action_v11_k8_no_emit.log` |
| k8_full | `logs/multi_action_v11_k8_full.log` |
| k16_no_emit | `logs/multi_action_v11_k16_no_emit.log` |
| **HTML 回放** | `logs/replay_html/<tag>_seed<N>/replay.html` |
| **4k 续跑** | `logs/v11_k8_no_emit_4k.log` |
| Replay JSON | `logs/replay_analyze/v11_<tag>_vs_v20.json` |
| Replay gate 一行 | `logs/replay_analyze/v11_<tag>_vs_v20.summary.txt` |
| v20 buffer | `data/v20_states_200g.npz` |
