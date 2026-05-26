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
| k8_full (K=8, R4 ON) | ✅ 800 upd + replay | spf **1.92** garr 25.24 bin0+1 **59.5%** — **比 no_emit 更差，R4 仍有害** |
| k16_no_emit (K=16, R4 OFF) | ⏳ ablation 队列 | 对照：只关 R4 是否足够 |
| v20 state buffer | ✅ **已采集** | `data/v20_states_200g.npz` |
| k8_no_emit 4k 续跑 | 🔜 GPU 空出后启动 | resume @799 → +3200 upd |

### replay 数字汇总（first-80，5 局 vs v20）

| metric | v9b | G1 | G2 | k8_no_emit | k8_full | 目标 |
|---|---|---|---|---|---|---|
| spf | 4.76 | 1.50 | 1.09 | **2.32** | 1.92 | > **4.76** |
| garr | 31.85 | 7.86 | 9.25 | **27.45** | 25.24 | > **31** |
| flip | 3.88% | 0.66% | 0.31% | 0.85% | 0.89% | > **3%** |
| bin0+bin1 | — | 62.5% | 75.6% | **52.7%** | 59.5% | < **40%** |
| bin7 | — | ~5% | ~5% | 16.5% | 10.3% | ~v20 26% |
| WR | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | — |

**结论**：关 R4 + K=8 有效；k8_full 更差 → **永久关 EMIT_LOG**。HTML 回放确认 early game 仍大量 1 舰 launch（pct 小 + 低 garr）。**主线：k8_no_emit 4k 续跑。**

**1 舰 launch 机制**：`ships = max(1, floor(驻军 × pct))`；驻军 ~10、pct=10% → 1 舰。训练 spf≈200 因 self-play 高 garr，vs v20 early 暴露。

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

---

## 6. Checklist（下班 / 次日）

### 6.1 5090 上今晚可做（GPU 空出后）

- [ ] **Phase A 收尾**：等 ablation 跑完 k16_no_emit（若还在跑）
  ```bash
  tail -f logs/v11_ablation.launcher.log
  ```
- [ ] **k16_no_emit replay**（若尚未跑）：
  ```bash
  bash scripts/quick_replay.sh \
      ckpt_multi_action_v11_k16_no_emit/ckpt_000799.pkl v11_k16_no_emit
  ```
- [ ] **Phase B 启动 k8_no_emit 4k**（主线，见 §2 Phase B 完整 env var + 命令）
- [ ] 确认 4k log 有输出：`tail -f logs/v11_k8_no_emit_4k.log`

### 6.2 每条 ckpt replay gate（数字）

- [ ] 跑 replay（K 自动探测 template）：
  ```bash
  bash scripts/quick_replay.sh <ckpt.pkl> <tag>
  # 产出 logs/replay_analyze/<tag>_vs_v20.json
  #       logs/replay_analyze/<tag>_vs_v20.summary.txt
  ```
- [ ] 对照 summary 一行阈值（§3.4）：
  - [ ] spf > **3**（sign-of-life）/ > **4.76**（beat v9b）/ > **10**（gate）
  - [ ] garr > **20** / > **31** / > **60**
  - [ ] flip > **2%** / > **6%**
  - [ ] e2+ > **5%**
- [ ] **已有 JSON 路径**（勿用尚未存在的 4k 路径）：

| tag | JSON |
|---|---|
| k8_no_emit | `logs/replay_analyze/v11_k8_no_emit_vs_v20.json` |
| k8_full | `logs/replay_analyze/v11_k8_full_vs_v20.json` |
| k16_no_emit | `logs/replay_analyze/v11_k16_no_emit_vs_v20.json` |
| 4k 最终 | `logs/replay_analyze/v11_k8_4k_u3999_vs_v20.json`（4k 跑完后） |

### 6.3 pct 分布 check（比 spf 更敏感）

- [ ] **单 agent 明细**（改 `path` 即可）：
  ```bash
  python -c "
  import json
  path = 'logs/replay_analyze/v11_k8_no_emit_vs_v20.json'
  d = json.load(open(path))
  fb = d['aggregate_by_window']['first_80turns']['player_0']
  vals, dist = fb['pct_bin_values'], fb['pct_bin_distribution']
  print('=== player_0 first-80 ===')
  for v, p in zip(vals, dist):
      print(f'  bin@{v:.2f}: {100*p:.1f}%')
  print(f'  bin0+bin1: {100*(dist[0]+dist[1]):.1f}%')
  print(f'  spf={fb[\"mean_ships_per_fleet\"]:.2f}  garr={fb[\"mean_garrison_my\"]:.2f}')
  "
  ```
- [ ] **ours vs v20 一行对比**：
  ```bash
  python -c "
  import json
  path = 'logs/replay_analyze/v11_k8_no_emit_vs_v20.json'  # 改 tag
  d = json.load(open(path))
  for tag, key in [('ours', 'player_0'), ('v20', 'player_1')]:
      fb = d['aggregate_by_window']['first_80turns'][key]
      dist = fb['pct_bin_distribution']
      print(f'{tag}: bin0+1={100*(dist[0]+dist[1]):.1f}%  '
            f'bin4-7={100*sum(dist[4:]):.1f}%  '
            f'spf={fb[\"mean_ships_per_fleet\"]:.2f}  garr={fb[\"mean_garrison_my\"]:.2f}')
  "
  ```
- [ ] pct 目标：**bin0+bin1 < 30%**（v20 ~10%）；bin4–7 占比上升

| | k8_no_emit | k8_full | v20 |
|---|---|---|---|
| bin0+bin1 | 52.7% | 59.5% | ~9.9% |
| spf | 2.32 | 1.92 | 17.0 |

### 6.4 HTML 可视化回放（肉眼看 1 舰 spam）

- [ ] export（若还没有 submission）：
  ```bash
  bash scripts/quick_replay.sh ckpt_.../ckpt_000799.pkl v11_k8_no_emit
  ```
- [ ] 生成 HTML：
  ```bash
  python -m orbit_wars_rl.scripts.replay_html \
      --agent-a submission_rl_v11_k8_no_emit.py \
      --agent-b submission_v20_0513.py \
      --seed 0 \
      --out-dir logs/replay_html/v11_k8_no_emit_seed0
  ```
- [ ] 浏览器打开 `logs/replay_html/.../replay.html`（5090 → scp 到本地）
- [ ] 重点看 **Step 20–50**：小蓝三角带 `1` = bin0 + 低 garr

### 6.5 训练途中监控（4k 跑起来后）

- [ ] 最新 upd：`grep "^upd " logs/v11_k8_no_emit_4k.log | tail -1`
- [ ] monitor_train：`python -m orbit_wars_rl.scripts.monitor_train --once logs/v11_k8_no_emit_4k.log`
- [ ] 健康阈值（§3.2）：ev>0.7、spf>150、garr>300、clip<0.35、kl<0.10
- [ ] **训练 spf 高 ≠ vs v20 好**；决策只看 replay

### 6.6 4k 中途 / 最终 gate

- [ ] @1999：`bash scripts/quick_replay.sh ckpt_multi_action_v11_k8_4k/ckpt_001999.pkl v11_k8_4k_u1999`
- [ ] @3999：`bash scripts/quick_replay.sh ckpt_multi_action_v11_k8_4k/ckpt_003999.pkl v11_k8_4k_u3999`
- [ ] 4k 后 pct check（改 path 为 `v11_k8_4k_u3999_vs_v20.json`）
- [ ] 决策（§2 Phase B 表）：PROMOTE / 再加 4k / Plan B

### 6.7 Plan B（4k pct 仍 bin0+1>40% 时）

- [ ] buffer 已就绪：`data/v20_states_200g.npz`
- [ ] 启动：`SHAPING_EMIT_LOG=0.0` + `multi_action_v11_buf.yaml`（§2 Phase C）
- [ ] 800 upd 后 replay + pct check

