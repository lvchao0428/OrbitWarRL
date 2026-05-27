# DAY6 进展

> **2026-05-27 PM 更新**。4k replay 完成：**garr/spf 升、bin0 恶化** → **Phase C Plan B buffer**（从 `ckpt_003199.pkl` 续跑）。

---

## 1. 当前进展

| 阶段 | 状态 | replay vs v20 (first-80) |
|---|---|---|
| v11 G1 (K=16, R4 ON) | ✅ 完成 | spf **1.50** garr 7.86 flip 0.66% — FAIL |
| v11 G2 (curriculum) | ✅ 完成 | spf **1.09** garr 9.25 flip 0.31% — FAIL，**跳过 G3** |
| **k8_no_emit** (K=8, R4 OFF) | ✅ 800 upd + replay | spf **2.32** garr **27.45** flip 0.85% — **方向对，upd 不够** |
| k8_full (K=8, R4 ON) | ✅ 800 upd + replay | spf **1.92** garr 25.24 bin0+1 **59.5%** — **比 no_emit 更差，R4 仍有害** |
| k16_no_emit (K=16, R4 OFF) | ✅ 800 upd + replay | spf **3.15** garr 26.86 bin0+1 **58.6%** — **略好于 k8，但 pct 仍差** |
| v20 state buffer | ✅ **已采集** | `data/v20_states_200g.npz` |
| **k8_no_emit 4k** | ✅ @3199 + replay | spf **3.43** garr **66.6** flip 1.40% bin0 **72%** — **部分进步，pct 更差** |
| **Plan B buffer** | 🔜 下一步 | 从 4k ckpt + v20 state reset 50% |

### replay 数字汇总（first-80，5 局 vs v20）

| metric | v9b | k8_no_emit | k16 | **k8_4k** | v20 | 目标 |
|---|---|---|---|---|---|---|
| spf | 4.76 | 2.32 | 3.15 | **3.43** | 18.7 | > **4.76** |
| garr | 31.9 | 27.5 | 26.9 | **66.6** ✅ | 204.6 | > **31** |
| flip | 3.88% | 0.85% | 0.83% | **1.40%** | 14.7% | > **3%** |
| bin0 | — | 25.4% | 38.3% | **72.1%** ❌ | 0.7% | ↓ |
| bin0+bin1 | — | 52.7% | 58.6% | **80.1%** ❌ | 4.4% | < **40%** |
| emit=1/turn | — | 36% | — | **84%** ❌ | 25% | ↓ |
| WR | — | 0/5 | 0/5 | **0/5** | 5/5 | — |

**4k vs @800 对比（first-80）**：

| | @800 | @4k | Δ | 解读 |
|---|---|---|---|---|
| spf | 2.32 | 3.43 | +48% | 有进步，仍远低于 v9b 4.76 |
| garr | 27.5 | 66.6 | +142% | **过 Day5 gate (>60)** |
| flip | 0.85% | 1.40% | +65% | 仍 fail |
| bin0 | 25.4% | 72.1% | **+187%** | pct head **退化**，不是改善 |
| emit=1 | 36% | 84% | +133% | vs v20 时退化为「每 turn 只发 1 次 + bin0」 |

**4k 训练 vs replay 鸿沟（仍成立）**：

| | 训练 @3199 | replay @3199 |
|---|---|---|
| spf | 190.6 | 3.43 |
| garr | 513 | 66.6 |
| e2+ | 78% | 14.5% (emit≥2) |

**结论**：4k self-play 抬高了 garrison 水平，但 **pct 更依赖 bin0、emit 更单发**——在 vs v20 分布外比 @800 更糟。**不再续跑纯 self-play**；主线切 **Plan B buffer**（让 pct 在 v20 中期 garr 下学习）。

**1 舰 launch 机制**：`ships = max(1, floor(驻军 × pct))`；驻军 ~10、pct=10% → 1 舰。训练 spf≈200 因 self-play 高 garr，vs v20 early 暴露。

---

## 2. 接下来动作（按顺序）

### Phase A — Ablation ✅ 已完成

三路 @799 + replay 已齐，见 §1 表。**R4 永久关闭，K=8 主线不变。**

```bash
column -t -s $'\t' logs/v11_ablation_summary.tsv
# k8_no_emit  spf_train=201.6
# k8_full       spf_train=140.7  (更差)
# k16_no_emit   spf_train=108.4  replay spf=3.15 (略好但 pct 仍差)
```

---

### Phase B — 4k 状态与 ckpt 排查

**训练其实跑完了，不是没训。** `logs/v11_k8_no_emit_4k.log` 有完整 3200 行（upd 0→3199），今天 13:27 结束，`charlie-ultra` 上 resume 自 `k8_no_emit/ckpt_000799.pkl`。

**ckpt 文件名易错**：保存用的是**本 run 的 upd 序号**（0–3199），不是全局 3999：

| 文档旧写法（错） | 实际最后 ckpt |
|---|---|
| `ckpt_003999.pkl` | **`ckpt_003199.pkl`** |

每 200 upd 存一次，应有 16 个：`ckpt_000199.pkl` … `ckpt_003199.pkl`。

**5090 上若目录空，先搜全盘再决定重训：**

```bash
cd ~/project/OrbitWarRL   # 确认 cwd

# 1) 目录里有什么
ls -la ckpt_multi_action_v11_k8_4k/

# 2) 是否写到了别的 cwd（相对路径 ./ckpt_...）
find ~/project -name 'ckpt_003199.pkl' 2>/dev/null
find ~/project -name 'ckpt_*k8_4k*' 2>/dev/null

# 3) log 里应有 [ckpt] saved ...（旧 log 无此行；新 run 会有）
grep '\[ckpt\]' logs/v11_k8_no_emit_4k.log | tail -5
```

若 `find` 也找不到 → ckpt 已丢，需**重跑 4k**（~14h）：

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
    --log-dir logs/v11_k8_no_emit_4k_r2 \
    --resume-from ckpt_multi_action_v11_k8_no_emit/ckpt_000799.pkl \
  > logs/v11_k8_no_emit_4k_r2.log 2>&1 &
echo "pid=$!"
# 启动后确认：ls ckpt_multi_action_v11_k8_4k/ 应在 upd~200 后出现 ckpt_000199.pkl
```

**找到 ckpt 后 replay gate：**

```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_k8_4k/ckpt_003199.pkl v11_k8_4k_u3199

# 中途 checkpoint
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_k8_4k/ckpt_001999.pkl v11_k8_4k_u1999
```

| 4k replay 结果 | 动作 |
|---|---|
| spf > 4.76 或 garr > 31 | PROMOTE |
| **实际：garr 66 ✅ spf 3.43 bin0 72%** | → **Phase C Plan B**（不再纯 self-play 续跑） |
| spf < 3 | Plan B 或回退 @800 ckpt |

---

### Phase C — Plan B buffer（**当前主线，5090 立即启动**）

**动机**：4k replay 证明 self-play 只抬 garr，pct 更锁 bin0。Buffer 用 v20 中期状态 reset，在真实 garrison 下重学 pct。

```bash
bash scripts/run_v11_plan_b.sh from4k
# resume: ckpt_multi_action_v11_k8_4k/ckpt_003199.pkl
# buffer: data/v20_states_200g.npz @ 50% reset
# ~800 upd ≈ 4h
```

800 upd 后 replay + pct check：

```bash
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_buf_from4k/ckpt_000799.pkl v11_buf_from4k
```

**Plan B gate（比 spf 更看 pct）**：

| 指标 | 当前 4k | Plan B 目标 |
|---|---|---|
| bin0 | 72% | **< 40%** |
| bin0+bin1 | 80% | **< 30%** |
| spf | 3.43 | > **4.76** |
| emit=1/turn | 84% | **< 50%** |

若 Plan B 800 upd 后 bin0 仍 > 50% → 提 `buffer_reset_ratio` 到 0.70 再跑 800 upd。

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

### 6.1 5090 上今晚可做

- [x] **Phase A 收尾**：ablation 三路 + replay 已完成
- [x] **k16_no_emit replay**：spf=3.15, bin0+1=58.6%
- [x] **Phase B 4k 训练**：@3199 完成（log 已同步）
- [x] **4k replay gate**：spf=3.43 garr=66.6 bin0=72% → **切 Plan B**
- [ ] **Phase C 启动**：`bash scripts/run_v11_plan_b.sh from4k`
- [ ] 800 upd 后 replay：`v11_buf_from4k` + pct check
- [ ] 若 bin0+1 仍 > 40%：**Phase C Plan B**
  ```bash
  bash scripts/run_v11_plan_b.sh from4k
  ```

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
| 4k 最终 | `logs/replay_analyze/v11_k8_4k_u3199_vs_v20.json` |
| Plan B | `logs/replay_analyze/v11_buf_from4k_vs_v20.json` |

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

| | k8_no_emit | k8_full | k16_no_emit | v20 |
|---|---|---|---|---|
| bin0+bin1 | 52.7% | 59.5% | 58.6% | ~9.9% |
| spf | 2.32 | 1.92 | 3.15 | 17.0 |

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
- [ ] @3199：`bash scripts/quick_replay.sh ckpt_multi_action_v11_k8_4k/ckpt_003199.pkl v11_k8_4k_u3199`
- [ ] replay 后 pct check（改 path 为 `v11_k8_4k_u3199_vs_v20.json`）
- [ ] 决策（§2 Phase B 表）：PROMOTE / 再加 4k / Plan B

### 6.7 Plan B（4k pct 仍 bin0+1>40% 时）

- [x] buffer 已就绪：`data/v20_states_200g.npz`
- [x] buf_from4k 800 upd 完成；replay spf/garr 过线，bin0 仍 ~61%
- [ ] **Plan B mix**（v20 + top10 平衡 buffer，仍只做 state 对齐、无 action BC）：

  **5090 一次性建 buffer：**
  ```bash
  bash scripts/build_mixed_buffer.sh
  # smoke: bash scripts/build_mixed_buffer.sh --smoke
  ```

  产出：
  - `data/top10_winner_states.npz` ← top10 JSON winner 视角 + flip aug
  - `data/mixed_v20_top10.npz` ← v20 与 top10 **等量** subsample 合并

  **训练（从 buf_from4k @799 resume）：**
  ```bash
  bash scripts/run_v11_plan_b_mix.sh
  tail -f logs/v11_buf_mix.log
  ```

  **800 upd 后 replay：**
  ```bash
  bash scripts/quick_replay.sh \
      ckpt_multi_action_v11_buf_mix/ckpt_000799.pkl v11_buf_mix
  ```

  配置：`orbit_wars_rl/configs/multi_action_v11_buf_mix.yaml`
  - `buffer_path: data/mixed_v20_top10.npz`
  - `buffer_reset_ratio: 0.50`（与 buf_from4k 一致）
  - 其余 shaping / K=8 / R4 off 同 `run_v11_plan_b.sh`

  若 bin0 仍 >45%：试 `buffer_reset_ratio: 0.70` 或 `--no-balance` 全量 top10（改 merge）

- [ ] 旧单源 Plan B：`bash scripts/run_v11_plan_b.sh from4k`（仅 v20 buffer，对照用）

