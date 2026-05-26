# DAY6 进展

> Day5 结论：v11 G1 vs v20 **0/5**，spf 1.50。根因：R4 EMIT_LOG + K=16 → 多 emit + 小 pct，vs v20 退化为 1 舰/launch。
> Day6：G2 已验证 curriculum 无效，主线切 **ablation**。Plan B buffer 代码已就绪。

---

## 数字对比

| metric | v9b | v11 G1 | **v11 G2** | Day6 gate |
|---|---|---|---|---|
| spf | 4.76 | 1.50 | **1.09** ↓ | > 4.76 |
| garr | 31.85 | 7.86 | **9.25** | > 31 |
| flip | 3.88% | 0.66% | **0.31%** ↓ | > 3% |
| bin0+bin1 | — | 62.5% | **75.6%** ↓ | — |
| emit≥2 | 2.8% | ~43% | **62.3%** | — |

**G2 结论：curriculum 无效，bin0+bin1 从 G1 的 62.5% 恶化到 75.6%，pct head 更退化。跳过 G3，直接 ablation。**

---

## 下一步：Ablation（现在启动）

```bash
nohup bash scripts/run_v11_ablation.sh \
  > logs/v11_ablation.launcher.log 2>&1 &
```

| tag | K | R4 emit | 假设 |
|---|---|---|---|
| `k8_no_emit` | 8 | **OFF** | 主假设：关 R4 + 降 K 修复 pct |
| `k8_full` | 8 | ON | K=8 本身是否不够？ |
| `k16_no_emit` | 16 | **OFF** | 只关 R4 是否足够？ |

每路完成后：
```bash
for v in k8_no_emit k8_full k16_no_emit; do
  bash scripts/quick_replay.sh \
    "ckpt_multi_action_v11_${v}/ckpt_000799.pkl" "v11_${v}"
done
```

### 决策树

| ablation 结果 | 动作 |
|---|---|
| `k8_no_emit` spf > 4.76 | overnight 4000 upd |
| `k8_full` 过，`k8_no_emit` 不过 | R4 不是主因，看 K；查 pct head |
| `k16_no_emit` 过，k8 不过 | K 不是主因，查 pct head / R2 |
| **三路全 FAIL** | → Plan B（立即启动 buffer 采集） |

---

## Plan B：State-Buffer Curriculum（代码已就绪，可提前 CPU 后台采集）

```bash
# Step 1：CPU 后台采集（~40 min，不占 GPU）
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  python -m orbit_wars_rl.bc.collect_states \
    --agent submission_v20_0513.py \
    --num-games 200 \
    --out data/v20_states_200g.npz \
  > logs/collect_states.log 2>&1 &

# Step 2：ablation fail 后训练（K=8 + R4 off + 30% buffer reset）
SHAPING_EMIT_LOG=0.0 \
nohup python -m orbit_wars_rl.scripts.train \
    --config orbit_wars_rl/configs/multi_action_v11_buf.yaml \
    --log-dir logs/v11_buf_run1 \
  > logs/v11_buf_run1.log 2>&1 &

# Step 3：800 upd gate
bash scripts/quick_replay.sh \
    ckpt_multi_action_v11_buf/ckpt_000799.pkl v11_buf
```

---

## Export 备忘

```bash
# 必须用 v11 template（v4 template → agent 静默返回 []）
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  python -m orbit_wars_rl.scripts.export_submission \
  --ckpt ckpt_multi_action_v11_<tag>/ckpt_000799.pkl \
  --template submission_rl_v11.py \
  --out submission_rl_v11_<tag>_u799.py
```

---

## 日志路径

| 用途 | 路径 |
|---|---|
| Ablation launcher | `logs/v11_ablation.launcher.log` |
| Ablation 汇总 | `logs/v11_ablation_summary.tsv` |
| Replay gate | `logs/replay_analyze/v11_<tag>_vs_v20.summary.txt` |
| Plan B buffer | `logs/collect_states.log` |
