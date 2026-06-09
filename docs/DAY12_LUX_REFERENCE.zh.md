# DAY12 — 参考 Frog Parade (Lux S3 第2名) 重构方案

> **2026-06-09 启动**  
> 基于 `kaggle-lux-2024/` 完整代码 + 获奖感言深度分析。  
> **核心思路**：简化系统，回归经证明有效的基础框架，然后规模化训练。

---

## 0. 差距诊断（Frog Parade vs 我们）

| 维度 | Frog Parade (Lux S3 #2) | 我们 (v11_f44) | 差距分析 |
|------|-------------------------|----------------|---------|
| **模型规模** | 10M params, d=256, 8 blocks | 680K, d=128, 2 layers | 15x 参数量 |
| **训练量** | 300M steps (~8天连续) | ~几M steps/实验 | 差一个数量级 |
| **Self-play** | 最简对称 (同模型双方, N=32) | frozen/strong/buffer/random 四重混合 | 过度复杂 |
| **Reward** | 先 MATCH_WINNER 跑稳 → 切 FINAL_WINNER | 7-8种 shaping 反复调参 | 策略畸形 |
| **Value Head** | Zero-sum softmax (训练时看双方) | 标准单方 value | 缺零和约束 |
| **gamma** | 0.9999-1.0 | 0.99-0.997 | 太低，500步 episode 信号消失 |
| **Test-time** | 3种 augmentation 平均后随机采样 | 确定性/无增强 | 缺策略平滑 |
| **总代码量** | ~10.8K Rust + ~6.5K Python | ~6K Python | 规模相当 |

### 核心教训

1. **所有成功的 RL 方案都先用 dense reward 跑稳，再切 sparse**  
   - Frog Parade: MATCH_WINNER → FINAL_WINNER  
   - Lux S3 第5名 (Kiwis): 小地图 dense → 大地图 sparse  
   - Lux S3 第1名: 动态 reward scaling  
   - **我们**: 从纯 +1/-1 cold start → 反复塌缩

2. **Self-play 越简单越好**  
   - Frog Parade: "I used the most basic form where the same model plays for both players"  
   - 我们的 4 种对手混合引入了 shape mismatch bug、buffer 污染训练指标等问题

3. **训练时间充足比复杂 shaping 更重要**  
   - 300M steps @ 430 SPS = ~8 天连续训练  
   - "mostly plateaued by 200M steps, but continued to exhibit small gradual improvements"

---

## 阶段一：简化 + 规模化（立即执行）

### 1.1 改动清单

| 改动 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| d_model | 128 | 256 | 匹配 Frog Parade; 表达容量不足是根因之一 |
| n_layers | 2 | 4 | 同上 |
| ff_dim | 512 | 1024 | 标准 4x d_model |
| gamma | 0.99 | 0.9999 | Frog Parade 用 0.9999-1.0; episode 500 步需要高折扣 |
| rollout_length | 256 | 64 | 缩短 rollout 降低 GPU 内存; Frog Parade 用 32 |
| num_envs | 128 | 64 | 降低到适配更大模型的内存需求 |
| self-play | 4 种混合 | **纯对称** | Frog Parade 证明最简 self-play 就够 |
| reward | CAPTURE shaping | **sparse ±1 (game winner)** | 先跑稳再加 |
| num_updates | 500 | **10000** | 连续跑 ~3 天 |
| warmup_updates | 0-50 | **100** | 前 100 upd vs random 学基础 |
| emit_hard_stop | true | **false** | 去掉限制，让模型自由探索 |
| flip_hard_mask | true | **false** | 同上 |

### 1.2 自播放模式变化

**旧模式**（runner.py 490-529 行）:
```
roll = random()
if roll < strong_ratio:      → vs strong_ckpt
elif roll < strong+frozen:    → vs frozen pool
elif roll < strong+frozen+buf: → vs buffer + self
else:                          → vs random
```

**新模式**（Frog Parade 式纯对称）:
```
if update < warmup:           → vs random
else:                          → vs self (同一个 params 控制双方)
```

### 1.3 新增文件

| 文件 | 内容 |
|------|------|
| `orbit_wars_rl/ppo/rollout_symmetric.py` | 对称 self-play rollout: 同 params 双方 |
| `orbit_wars_rl/configs/multi_action_v12_lux.yaml` | 新配置 |
| `scripts/run_v12_lux.sh` | 训练脚本 |
| `submission_rl_v12_lux.py` | 提交模板 |

### 1.4 不变的部分

- 特征维度 (planet=33, fleet=10, global=18)
- 动作空间 (K=8 autoregressive)
- pair features (dst/pct/emit pair feats)
- 推理导出流程 (numpy_forward)

---

## 阶段二：Zero-Sum Value Head（阶段一启动后并行开发）

### 2.1 设计

参考 `kaggle-lux-2024/python/rux_ai_s3/models/critic_heads.py` 的 `ZeroSumCriticHead`:

```python
# 训练时: value head 同时看双方 obs，输出 softmax 归一化的胜率
# 推理时: value head 不计算 (omit_value=True)

class ZeroSumValueHead:
    def __call__(self, obs_p0, obs_p1) -> (v0, v1):
        raw_0 = value_mlp(obs_p0)  # scalar
        raw_1 = value_mlp(obs_p1)  # scalar  
        probs = softmax([raw_0, raw_1])  # 约束 v0 + v1 = 1
        return 2 * probs[0] - 1, 2 * probs[1] - 1  # map to [-1, 1]
```

### 2.2 为什么有效

- **零和约束**: V(p0) + V(p1) ≡ 0，GAE advantage 更准确
- **"作弊"**: 训练时看到对方 obs 让 value 估计更稳定（推理时不用 value）
- **对称 self-play 天然支持**: 同一个 forward pass 就能得到双方 value

### 2.3 改动范围

| 文件 | 改动 |
|------|------|
| `orbit_wars_rl/net/heads.py` | 新增 `ZeroSumValueHead` |
| `orbit_wars_rl/net/model.py` | ActorCritic 添加 `critic_mode: str` 参数 |
| `orbit_wars_rl/ppo/rollout_symmetric.py` | 传递双方 obs 给 value head |
| `orbit_wars_rl/ppo/update.py` | 处理双方 value 的 GAE |

---

## 阶段三：Test-time 增强（阶段一出结果后）

### 3.1 数据增强

Orbit Wars 地图有 4-fold 对称性（关于中心点的旋转+镜像）。

| 增强 | 空间变换 | 动作映射 |
|------|---------|---------|
| Identity | 无 | 无 |
| Rotate 180° | (x,y) → (100-x, 100-y) | planet_idx 重映射 |
| Mirror X | (x,y) → (100-x, y) | planet_idx 重映射 |
| Mirror Y | (x,y) → (x, 100-y) | planet_idx 重映射 |

### 3.2 推理流程

```python
# 4 个视角各做一次 forward pass
for aug in [identity, rot180, mirror_x, mirror_y]:
    obs_aug = aug.transform_obs(obs)
    logits_aug = model(obs_aug)
    logits_canonical = aug.inverse_transform_logits(logits_aug)
    all_logits.append(logits_canonical)

# 平均 log-probs 后采样
avg_logits = mean(all_logits)
action = sample(avg_logits)
```

### 3.3 改动范围

| 文件 | 改动 |
|------|------|
| `orbit_wars_rl/inference/augmentation.py` | 增强变换和逆变换 |
| `orbit_wars_rl/inference/numpy_forward.py` | 多视角平均推理 |
| `submission_rl_v12_lux.py` | 启用增强 |

---

## 执行顺序

```
Day12 AM: 阶段一完整实现 + smoke test + 远程启动训练
Day12 PM: 阶段二 zero-sum value head (训练中并行开发)
Day13:    检查训练曲线, 若趋势正确 → 阶段三 augmentation
Day14:    根据 @3000+ 结果决策:
          - WR vs v20 > 0 → 继续长训/微调
          - WR vs v20 = 0 但指标改善 → 加回微量 CAPTURE shaping
          - 全线 fail → 回退分析
```

---

## 关键超参速查

| 参数 | Frog Parade | 我们 v12 | 说明 |
|------|------------|---------|------|
| d_model | 256 | 256 | 匹配 |
| n_blocks/layers | 8 (CNN) | 4 (Transformer) | Transformer 效率更高 |
| total params | 10M | ~3-5M | 受限于 entity transformer |
| gamma | 0.9999 | 0.9999 | 匹配 |
| gae_lambda | 0.85 | 0.95 | 保持我们的值，Frog Parade 用 0.85 |
| rollout N | 32 | 64 | 稍长以补偿 env 数量减少 |
| num_envs | 64 × 2 players | 64 | 对称 self-play |
| lr | linear decay | warmup-cosine | 保持我们的 schedule |
| clip_eps | 0.2 | 0.2 | 匹配 |
| update_epochs | ? | 4 | 标准值 |
| minibatches | ? | 8 | 适配 batch size |
