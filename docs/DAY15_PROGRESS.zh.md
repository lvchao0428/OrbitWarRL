# DAY15 — v14d 三阶段 Curriculum 训练

> **2026-06-11 Day15**  
> **现状**：v14 系列（v14/v14b/v14c）在 5090 上已跑完一轮；v14c 聚合 flip≈11% 但 replay 显示开局乱射、0 capture，远不如 v13c（seed0 281 步胜）。根因是 self-play 开局乱射 → 走不到占点奖励链。  
> **今日动作**：启动 **v14d 三阶段 curriculum**（囤兵 → 占点 → 微调），从头训练，5090 无人值守。

---

## 0. Day14 结论摘要

| 版本 | vs v20 | 战术观感 |
|------|--------|---------|
| v13c_final | 1/4 胜 | 开局囤 21 回合后齐射，能占点；有射太阳等 bug |
| v14c u1599 | 0/4 负 | turn 4 乱射，15 次发射 0 capture，「没头苍蝇」 |
| v14 架构 | 16-bin pct + 6-dim pair | **参数空间保留**，缺 curriculum |

**核心洞察**：sparse reward 下，若开局策略不对，永远学不到 capture/胜利轨迹；但 self-play 里可以学到囤兵局部最优（v14c 末期 spf≈370），对 v20 无效。

---

## 1. v14d Curriculum 设计

### 三阶段概览

```
Phase A — 囤兵/耐心 (u≤1200+ext)
  allow_hold=true, force_emit_worth_it=false
  HOLD_BONUS=0.04, RELEASE_K=30, CAPTURE=0
  Gate: garr≥45, z0∈[0.30,0.70], spf∈[20,90]

Phase B — 占点/进攻 (u≤1500+ext, resume A)
  force_emit_worth_it=true, CAPTURE=0.10
  Gate: vs v20 flip≥8%, spf≥15, garr≥18 或 有胜场

Phase C — 战术微调 (u≤2000+ext, resume B)
  CAPTURE=0.05, HOLD=0, lr↓, entropy↓
  Gate: WLD≥1/4 或 flip≥10% + spf≥18 + z0∈[20%,45%]
```

### 文件

| 文件 | 说明 |
|------|------|
| `orbit_wars_rl/configs/multi_action_v14d_phase_{a,b,c}.yaml` | 三阶段配置 |
| `scripts/v14d_curriculum.sh` | 主控：顺序跑 A→B→C，gate 不通过则 extend |
| `scripts/v14d_gate_check.py` | Gate 判定 |
| `scripts/monitor_v14d.sh` | 远端状态轮询 |
| `logs/v14d_curriculum.state.json` | 当前阶段 / 状态 |

### 启动（5090）

```bash
cd ~/project/OrbitWarRL
nohup bash scripts/v14d_curriculum.sh > logs/v14d_curriculum.log 2>&1 &
tail -f logs/v14d_curriculum.log
bash scripts/monitor_v14d.sh
```

### Checkpoint 目录

- `ckpt_multi_action_v14d_a/` — Phase A
- `ckpt_multi_action_v14d_b/` — Phase B（从 A resume）
- `ckpt_multi_action_v14d_c/` — Phase C（从 B resume）

---

## 2. 执行进展

> 以下由 curriculum / monitor 自动更新；Day15 启动后填写。

| 阶段 | 状态 | 开始 | Gate | 备注 |
|------|------|------|------|------|
| A 囤兵 | **运行中** | 2026-06-11 00:20 CST | — | seed=1500, u≤1200, HOLD=0.04 |
| B 占点 | 待 A | — | — | resume best A |
| C 微调 | 待 B | — | — | resume best B |

5090 启动: `nohup bash scripts/v14d_curriculum.sh` (PID train ~1985061)  
Git: `6eb27a4` pushed + rsync 已同步。

---

## 3. 监控命令

```bash
# 本地
bash scripts/monitor_v14d.sh
bash scripts/monitor_v14d.sh --watch

# 远端
ssh charlie@www.ultrapp.online "tail -5 ~/project/OrbitWarRL/logs/v14d_curriculum.log"
ssh charlie@www.ultrapp.online "cat ~/project/OrbitWarRL/logs/v14d_curriculum.state.json"
ssh charlie@www.ultrapp.online "grep eval_vs_v20 ~/project/OrbitWarRL/logs/v14d_phase_*.log | tail -5"
```

---

## 4. 预期与停训条件

**符合预期（结束 curriculum）**：
- Phase C gate PASS，或 final eval vs v20 出现 ≥1 胜
- HTML replay seed0：首攻 turn≥15，captures>0，观感接近 v13c

**需人工介入**：
- Phase A 延长后仍 garr<30 → 加大 HOLD_BONUS
- Phase B flip 长期 <5% → 检查 export / worth_it
- 自博弈 spf>150 且 vs v20 恶化 → 提前进 Phase C 或加 RELEASE 惩罚

---

## 5. 与 v13c 的关系

v14d **从头训练**（不 resume v13c），但 **Phase A 模仿 v13c 开局节奏**（先囤后打）。  
v13c_final 仍是战术参考上界；curriculum 完成后用同 seed replay 三角对比。
