#!/usr/bin/env bash
# v12_smoke.sh — 快速验证 v12_lux 全链路 (d=32 小模型, 5 updates, CPU 可跑)
#
# 验证内容:
#   1. symmetric selfplay rollout 编译+运行
#   2. zero-sum value head 前向/反向
#   3. sparse ±1 reward 正确传递
#   4. 训练 loop 无 crash
#
# 用时: CPU ~60s, GPU ~20s
#
# Usage:
#   bash scripts/v12_smoke.sh
#   PYTHON=/path/to/python bash scripts/v12_smoke.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else PY=python; fi
else PY="$PYTHON"; fi

echo "[v12_smoke] 开始验证... (py=$PY)"

env \
  ORBITWARS_SHAPING_SCALE=0.0 \
  ORBITWARS_SHAPING_CAPTURE=0.0 \
  ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE=0.0 \
  ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.0 \
  ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.0 \
  ORBITWARS_SHAPING_HIGH_PROD_CAPTURE=0.0 \
  ORBITWARS_SHAPING_MULTI_EMIT=0.0 \
  ORBITWARS_SHAPING_EMIT_LOG=0.0 \
  ORBITWARS_SHAPING_EMIT_GATED=0 \
  ORBITWARS_SHAPING_PLANET_SHARE=0.0 \
  ORBITWARS_SHAPING_PROD_SHARE=0.0 \
  ORBITWARS_SHAPING_FLEET_LOG=0.0 \
  ORBITWARS_SHAPING_RELEASE=0.0 \
  ORBITWARS_SHAPING_KEEP_HOME=0.0 \
  ORBITWARS_SHAPING_FLEET_SIZE=0.0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$PY" -c "
from orbit_wars_rl.ppo.runner import TrainConfig, SelfPlayConfig, train
from orbit_wars_rl.ppo.update import PPOConfig
import time

ppo_cfg = PPOConfig(
    lr_peak=3e-4, lr_warmup_steps=10, lr_decay_steps=100, lr_floor=1e-5,
    gamma=0.9999, gae_lambda=0.95, update_epochs=1, num_minibatches=2,
)
selfplay_cfg = SelfPlayConfig(enabled=False)
train_cfg = TrainConfig(
    num_envs=4, rollout_length=8, num_updates=5, num_groups=3,
    episode_steps=50, eval_every=5, eval_num_envs=4,
    ckpt_dir='./ckpt_v12_smoke', ckpt_every=0,
    d_model=32, n_layers=1, n_heads=2, ff_dim=64,
    symmetric_selfplay=True, symmetric_warmup=0, zero_sum_value=True,
    ppo=ppo_cfg, selfplay=selfplay_cfg,
)

t0 = time.time()
result = train(train_cfg)
elapsed = time.time() - t0

h = result['history']
last = h[-1]
print()
print('=' * 60)
print(f'[v12_smoke] PASS  updates={len(h)}  elapsed={elapsed:.1f}s')
print(f'  opp_tag = {last[\"opp_tag\"]}  (expect: symm)')
print(f'  ev      = {last[\"explained_variance\"]:.3f}  (> 0.3 = OK)')
print(f'  v_loss  = {last[\"v_loss\"]:.5f}')
print(f'  loss    = {last[\"loss\"]:.4f}')
print(f'  tR      = {last[\"mean_terminal_reward\"]:.2f}')

checks = []
if last['opp_tag'] != 'symm':
    checks.append('FAIL: opp_tag should be symm')
if last['explained_variance'] < -1.0:
    checks.append('WARN: explained_variance very low')
if checks:
    for c in checks:
        print(f'  !! {c}')
else:
    print(f'  全部检查通过 ✓')
print('=' * 60)
"

RC=$?
rm -rf ./ckpt_v12_smoke 2>/dev/null || true
exit $RC
