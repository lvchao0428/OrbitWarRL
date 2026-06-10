# Archive — 与当前路线无关的历史代码

> 归档时间: 2026-06-10 (Day14)
> 当前活跃路线: **v12_lux_b** (大模型 symmetric self-play) + **v14** (精细化兵力分配)

本目录包含 Day1–Day12 期间的实验代码、早期 submission、历史配置等。
这些代码不影响当前训练和评测流程，归档后可减少项目根目录噪声。

## 目录结构

| 子目录 | 内容 |
|--------|------|
| `scripts_legacy/` | v11 系列评测脚本 (run_f31_eval ~ run_f44_eval), v9 训练, BC pipeline 等 |
| `configs_legacy/` | MVP/v4–v9/v11 系列 YAML 配置 (~50 个) |
| `submissions_legacy/` | v1–v11 导出的 submission agent (~50 个, 含权重) |
| `docs_legacy/` | Day1–Day9 进展日记, 旧 runbook |
| `references_legacy/` | 参考方案 (halite3, planet-wars, lux-s2) |
| `kaggle-lux-2024/` | Frog Parade Lux S3 参考代码 (已提取完关键思路) |
| `bc_module/` | Behavior Cloning 模块 (Day3 已取消该方向) |
| `ckpt_*/` | 旧版本 checkpoint 目录 (mvp, selfplay, v11_f26/f29/f38/f44 等) |
| `rollout_4p.py` | 4-player rollout (未使用) |
| `*.py` | 旧版诊断脚本 |
| `*.txt / *.ipynb` | 早期数据集/笔记 |

## 如何恢复

如需回退某个实验:
```bash
cp _archive/configs_legacy/multi_action_v11_f29.yaml orbit_wars_rl/configs/
cp _archive/scripts_legacy/run_f29_eval.sh scripts/
```
