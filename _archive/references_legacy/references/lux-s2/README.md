# Lux AI (Season 2 / NeurIPS) — reading map for Orbit Wars

Lux is **1v1 multi-unit resource RTS** on Kaggle with a mature SDK and public top solutions. Mechanics differ from Orbit Wars (grid/tiles vs continuous orbit), but **macro allocation, opponent modeling, and time-budgeted search** transfer well.

## Official links

- **Site:** [Lux AI Challenge](https://www.lux-ai.org/)
- **Season 2 design / engine:** [Lux-AI-Challenge/Lux-Design-S2](https://github.com/Lux-AI-Challenge/Lux-Design-S2)

## Public strong solutions

- **ryandy (1st Lux AI Season 2):** [ryandy/Lux-S2-public](https://github.com/ryandy/Lux-S2-public)
- **NeurIPS variant:** [ryandy/Lux-S2-neurips-public](https://github.com/ryandy/Lux-S2-neurips-public)

Browse for: **resource routing**, **factory/robot allocation**, **risk/reward scoring**, **endgame heuristics**.

## Kaggle discussion (RL / Jux)

- Example: [Lux AI Season 2 NeurIPS Stage 2 — PPO / Jux discussion](https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2/discussion/459891)

## Map to this repo (`orbit_wars_bot/`)

| Lux idea | Orbit Wars hook |
|----------|------------------|
| Macro “where to invest next” | `allocation/scoring.py` — planet value, threat, ETA |
| Unit assignment under cap | `GameState` + multi-source sends in `archive/legacy/submissions/submission_v6.py` |
| Rollout / limited-depth search | `simulation/forward.py` |
| RL for policy | `orbit_wars_bot/rl/` — start with **macro parameters** or **discrete src×dst on top-K** |

### Checklist

- [ ] Skim Lux-S2-public `src` layout for scoring vs simulation split.
- [ ] Read one competition postmortem on **time limits** (parallel to `actTimeout` 1s in Orbit Wars).
