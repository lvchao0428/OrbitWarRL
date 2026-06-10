# Halite III — reading map for Orbit Wars

Halite III (Two Sigma) is **spatial mining and fleet routing** with strong writeups and open-source bots. Good for **potential fields, shortest-path style valuation, and scheduling** — analogous to choosing **which planet to hit next** and **when** in Orbit Wars.

## Articles

- **Two Sigma:** [Best practices from building a machine learning bot for Halite](https://www.twosigma.com/articles/best-practices-from-building-a-machine-learning-bot-for-halite/)

## Open-source bots (search / examples)

Community references (clone as needed; do not vendor into this repo by default):

- Strong **rules + navigation** bots on GitHub under topics `halite3` (e.g. search “halite3 bot github”).
- Example pattern: **Dijkstra/BFS** from ship positions to score dropoff targets — in Orbit Wars, substitute **geodesic-like cost** = ETA from `fleet_speed` + intercept + **sun avoidance**.

## Map to Orbit Wars

| Halite idea | Orbit Wars hook |
|-------------|------------------|
| “Value per square / distance tradeoff” | `allocation/scoring.py` — `production`, `ships`, `dist`, `net_threat` |
| Fleet drop timing | `_coordinated_attack` arrival alignment; expand with `simulation/forward.py` |
| ML vs heuristics | RL in `rl/` — hybrid with `archive/legacy/submissions/submission_v6.py` geometry (`safe_aim`) |

### Checklist

- [ ] Read Two Sigma article section on **why pure RL was expensive** vs structured heuristics.
- [ ] Pick one Halite bot and trace its **single-step scoring function**; mirror structure in `scoring.py`.
