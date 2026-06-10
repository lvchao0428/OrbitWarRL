# Planet Wars (2010) — local mirror

This folder is an offline copy of Daniel Hartmeier’s page about the Google AI Challenge 2010: Planet Wars.

- **Source (live):** [Google AI challenge 2010: Planet Wars](https://www.benzedrine.ch/planetwars.html)
- **Local files:** `planetwars.html`, `logo.jpg`, `planetwars.png` (screenshot). Open `planetwars.html` in a browser from this directory so images resolve.

The unofficial TCP test server mentioned on the page is **shut down**; treat the statistics as historical only.

## Champion and strong solutions (read these for Orbit Wars)

| Rank | Author | Link | Why it matters for Orbit Wars |
|------|--------|------|-------------------------------|
| 1 | Gábor Melis (bocsimacko) | [Planet Wars Post-Mortem](http://quotenil.com/Planet-Wars-Post-Mortem.html) | **Multi-planet coordinated attacks**, simulating future states (`FUTURE`), evaluating move *steps* under branching — maps to **global allocation + short-horizon rollouts** for fleets and orbital timing. |
| 2 | Iouri Khramtsov | [Planet Wars Entry (C++)](http://iouri-khramtsov.blogspot.com/2010/11/google-ai-challenge-planet-wars-entry.html) | Strong engineering perspective on search/eval; useful when implementing **forward simulation** in `orbit_wars_bot/simulation/`. |

### Mapping concepts → this repo

- **Multi-line coordination:** `_coordinated_attack` and expansion in [`submission_v6.py`](../../archive/legacy/submissions/submission_v6.py); extend via `orbit_wars_bot/allocation/scoring.py`.
- **Simulation / “what if”:** Planet Wars–style rollouts → `orbit_wars_bot/simulation/forward.py` (env-backed or lightweight geometry).
- **Fleets in flight:** Orbit Wars exposes `fleets` and turn order in Kaggle specs (`archive/legacy/docs/` 等处有镜像文本); when simulating, mirror **production → movement → combat** order.

## Official site archival

- Original: [planetwars.aichallenge.org](http://planetwars.aichallenge.org/) may be offline.
- Use [Wayback Machine](https://web.archive.org/) for rankings, starter packages, and forums if links 404.

## Other links from the mirrored page

- [Final rankings (historical)](http://planetwars.aichallenge.org/rankings.php)
- [space.invader — genetic programming (rank 277)](http://forums.aichallenge.org/viewtopic.php?f=17&t=1136)
- Author’s own bot sources (on live site under `/planetwars/`): `MyBot.cc`, `board.h`, `board.cc` — optional to mirror later if needed.
