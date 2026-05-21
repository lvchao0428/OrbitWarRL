"""Orbit Wars v20.0 - Regional Graph + Multi-Hop Planning (single submission file).

Evolution of v19 (opening reserve / factory defer / sync capture). Regional utilities,
clustering, timeline, multi-hop scaffold, unified ``capture_edge_score``, pragmatic
action UCB, and threat-aware surplus live in this module only (no separate regional import).
"""

from __future__ import annotations

import base64
import io
import math
import os
import random
import time
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy.cluster.hierarchy import fclusterdata as _scipy_fclusterdata
except ImportError:  # noqa: SIM105 — Kaggle notebook/kernel 常无 scipy，仅保证可导入
    _scipy_fclusterdata = None  # type: ignore[misc, assignment]

# Set by eval / rollout wrappers (`v20@rush`); Kaggle never touches this.
ORB_STRATEGY_PROFILE: ContextVar[Optional[str]] = ContextVar(
    "ORB_STRATEGY_PROFILE", default=None
)


# ╔═══ region 0: constants & helpers ════════════════════════════════════════╗

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
# Extra clearance for “direct lane” filters (target_score / _emit) and regional detour heuristic.
SUN_PATH_MARGIN = 3.5
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0
DEFAULT_EPISODE_STEPS = 500

MAX_TOTAL_MOVES = 26
# Multi-source capture: start with a tight ETA band around the fastest source; if
# the band cannot meet ``required`` ships, widen up to SYNC_ETA_WINDOW_MAX so a
# fat neutral (e.g. 45-stack) can still receive a delayed third source. Regional
# score was fine — silent failure was purely contributor shortage under narrow sync.
SYNC_ETA_WINDOW = 5
SYNC_ETA_WINDOW_MAX = 26
MAX_SOURCES_PER_TARGET = 8
MAX_TARGETS_PER_PLAN = 3      # 集中力量：最多同时打3个目标
# expand/balanced：己方星球很少时，禁止同一出兵星在本 plan 里打两个不同目标（避免 8+12 拆开浪费）。
ONE_OUTBOUND_DST_PER_SOURCE_UNTIL_N_WORLDS = 3
ABS_MIN_BATCH = 8  # 提高下限减少碎片化小舰队（配合 early 略降 growth_lock）
# Expand / balanced: edge scores below this were dropped from ranked targets — too
# aggressive vs regional cross-zone tax; leaves fat planets idle on contested greys.
EXPAND_RANK_SCORE_FLOOR = -44.0
# Neutral + is_safe_investment False: still try capture if pooled surplus is huge
# (first wave failed vs orange but HQ stack is still rich).
NEUTRAL_BRUTE_SLACK_MUL = 2
NEUTRAL_BRUTE_SLACK_MIN = 52
# First step / solo HQ: cap reserve so opening can peel ~ships+1 vs neutral-20
# (symmetric ladder seeds; see tools/sim_first_turn_opening.py).
OPENING_FIRST_CAPTURE_SEND = 21
# While still one friendly planet and no inbound siege, cap reserve so a ~20-stack
# neutral can be taken as soon as HQ reaches OPENING_FIRST_CAPTURE_SEND (+ batch
# rules), not only on step 0. (Default start is ~10 ships; peeling ramps up later.)
OPENING_SOLO_HQ_RESERVE_LAST_STEP = 48

# Bocsimacko (2010 Planet Wars champion) caps per-planet scoring at the
# horizon - distant production is heavily discounted instead of accumulated
# linearly to game-end. Combined with the small enemy-ship positional pen,
# this biases the bot toward proximate, certain gains.
HORIZON_TURNS = 60
ENEMY_SHIP_PEN_COEFF = 0.0008  # tiny - only breaks ties

# Orbital distance (initial) from sun for “four inner clusters” on symmetric ladder
# maps. Used to garrison and avoid stripping these worlds while they rotate into
# enemy arcs (user: clockwise → need mass, not dribbling to passing neutrals).
INNER_SUN_BELT_R = 33.0


def _get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    if ships <= 1:
        return 1.0
    spd = 1.0 + (max_speed - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
    return min(spd, max_speed)


def point_segment_distance(px, py, ax, ay, bx, by) -> float:
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def segment_hits_sun(ax, ay, bx, by, margin: float = 1.25) -> bool:
    return point_segment_distance(SUN_X, SUN_Y, ax, ay, bx, by) < SUN_RADIUS + margin


def swept_pair_hit(
    ax: float, ay: float, bx: float, by: float,
    p0x: float, p0y: float, p1x: float, p1y: float, r: float,
) -> bool:
    """True iff segment A->B and segment P0->P1 come within r (orbit_wars.py)."""
    d0x, d0y = ax - p0x, ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


# ╔═══ region 0b: v20 regional graph (inlined; no external module) ══════════╗


@dataclass
class Region:
    """Spatial region cluster (centroid + optional bookkeeping fields)."""
    id: int
    center: Tuple[float, float]
    my_planets: List[int] = field(default_factory=list)
    enemy_planets: List[int] = field(default_factory=list)
    external_planets: List[int] = field(default_factory=list)
    production_rate: int = 0


@dataclass
class Wave:
    target_id: int
    required_ships: int
    launch_turn: int
    sources: List[int] = field(default_factory=list)
    expected_arrival: int = 0


class RegionalGraph:
    """Geographic clustering (4 regions) + cached path lengths (sun-aware penalty)."""

    def __init__(self, planets: List, spawn_positions: Optional[List[Tuple[float, float]]] = None):
        self.planets = planets
        self.regions: Dict[int, Region] = {}
        self.planet_to_region: Dict[int, int] = {}
        self.dijkstra_cache: Dict[Tuple[int, int], Tuple[float, int]] = {}

        coords = np.array([[p.x, p.y] for p in planets])
        try:
            if _scipy_fclusterdata is None:
                raise RuntimeError("scipy not available")
            cluster_labels = _scipy_fclusterdata(
                coords, t=4, criterion="maxclust", method="complete"
            )
            self._build_regions(cluster_labels, spawn_positions)
        except Exception:
            self._build_regions_fallback(spawn_positions)

        self._precompute_dijkstra()

    def _build_regions(self, cluster_labels: np.ndarray, spawn_positions: Optional[List] = None):
        clusters: Dict[int, List] = {}
        for planet, cluster_id in zip(self.planets, cluster_labels):
            clusters.setdefault(cluster_id, []).append(planet)

        region_list: List[Region] = []
        for cluster_id in sorted(clusters.keys())[:4]:
            planets_in_cluster = clusters[cluster_id]
            center_x = sum(p.x for p in planets_in_cluster) / len(planets_in_cluster)
            center_y = sum(p.y for p in planets_in_cluster) / len(planets_in_cluster)
            region = Region(id=len(region_list), center=(center_x, center_y))
            region_list.append(region)
            for p in planets_in_cluster:
                self.planet_to_region[p.id] = region.id

        self.regions = {r.id: r for r in region_list}

    def _build_regions_fallback(self, spawn_positions: Optional[List] = None):
        if spawn_positions and len(spawn_positions) >= 2:
            spawn_array = np.array(
                spawn_positions[:4] if len(spawn_positions) >= 4 else spawn_positions * 2
            )
        else:
            spawn_array = np.array([(25, 25), (75, 25), (25, 75), (75, 75)])

        for planet in self.planets:
            distances = [math.hypot(planet.x - sp[0], planet.y - sp[1]) for sp in spawn_array]
            nearest_region = distances.index(min(distances)) % 4
            self.planet_to_region[planet.id] = nearest_region

        for i in range(4):
            self.regions[i] = Region(id=i, center=tuple(spawn_array[i]))

    def _precompute_dijkstra(self):
        planet_ids = [p.id for p in self.planets]
        for src_id in planet_ids:
            for dst_id in planet_ids:
                if src_id != dst_id:
                    distance, steps = self._dijkstra_impl(src_id, dst_id)
                    self.dijkstra_cache[(src_id, dst_id)] = (distance, steps)

    def _dijkstra_impl(self, src_id: int, dst_id: int) -> Tuple[float, int]:
        src = next((p for p in self.planets if p.id == src_id), None)
        dst = next((p for p in self.planets if p.id == dst_id), None)
        if not src or not dst:
            return 999.0, 999

        direct_dist = math.hypot(src.x - dst.x, src.y - dst.y)
        if point_segment_distance(SUN_X, SUN_Y, src.x, src.y, dst.x, dst.y) < SUN_RADIUS + SUN_PATH_MARGIN:
            total_dist = direct_dist + SUN_RADIUS * 2
        else:
            total_dist = direct_dist
        steps = max(1, int(math.ceil(total_dist / 2.0)))
        return total_dist, steps

    def dijkstra(self, src_id: int, dst_id: int) -> Tuple[float, int]:
        key = (src_id, dst_id)
        if key in self.dijkstra_cache:
            return self.dijkstra_cache[key]
        return 999.0, 999

    def in_same_region(self, pid1: int, pid2: int) -> bool:
        return self.planet_to_region.get(pid1, -1) == self.planet_to_region.get(pid2, -1)

    def get_region_by_id(self, region_id: int) -> Optional[Region]:
        return self.regions.get(region_id)

    def get_region_by_planet(self, planet_id: int) -> Optional[Region]:
        region_id = self.planet_to_region.get(planet_id, -1)
        return self.regions.get(region_id)

    def region_production(self, region_id: int, my_control: Set[int]) -> int:
        if region_id not in self.regions:
            return 0
        production = 0
        for planet in self.planets:
            if planet.id in my_control and self.planet_to_region.get(planet.id) == region_id:
                production += planet.production
        return production

    def region_threat(self, region_id: int, enemy_planets: Sequence) -> float:
        region = self.regions.get(region_id)
        if not region:
            return 0.0
        rcx, rcy = region.center
        total = 0.0
        for p in enemy_planets:
            pr = self.planet_to_region.get(p.id, -1)
            prod = float(getattr(p, "production", 0) or 0)
            ships = int(getattr(p, "ships", 0) or 0)
            if pr == region_id:
                total += prod * 2.2 + math.sqrt(max(1, ships)) * 0.25
            else:
                d = math.hypot(p.x - rcx, p.y - rcy)
                if d < 42.0:
                    w = max(0.0, (42.0 - d) / 42.0)
                    total += w * (prod * 1.4 + math.sqrt(max(1, ships)) * 0.12)
        return total

    def get_all_regions(self) -> List[Region]:
        return list(self.regions.values())

    def get_planets_in_region(self, region_id: int) -> List:
        return [p for p in self.planets if self.planet_to_region.get(p.id) == region_id]


class ProductionTimeline:
    def __init__(self, planets: List, my_control: Set[int]):
        self.planets = planets
        self.my_control = my_control

    def predict_surplus(self, planet_ids: List[int], turns_ahead: int) -> List[int]:
        surplus_per_turn: List[int] = []
        for turn in range(turns_ahead):
            production = sum(
                p.production for p in self.planets
                if p.id in planet_ids and p.id in self.my_control
            )
            accumulated = production * (turn + 1)
            available = int(accumulated * 0.8)
            surplus_per_turn.append(available)
        return surplus_per_turn

    def can_support_wave(self, sources: List[int], required: int, launch_turn: int) -> bool:
        surpluses = self.predict_surplus(sources, launch_turn + 1)
        if launch_turn < len(surpluses):
            return surpluses[launch_turn] >= required
        return False


def calculate_safe_surplus(my_planets: List, my_production: int, enemy_threats: Dict) -> int:
    max_threat = max(enemy_threats.values()) if enemy_threats else 0
    defensive_requirement = int(max_threat * 1.5)
    safe_surplus = my_production - defensive_requirement
    return max(0, int(safe_surplus * 0.65))


class MultiHopPlanner:
    def __init__(self, regional_graph: RegionalGraph, production_timeline: ProductionTimeline):
        self.regional_graph = regional_graph
        self.timeline = production_timeline

    def plan_attack_sequence(
        self, target_id: int, my_region_id: int, budget_turns: int = 5, max_hops: int = 3
    ) -> List[Wave]:
        target_planet = next((p for p in self.regional_graph.planets if p.id == target_id), None)
        if not target_planet:
            return []
        my_sources = [
            p for p in self.regional_graph.planets
            if self.regional_graph.planet_to_region.get(p.id) == my_region_id
        ]
        if not my_sources:
            return []
        source_planet = my_sources[0]
        _distance, steps = self.regional_graph.dijkstra(source_planet.id, target_id)
        wave = Wave(
            target_id=target_id,
            required_ships=int(target_planet.ships + target_planet.production * 2),
            launch_turn=0,
            sources=[s.id for s in my_sources[:3]],
            expected_arrival=steps,
        )
        return [wave]


# ╔═══ region 1: data classes ═══════════════════════════════════════════════╗

class Planet:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production",
                 "initial_x", "initial_y", "is_comet")

    def __init__(self, id, owner, x, y, radius, ships, production,
                 initial_x=0.0, initial_y=0.0, is_comet=False):
        self.id = id; self.owner = owner
        self.x = x; self.y = y; self.radius = radius
        self.ships = ships; self.production = production
        self.initial_x = initial_x; self.initial_y = initial_y
        self.is_comet = is_comet

    def dist(self, o: "Planet") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def dist_xy(self, x, y) -> float:
        return math.hypot(self.x - x, self.y - y)


class Fleet:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.angle = angle; self.from_planet_id = from_planet_id; self.ships = ships


def _combat(owner: int, garrison: int, arrivals: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not arrivals:
        return owner, max(0, int(garrison))
    by_owner: Dict[int, int] = defaultdict(int)
    for o, s in arrivals:
        if s > 0:
            by_owner[int(o)] += int(s)
    if not by_owner:
        return owner, max(0, int(garrison))
    forces = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    if len(forces) >= 2 and forces[0][1] == forces[1][1]:
        return owner, max(0, int(garrison))
    atk_owner, atk_ships = forces[0]
    second = forces[1][1] if len(forces) >= 2 else 0
    survivor = atk_ships - second
    if survivor <= 0:
        return owner, max(0, int(garrison))
    if atk_owner == owner:
        return owner, max(0, int(garrison) + survivor)
    if survivor > garrison:
        return atk_owner, survivor - int(garrison)
    return owner, int(garrison) - survivor


# ╔═══ region 2: GameState ══════════════════════════════════════════════════╗

class GameState:
    """Parses obs once. Holds planets/fleets and incoming-fleet metadata."""

    def __init__(self, obs, config=None):
        self.my_id = int(_get(obs, "player", 0) or 0)
        self.ang_vel = float(_get(obs, "angular_velocity", 0.0) or 0.0)
        self.step = int(_get(obs, "step", 0) or 0)

        cfg = _get(obs, "configuration", None) or config or {}
        self.max_speed = float(_get(cfg, "shipSpeed", DEFAULT_MAX_SHIP_SPEED)
                               or DEFAULT_MAX_SHIP_SPEED)
        self.episode_steps = int(_get(cfg, "episodeSteps", DEFAULT_EPISODE_STEPS)
                                 or DEFAULT_EPISODE_STEPS)

        comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
        self.comet_paths: Dict[int, Tuple[List[Tuple[float, float]], int]] = {}
        self._parse_comets(_get(obs, "comets", []) or [])

        initial_xy: Dict[int, Tuple[float, float]] = {}
        for row in _get(obs, "initial_planets", []) or []:
            initial_xy[int(row[0])] = (float(row[2]), float(row[3]))

        self.planets: List[Planet] = []
        for row in _get(obs, "planets", []) or []:
            pid = int(row[0])
            ix, iy = initial_xy.get(pid, (float(row[2]), float(row[3])))
            self.planets.append(Planet(pid, int(row[1]), float(row[2]), float(row[3]),
                                       float(row[4]), int(row[5]), int(row[6]),
                                       ix, iy, pid in comet_ids))

        self.fleets: List[Fleet] = []
        for row in _get(obs, "fleets", []) or []:
            self.fleets.append(Fleet(int(row[0]), int(row[1]), float(row[2]),
                                     float(row[3]), float(row[4]),
                                     int(row[5]), int(row[6])))

        self._pm = {p.id: p for p in self.planets}
        self.my_pl = [p for p in self.planets if p.owner == self.my_id]
        self.en_pl = [p for p in self.planets if p.owner not in (-1, self.my_id)]
        self.neu_pl = [p for p in self.planets if p.owner == -1]
        self.en_ids = sorted({p.owner for p in self.en_pl})

        self.fleet_target: Dict[int, Optional[Tuple[int, int]]] = {}
        self.arrivals: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.incoming: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for f in self.fleets:
            pred = self._predict_fleet_target(f)
            self.fleet_target[f.id] = pred
            if pred is not None:
                tid, eta = pred
                self.arrivals[tid].append((eta, f.owner, f.ships))
                self.incoming[tid][f.owner] += f.ships

    # ── parsing helpers ──
    def _parse_comets(self, groups) -> None:
        for group in groups:
            ids = _get(group, "planet_ids", []) or []
            paths = _get(group, "paths", []) or []
            idx = int(_get(group, "path_index", 0) or 0)
            for i, pid_raw in enumerate(ids):
                if i >= len(paths):
                    continue
                path: List[Tuple[float, float]] = []
                for pt in paths[i] or []:
                    if len(pt) >= 2:
                        path.append((float(pt[0]), float(pt[1])))
                if path:
                    self.comet_paths[int(pid_raw)] = (path, idx)

    # ── geometry / lookup ──
    def get(self, pid: int) -> Optional[Planet]:
        return self._pm.get(pid)

    def is_orbiting(self, p: Planet) -> bool:
        if p.is_comet:
            return False
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        return r + p.radius < ROTATION_RADIUS_LIMIT and abs(self.ang_vel) > 1e-12

    def planet_pos_at(self, p: Planet, t: int) -> Tuple[float, float]:
        if p.is_comet and p.id in self.comet_paths:
            path, idx = self.comet_paths[p.id]
            j = idx + max(0, int(t))
            return path[min(j, len(path) - 1)]
        if not self.is_orbiting(p):
            return p.x, p.y
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
        a1 = a0 + self.ang_vel * (self.step + int(t))
        return SUN_X + r * math.cos(a1), SUN_Y + r * math.sin(a1)

    def planet_motion_segment(self, p: Planet, k: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """(old_pos, new_pos) for planet ``p`` during fleet's k-th move (k>=1), per orbit_wars."""
        if p.is_comet and p.id in self.comet_paths:
            path, idx = self.comet_paths[p.id]
            if k == 1:
                p_old = (p.x, p.y)
                if idx + 1 >= len(path):
                    p_new = p_old
                else:
                    p_new = (float(path[idx + 1][0]), float(path[idx + 1][1]))
                return p_old, p_new
            j0 = min(idx + k - 1, len(path) - 1)
            j1 = min(idx + k, len(path) - 1)
            return ((float(path[j0][0]), float(path[j0][1])),
                    (float(path[j1][0]), float(path[j1][1])))
        if not self.is_orbiting(p):
            xy = (p.x, p.y)
            return xy, xy
        if k == 1:
            p_old = (p.x, p.y)
            r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
            a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
            a1 = a0 + self.ang_vel * self.step
            p_new = (SUN_X + r * math.cos(a1), SUN_Y + r * math.sin(a1))
            return p_old, p_new
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
        old_i = self.step + k - 2
        new_i = self.step + k - 1
        a_old = a0 + self.ang_vel * old_i
        a_new = a0 + self.ang_vel * new_i
        return ((SUN_X + r * math.cos(a_old), SUN_Y + r * math.sin(a_old)),
                (SUN_X + r * math.cos(a_new), SUN_Y + r * math.sin(a_new)))

    def comet_turns_left(self, p: Planet) -> int:
        if not p.is_comet or p.id not in self.comet_paths:
            return 999
        path, idx = self.comet_paths[p.id]
        return max(0, len(path) - idx - 1)

    def _predict_fleet_target(self, f: Fleet, max_steps: int = 220) -> Optional[Tuple[int, int]]:
        spd = fleet_speed(f.ships, self.max_speed)
        cx, cy = f.x, f.y
        dx, dy = math.cos(f.angle) * spd, math.sin(f.angle) * spd
        for t in range(1, max_steps + 1):
            nx, ny = cx + dx, cy + dy
            if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD):
                return None
            if point_segment_distance(SUN_X, SUN_Y, cx, cy, nx, ny) < SUN_RADIUS:
                return None
            best_pid, best_d = None, float("inf")
            for p in self.planets:
                px, py = self.planet_pos_at(p, t)
                d = point_segment_distance(px, py, cx, cy, nx, ny)
                if d < p.radius and d < best_d:
                    best_pid, best_d = p.id, d
            if best_pid is not None:
                return best_pid, t
            cx, cy = nx, ny
        return None

    # ── aggregates ──
    def total_ships(self, owner: int) -> int:
        return (sum(p.ships for p in self.planets if p.owner == owner) +
                sum(f.ships for f in self.fleets if f.owner == owner))

    def centroid(self) -> Tuple[float, float]:
        if not self.my_pl:
            return SUN_X, SUN_Y
        return (sum(p.x for p in self.my_pl) / len(self.my_pl),
                sum(p.y for p in self.my_pl) / len(self.my_pl))

    def net_threat(self, p: Planet) -> int:
        inc = self.incoming.get(p.id, {})
        attackers = sum(v for k, v in inc.items() if k not in (-1, self.my_id))
        return attackers - inc.get(self.my_id, 0)

    def phase(self) -> str:
        progress = self.step / max(1, self.episode_steps)
        if progress < 0.18:
            return "early"
        if progress < 0.64:
            return "mid"
        return "late"

    def turns_left(self) -> int:
        return max(1, self.episode_steps - self.step)

    def enemy_incoming(self, pid: int) -> int:
        inc = self.incoming.get(pid, {})
        return sum(v for k, v in inc.items() if k not in (-1, self.my_id))

    def effective_garrison(self, p: Planet) -> int:
        if p.owner not in self.en_ids:
            return p.ships
        out = 0
        for f in self.fleets:
            if f.owner != p.owner:
                continue
            t = self.fleet_target.get(f.id)
            if t is not None and t[0] != p.id:
                out += f.ships
        return max(0, p.ships - out)


def is_sun_belt_planet(state: GameState, p: Planet) -> bool:
    """Inner rotating ring around the sun (not comets, not static far-outs)."""
    if p.is_comet or not state.is_orbiting(p):
        return False
    r0 = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
    return r0 + float(p.radius) <= INNER_SUN_BELT_R


# ╔═══ region 3: Snapshot & geometry ════════════════════════════════════════╗

def lead_intercept(state: GameState, src: Planet, dst: Planet, ships: int,
                   iters: int = 12) -> Tuple[float, float, int, float]:
    """Iteratively converge on intercept of ``dst`` **disk center** from ``src``.

    Each step uses ``launch_origin`` (rim + ENGINE_LAUNCH_PAD) as the true start,
    so bearing/ETA match the engine and moving orbit targets get a proper lead
    (提前量). Orbiting, non-comet targets apply ``ORBIT_AIM_LEAD_STEPS`` to push
    the aim along orbit after convergence (often ~1–2t earlier contact). Returns
    ``(tx, ty, eta, aim_angle)`` where ``aim_angle`` is from rim toward the
    predicted center.
    """
    spd = fleet_speed(max(1, ships), state.max_speed)
    eta = max(1, int(math.ceil(math.hypot(dst.x - src.x, dst.y - src.y) / spd)))
    angle = math.atan2(dst.y - src.y, dst.x - src.x)
    n = max(iters, 10)
    for _ in range(n):
        tx, ty = state.planet_pos_at(dst, eta)
        lx, ly = launch_origin(src, angle)
        dist = math.hypot(tx - lx, ty - ly)
        new_eta = max(1, int(math.ceil(dist / spd))) if dist > 1e-9 else 1
        new_angle = math.atan2(ty - ly, tx - lx)
        if new_eta == eta and abs(new_angle - angle) < 1e-4:
            break
        eta = new_eta
        angle = new_angle
    lead = ORBIT_AIM_LEAD_STEPS if (state.is_orbiting(dst) and not dst.is_comet) else 0
    tx, ty = state.planet_pos_at(dst, eta + lead)
    _MARGIN = 1.0
    tx = max(_MARGIN, min(BOARD - _MARGIN, tx))
    ty = max(_MARGIN, min(BOARD - _MARGIN, ty))
    lx, ly = launch_origin(src, angle)
    angle = math.atan2(ty - ly, tx - lx)
    eta = max(1, int(math.ceil(math.hypot(tx - lx, ty - ly) / spd)))
    return tx, ty, eta, angle


def _ray_safe(sx: float, sy: float, angle: float, spd: float, min_flight: int = 0) -> bool:
    """Check if a ray from (sx,sy) at angle with speed spd is safe.
    - Never enters the sun (entire flight)
    - If min_flight > 0, must stay in-board for at least that many steps
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for e in range(1, 80):
        ex = sx + cos_a * spd * e
        ey = sy + sin_a * spd * e
        if not (0.0 <= ex <= BOARD and 0.0 <= ey <= BOARD):
            if e <= min_flight:
                return False  # exits board before reaching target
            return True
        if math.hypot(ex - SUN_X, ey - SUN_Y) < SUN_RADIUS + 1.0:
            return False
    return True


LAUNCH_SIM_MAX_STEPS = 220
# Must match kaggle_environments/envs/orbit_wars/orbit_wars.py fleet spawn.
ENGINE_LAUNCH_PAD = 0.1
# Discrete ETA + rim spawn tends to under-lead orbiting neutrals; nudge the aim
# point forward along orbit so fleets meet the center ~1–2 steps earlier in practice.
ORBIT_AIM_LEAD_STEPS = 2


def launch_origin(src: Planet, angle: float) -> Tuple[float, float]:
    """Point just outside `src`'s disk on the launch bearing (engine uses radius+0.1)."""
    r = float(src.radius) + ENGINE_LAUNCH_PAD
    return src.x + math.cos(angle) * r, src.y + math.sin(angle) * r


def launch_intercept_step(
    state: GameState,
    src_x: float,
    src_y: float,
    angle: float,
    ships: int,
    target_id: int,
    max_steps: int = LAUNCH_SIM_MAX_STEPS,
    ignore_planet_id: Optional[int] = None,
) -> Optional[int]:
    """First simulation step (1-based) where fleet hits ``target_id``, else None."""
    spd = fleet_speed(max(1, int(ships)), state.max_speed)
    cx, cy = float(src_x), float(src_y)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for k in range(1, max_steps + 1):
        fx0, fy0 = cx, cy
        fx1, fy1 = cx + cos_a * spd, cy + sin_a * spd
        hit_id: Optional[int] = None
        for p in state.planets:
            if ignore_planet_id is not None and p.id == ignore_planet_id:
                continue
            p0, p1 = state.planet_motion_segment(p, k)
            if p0[0] < 0:
                continue
            if swept_pair_hit(fx0, fy0, fx1, fy1,
                              p0[0], p0[1], p1[0], p1[1], p.radius):
                hit_id = p.id
                break
        if hit_id is not None:
            return k if hit_id == target_id else None
        if not (0.0 <= fx1 <= BOARD and 0.0 <= fy1 <= BOARD):
            return None
        if point_segment_distance(SUN_X, SUN_Y, fx0, fy0, fx1, fy1) < SUN_RADIUS:
            return None
        cx, cy = fx1, fy1
    return None


def launch_hits_target_first(
    state: GameState,
    src_x: float,
    src_y: float,
    angle: float,
    ships: int,
    target_id: int,
    max_steps: int = LAUNCH_SIM_MAX_STEPS,
    ignore_planet_id: Optional[int] = None,
) -> bool:
    """True iff the first planetary contact (orbit_wars swept collision order) is target_id.

    Uses the same relative-motion segment test as the official environment, not
    a frozen planet at integer steps.
    """
    return launch_intercept_step(
        state, src_x, src_y, angle, ships, target_id,
        max_steps=max_steps, ignore_planet_id=ignore_planet_id,
    ) is not None


def safe_aim(state: GameState, src: Planet, dst: Planet, ships: int) -> Tuple[float, int]:
    """Return (angle, eta). Pick rim bearing with **earliest** stepped intercept.

    Orbits can admit two families of feasible shots (short meet vs long chase).
    Taking the first passing candidate used to lock in the slow clockwise chase;
    we now minimize intercept time so fleets prefer the short arc (user: CCW vs CW).
    """
    nships = max(1, int(ships))
    spd = fleet_speed(nships, state.max_speed)
    tx, ty, eta_hint, angle0 = lead_intercept(state, src, dst, nships)
    tx = max(1.0, min(BOARD - 1.0, tx))
    ty = max(1.0, min(BOARD - 1.0, ty))
    lx0, ly0 = launch_origin(src, angle0)
    angle0 = math.atan2(ty - ly0, tx - lx0)
    tid = dst.id

    deltas = (0.15, -0.15, 0.30, -0.30, 0.50, -0.50, 0.75, -0.75,
              1.05, -1.05, 1.40, -1.40, 1.80, -1.80, 2.20, -2.20,
              2.60, -2.60, 2.85, -2.85, 3.00, -3.00, 3.02, -3.02,
              3.14, -3.14)
    cands: List[float] = [angle0, angle0 + math.pi]
    for d in deltas:
        cands.append(angle0 + d)

    def best_from_pool(pool: List[float], min_flight: int) -> Optional[Tuple[float, int]]:
        best_s: Optional[int] = None
        best_a: Optional[float] = None
        mf = max(1, min_flight)
        for a in pool:
            lx, ly = launch_origin(src, a)
            if not _ray_safe(lx, ly, a, spd, min_flight=mf):
                continue
            st = launch_intercept_step(
                state, lx, ly, a, nships, tid,
                max_steps=LAUNCH_SIM_MAX_STEPS, ignore_planet_id=None)
            if st is None:
                continue
            if best_s is None or st < best_s:
                best_s, best_a = st, a
        if best_a is None or best_s is None:
            return None
        return best_a, int(best_s)

    got = best_from_pool(cands, 1)
    if got is not None:
        return got

    got = best_from_pool(cands, max(1, eta_hint - 1))
    if got is not None:
        return got

    safe_corners = [(2.0, 2.0), (2.0, 98.0), (98.0, 2.0), (98.0, 98.0)]
    corner = max(safe_corners, key=lambda c: math.hypot(c[0] - src.x, c[1] - src.y))
    ca = math.atan2(corner[1] - src.y, corner[0] - src.x)
    got = best_from_pool([ca], max(1, eta_hint - 1))
    if got is not None:
        return got

    for a in cands:
        lx, ly = launch_origin(src, a)
        if _ray_safe(lx, ly, a, spd, min_flight=max(1, eta_hint - 1)):
            return a, int(eta_hint)
    lx_ca, ly_ca = launch_origin(src, ca)
    if _ray_safe(lx_ca, ly_ca, ca, spd, min_flight=max(1, eta_hint - 1)):
        return ca, int(eta_hint)
    return angle0, int(eta_hint)


def target_state_at(state: GameState, dst: Planet, eta: int) -> Tuple[int, int]:
    """Project (owner, ships) of `dst` after `eta` turns including in-flight arrivals."""
    owner = dst.owner
    ships = int(dst.ships)
    by_turn: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for arr_eta, arr_owner, arr_ships in state.arrivals.get(dst.id, []):
        if 1 <= arr_eta <= eta:
            by_turn[arr_eta].append((arr_owner, arr_ships))
    for t in range(1, max(1, eta) + 1):
        if owner >= 0:
            ships += dst.production
        if by_turn.get(t):
            owner, ships = _combat(owner, ships, by_turn[t])
    return owner, max(0, ships)


def my_inbound_ships_to(state: GameState, planet_id: int) -> int:
    """Our fleets already in flight toward ``planet_id`` (obs snapshot)."""
    s = 0
    mi = state.my_id
    for f in state.fleets:
        if f.owner != mi:
            continue
        t = state.fleet_target.get(f.id)
        if t is not None and t[0] == planet_id:
            s += f.ships
    return int(s)


def neutral_wave_wins(state: GameState, dst: Planet, eta: int,
                       send: int, other_my_inbound: int) -> bool:
    """True if ``send`` + inbound allies strictly beats projected grey garrison at eta."""
    if dst.owner != -1:
        return True
    _, gar = target_state_at(state, dst, eta)
    return (send + other_my_inbound) > gar


def capture_need(state: GameState, src: Planet, dst: Planet,
                 margin: Optional[int] = None) -> Tuple[int, int]:
    """Iterative estimate of (need, eta) to capture dst from src."""
    early = state.phase() == "early"
    if dst.owner == -1:
        pad0 = 1 if early else 2
    else:
        pad0 = 8
    need = max(ABS_MIN_BATCH, dst.ships + pad0)
    eta = 1
    for _ in range(4):
        _, _, eta, _ = lead_intercept(state, src, dst, need)
        owner, ships = target_state_at(state, dst, eta)
        if owner == state.my_id:
            need = max(ABS_MIN_BATCH, state.net_threat(dst) + 4)
        else:
            if margin is not None:
                base_margin = margin
            elif owner == -1:
                # Early: tighter neutral padding so first waves leave closer to
                # opponents' razor timings (~ships+1) instead of +3 lag.
                base_margin = 1 if early else 3
            else:
                base_margin = 8 + min(6, dst.production)
            extra_prod = (dst.production * eta) if owner not in (-1, state.my_id) else 0
            need = ships + base_margin + extra_prod // 4
        need = max(ABS_MIN_BATCH, int(need))
    return need, eta


@dataclass
class Snapshot:
    """Per-turn precomputed view. Planners read from here; they never recompute
    surplus / reserve themselves. Mutate by calling `subtract(pid, ships)`
    AFTER an action commits."""

    state: GameState
    policy: "PhasePolicy"

    surplus: Dict[int, int] = field(default_factory=dict)
    reserve: Dict[int, int] = field(default_factory=dict)
    centroid: Tuple[float, float] = (SUN_X, SUN_Y)
    nearest_enemy_dist: Dict[int, float] = field(default_factory=dict)
    used: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    # v13: predicted earliest enemy fleet arrival eta per friendly planet id.
    # Only includes planets where an enemy fleet is in-flight and will arrive
    # within THREAT_HORIZON_WINDOW turns.
    threat_horizon: Dict[int, int] = field(default_factory=dict)
    # Same-turn commits toward grey id (obs does not yet list our launched fleets).
    pending_neutral_wave: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def build(cls, state: GameState, policy: "PhasePolicy") -> "Snapshot":
        snap = cls(state=state, policy=policy)
        snap.centroid = state.centroid()
        # v13: build threat_horizon first so _reserve can read it.
        # v14: expanded window to 8 turns (was 3) for earlier reinforcement.
        for f in state.fleets:
            if f.owner in (-1, state.my_id):
                continue
            target = state.fleet_target.get(f.id)
            if not target:
                continue
            tid, eta_to_planet = target
            dst = state.get(tid)
            if dst is None or dst.owner != state.my_id:
                continue
            prev = snap.threat_horizon.get(dst.id, 999)
            if eta_to_planet < prev:
                snap.threat_horizon[dst.id] = eta_to_planet
        for p in state.my_pl:
            snap.nearest_enemy_dist[p.id] = (
                min((p.dist(e) for e in state.en_pl), default=999.0))
            snap.reserve[p.id] = snap._reserve(p)
            snap.surplus[p.id] = max(0, p.ships - snap.reserve[p.id])
        return snap

    def _reserve(self, p: Planet) -> int:
        s = self.state
        threat = max(0, s.net_threat(p))
        if p.is_comet:
            ttl = s.comet_turns_left(p)
            return max(threat + 2, 2 if ttl > 10 else p.ships)
        ned = self.nearest_enemy_dist.get(p.id, 999.0)
        # v20: early phase — relax front_lock so nearby neutrals can be grabbed.
        # Only lock hard when enemy is very close (< 20) or threat is active.
        if self.state.phase() == "early":
            front_lock = (8 + p.production * 3) if ned < 20 else 0
        else:
            front_lock = ((10 + p.production * 4) if ned < 20
                          else (5 + p.production * 3) if ned < 36 else 0)
        growth_lock = p.production * self.policy.reserve_growth_mul
        floor_threat_pad = 4 if self.state.phase() == "early" else 6
        floor_idle = 2 if self.state.phase() == "early" else 3
        base = max(threat + floor_threat_pad, growth_lock, front_lock, floor_idle)
        # v13/v14: if an enemy fleet will arrive within 8 turns AND this planet
        # has high production, keep extra buffer to defend it proactively.
        # Window expanded from 3 to 8 in v14 for earlier reinforcement.
        horizon_eta = self.threat_horizon.get(p.id, 999)
        if horizon_eta <= 8 and p.production >= 3:
            # Enemy fleet targeting us soon — keep extra pad, but do NOT lock the
            # entire stack (old ``ships + prod*eta`` made reserve > ships → surplus 0).
            proactive = (threat + p.production * min(horizon_eta, 4)
                         + 6 + max(0, 10 - horizon_eta))
            base = max(base, proactive)

        # Inner sun belt: multi-planet — hoard before rotation carries us into enemy.
        if (
            len(s.my_pl) >= 2
            and is_sun_belt_planet(s, p)
            and not p.is_comet
        ):
            en_belt = any(is_sun_belt_planet(s, e) for e in s.en_pl)
            if en_belt or s.phase() != "early":
                belt_pad = 8 + p.production * 5
                if s.phase() in ("mid", "late"):
                    belt_pad += 8
                base = max(base, threat + belt_pad)

        if (
            self.state.phase() == "early"
            and s.step <= OPENING_SOLO_HQ_RESERVE_LAST_STEP
            and len(s.my_pl) == 1
            and threat == 0
            and not p.is_comet
        ):
            peel = OPENING_FIRST_CAPTURE_SEND
            if p.ships >= peel:
                horizon_eta = self.threat_horizon.get(p.id, 999)
                if horizon_eta <= 8 and p.production >= 3:
                    # Enemy fleet actually inbound: keep slack vs growth_lock / peel math.
                    base = min(base, max(threat + 2, p.ships - peel))
                else:
                    # Race first factory: do not let growth_lock sit on 3–4 ships while
                    # 22-on-HQ cannot peel 21 (user: same turn as screenshot 2).
                    base = min(base, threat + 1)
        return base

    def avail(self, pid: int) -> int:
        """Surplus minus already-used in this turn."""
        return max(0, self.surplus.get(pid, 0) - self.used.get(pid, 0))

    def subtract(self, pid: int, ships: int) -> None:
        self.used[pid] += int(ships)

    def calculate_safe_surplus_v20(self, regional_graph: Optional[RegionalGraph] = None) -> int:
        """Ship budget for strategic commits from current per-planet avail pools."""
        avail_total = sum(self.avail(p.id) for p in self.state.my_pl)
        if avail_total <= ABS_MIN_BATCH:
            return 0
        if not regional_graph:
            return max(ABS_MIN_BATCH, int(avail_total * 0.65))

        threats = [
            regional_graph.region_threat(rid, self.state.en_pl)
            for rid in regional_graph.regions
        ]
        max_t = max(threats) if threats else 0.0
        my_prod = float(sum(p.production for p in self.state.my_pl))
        pressure = max_t / max(my_prod, 1.0)
        pressure = min(1.35, pressure)
        alloc_frac = max(0.45, min(0.82, 0.82 - pressure * 0.28))
        return max(ABS_MIN_BATCH, int(avail_total * alloc_frac))

    def is_safe_investment(self, dst: Planet, eta: int) -> bool:
        """Bocsimacko `safe-to-invest-p` port (player.lisp:1079).

        Returns True unless this looks like a clearly losing trade:
          - Enemy is much closer AND has overwhelming nearby firepower.
          - Or our friendly planets are under net inbound threat that
            already exceeds our total surplus (i.e. we cannot afford to
            send anything outward without losing a homeworld).

        Conservative - only filters obvious losers, not borderline cases.
        Original Lisp version uses time-vector arrivals; we approximate.
        """
        s = self.state
        # Defensive triage: aggregate friendly inbound threat and surplus.
        net_threat = sum(max(0, s.net_threat(p)) for p in s.my_pl)
        my_total_surplus = sum(self.surplus.values())
        if net_threat > my_total_surplus * 1.10 and my_total_surplus > 0:
            # We are already underwater on defense - no time for expansion.
            return False
        # Enemy proximity / power dominance check.
        my_reach = min((m.dist(dst) for m in s.my_pl), default=999.0)
        en_reach = min((e.dist(dst) for e in s.en_pl), default=999.0)
        early = s.phase() == "early"
        reach_factor = 0.56 if early else 0.70
        power_factor = 1.72 if early else 1.60
        if en_reach < my_reach * reach_factor:
            en_local = sum(e.ships for e in s.en_pl
                           if e.dist(dst) < my_reach * 1.20)
            if en_local > (dst.ships + dst.production * eta) * power_factor:
                return False
        return True


# ╔═══ region 4: PhasePolicy ════════════════════════════════════════════════╗
#
# ── How to iterate parameters (offline → Kaggle) ───────────────────────────
#
# 1) Canonical knobs live only in PHASE_TABLE below (early / mid / late).
#    Typical groups:
#    - Economy vs aggression: reserve_growth_mul, cost_pen_mul,
#      cost_pen_neutral_mul, urgent_attack_ratio, mode_order.
#    - Search cost: mcts_* , pragmatic_mcts_* , neural_modifier_strength ,
#      sim_steps , tempo_floor.
#    - Commit gates (were magic numbers): region_pressure_ratio,
#      safe_surplus_ship_mult — used when regional_graph is active in
#      PlanArbiter.commit_best; baseline_commit_margin vs idle baseline.
#    - Pessimistic 1-enemy-rollout rerank: paranoid_score_budget_ms (early=0),
#      paranoid_plan_top_k , paranoid_steps , paranoid_blend — see
#      score_plan_actions_paranoid + PlanArbiter.score_with_modifiers .
#
# 2) Sweep without editing the table: export ORB_REGION_PRESSURE_RATIO ,
#    ORB_SAFE_SURPLUS_SHIP_MULT , ORB_BASELINE_COMMIT_MARGIN per process ;
#    _merged_phase_row() overlays them onto PHASE_TABLE reads. Useful with
#    tools/sweep_commit_gates.py wrapping scripts/eval_head2head.py .
#
# 3) Scripted eval: scripts/eval_head2head.py --a v20 --b v17 --seeds 0-19 .
#    Style overlays (same file, ContextVar-safe): --a v20@rush ,
#    v20@turtle — ORB_STRATEGY_PROFILE merges _STRATEGY_PROFILE_DELTAS .
#
# 4) Smoked pessimistic math (no episode): python3 tools/paranoid_score.py
#
# 5) RL self-play opponent mix (tools/rollout_worker.py): --opponent-mix
#    "self:0.5,v13:0.3,v20@expand:0.2" — weights normalized; tokens are load
#    paths or ``self``.
#
# 6) Acceptance: prefer 20+ double-seeded games; watch step wall-time (arbiter
#    skips paranoid when over paranoid_score_budget_ms).
#
# Single source of truth for phase-dependent tuning. Adjusting strategy = edit
# one row (or overlay env/profile as above).

PHASE_TABLE: Dict[str, Dict[str, object]] = {
    "early": dict(
        reserve_growth_mul=1,
        cost_pen_mul=0.75,
        cost_pen_neutral_mul=0.60,
        urgent_attack_ratio=2.65,
        urgent_attack_min_prod=3,
        mode_order=["expand", "aggro", "counter", "balanced", "comet"],
        mcts_budget_ms=0,
        mcts_max_iters=0,
        pragmatic_mcts_budget_ms=0,
        pragmatic_mcts_max_iters=0,
        pragmatic_mcts_top_k=6,
        pragmatic_mcts_rollout_steps=12,
        neural_modifier_strength=0.10,
        recapture_mul=1.2,
        approach_weight=1.8,
        sim_steps=8,
        tempo_floor=1,
        region_pressure_ratio=0.72,
        safe_surplus_ship_mult=1.55,
        baseline_commit_margin=0.10,
        paranoid_score_budget_ms=0,
        paranoid_plan_top_k=4,
        paranoid_steps=8,
        paranoid_blend=0.0,
    ),
    "mid": dict(
        reserve_growth_mul=3,
        cost_pen_mul=0.80,
        cost_pen_neutral_mul=0.74,
        urgent_attack_ratio=1.05,
        urgent_attack_min_prod=4,
        mode_order=["aggro", "expand", "counter", "balanced", "comet"],
        mcts_budget_ms=120,
        mcts_max_iters=50,
        pragmatic_mcts_budget_ms=55,
        pragmatic_mcts_max_iters=28,
        pragmatic_mcts_top_k=6,
        pragmatic_mcts_rollout_steps=14,
        neural_modifier_strength=0.12,
        recapture_mul=1.10,
        approach_weight=1.55,
        sim_steps=8,
        tempo_floor=2,
        region_pressure_ratio=0.72,
        safe_surplus_ship_mult=1.55,
        baseline_commit_margin=0.10,
        paranoid_score_budget_ms=40,
        paranoid_plan_top_k=5,
        paranoid_steps=8,
        paranoid_blend=0.42,
    ),
    "late": dict(
        reserve_growth_mul=4,
        cost_pen_mul=0.78,
        cost_pen_neutral_mul=0.70,
        urgent_attack_ratio=0.90,
        urgent_attack_min_prod=3,
        mode_order=["aggro", "counter", "expand", "balanced", "comet"],
        mcts_budget_ms=200,
        mcts_max_iters=80,
        pragmatic_mcts_budget_ms=85,
        pragmatic_mcts_max_iters=36,
        pragmatic_mcts_top_k=8,
        pragmatic_mcts_rollout_steps=16,
        neural_modifier_strength=0.15,
        recapture_mul=1.18,
        approach_weight=1.40,
        sim_steps=10,
        tempo_floor=1,
        region_pressure_ratio=0.72,
        safe_surplus_ship_mult=1.55,
        baseline_commit_margin=0.10,
        paranoid_score_budget_ms=62,
        paranoid_plan_top_k=5,
        paranoid_steps=10,
        paranoid_blend=0.48,
    ),
}

# Optional per-profile overrides for local eval / rollouts via ORB_STRATEGY_PROFILE.
# Naming mirrors Planet-Wars-style archetypes (Java PW bots cannot run here);
# README lists PW ↔ profile mapping.
_STRATEGY_PROFILE_DELTAS: Dict[str, Dict[str, object]] = {
    "turtle": dict(
        reserve_growth_mul_delta=+2,
        urgent_attack_ratio_delta=+0.35,
        cost_pen_mul_delta=+0.06,
        cost_pen_neutral_mul_delta=+0.05,
    ),
    "rush": dict(
        reserve_growth_mul_delta=-1,
        urgent_attack_ratio_delta=-0.20,
        cost_pen_mul_delta=-0.06,
        mode_order_override=["aggro", "counter", "expand", "balanced", "comet"],
    ),
    "expand": dict(
        cost_pen_mul_delta=-0.10,
        cost_pen_neutral_mul_delta=-0.08,
        mode_order_override=["expand", "balanced", "aggro", "counter", "comet"],
    ),
    "greedy_prod": dict(
        urgent_attack_min_prod_delta=-1,
        recapture_mul_delta=+0.12,
        cost_pen_mul_delta=-0.04,
    ),
    "dual": dict(
        urgent_attack_ratio_delta=+0.12,
        mode_order_override=["expand", "aggro", "counter", "balanced", "comet"],
    ),
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _merged_phase_row(ph: str) -> Dict[str, object]:
    """Merge PHASE_TABLE[ph] with profile deltas and optional env gate floats.

    When set in the environment, ORB_REGION_PRESSURE_RATIO,
    ORB_SAFE_SURPLUS_SHIP_MULT, ORB_BASELINE_COMMIT_MARGIN override the row.
    """
    row: Dict[str, object] = dict(PHASE_TABLE[ph])
    prof = ORB_STRATEGY_PROFILE.get(None)
    if isinstance(prof, str) and prof.strip():
        deltas = _STRATEGY_PROFILE_DELTAS.get(prof.strip().lower())
        if deltas:
            for k, dv in deltas.items():
                if k == "mode_order_override":
                    row["mode_order"] = list(dv)  # type: ignore[arg-type]
                    continue
                if k.endswith("_delta"):
                    base_key = k[:-6]
                    old = row.get(base_key)
                    if old is None:
                        continue
                    if isinstance(old, bool):
                        continue
                    if isinstance(old, int):
                        row[base_key] = int(old) + int(dv)  # type: ignore[arg-type]
                    elif isinstance(old, float):
                        row[base_key] = float(old) + float(dv)  # type: ignore[arg-type]
            rgm = int(row.get("reserve_growth_mul", 2))
            row["reserve_growth_mul"] = max(1, min(8, rgm))
    row["region_pressure_ratio"] = _env_float(
        "ORB_REGION_PRESSURE_RATIO", float(row.get("region_pressure_ratio", 0.72))
    )
    row["safe_surplus_ship_mult"] = _env_float(
        "ORB_SAFE_SURPLUS_SHIP_MULT", float(row.get("safe_surplus_ship_mult", 1.55))
    )
    row["baseline_commit_margin"] = _env_float(
        "ORB_BASELINE_COMMIT_MARGIN", float(row.get("baseline_commit_margin", 0.10))
    )
    return row


@dataclass
class PhasePolicy:
    """Resolved view of PHASE_TABLE for the current step."""
    phase: str
    reserve_growth_mul: int
    cost_pen_mul: float
    cost_pen_neutral_mul: float
    urgent_attack_ratio: float
    urgent_attack_min_prod: int
    mode_order: List[str]
    mcts_budget_ms: float
    mcts_max_iters: int
    pragmatic_mcts_budget_ms: float
    pragmatic_mcts_max_iters: int
    pragmatic_mcts_top_k: int
    pragmatic_mcts_rollout_steps: int
    neural_modifier_strength: float
    recapture_mul: float
    approach_weight: float
    sim_steps: int
    tempo_floor: int
    region_pressure_ratio: float
    safe_surplus_ship_mult: float
    baseline_commit_margin: float
    paranoid_score_budget_ms: float
    paranoid_plan_top_k: int
    paranoid_steps: int
    paranoid_blend: float

    @classmethod
    def for_state(cls, state: GameState) -> "PhasePolicy":
        ph = state.phase()
        row = _merged_phase_row(ph)
        return cls(
            phase=ph,
            reserve_growth_mul=int(row["reserve_growth_mul"]),
            cost_pen_mul=float(row["cost_pen_mul"]),
            cost_pen_neutral_mul=float(row["cost_pen_neutral_mul"]),
            urgent_attack_ratio=float(row["urgent_attack_ratio"]),
            urgent_attack_min_prod=int(row["urgent_attack_min_prod"]),
            mode_order=list(row["mode_order"]),  # type: ignore[arg-type]
            mcts_budget_ms=float(row["mcts_budget_ms"]),
            mcts_max_iters=int(row["mcts_max_iters"]),
            pragmatic_mcts_budget_ms=float(row["pragmatic_mcts_budget_ms"]),
            pragmatic_mcts_max_iters=int(row["pragmatic_mcts_max_iters"]),
            pragmatic_mcts_top_k=int(row["pragmatic_mcts_top_k"]),
            pragmatic_mcts_rollout_steps=int(row["pragmatic_mcts_rollout_steps"]),
            neural_modifier_strength=float(row["neural_modifier_strength"]),
            recapture_mul=float(row["recapture_mul"]),
            approach_weight=float(row["approach_weight"]),
            sim_steps=int(row["sim_steps"]),
            tempo_floor=int(row["tempo_floor"]),
            region_pressure_ratio=float(row["region_pressure_ratio"]),
            safe_surplus_ship_mult=float(row["safe_surplus_ship_mult"]),
            baseline_commit_margin=float(row["baseline_commit_margin"]),
            paranoid_score_budget_ms=float(row["paranoid_score_budget_ms"]),
            paranoid_plan_top_k=int(row["paranoid_plan_top_k"]),
            paranoid_steps=int(row["paranoid_steps"]),
            paranoid_blend=float(row["paranoid_blend"]),
        )


# ╔═══ region 5: scoring ════════════════════════════════════════════════════╗

def approach_bonus(snap: Snapshot, dst: Planet, eta: int) -> float:
    if not snap.state.is_orbiting(dst):
        return 0.0
    cx, cy = snap.centroid
    now_d = math.hypot(dst.x - cx, dst.y - cy)
    fx, fy = snap.state.planet_pos_at(dst, eta)
    fut_d = math.hypot(fx - cx, fy - cy)
    gain = now_d - fut_d
    return max(-28.0, min(34.0, gain * snap.policy.approach_weight + dst.production * 1.2))


def orbit_arc_strategic_score(state: GameState, dst: Planet, eta: int) -> float:
    """Prefer targets that, by arrival time, drift toward our mass / away from enemy.

    Penalizes chasing passing neutrals that rotate into the opponent's arc (user:
    「占了也快进对面半场 = 白给」).
    """
    if dst.is_comet or not state.is_orbiting(dst):
        return 0.0
    if not state.en_pl or not state.my_pl:
        return 0.0
    t = max(1, min(int(eta), 56))
    fx, fy = state.planet_pos_at(dst, t)
    now_d_my = min(dst.dist(m) for m in state.my_pl)
    fut_d_my = min(
        math.hypot(fx - state.planet_pos_at(m, t)[0], fy - state.planet_pos_at(m, t)[1])
        for m in state.my_pl
    )
    now_d_en = min(dst.dist(e) for e in state.en_pl)
    fut_d_en = min(
        math.hypot(fx - state.planet_pos_at(e, t)[0], fy - state.planet_pos_at(e, t)[1])
        for e in state.en_pl
    )
    # Positive: closing on our worlds. Positive toward_en_closing: closing on enemy (bad).
    toward_us = now_d_my - fut_d_my
    toward_en_closing = now_d_en - fut_d_en
    raw = 1.18 * toward_us - 1.52 * max(0.0, toward_en_closing)
    return max(-46.0, min(52.0, raw))


def enemy_eta_power(state: GameState, dst: Planet) -> Tuple[int, int]:
    best_eta, best_power = 999, 0
    for e in state.en_pl:
        probe = max(1, min(e.ships, max(5, e.ships * 2 // 3)))
        _, _, eta, _ = lead_intercept(state, e, dst, probe)
        if eta < best_eta:
            best_eta, best_power = eta, e.ships
    return best_eta, best_power


def recapture_bonus(snap: Snapshot, dst: Planet) -> float:
    if dst.owner not in snap.state.en_ids:
        return 0.0
    cx, cy = snap.centroid
    d = max(1.0, math.hypot(dst.x - cx, dst.y - cy))
    return dst.production * 14.0 / (1.0 + d * 0.04) * snap.policy.recapture_mul


def contest_penalty(state: GameState, dst: Planet) -> float:
    en_inc = state.enemy_incoming(dst.id)
    if en_inc <= 0:
        return 0.0
    return min(40.0, en_inc * 0.6)



def target_score(snap: Snapshot, src: Planet, dst: Planet) -> Tuple[float, int, int]:
    """Score (score, need, eta) — v20: distance-dominant scoring (v19 lineage).

    核心原则（模仿 top 方案）：
    - 近处目标天然占优（eta 是强惩罚因子）
    - 高产值星球有额外加成
    - 路径穿越太阳的目标直接判死（不调 safe_aim，纯几何检查）
    - 不跨越太阳去打远处目标
    """
    state = snap.state
    if dst.owner == state.my_id or src.id == dst.id:
        return -1e18, 0, 0

    need, eta = capture_need(state, src, dst)
    if need <= 0:
        return -1e18, 0, eta

    # 球心连线擦过太阳：启发式上偏难，但弹道可从边缘绕行（见 _emit / safe_aim）。
    # 旧逻辑直接 -1e18 使内环球长期进不了排序；改为扣分，交给真轨迹门。
    sun_detour_pen = 0.0
    sun_dist = point_segment_distance(SUN_X, SUN_Y, src.x, src.y, dst.x, dst.y)
    if sun_dist < SUN_RADIUS + SUN_PATH_MARGIN:
        sun_detour_pen = 42.0 + eta * 0.30

    raw_turns = max(1, state.turns_left() - eta)
    turns = min(raw_turns, HORIZON_TURNS)
    if dst.is_comet:
        turns = min(turns, max(0, state.comet_turns_left(dst) - eta), 60)
        if turns <= 8:
            return -1e18, need, eta

    is_neu = dst.owner == -1
    is_en = dst.owner not in (-1, state.my_id)

    # Snipe-aware sizing for neutrals
    if is_neu and dst.production > 0 and dst.ships > dst.production:
        e_eta_first, _ = enemy_eta_power(state, dst)
        if 0 < e_eta_first < eta:
            owner_after, ships_after = target_state_at(state, dst, e_eta_first + 1)
            if owner_after not in (-1, state.my_id):
                snipe_eta = max(eta, e_eta_first + 1)
                snipe_need = ships_after + 8 + min(6, dst.production)
                snipe_need += dst.production * snipe_eta // 5
                if snipe_need > need:
                    need = max(need, snipe_need)
                    eta = snipe_eta

    # === 打分（distance-dominant） ===

    # 产值回报：占据后每回合收益 × 收益轮数，但用 min(turns, 30) 压缩远期
    prod_value = dst.production * min(turns, 30)

    # 高产加成：prod>=5 的星球是战略目标
    prod_bonus = 0.0
    if dst.production >= 5:
        prod_bonus = 40.0 + dst.production * 5.0
    elif dst.production >= 3:
        prod_bonus = 15.0 + dst.production * 3.0
    elif dst.production >= 1:
        prod_bonus = dst.production * 2.0

    # 敌方星球额外价值（夺取=削弱对手+增强自己）；贴身弱敌强推（expand 曾漏掉 >20 驻军的邻球）
    enemy_bonus = 30.0 if is_en else 0.0
    if is_en and src.dist(dst) < 20.0:
        enemy_bonus += 35.0
    comet_bonus = 12.0 if dst.is_comet else 0.0
    rec_bonus = recapture_bonus(snap, dst)
    # Early race for big factories: nudge expand ordering toward prod>=4 neutrals.
    early_hot_neutral = (
        22.0 if (is_neu and state.phase() == "early" and dst.production >= 4) else 0.0
    )
    # Value-per-commitment: prod^2 / need  favors factories over low-prod mites when
    # distances are similar (common case: 20@prod4 vs 14@prod1).
    neutral_mfg = 0.0
    if is_neu:
        neutral_mfg = 48.0 * float(dst.production * dst.production) / max(1.0, float(need))

    # 距离惩罚：eta 的 **强** 衰减——这是与旧版最大的区别
    # eta=5 → 0.67, eta=10 → 0.50, eta=20 → 0.37, eta=30 → 0.31
    distance_decay = 1.0 / (1.0 + eta * 0.10)

    # 兵力成本：需要的兵越多，性价比越低
    cost_pen = 0.0
    if eta > 3:
        cost_mul = snap.policy.cost_pen_neutral_mul if is_neu else snap.policy.cost_pen_mul
        cost_pen = cost_mul * need

    # Sniping risk
    snipe_pen = 0.0
    if is_neu:
        e_eta, e_pow = enemy_eta_power(state, dst)
        if e_eta <= eta + 1 and e_pow > max(0, need - 4):
            snipe_pen = 30.0
        elif e_eta <= eta + 2 and e_pow > need + 5:
            snipe_pen = 15.0

    # 1 产「蚊子球」：前期主动压低排序，避免和轨道上 3–4 产厂星抢优先（用户反馈先占 14@1 再等 20@4）。
    mite_neutral_pen = 0.0
    if is_neu and state.phase() == "early" and dst.production <= 1:
        mite_neutral_pen = 38.0

    orbit_arc = 0.0
    approach_adj = 0.0
    fat_local_neu = 0.0
    finish_neu = 0.0
    if is_neu:
        if state.en_pl:
            orbit_arc = orbit_arc_strategic_score(state, dst, eta)
            approach_adj = 0.48 * approach_bonus(snap, dst, eta)
        if dst.ships >= 38:
            d_anchor = min(dst.dist(m) for m in state.my_pl)
            if d_anchor < 36.0:
                # 身边大灰（如 59）：优先吃近处高驻军工厂，少去追「路过」小灰
                fat_local_neu = 16.0 + (36.0 - d_anchor) * 1.65 + min(22.0, dst.ships * 0.08)
        oth = my_inbound_ships_to(state, dst.id)
        if oth > 0 and oth <= dst.ships + 4:
            # 已有己方舰队在途但未够占领：优先补刀，避免下回合改打远处浪费产兵
            gap = (dst.ships + 1) - oth
            if 1 <= gap <= 18:
                finish_neu = 52.0 + (18 - gap) * 1.5

    score = (
        prod_value + prod_bonus + enemy_bonus + comet_bonus + rec_bonus
        + early_hot_neutral + neutral_mfg + fat_local_neu + finish_neu
        + orbit_arc + approach_adj
    ) * distance_decay
    score -= cost_pen + snipe_pen + mite_neutral_pen + sun_detour_pen
    # 内环球中立：战略优先级（旋转进对手弧前须占）
    if is_neu and is_sun_belt_planet(state, dst):
        score += 38.0 * distance_decay
    return score, need, eta


def regional_capture_adjustment(
    snap: Snapshot,
    src: Planet,
    dst: Planet,
    regional_graph: RegionalGraph,
    eta: int,
) -> float:
    """Additive regional layer: cohesion bonus or cross-zone expedition tax."""
    state = snap.state
    my_ids = {p.id for p in state.my_pl}
    rid_s = regional_graph.planet_to_region.get(src.id, -1)
    rid_d = regional_graph.planet_to_region.get(dst.id, -1)
    if rid_s < 0 or rid_d < 0:
        return 0.0

    ddist, _ = regional_graph.dijkstra(src.id, dst.id)
    dist_cost = 0.12 * min(ddist, 80.0)

    if rid_s == rid_d:
        my_prod = regional_graph.region_production(rid_d, my_ids)
        bonus = 8.0 + min(22.0, float(my_prod) * 1.55)
        pot = 0.0
        for p in regional_graph.get_planets_in_region(rid_d):
            if p.id == dst.id:
                continue
            if p.owner == -1:
                pot += float(p.production) + 0.5
            elif p.owner in state.en_ids:
                eg = state.effective_garrison(p)
                if eg < p.ships * 0.65:
                    pot += float(p.production) * 0.6
        bonus += min(16.0, pot * 1.8)
        return bonus - dist_cost * 0.35

    cross = 10.0 + 0.045 * float(eta * eta)
    if is_sun_belt_planet(state, dst) and dst.owner == -1:
        cross *= 0.38
    return -cross - dist_cost


def capture_edge_score(
    snap: Snapshot,
    src: Planet,
    dst: Planet,
    regional_graph: Optional[RegionalGraph] = None,
) -> Tuple[float, int, int]:
    """Unified capture ranking: heuristic target_score + optional regional layer."""
    base, need, eta = target_score(snap, src, dst)
    if regional_graph is None or base <= -1e17:
        return base, need, eta
    adj = regional_capture_adjustment(snap, src, dst, regional_graph, eta)
    return base + adj, need, eta


def target_value_in_region(
    snap: Snapshot,
    src: Planet,
    dst: Planet,
    regional_graph: Optional[RegionalGraph] = None,
) -> float:
    """Same scalar as ``capture_edge_score`` first component (compat / tooling)."""
    sc, _, _ = capture_edge_score(snap, src, dst, regional_graph)
    return sc



def elite_eval(state: GameState) -> float:
    """Static positional eval - reserved for NeuralVal feature engineering."""
    mi = state.my_id
    ms = state.total_ships(mi)
    es = sum(state.total_ships(e) for e in state.en_ids) + 1e-9
    mp = sum(p.production for p in state.my_pl)
    ep = sum(p.production for p in state.en_pl)
    mc = len(state.my_pl); ec = len(state.en_pl)
    threat = sum(max(0, state.net_threat(p)) for p in state.my_pl)
    mf = sum(f.ships for f in state.fleets if f.owner == mi)
    ef = sum(f.ships for f in state.fleets if f.owner not in (-1, mi))
    border = sum((35 - m.dist(e)) / 35 * m.production
                 for m in state.my_pl for e in state.en_pl if m.dist(e) < 35)
    ndeny = sum(n.production for n in state.neu_pl
                if any(n.dist(e) < 25 for e in state.en_pl)
                and not any(n.dist(m) < 25 for m in state.my_pl))
    return ((ms - es) + 48.0 * (mp - ep) + 20.0 * (mc - ec)
            - 2.8 * threat + 9.0 * border + 0.45 * (mf - ef) - 12.0 * ndeny)


# ╔═══ region 6: SimP / SimF / sim_step (forward simulator) ═════════════════╗

class SimP:
    __slots__ = ("id", "owner", "ships", "production")

    def __init__(self, p):
        self.id = p.id; self.owner = p.owner
        self.ships = p.ships; self.production = p.production

    def copy(self) -> "SimP":
        s = SimP.__new__(SimP)
        s.id = self.id; s.owner = self.owner
        s.ships = self.ships; s.production = self.production
        return s


class SimF:
    __slots__ = ("owner", "tid", "ships", "eta")

    def __init__(self, owner, tid, ships, eta):
        self.owner = owner; self.tid = tid; self.ships = ships; self.eta = eta

    def copy(self) -> "SimF":
        return SimF(self.owner, self.tid, self.ships, self.eta)


def clone_sim(state: GameState) -> Tuple[Dict[int, SimP], List[SimF]]:
    planets = {p.id: SimP(p) for p in state.planets}
    fleets: List[SimF] = []
    for f in state.fleets:
        t = state.fleet_target.get(f.id)
        if t:
            fleets.append(SimF(f.owner, t[0], f.ships, t[1]))
    return planets, fleets


def copy_sim(planets: Dict[int, SimP], fleets: List[SimF]):
    return ({pid: p.copy() for pid, p in planets.items()},
            [f.copy() for f in fleets])


def sim_step(planets: Dict[int, SimP], fleets: List[SimF]) -> None:
    for p in planets.values():
        if p.owner >= 0:
            p.ships += p.production
    by_target: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    nxt: List[SimF] = []
    for f in fleets:
        f.eta -= 1
        if f.eta <= 0:
            by_target[f.tid].append((f.owner, f.ships))
        else:
            nxt.append(f)
    for tid, arrivals in by_target.items():
        p = planets.get(tid)
        if p:
            p.owner, p.ships = _combat(p.owner, p.ships, arrivals)
    fleets[:] = nxt


def eval_sim_planets(state: GameState, planets: Dict[int, SimP], fleets: List[SimF]) -> float:
    mi = state.my_id
    ms = sum(p.ships for p in planets.values() if p.owner == mi)
    ms += sum(f.ships for f in fleets if f.owner == mi)
    es = sum(p.ships for p in planets.values() if p.owner not in (-1, mi))
    es += sum(f.ships for f in fleets if f.owner not in (-1, mi))
    mp = sum(p.production for p in planets.values() if p.owner == mi)
    ep = sum(p.production for p in planets.values() if p.owner not in (-1, mi))
    mc = sum(1 for p in planets.values() if p.owner == mi)
    ec = sum(1 for p in planets.values() if p.owner not in (-1, mi))
    return (ms - es) + 48.0 * (mp - ep) + 20.0 * (mc - ec)


# ╔═══ region 7: Plan + planners ════════════════════════════════════════════╗

@dataclass
class Plan:
    actions: List[Tuple[int, int, int]] = field(default_factory=list)
    score: float = 0.0
    tag: str = ""
    urgent: bool = False  # if True, commit before scoring strategic plans


def _prep_sim_own_launches(state: GameState, actions: List[Tuple[int, int, int]],
                           tempo_floor: int) -> Tuple[Dict[int, SimP], List[SimF]]:
    """Clone + apply our launches + tempo_floor idle sim (before eval window)."""
    planets, fleets = clone_sim(state)
    used: Dict[int, int] = defaultdict(int)
    for sid, did, ships in actions:
        sp = state.get(sid); dp = state.get(did); sim_src = planets.get(sid)
        if sp is None or dp is None or sim_src is None or sim_src.owner != state.my_id:
            continue
        send = min(int(ships), max(0, sim_src.ships - ABS_MIN_BATCH - used[sid]))
        if send <= 0:
            continue
        _, eta = safe_aim(state, sp, dp, send)
        sim_src.ships -= send
        used[sid] += send
        fleets.append(SimF(state.my_id, did, send, eta))
    for _ in range(max(0, tempo_floor - 1)):
        sim_step(planets, fleets)
    return planets, fleets


def _rollout_eval(state: GameState, planets: Dict[int, SimP], fleets: List[SimF],
                  steps: int) -> float:
    p, f = copy_sim(planets, fleets)
    for _ in range(steps):
        sim_step(p, f)
    return eval_sim_planets(state, p, f)


def score_plan_actions(state: GameState, actions: List[Tuple[int, int, int]],
                       steps: int = 8, tempo_floor: int = 1) -> float:
    """Score by simulating `steps` turns after applying actions to a clone.

    `tempo_floor` (Bocsimacko `min-turn-to-depart`): run this many *idle*
    sim steps BEFORE the regular eval window, simulating no follow-up fleets
    yet. Plans that depend on chained reinforcements that we have not planned
    for in `actions` get penalised because their targets do not materialise
    within the eval window without those follow-ups. tempo_floor=1 means
    evaluate as-is.
    """
    planets, fleets = _prep_sim_own_launches(state, actions, tempo_floor)
    return _rollout_eval(state, planets, fleets, steps)


def _primary_enemy_id(state: GameState) -> Optional[int]:
    if not state.en_ids:
        return None
    return max(state.en_ids, key=lambda e: state.total_ships(e))


def _try_append_enemy_fleet(
    state: GameState,
    planets: Dict[int, SimP],
    fleets: List[SimF],
    enemy_id: int,
    src_id: int,
    dst_id: int,
    want_ships: int,
) -> bool:
    sp = planets.get(src_id)
    dp = planets.get(dst_id)
    geo_s = state.get(src_id)
    geo_d = state.get(dst_id)
    if sp is None or dp is None or geo_s is None or geo_d is None:
        return False
    if sp.owner != enemy_id:
        return False
    max_send = max(0, sp.ships - ABS_MIN_BATCH)
    send = min(max(int(want_ships), ABS_MIN_BATCH), max_send)
    if send < ABS_MIN_BATCH:
        return False
    _, eta = safe_aim(state, geo_s, geo_d, send)
    sp.ships -= send
    fleets.append(SimF(enemy_id, dst_id, send, eta))
    return True


def _inject_paranoid_profile(
    state: GameState,
    planets: Dict[int, SimP],
    fleets: List[SimF],
    enemy_id: int,
    profile_key: str,
) -> None:
    """Mutate planets/fleets with one pessimistic opponent launch."""
    mi = state.my_id
    enemy_pids = [pid for pid, sp in planets.items()
                  if sp.owner == enemy_id and sp.ships >= ABS_MIN_BATCH * 2]
    my_pids = [pid for pid, sp in planets.items() if sp.owner == mi]
    neu_pids = [pid for pid, sp in planets.items() if sp.owner == -1]
    cx, cy = state.centroid()

    want = ABS_MIN_BATCH * 8

    if profile_key == "nearest_my":
        best = None
        for eid in enemy_pids:
            ge = state.get(eid)
            if ge is None:
                continue
            for mid in my_pids:
                gm = state.get(mid)
                if gm is None:
                    continue
                d = ge.dist(gm)
                if best is None or d < best[0]:
                    best = (d, eid, mid)
        if best:
            _, eid, mid = best
            _try_append_enemy_fleet(state, planets, fleets, enemy_id, eid, mid,
                                    want)

    elif profile_key == "contest_neutral" and neu_pids:
        neu_pids_sorted = sorted(
            neu_pids,
            key=lambda nid: math.hypot(state.get(nid).x - cx, state.get(nid).y - cy)
            if state.get(nid) else 999)
        tgt = None
        for nid in neu_pids_sorted[:6]:
            gn = state.get(nid)
            if gn is None:
                continue
            best_ep = None
            for eid in enemy_pids:
                ge = state.get(eid)
                if ge is None:
                    continue
                d = ge.dist(gn)
                if best_ep is None or d < best_ep[0]:
                    best_ep = (d, eid)
            if best_ep:
                tgt = (best_ep[1], nid)
                break
        if tgt:
            _try_append_enemy_fleet(state, planets, fleets, enemy_id, tgt[0], tgt[1],
                                    want)

    elif profile_key == "strike_high_prod_my" and my_pids:
        mid = max(my_pids, key=lambda pid: state.get(pid).production
                   if state.get(pid) else 0)
        gm = state.get(mid)
        if gm is None:
            return
        best_ep = None
        for eid in enemy_pids:
            ge = state.get(eid)
            if ge is None:
                continue
            d = ge.dist(gm)
            if best_ep is None or d < best_ep[0]:
                best_ep = (d, eid)
        if best_ep:
            _try_append_enemy_fleet(state, planets, fleets, enemy_id,
                                    best_ep[1], mid,
                                    max(want, ABS_MIN_BATCH * 12))


def score_plan_actions_paranoid(
    state: GameState,
    actions: List[Tuple[int, int, int]],
    steps: int,
    tempo_floor: int,
    par_steps: Optional[int] = None,
) -> Tuple[float, float]:
    """Return ``(baseline_sim, pessimistic_sim)``

    Baseline ignores fresh opponent fleets; pessimistic inserts up to three
    single-launch pessimistic envelopes from the strongest enemy, then takes
    the worst ``eval_sim_planets`` among them (and the no-injection path).
    """
    if par_steps is None:
        par_steps = steps
    eid = _primary_enemy_id(state)
    planets0, fleets0 = _prep_sim_own_launches(state, actions, tempo_floor)

    baseline = _rollout_eval(state, planets0, fleets0, steps)
    if eid is None:
        return baseline, baseline

    worst = baseline
    for key in ("nearest_my", "contest_neutral", "strike_high_prod_my"):
        p, f = copy_sim(planets0, fleets0)
        _inject_paranoid_profile(state, p, f, eid, key)
        worst = min(worst, _rollout_eval(state, p, f, par_steps))

    return baseline, worst


def blended_paranoid_sim(
    state: GameState,
    actions: List[Tuple[int, int, int]],
    *,
    steps: int,
    tempo_floor: int,
    par_steps: int,
    blend: float,
) -> float:
    """``base + blend * (pessimistic - base)`` for arbiter blending."""
    b, pe = score_plan_actions_paranoid(state, actions, steps, tempo_floor,
                                        par_steps=par_steps)
    if blend <= 1e-9:
        return b
    return b + blend * (pe - b)


# ── DefensePlanner ───────────────────────────────────────────────────────────

class DefensePlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        actions: List[Tuple[int, int, int]] = []
        score = 0.0
        local_used: Dict[int, int] = defaultdict(int)

        # Combine reactive (net_threat > 0) and proactive (threat_horizon <= 8
        # on high-production planets) targets. Window expanded to 8 in v14.
        reactive = {p for p in state.my_pl if state.net_threat(p) > 0}
        proactive = {state.get(pid) for pid, eta in snap.threat_horizon.items()
                     if eta <= 8}
        proactive = {p for p in proactive if p is not None and p.production >= 3
                     and p not in reactive}
        all_targets = (sorted(reactive, key=lambda p: -state.net_threat(p))
                       + sorted(proactive, key=lambda p: -p.production))

        for tgt in all_targets:
            if tgt in reactive:
                threat = state.net_threat(tgt)
                need = threat + max(5, tgt.production * 2)
            else:
                # Proactive: reinforce to absorb the incoming fleet.
                incoming = sum(f.ships for f in state.fleets
                               if f.owner not in (-1, state.my_id)
                               and state.fleet_target.get(f.id, (None,))[0] == tgt.id)
                need = max(incoming - tgt.ships + tgt.production * 3, ABS_MIN_BATCH)
            helpers = sorted((p for p in state.my_pl if p.id != tgt.id),
                             key=lambda p: p.dist(tgt))
            for src in helpers:
                if need <= 0:
                    break
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                send = min(avail, need)
                if send < ABS_MIN_BATCH:
                    continue
                actions.append((src.id, tgt.id, send))
                local_used[src.id] += send
                need -= send
                score += send * 4.0
        return Plan(actions, score, "defense", urgent=True)


# ── InterceptPlanner ─────────────────────────────────────────────────────────

class InterceptPlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        actions: List[Tuple[int, int, int]] = []
        score = 0.0
        local_used: Dict[int, int] = defaultdict(int)

        for f in sorted(state.fleets, key=lambda x: -x.ships):
            if f.owner in (-1, state.my_id) or f.ships < 10:
                continue
            target = state.fleet_target.get(f.id)
            if not target:
                continue
            tid, eta_to_planet = target
            dst = state.get(tid)
            if dst is None or dst.owner != state.my_id:
                continue
            helpers = sorted((p for p in state.my_pl if p.id != dst.id),
                             key=lambda p: p.dist(dst))
            need = f.ships + 4
            for src in helpers[:4]:
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                if avail < ABS_MIN_BATCH:
                    continue
                _, _, eta, _ = lead_intercept(state, src, dst, min(avail, need))
                if eta > eta_to_planet + 1:
                    continue
                send = min(avail, need)
                if send < ABS_MIN_BATCH:
                    continue
                actions.append((src.id, dst.id, send))
                local_used[src.id] += send
                need -= send
                score += send * 2.5
                if need <= 0:
                    break
        return Plan(actions, score, "intercept", urgent=True)


# ── Capture-style planner (shared by Expand and Attack) ──────────────────────

def _target_pool(state: GameState, mode: str,
                 diplo: Optional["DiplomacyEngine"] = None) -> List[Planet]:
    if mode == "expand":
        neu = sorted(state.neu_pl, key=lambda p: (-p.production, -p.ships))
        # v20: 旧阈值 ships<=20 会漏掉邻接「26~40兵+3产」等可吃口袋，表现为一格之隔从不打。
        soft_en: List[Planet] = []
        for p in state.en_pl:
            if p.ships <= 20 or p.production <= 2:
                soft_en.append(p)
                continue
            if state.phase() != "early":
                eg = state.effective_garrison(p)
                if eg <= 44 and p.ships <= 52:
                    soft_en.append(p)
        return neu + soft_en
    if mode == "aggro":
        return sorted(state.en_pl,
                      key=lambda p: (state.effective_garrison(p) - p.production * 5,
                                     p.dist_xy(SUN_X, SUN_Y)))
    if mode == "comet":
        return [p for p in state.planets if p.is_comet and p.owner != state.my_id]
    if mode == "counter":
        return [p for p in state.en_pl if state.effective_garrison(p) < p.ships * 0.55]
    if mode == "diplo":
        return state.en_pl
    return state.neu_pl + state.en_pl


def _build_capture_plan(snap: Snapshot, mode: str,
                        diplo: Optional["DiplomacyEngine"] = None,
                        regional_graph: Optional[RegionalGraph] = None,
                        diplo_target: Optional[Planet] = None) -> Plan:
    """Generic build: ranked targets + multi-source ETA-synced batches.
    Does NOT mutate snap.used; the Arbiter does that on commit."""
    state = snap.state
    actions: List[Tuple[int, int, int]] = []
    score = 0.0
    local_used: Dict[int, int] = defaultdict(int)
    target_done: set = set()

    targets = _target_pool(state, mode, diplo)
    if mode == "diplo" and diplo_target is not None:
        targets = [diplo_target] + [p for p in targets if p.id != diplo_target.id]

    # Rank.
    ranked: List[Tuple[float, Planet]] = []
    # Rank (aggro/counter/diplo keep tight floor; expand/balanced need looser floor).
    rank_floor = EXPAND_RANK_SCORE_FLOOR if mode in ("expand", "balanced") else -31.0
    for dst in targets:
        best_sc = -1e18
        for src in state.my_pl:
            sc, _, _ = capture_edge_score(snap, src, dst, regional_graph)
            if mode == "aggro":
                sc += recapture_bonus(snap, dst) * 0.5
            if mode == "counter":
                sc += 38.0 if snap.policy.phase == "mid" else 30.0
            if mode == "diplo":
                sc += 25.0 if dst is diplo_target else 0.0
            if diplo:
                sc += diplo.leader_penalty(dst)
            if sc > best_sc:
                best_sc = sc
        if best_sc > rank_floor:
            ranked.append((best_sc, dst))
    ranked.sort(key=lambda x: -x[0])

    source_dst_lock: Dict[int, int] = {}
    split_lock = (
        mode in ("expand", "balanced")
        and len(state.my_pl) <= ONE_OUTBOUND_DST_PER_SOURCE_UNTIL_N_WORLDS
    )

    for _, dst in ranked[:MAX_TARGETS_PER_PLAN]:
        if len(actions) >= MAX_TOTAL_MOVES or dst.id in target_done:
            continue

        # Defer low-prod neutrals while a prod>=3 factory grey is catchable within a
        # few turns of HQ production — don't burn the first wave on 14@1 when 20@3/4
        # is about to become peelable (same orbit / similar eta).
        if mode == "expand" and dst.owner == -1 and dst.production <= 2:
            pool_rem = sum(
                max(0, snap.avail(p.id) - local_used[p.id]) for p in state.my_pl)
            my_prod_sum = sum(p.production for p in state.my_pl)
            defer_low_yield = False
            for h in state.neu_pl:
                if h.production < 3 or h.id == dst.id:
                    continue
                if not any(
                    point_segment_distance(SUN_X, SUN_Y, sx.x, sx.y, h.x, h.y)
                    >= SUN_RADIUS + SUN_PATH_MARGIN
                    for sx in state.my_pl
                ):
                    continue
                min_need = min(
                    capture_need(state, s, h)[0] for s in state.my_pl)
                # prod==1 mites: wait longer toward factory; prod==2 slightly longer than old slack.
                if dst.production <= 1:
                    wait_budget = max(my_prod_sum * 6, 20 + my_prod_sum * 2, 24)
                else:
                    wait_budget = max(my_prod_sum * 4, 12 + my_prod_sum * 2)
                if pool_rem < min_need <= pool_rem + wait_budget:
                    defer_low_yield = True
                    break
            if defer_low_yield:
                continue

        contributors: List[Tuple[int, Planet, int, int, float]] = []
        en_belt = any(is_sun_belt_planet(state, e) for e in state.en_pl)
        for src in state.my_pl:
            if split_lock:
                locked_d = source_dst_lock.get(src.id)
                if locked_d is not None and locked_d != dst.id:
                    continue
            avail = max(0, snap.avail(src.id) - local_used[src.id])
            # Do not strip inner sun-belt planets for outer “passing” neutrals once
            # the mid-game contest starts (rotation drags these into enemy arcs).
            if mode in ("expand", "balanced") and is_sun_belt_planet(state, src):
                if dst.owner == -1 and not is_sun_belt_planet(state, dst):
                    if len(state.my_pl) >= 2 and (en_belt or state.phase() != "early"):
                        hoard = 14 + src.production * 5
                        if state.phase() in ("mid", "late"):
                            hoard += 8
                        avail = max(0, avail - hoard)
            # v14: lowered threshold to ABS_MIN_BATCH (was max(ABS_MIN_BATCH, prod*2))
            # so all planets with any surplus can join coordinated captures.
            if avail < ABS_MIN_BATCH:
                continue
            need, eta = capture_need(state, src, dst)
            sc, _, _ = capture_edge_score(snap, src, dst, regional_graph)
            contributors.append((eta, src, avail, need, sc))
        if not contributors:
            continue

        contributors.sort(key=lambda x: (x[0], -x[4], -x[2]))
        eta0 = contributors[0][0]
        min_eta = eta0

        staged_ok: List[Tuple[int, int, int]] = []
        group_used: List[Tuple[int, Planet, int, int, float]] = []
        required_ok = 0
        window = SYNC_ETA_WINDOW
        while window <= SYNC_ETA_WINDOW_MAX:
            group = [c for c in contributors if c[0] <= eta0 + window][
                :MAX_SOURCES_PER_TARGET]
            if not group:
                break

            group_eta = max(c[0] for c in group)
            # v16: garrison uses fastest arrival; enemy production uses (group-min) gap.
            if dst.owner == -1 and group_eta > 3 and not snap.is_safe_investment(
                    dst, group_eta):
                slack_pool = sum(
                    max(0, snap.avail(p.id) - local_used[p.id]) for p in state.my_pl)
                brute_need = max(
                    NEUTRAL_BRUTE_SLACK_MIN,
                    int(dst.ships * NEUTRAL_BRUTE_SLACK_MUL + dst.production * 6 + 24),
                )
                if slack_pool < brute_need:
                    break
            owner, garrison = target_state_at(state, dst, min_eta)
            if owner == state.my_id:
                break
            if owner == -1:
                if snap.policy.phase == "early" and min_eta <= 2:
                    neu_pad = 1
                elif snap.policy.phase == "early" and min_eta <= 12:
                    # Match ``capture_need`` early neutral margin (+1), not +2 — otherwise
                    # ``required`` is 22 vs ships+1==21 and solo HQ fails staging one turn early.
                    neu_pad = 1
                else:
                    neu_pad = 3
            else:
                neu_pad = 8 + min(6, dst.production)
            required = garrison + neu_pad
            if owner not in (-1, state.my_id):
                required += dst.production * max(0, group_eta - min_eta) // 3

            if dst.is_comet and state.comet_turns_left(dst) <= group_eta + 5:
                break

            total_group_avail = sum(c[2] for c in group)
            if total_group_avail < required:
                window += max(4, SYNC_ETA_WINDOW)
                continue

            sent = 0
            staged: List[Tuple[int, int, int]] = []
            for _, src, avail, _, _ in group:
                if sent >= required:
                    break
                send = min(avail, required - sent)
                if send < ABS_MIN_BATCH and sent + send < required:
                    continue
                send = max(send, ABS_MIN_BATCH) if send >= ABS_MIN_BATCH else send
                staged.append((src.id, dst.id, send))
                sent += send
            if sent >= required:
                staged_ok = staged
                group_used = group
                required_ok = required
                break

            window += max(4, SYNC_ETA_WINDOW)

        if not staged_ok:
            continue

        for sid, did, send in staged_ok:
            actions.append((sid, did, send))
            local_used[sid] += send
            if split_lock:
                source_dst_lock[sid] = dst.id
        target_done.add(dst.id)
        score += sum(c[4] for c in group_used[:len(staged_ok)]) + required_ok * (
            1.5 if dst.owner != -1 else 0.9)

    return Plan(actions, score, mode)


# ╔═══ region 7b: pragmatic action-space UCB ══════════════════════════════════╗


class _PragmaticNode:
    __slots__ = ("action_idx", "parent", "children", "visits", "value", "_untried")

    def __init__(self, action_idx: Optional[int], parent: Optional["_PragmaticNode"], n_choices: int):
        self.action_idx = action_idx
        self.parent = parent
        self.children: List["_PragmaticNode"] = []
        self.visits = 0
        self.value = 0.0
        self._untried = list(range(n_choices))

    def ucb1(self, c: float = 1.35) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value / self.visits
        return (self.value / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = 1.35) -> "_PragmaticNode":
        return max(self.children, key=lambda n: n.ucb1(c))


def pragmatic_candidate_actions(
    snap: Snapshot,
    regional_graph: Optional[RegionalGraph],
    top_k: int,
) -> List[Tuple[int, int, int]]:
    """Top-K single-edge moves ranked by capture_edge_score."""
    state = snap.state
    pool: List[Planet] = []
    seen: Set[int] = set()
    for p in sorted(state.neu_pl, key=lambda x: (-x.production, -x.ships)):
        if p.id not in seen:
            pool.append(p)
            seen.add(p.id)
    for p in state.en_pl:
        if p.id not in seen and (p.ships <= 28 or p.production <= 3):
            pool.append(p)
            seen.add(p.id)

    ranked_edges: List[Tuple[float, Tuple[int, int, int]]] = []
    for dst in pool[:36]:
        best_sc = -1e18
        best_pack: Optional[Tuple[int, int, int]] = None
        for src in state.my_pl:
            sc, need, _eta = capture_edge_score(snap, src, dst, regional_graph)
            if sc <= best_sc:
                continue
            avail = snap.avail(src.id)
            if avail < ABS_MIN_BATCH:
                continue
            send = min(avail, max(ABS_MIN_BATCH, min(int(need), avail)))
            if send < ABS_MIN_BATCH:
                continue
            best_sc = sc
            best_pack = (src.id, dst.id, send)
        if best_pack is not None and best_sc > EXPAND_RANK_SCORE_FLOOR:
            ranked_edges.append((best_sc, best_pack))

    ranked_edges.sort(key=lambda x: -x[0])
    out: List[Tuple[int, int, int]] = []
    picked: Set[Tuple[int, int]] = set()
    for _sc, triple in ranked_edges:
        sid, did, _ = triple
        if (sid, did) in picked:
            continue
        picked.add((sid, did))
        out.append(triple)
        if len(out) >= top_k:
            break
    return out


class PragmaticActionUCB:
    """UCB1 over top-K atomic moves; rollout via short score_plan_actions."""

    def __init__(
        self,
        state: GameState,
        snap: Snapshot,
        regional_graph: Optional[RegionalGraph],
        candidates: List[Tuple[int, int, int]],
        budget_ms: float,
        max_iters: int,
        rollout_steps: int,
        tempo_floor: int,
    ):
        self.state = state
        self.snap = snap
        self.regional_graph = regional_graph
        self.candidates = candidates
        self.budget_ms = max(0.0, budget_ms)
        self.max_iters = max(0, max_iters)
        self.rollout_steps = max(4, rollout_steps)
        self.tempo_floor = tempo_floor

    def evaluate(self) -> Tuple[Dict[int, float], Optional[int]]:
        n = len(self.candidates)
        if n == 0 or self.budget_ms <= 0 or self.max_iters <= 0:
            return {}, None

        root = _PragmaticNode(action_idx=None, parent=None, n_choices=n)
        deadline = time.time() * 1000.0 + self.budget_ms
        iters = 0
        while iters < self.max_iters and time.time() * 1000.0 < deadline:
            if root._untried:
                idx = root._untried.pop()
                child = _PragmaticNode(action_idx=idx, parent=root, n_choices=0)
                root.children.append(child)
                node = child
            elif root.children:
                node = root.best_child()
            else:
                break

            act = self.candidates[node.action_idx]
            val = score_plan_actions(
                self.state,
                [act],
                steps=self.rollout_steps,
                tempo_floor=self.tempo_floor,
            )
            cur: Optional[_PragmaticNode] = node
            while cur is not None:
                cur.visits += 1
                cur.value += val
                cur = cur.parent
            iters += 1

        out: Dict[int, float] = {}
        best_idx_out: Optional[int] = None
        best_visits = -1
        best_mean = -1e30
        for c in root.children:
            if c.visits <= 0 or c.action_idx is None:
                continue
            mean_v = c.value / c.visits
            out[c.action_idx] = mean_v
            if c.visits > best_visits or (
                c.visits == best_visits and mean_v > best_mean
            ):
                best_visits = c.visits
                best_mean = mean_v
                best_idx_out = c.action_idx

        return out, best_idx_out


class ExpandPlanner:
    """Wraps capture builder for neutral / weak-target modes."""

    @staticmethod
    def plan(snap: Snapshot, mode: str = "expand",
             diplo: Optional["DiplomacyEngine"] = None,
             regional_graph: Optional[RegionalGraph] = None) -> Plan:
        return _build_capture_plan(snap, mode, diplo=diplo, regional_graph=regional_graph)


class AttackPlanner:
    """Wraps capture builder for aggressive enemy-target modes."""

    @staticmethod
    def plan(snap: Snapshot, mode: str = "aggro",
             diplo: Optional["DiplomacyEngine"] = None,
             diplo_target: Optional[Planet] = None,
             regional_graph: Optional[RegionalGraph] = None) -> Plan:
        return _build_capture_plan(snap, mode, diplo=diplo, diplo_target=diplo_target,
                                   regional_graph=regional_graph)


# ── OpponentModel + DiplomacyEngine ──────────────────────────────────────────

class OpponentModel:
    """Cross-turn enemy stats. Shared instance across the agent's lifetime."""

    def __init__(self):
        self.ship_h: Dict[int, List[int]] = defaultdict(list)
        self.planet_h: Dict[int, List[int]] = defaultdict(list)
        self.atk_count: Dict[int, int] = defaultdict(int)

    def update(self, state: GameState):
        for eid in state.en_ids:
            self.ship_h[eid].append(state.total_ships(eid))
            self.planet_h[eid].append(len([p for p in state.planets if p.owner == eid]))
        for f in state.fleets:
            if f.owner in state.en_ids:
                t = state.fleet_target.get(f.id)
                if t:
                    tp = state.get(t[0])
                    if tp and tp.owner == state.my_id:
                        self.atk_count[f.owner] += 1

    def aggression(self, eid: int) -> float:
        n = self.atk_count.get(eid, 0)
        t = max(len(self.ship_h.get(eid, [1])), 1)
        return min(n / t * 6, 1.0)


_GLOBAL_OPP = OpponentModel()


class DiplomacyEngine:
    LEADER = "LEADER"
    MID = "MID"
    WEAK = "WEAK"

    def __init__(self, state: GameState, opp: OpponentModel):
        self.state = state; self.opp = opp

    def power(self, eid: int) -> float:
        s = self.state
        prod = sum(p.production for p in s.planets if p.owner == eid)
        return s.total_ships(eid) + prod * 24.0

    def threat_to_us(self, eid: int) -> float:
        s = self.state
        ep = [p for p in s.planets if p.owner == eid]
        if not ep:
            return 0.0
        prox = sum(1.0 / max(m.dist(e), 1.0) for m in s.my_pl for e in ep)
        aggr = 1.0 + self.opp.aggression(eid)
        return (prox * 38.0 + s.total_ships(eid) * 0.65) * aggr

    def rank(self) -> List[Tuple[str, int, float]]:
        if not self.state.en_ids:
            return []
        scored = [(self.threat_to_us(e), e) for e in self.state.en_ids]
        scored.sort(reverse=True)
        powers = {e: self.power(e) for _, e in scored}
        max_power = max(powers.values()) if powers else 1.0
        result = []
        for thr, eid in scored:
            if len(self.state.en_ids) == 1:
                tag = self.LEADER
            elif powers[eid] >= max_power * 0.75:
                tag = self.LEADER
            elif powers[eid] >= max_power * 0.40:
                tag = self.MID
            else:
                tag = self.WEAK
            result.append((tag, eid, thr))
        return result

    def primary_target(self) -> Optional[Planet]:
        ranked = self.rank()
        if not ranked:
            return None
        leader_ids = [eid for tag, eid, _ in ranked if tag == self.LEADER]
        target_eid = leader_ids[0] if leader_ids else ranked[0][1]
        ep = [p for p in self.state.planets if p.owner == target_eid]
        if not ep:
            return None
        cx, cy = self.state.centroid()
        return min(ep, key=lambda p: self.state.effective_garrison(p)
                                     + math.hypot(p.x - cx, p.y - cy) * 0.38)

    def leader_penalty(self, dst: Planet) -> float:
        ranked = self.rank()
        if len(ranked) <= 1:
            return 0.0
        leaders = {eid for tag, eid, _ in ranked if tag == self.LEADER}
        if dst.owner in leaders:
            return 15.0
        leader_power = sum(self.power(e) for e in leaders)
        our_power = self.state.total_ships(self.state.my_id)
        if leader_power > our_power * 1.2:
            return -20.0
        return 0.0


# ── Redistribution & late dump (fallback fillers) ────────────────────────────

class RedistributionPlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        if len(state.my_pl) < 2 or not state.en_pl:
            return Plan([], 0.0, "redistribute")

        def ned(p):
            return min((p.dist(e) for e in state.en_pl), default=999.0)

        ordered = sorted(state.my_pl, key=ned)
        fc = max(1, min(len(ordered) - 1, len(ordered) // 2 + 1))
        fronts = ordered[:fc]
        en_belt = any(is_sun_belt_planet(state, e) for e in state.en_pl)
        rears = []
        for p in ordered[fc:]:
            if state.net_threat(p) > 0:
                continue
            if is_sun_belt_planet(state, p) and (
                en_belt or state.phase() != "early"
            ):
                # Belt rears are usually reserves for rotation — but huge stacks
                # must still feed fronts when strategic commit is capped.
                stack = max(0, snap.surplus.get(p.id, 0))
                if stack < 52:
                    continue
            rears.append(p)
        actions: List[Tuple[int, int, int]] = []
        local_used: Dict[int, int] = defaultdict(int)
        for rear in rears[:3]:
            avail = max(0, snap.avail(rear.id) - local_used[rear.id])
            # v17: raise threshold - only redistribute when rear has a meaningful
            # surplus (>= 20 ships). Small trickles just clutter the board.
            if avail < max(20, rear.production * 5):
                continue
            dst = min(fronts, key=lambda f: rear.dist(f))
            # Send at most 40% of surplus so rear keeps a buffer.
            send = max(ABS_MIN_BATCH, min(avail, max(12, int(avail * 0.40))))
            if send >= ABS_MIN_BATCH:
                actions.append((rear.id, dst.id, send))
                local_used[rear.id] += send
        return Plan(actions, float(sum(a[2] for a in actions)) * 0.25, "redistribute")


class UrgentHighProdPlanner:
    """Pre-emptively capture high-production nearby enemy planets BEFORE the
    strategic ranking. Mirrors v10's `build_urgent_highprod_plan`. Activation
    is gated by `policy.urgent_attack_*` so early game stays expansion-only."""

    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        policy = snap.policy
        if not state.en_pl or policy.urgent_attack_ratio >= 999.0:
            return Plan([], 0.0, "urgent_hp", urgent=True)

        my_total = state.total_ships(state.my_id)
        en_total = sum(state.total_ships(e) for e in state.en_ids)
        if my_total < en_total * policy.urgent_attack_ratio:
            return Plan([], 0.0, "urgent_hp", urgent=True)

        cx, cy = snap.centroid
        targets = sorted(
            [p for p in state.en_pl
             if p.production >= policy.urgent_attack_min_prod
             and math.hypot(p.x - cx, p.y - cy) < 40],
            key=lambda p: -(p.production / max(1.0, math.hypot(p.x - cx, p.y - cy))),
        )
        if not targets:
            return Plan([], 0.0, "urgent_hp", urgent=True)

        actions: List[Tuple[int, int, int]] = []
        local_used: Dict[int, int] = defaultdict(int)
        for dst in targets[:3]:
            contributors: List[Tuple[int, Planet, int, int]] = []
            for src in state.my_pl:
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                if avail < ABS_MIN_BATCH:
                    continue
                need, eta = capture_need(state, src, dst)
                contributors.append((eta, src, avail, need))
            if not contributors:
                continue
            contributors.sort(key=lambda x: x[0])
            eta0 = contributors[0][0]
            group = [c for c in contributors
                     if c[0] <= eta0 + SYNC_ETA_WINDOW][:MAX_SOURCES_PER_TARGET]
            group_eta = max(c[0] for c in group)
            min_eta = eta0
            owner, garrison = target_state_at(state, dst, min_eta)
            if owner == state.my_id:
                continue
            required = (garrison + 8 + min(6, dst.production)
                        + dst.production * max(0, group_eta - min_eta) // 3)
            sent = 0
            staged: List[Tuple[int, int, int]] = []
            for _, src, avail, _ in group:
                if sent >= required:
                    break
                send = min(avail, required - sent)
                if send < ABS_MIN_BATCH and sent + send < required:
                    continue
                send = max(send, ABS_MIN_BATCH) if send >= ABS_MIN_BATCH else send
                staged.append((src.id, dst.id, send))
                sent += send
            if sent < required:
                continue
            for sid, did, send in staged:
                actions.append((sid, did, send))
                local_used[sid] += send
        return Plan(actions, float(sum(a[2] for a in actions)) * 1.0,
                    "urgent_hp", urgent=True)


class LateDumpPlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        if state.phase() != "late" or not state.en_pl:
            return Plan([], 0.0, "late_dump")
        weak = min(state.en_pl,
                   key=lambda p: state.effective_garrison(p) + p.production * 2)
        actions: List[Tuple[int, int, int]] = []
        local_used: Dict[int, int] = defaultdict(int)
        for src in sorted(state.my_pl, key=lambda p: -snap.avail(p.id)):
            avail = max(0, snap.avail(src.id) - local_used[src.id])
            need, _ = capture_need(state, src, weak)
            if avail >= need + ABS_MIN_BATCH:
                send = min(avail, need + 12)
                actions.append((src.id, weak.id, send))
                local_used[src.id] += send
        return Plan(actions, float(sum(a[2] for a in actions)) * 0.5, "late_dump")


# ╔═══ region 8: MCTSEngine - plan-level tree search ════════════════════════╗

class _MCTSNode:
    __slots__ = ("plan_idx", "parent", "children", "visits", "value", "_untried")

    def __init__(self, plan_idx, parent, n_choices):
        self.plan_idx = plan_idx
        self.parent: Optional[_MCTSNode] = parent
        self.children: List[_MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self._untried = list(range(n_choices))

    def ucb1(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value / self.visits
        return (self.value / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = 1.41) -> "_MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))

    def fully_expanded(self) -> bool:
        return len(self._untried) == 0


class MCTSEngine:
    """Plan-level MCTS. Given a list of candidate plans, runs UCB1 over them and
    returns a per-plan bonus reflecting expected outcome above the rest.

    The 'rollout' is the score_plan_actions value applied with `rollout_steps`
    sim turns. We don't expand deeper than 1 level (no branching after a plan
    is chosen) - this keeps cost predictable and the bonus interpretable."""

    def __init__(self, state: GameState, plans: List[Plan],
                 budget_ms: float, max_iters: int,
                 rollout_steps: int = 12):
        self.state = state
        self.plans = plans
        self.budget_ms = max(0.0, budget_ms)
        self.max_iters = max(0, max_iters)
        self.rollout_steps = rollout_steps

    def _rollout(self, plan_idx: int) -> float:
        plan = self.plans[plan_idx]
        return score_plan_actions(self.state, plan.actions, steps=self.rollout_steps)

    def evaluate(self) -> Dict[int, float]:
        """Return {plan_idx: mean_value}. Empty dict if budget == 0 or no plans."""
        n = len(self.plans)
        if n == 0 or self.budget_ms <= 0 or self.max_iters <= 0:
            return {}

        root = _MCTSNode(plan_idx=None, parent=None, n_choices=n)
        deadline = time.time() * 1000.0 + self.budget_ms
        iters = 0
        while iters < self.max_iters and time.time() * 1000.0 < deadline:
            # Selection / expansion (depth 1).
            if root._untried:
                idx = root._untried.pop()
                child = _MCTSNode(plan_idx=idx, parent=root, n_choices=0)
                root.children.append(child)
                node = child
            elif root.children:
                node = root.best_child()
            else:
                break

            val = self._rollout(node.plan_idx)
            cur: Optional[_MCTSNode] = node
            while cur is not None:
                cur.visits += 1
                cur.value += val
                cur = cur.parent
            iters += 1

        out: Dict[int, float] = {}
        for c in root.children:
            if c.visits > 0:
                out[c.plan_idx] = c.value / c.visits
        return out


# ╔═══ region 9: NeuralVal - multiplicative score modifier ══════════════════╗

# Inline pre-trained weights from v10. Same 14→64→32→1 architecture; lifted
# verbatim because retraining is out-of-scope for v11's framework rebuild.
_NEURAL_WEIGHTS_B64 = "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"  # left empty: NeuralVal will use random init (still safe as a small modifier).


class NeuralVal:
    """Score modifier in [-strength, +strength]. NEVER overrides plans; just
    nudges the arbiter's ranking by `(1 + strength * predict)`."""

    N_FEAT = 14

    def __init__(self):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.2, (64, self.N_FEAT)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.normal(0, 0.2, (32, 64)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = rng.normal(0, 0.2, (1, 32)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)
        self._try_load_inline()

    def _try_load_inline(self):
        if not _NEURAL_WEIGHTS_B64:
            return
        try:
            raw = base64.b64decode(_NEURAL_WEIGHTS_B64)
            d = np.load(io.BytesIO(raw), allow_pickle=True).item()
            self.W1 = d["W1"]; self.b1 = d["b1"]
            self.W2 = d["W2"]; self.b2 = d["b2"]
            self.W3 = d["W3"]; self.b3 = d["b3"]
        except Exception:
            pass

    def feat(self, state: GameState) -> np.ndarray:
        mi = state.my_id
        total = sum(state.total_ships(o) for o in [mi] + state.en_ids) + 1e-6
        my_sh = state.total_ships(mi) / total
        en_sh = sum(state.total_ships(e) for e in state.en_ids) / total
        my_prod = sum(p.production for p in state.my_pl)
        en_prod = sum(p.production for p in state.en_pl)
        prod_ratio = my_prod / max(my_prod + en_prod + 1, 1)
        n_planets = max(len(state.planets), 1)
        planet_ratio = len(state.my_pl) / n_planets
        cx, cy = state.centroid()
        min_en_dist = min((math.hypot(p.x - cx, p.y - cy)
                           for p in state.en_pl), default=100.0) / 100.0
        phase_enc = {"early": 0.0, "mid": 0.5, "late": 1.0}[state.phase()]
        tl = state.turns_left() / max(state.episode_steps, 1)
        fronts = [p for p in state.my_pl
                  if any(p.dist(e) < 35 for e in state.en_pl)]
        front_ratio = len(fronts) / max(len(state.my_pl), 1)
        comet_cnt = sum(1 for p in state.planets
                        if p.is_comet and p.owner != mi) / max(n_planets, 1)
        en_fleet = sum(f.ships for f in state.fleets
                       if f.owner not in (-1, mi))
        en_fleet_ratio = en_fleet / max(state.total_ships(mi) + 1, 1)
        net_thr = sum(max(0, state.net_threat(p)) for p in state.my_pl)
        net_thr_ratio = net_thr / max(state.total_ships(mi) + 1, 1)
        ee = float(np.tanh(elite_eval(state) / 500.0))
        border = sum((35 - m.dist(e)) / 35 * m.production
                     for m in state.my_pl for e in state.en_pl
                     if m.dist(e) < 35)
        border_norm = float(np.tanh(border / 100.0))
        return np.array([
            my_sh, en_sh, prod_ratio, planet_ratio,
            min_en_dist, phase_enc, tl, front_ratio,
            comet_cnt, en_fleet_ratio, net_thr_ratio,
            ee, border_norm,
            float(len(state.en_ids) > 1),
        ], dtype=np.float32)

    def predict(self, state: GameState) -> float:
        try:
            x = self.feat(state)
            h1 = np.maximum(0.0, self.W1 @ x + self.b1)
            h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
            out = float(np.tanh(self.W3 @ h2 + self.b3)[0])
            return out
        except Exception:
            return 0.0

    def score_modifier(self, state: GameState, strength: float) -> float:
        """Returns multiplicative factor in [1-strength, 1+strength]."""
        return 1.0 + strength * self.predict(state)


_GLOBAL_NEURAL = NeuralVal()


# ╔═══ region 10: PlanArbiter ═══════════════════════════════════════════════╗

class PlanArbiter:
    """Single source of decisions. Collects candidate plans, scores them with
    sim + MCTS bonus + Neural modifier, then commits to `moves` honouring the
    Snapshot's per-source surplus."""

    def __init__(self, snap: Snapshot, diplo: DiplomacyEngine,
                 neural: NeuralVal,
                 elapsed_ms_fn,
                 deadline_ms: float = 920.0,
                 regional_graph: Optional[RegionalGraph] = None,
                 multi_hop_planner: Optional[MultiHopPlanner] = None):
        self.snap = snap
        self.policy = snap.policy
        self.diplo = diplo
        self.neural = neural
        self.elapsed_ms = elapsed_ms_fn
        self.deadline_ms = deadline_ms
        self.moves: List[List] = []
        # v20: regional awareness
        self.regional_graph = regional_graph
        self.multi_hop_planner = multi_hop_planner

    # ── 1. urgent: defense + intercept (commit immediately) ──────────────────

    def commit_urgent(self) -> None:
        # Defense + intercept run with `urgent=True` (allowed to dip into reserve).
        for planner in (DefensePlanner, InterceptPlanner):
            if self._out_of_time(self.deadline_ms - 320):
                return
            self._commit_plan(planner.plan(self.snap), urgent=True)
        # Urgent high-production attacks: phase-gated, NOT urgent w.r.t. reserves
        # (must respect normal surplus to avoid leaving home defenseless).
        if self._out_of_time(self.deadline_ms - 280):
            return
        self._commit_plan(UrgentHighProdPlanner.plan(self.snap), urgent=False)

    # ── 2. strategic: collect, score, commit best ────────────────────────────

    def collect_strategic(self) -> List[Plan]:
        plans: List[Plan] = []
        diplo_tgt = self.diplo.primary_target()

        rg = self.regional_graph  # shorthand, may be None

        # Comet always considered if any non-allied comet exists.
        if any(p.is_comet and p.owner != self.snap.state.my_id
               for p in self.snap.state.planets):
            plans.append(ExpandPlanner.plan(self.snap, "comet", diplo=self.diplo,
                                            regional_graph=rg))

        for mode in self.policy.mode_order:
            if mode == "comet":
                continue  # already handled
            if self._out_of_time(self.deadline_ms - 200):
                break
            if mode in ("expand", "balanced"):
                plans.append(ExpandPlanner.plan(self.snap, mode, diplo=self.diplo,
                                                regional_graph=rg))
            elif mode == "diplo" and diplo_tgt is not None:
                plans.append(AttackPlanner.plan(self.snap, "diplo",
                                                diplo=self.diplo,
                                                diplo_target=diplo_tgt,
                                                regional_graph=rg))
            elif mode in ("aggro", "counter"):
                plans.append(AttackPlanner.plan(self.snap, mode, diplo=self.diplo,
                                                regional_graph=rg))

        # Optional diplo even if not in mode_order (always available).
        if diplo_tgt is not None and not any(p.tag == "diplo" for p in plans):
            plans.append(AttackPlanner.plan(self.snap, "diplo",
                                            diplo=self.diplo,
                                            diplo_target=diplo_tgt,
                                            regional_graph=rg))

        # Drop empty plans.
        return [p for p in plans if p.actions]

    def score_with_modifiers(self, plans: List[Plan]) -> List[Tuple[float, Plan]]:
        if not plans:
            return []

        # 1. Base sim scores (cheap).
        sim_steps = self.policy.sim_steps
        base_rows: List[List] = []
        st = self.snap.state
        for plan in plans:
            sim_b = score_plan_actions(st, plan.actions,
                                      steps=sim_steps,
                                      tempo_floor=self.policy.tempo_floor)
            base_rows.append([plan.score + sim_b, plan, sim_b])

        base_rows.sort(key=lambda r: -r[0])

        # 1b. Optional pessimistic 1-enemy-rollout refinement (timed budget).
        if (
            self.policy.paranoid_score_budget_ms >= 18.0
            and self.policy.paranoid_blend > 1e-6
        ):
            t0_par = self.elapsed_ms()
            lim = max(1, min(self.policy.paranoid_plan_top_k, len(base_rows)))
            pb = float(self.policy.paranoid_score_budget_ms)
            blend = float(self.policy.paranoid_blend)
            psteps = max(4, min(self.policy.paranoid_steps, sim_steps + 8))
            for i in range(lim):
                if self.elapsed_ms() - t0_par >= pb:
                    break
                s_tot, plan, sim_b = base_rows[i]
                _, pessim = score_plan_actions_paranoid(
                    st, plan.actions, sim_steps,
                    tempo_floor=self.policy.tempo_floor,
                    par_steps=psteps)
                adj = sim_b + blend * (pessim - sim_b)
                base_rows[i][0] = plan.score + adj
            base_rows.sort(key=lambda r: -r[0])

        base: List[Tuple[float, Plan]] = [(r[0], r[1]) for r in base_rows]

        # 2. MCTS bonus on top-3.
        top_k = min(3, len(base))
        mcts_bonus: Dict[int, float] = {}
        if top_k > 0 and self.policy.mcts_budget_ms > 0:
            elapsed_now = self.elapsed_ms()
            remaining = max(0.0, self.deadline_ms - elapsed_now - 60.0)
            budget = min(self.policy.mcts_budget_ms, remaining)
            if budget > 30.0:
                top_plans = [p for _, p in base[:top_k]]
                mcts = MCTSEngine(self.snap.state, top_plans,
                                  budget_ms=budget,
                                  max_iters=self.policy.mcts_max_iters,
                                  rollout_steps=max(sim_steps, 10))
                mcts_vals = mcts.evaluate()
                if mcts_vals:
                    # Normalise around 0: subtract mean, scale to ~base score range.
                    mu = sum(mcts_vals.values()) / len(mcts_vals)
                    for idx, v in mcts_vals.items():
                        mcts_bonus[idx] = (v - mu) * 0.5

        prag_bonus: Dict[int, float] = {}
        if self.policy.pragmatic_mcts_budget_ms > 0 and self.policy.pragmatic_mcts_max_iters > 0:
            elapsed_p = self.elapsed_ms()
            rem_p = max(0.0, self.deadline_ms - elapsed_p - 40.0)
            pbudget = min(self.policy.pragmatic_mcts_budget_ms, rem_p)
            if pbudget > 22.0:
                cands = pragmatic_candidate_actions(
                    self.snap,
                    self.regional_graph,
                    self.policy.pragmatic_mcts_top_k,
                )
                if cands:
                    prag = PragmaticActionUCB(
                        self.snap.state,
                        self.snap,
                        self.regional_graph,
                        cands,
                        pbudget,
                        self.policy.pragmatic_mcts_max_iters,
                        self.policy.pragmatic_mcts_rollout_steps,
                        self.policy.tempo_floor,
                    )
                    means, star_idx = prag.evaluate()
                    if means and star_idx is not None:
                        mu_p = sum(means.values()) / len(means)
                        sid_st, did_st, _ = cands[star_idx]
                        spread = (means[star_idx] - mu_p) * 0.42
                        for i, (_, plan) in enumerate(base):
                            if any(a[0] == sid_st and a[1] == did_st for a in plan.actions):
                                prag_bonus[i] = spread

        # 3. Neural modifier (state-only, applied multiplicatively).
        modifier = self.neural.score_modifier(self.snap.state,
                                              self.policy.neural_modifier_strength)

        scored: List[Tuple[float, Plan]] = []
        for i, (s, plan) in enumerate(base):
            bonus = mcts_bonus.get(i, 0.0) + prag_bonus.get(i, 0.0)
            final = s * modifier + bonus
            scored.append((final, plan))
        scored.sort(key=lambda x: -x[0])
        return scored

    def _trim_plan_to_ship_budget(self, plan: Plan, max_total: int) -> Plan:
        """Shrink planned sends to at most ``max_total`` ships (greedy prefix)."""
        if max_total < ABS_MIN_BATCH or not plan.actions:
            return Plan([], plan.score, plan.tag, urgent=plan.urgent)
        remain = int(max_total)
        out: List[Tuple[int, int, int]] = []
        for sid, did, ships in plan.actions:
            if remain < ABS_MIN_BATCH:
                break
            take = min(int(ships), remain)
            if take >= ABS_MIN_BATCH:
                out.append((sid, did, take))
                remain -= take
        return Plan(out, plan.score, plan.tag, urgent=plan.urgent)

    def commit_best(self, scored: List[Tuple[float, Plan]]) -> None:
        """Commit the single best strategic plan only, gated by position eval.

        v17 Position Gate: in mid/late phases, compute a baseline score for
        "do nothing this turn". Only commit if the best plan's sim score
        exceeds the baseline. Early phase always commits (must expand fast).
        """
        if not scored:
            return
        best_score, best_plan = scored[0]

        if (
            self.regional_graph is not None
            and self.snap.policy.phase != "early"
            and best_plan.actions
        ):
            threats = [
                self.regional_graph.region_threat(rid, self.snap.state.en_pl)
                for rid in self.regional_graph.regions
            ]
            max_t = max(threats) if threats else 0.0
            my_prod = float(sum(p.production for p in self.snap.state.my_pl))
            rp = float(self.snap.policy.region_pressure_ratio)
            ssm = float(self.snap.policy.safe_surplus_ship_mult)
            if max_t > my_prod * rp:
                budget = self.snap.calculate_safe_surplus_v20(self.regional_graph)
                limit = int(budget * ssm) + ABS_MIN_BATCH * 4
                ship_sum = sum(a[2] for a in best_plan.actions)
                if ship_sum > limit:
                    trimmed = self._trim_plan_to_ship_budget(best_plan, limit)
                    if not trimmed.actions:
                        return
                    best_plan = trimmed
                    best_score = trimmed.score

        # Early phase: always commit - expansion is critical.
        if self.snap.state.phase() != "early":
            baseline = score_plan_actions(self.snap.state, [],
                                          steps=self.policy.sim_steps,
                                          tempo_floor=self.policy.tempo_floor)
            margin = float(self.snap.policy.baseline_commit_margin)
            commit_bonus = 0.0
            st0 = self.snap.state
            if best_plan.tag in ("expand", "balanced") and best_plan.actions:
                if any(
                    st0.get(a[1]) is not None and st0.get(a[1]).owner == -1
                    for a in best_plan.actions
                ):
                    # Short-horizon sim often punishes sends vs contested grey; avoid idle 50+ stacks.
                    commit_bonus = 6.0
            if best_score + commit_bonus <= baseline + margin:
                return
        self._commit_plan(best_plan, urgent=False)

    # ── 3. fallback: redistribution + late dump ──────────────────────────────

    def commit_fallback(self) -> None:
        for planner in (RedistributionPlanner, LateDumpPlanner):
            if self._out_of_time(self.deadline_ms - 30):
                return
            self._commit_plan(planner.plan(self.snap), urgent=False)

    # ── helpers ──

    def _commit_plan(self, plan: Plan, urgent: bool) -> None:
        for sid, did, ships in plan.actions:
            if len(self.moves) >= MAX_TOTAL_MOVES:
                return
            self._emit(sid, did, ships, urgent=urgent)

    def _emit(self, sid: int, did: int, ships: int, urgent: bool) -> bool:
        snap = self.snap
        state = snap.state
        src = state.get(sid); dst = state.get(did)
        if src is None or dst is None or src.owner != state.my_id:
            return False

        if urgent:
            cap = max(0, src.ships - ABS_MIN_BATCH - snap.used.get(sid, 0))
        else:
            cap = snap.avail(sid)
        send_cap = min(int(ships), cap)
        if send_cap < ABS_MIN_BATCH:
            return False

        retry_sends: List[int] = [send_cap]
        for delta in (-8, 8, -16, 16, -4, 12):
            s2 = send_cap + delta
            if s2 >= ABS_MIN_BATCH and s2 <= cap and s2 not in retry_sends:
                retry_sends.append(s2)

        direct_angle = math.atan2(dst.y - src.y, dst.x - src.x)
        chord_clips_sun = (
            point_segment_distance(SUN_X, SUN_Y, src.x, src.y, dst.x, dst.y)
            < SUN_RADIUS + SUN_PATH_MARGIN
        )
        ang_tol = 1.55 if chord_clips_sun else 1.20
        for send in retry_sends:
            angle, eta = safe_aim(state, src, dst, send)
            spd = fleet_speed(send, state.max_speed)
            lx0, ly0 = launch_origin(src, angle)
            if not _ray_safe(lx0, ly0, angle, spd, min_flight=max(1, eta - 1)):
                continue
            angle_diff = abs(angle - direct_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > ang_tol:
                continue
            lx, ly = launch_origin(src, angle)
            if not launch_hits_target_first(state, lx, ly, angle, send, did,
                                            ignore_planet_id=None):
                continue
            if dst.owner == -1 and not urgent:
                queued = snap.pending_neutral_wave.get(did, 0)
                oth = my_inbound_ships_to(state, did) + queued
                if not neutral_wave_wins(state, dst, eta, send, oth):
                    continue
            self.moves.append([sid, float(angle), int(send)])
            snap.subtract(sid, send)
            if dst.owner == -1:
                snap.pending_neutral_wave[did] = snap.pending_neutral_wave.get(did, 0) + send
            return True
        return False


    def _out_of_time(self, threshold_ms: float) -> bool:
        return self.elapsed_ms() > threshold_ms


# ╔═══ region 11: agent() entry ═════════════════════════════════════════════╗

def agent(obs, config=None):
    """Kaggle-required entry. Returns list of [src_id, angle, ships] moves."""
    global _GLOBAL_OPP, _GLOBAL_NEURAL
    t0 = time.time()
    elapsed = lambda: (time.time() - t0) * 1000.0

    try:
        state = GameState(obs, config)
        if not state.my_pl:
            return []

        _GLOBAL_OPP.update(state)
        policy = PhasePolicy.for_state(state)
        snap = Snapshot.build(state, policy)
        diplo = DiplomacyEngine(state, _GLOBAL_OPP)

        # v20: Initialize regional graph and multi-hop planner
        regional_graph = None
        multi_hop_planner = None
        try:
            spawn_positions = config.get("spawn_positions", []) if config else []
            regional_graph = RegionalGraph(state.planets, spawn_positions)
            timeline = ProductionTimeline(state.planets, set(p.id for p in state.my_pl))
            multi_hop_planner = MultiHopPlanner(regional_graph, timeline)
        except Exception:
            regional_graph = None
            multi_hop_planner = None

        arbiter = PlanArbiter(snap, diplo, _GLOBAL_NEURAL,
                              elapsed_ms_fn=elapsed,
                              deadline_ms=920.0,
                              regional_graph=regional_graph,
                              multi_hop_planner=multi_hop_planner)

        # Pipeline.
        arbiter.commit_urgent()
        plans = arbiter.collect_strategic()
        scored = arbiter.score_with_modifiers(plans)
        arbiter.commit_best(scored)
        arbiter.commit_fallback()

        return arbiter.moves
    except Exception:
        return []
