"""Patch submission_rl_v11_f37.py -> submission_rl_v13c.py for v13c dims."""
import re

path = 'submission_rl_v13c.py'
with open(path) as f:
    src = f.read()

# 1) Add FLEET_SPEED constant after LEAD_TIMES line
src = src.replace(
    'LEAD_TIMES = (15.0, 30.0)',
    'LEAD_TIMES = (15.0, 30.0)\nFLEET_SPEED = 6.0\nETA_W1 = 5.0\nETA_W2 = 15.0',
)

# 2) Add fleet arrival prediction code BEFORE the planet_feats = np.stack(...)
old_stack = """    planet_feats = np.stack(
        [
            is_mine.astype(np.float32), is_enemy.astype(np.float32), is_neutral.astype(np.float32),
            x_norm, y_norm, radius_norm, log_ships, prod_norm, dist_sun,
            in_friend_norm, in_foe_norm, is_padding,
            is_orbiting_f, orbit_phase_norm, orbit_radius_norm,
            lead_x_15_norm, lead_y_15_norm, lead_x_30_norm, lead_y_30_norm,
            threat_ratio, net_inbound, eta_foe_min,
            flip_cost_norm, friendly_surplus, capturable_bin3,
            needed_pct_norm, capturable_bin5, weak_target_score,
            garrison_rank, safe_surplus_norm, is_strong_source,
            prod_per_need, v20_target_score,
        ],
        axis=-1,
    ).astype(np.float32)"""

new_code = """    # === B2: Fleet Arrival Prediction (dims 33-38) ===
    f_dx_p = planet_x[None, :] - fleet_x[:, None]  # [F, P]
    f_dy_p = planet_y[None, :] - fleet_y[:, None]
    f_dist_p = np.sqrt(f_dx_p * f_dx_p + f_dy_p * f_dy_p + 1e-6)
    f_eta_p = f_dist_p / FLEET_SPEED
    f_dxn_p = np.cos(fleet_angle)
    f_dyn_p = np.sin(fleet_angle)
    f_proj_p = f_dx_p * f_dxn_p[:, None] + f_dy_p * f_dyn_p[:, None]
    headed_p = (f_proj_p > 0.0) & fleet_mask[:, None]
    in_w1_p = headed_p & (f_eta_p <= ETA_W1)
    in_w2_p = headed_p & (f_eta_p <= ETA_W2)
    f_ships_f = np.maximum(fleet_ships, 0).astype(np.float32)
    f_is_mine_b = (fleet_owner == player) & fleet_mask
    f_is_opp_b = (fleet_owner >= 0) & (fleet_owner != player) & fleet_mask
    friendly_w1 = (f_ships_f[:, None] * in_w1_p * f_is_mine_b[:, None]).sum(axis=0)
    friendly_eta_w1 = np.log1p(friendly_w1) / 8.0
    friendly_w2 = (f_ships_f[:, None] * in_w2_p * f_is_mine_b[:, None]).sum(axis=0)
    friendly_eta_w2 = np.log1p(friendly_w2) / 8.0
    enemy_w1 = (f_ships_f[:, None] * in_w1_p * f_is_opp_b[:, None]).sum(axis=0)
    enemy_eta_w1 = np.log1p(enemy_w1) / 8.0
    enemy_w2 = (f_ships_f[:, None] * in_w2_p * f_is_opp_b[:, None]).sum(axis=0)
    enemy_eta_w2 = np.log1p(enemy_w2) / 8.0
    my_ships_at_t5 = np.where(is_mine, planet_ships.astype(np.float32), 0.0) + friendly_w1 - enemy_w1
    my_total_garr_f = max(float(my_total_garrison), 1.0)
    net_garrison_t5 = np.clip(my_ships_at_t5 / my_total_garr_f, -1.0, 1.0) * is_mine.astype(np.float32)
    my_ships_at_t15 = np.where(is_mine, planet_ships.astype(np.float32), 0.0) + friendly_w2 - enemy_w2
    net_garrison_t15 = np.clip(my_ships_at_t15 / my_total_garr_f, -1.0, 1.0) * is_mine.astype(np.float32)

    planet_feats = np.stack(
        [
            is_mine.astype(np.float32), is_enemy.astype(np.float32), is_neutral.astype(np.float32),
            x_norm, y_norm, radius_norm, log_ships, prod_norm, dist_sun,
            in_friend_norm, in_foe_norm, is_padding,
            is_orbiting_f, orbit_phase_norm, orbit_radius_norm,
            lead_x_15_norm, lead_y_15_norm, lead_x_30_norm, lead_y_30_norm,
            threat_ratio, net_inbound, eta_foe_min,
            flip_cost_norm, friendly_surplus, capturable_bin3,
            needed_pct_norm, capturable_bin5, weak_target_score,
            garrison_rank, safe_surplus_norm, is_strong_source,
            prod_per_need, v20_target_score,
            friendly_eta_w1, friendly_eta_w2, enemy_eta_w1, enemy_eta_w2,
            net_garrison_t5, net_garrison_t15,
        ],
        axis=-1,
    ).astype(np.float32)"""

assert old_stack in src, f'Could not find planet_feats stack!'
src = src.replace(old_stack, new_code)

# 3) Add 6 new global features (dims 18-23)
old_global = """    global_feats = np.array(
        [
            step_norm,
            _player_ships(player) / total_ships, _player_ships(opp) / total_ships,
            _player_planets(player) / total_planets, _player_planets(opp) / total_planets,
            _player_prod(player) / total_prod, _player_prod(opp) / total_prod,
            is_early, is_mid, is_late,
            av_norm,
            np.log1p(my_garr_total) / 10.0,
            n_fleets_mine / MAX_FLEETS,
            n_fleets_enemy / MAX_FLEETS,
            max_garr_norm,
            n_weak_targets_norm,
            ships_to_capture_all_weak_norm,
            min_effective_fleet_norm,
        ],
        dtype=np.float32,
    )"""

new_global = """    # === B1: Temporal proxy globals (dims 18-23) ===
    my_prod_total = float(_player_prod(player))
    my_fleet_total = float(fleet_ships[f_is_mine_b].sum()) if f_is_mine_b.any() else 0.0
    my_total_all = max(my_garr_total + my_fleet_total, 1.0)
    foe_garr_total = float(planet_ships[is_enemy].sum()) if is_enemy.any() else 0.0
    foe_prod_total = float(_player_prod(opp))
    foe_fleet_total = float(fleet_ships[(fleet_owner >= 0) & (fleet_owner != player) & fleet_mask].sum())
    garr_to_prod = min(my_garr_total / max(my_prod_total * 10.0, 1.0), 1.0)
    fleet_mass_ratio_g = my_fleet_total / my_total_all
    prod_advantage_g = float(np.clip((my_prod_total - foe_prod_total) / max(my_prod_total + foe_prod_total, 1.0), -1.0, 1.0))
    garr_advantage_g = float(np.clip((my_garr_total - foe_garr_total) / max(my_garr_total + foe_garr_total, 1.0), -1.0, 1.0))
    growth_potential_g = min(my_prod_total / max(my_garr_total, 1.0), 1.0)
    threat_pressure_g = min(foe_fleet_total / max(my_garr_total, 1.0), 2.0) / 2.0

    global_feats = np.array(
        [
            step_norm,
            _player_ships(player) / total_ships, _player_ships(opp) / total_ships,
            _player_planets(player) / total_planets, _player_planets(opp) / total_planets,
            _player_prod(player) / total_prod, _player_prod(opp) / total_prod,
            is_early, is_mid, is_late,
            av_norm,
            np.log1p(my_garr_total) / 10.0,
            n_fleets_mine / MAX_FLEETS,
            n_fleets_enemy / MAX_FLEETS,
            max_garr_norm,
            n_weak_targets_norm,
            ships_to_capture_all_weak_norm,
            min_effective_fleet_norm,
            garr_to_prod,
            fleet_mass_ratio_g,
            prod_advantage_g,
            garr_advantage_g,
            growth_potential_g,
            threat_pressure_g,
        ],
        dtype=np.float32,
    )"""

assert old_global in src, f'Could not find global_feats array!'
src = src.replace(old_global, new_global)

with open(path, 'w') as f:
    f.write(src)
print('Patch applied successfully.')
print(f'File size: {len(src)} chars')
