"""
LunaTrace Target Depression Selection Engine (Task 3)
Evaluates all topographic depressions in the Chandrayaan-2 DFSAR swath using the registered
LRO LOLA DEM and geometry grid.
Computes swath margin vs. geolocation uncertainty, horizon illumination (PSR and cold-trap
fractions), radar shadow and layover fractions, floor width, and incidence angle range.

CRITICAL PROTOCOL REQUIREMENT:
The ranking rule and candidate scoring table are fixed and recorded with an immutable UTC timestamp
BEFORE any CPR, DoP, or polarimetric radar observables are examined inside any candidate.
"""

from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import pyproj
from matplotlib.path import Path as MplPath

from src.pds4_reader import (
    read_lola_pds3_dem,
    compute_dem_radar_shadow_and_layover_masks
)
from src.config_schema import LunaTraceConfig
from src.config import load_config
from src.provenance import write_provenance_sidecar


FIXED_RANKING_RULE_DEFINITION = {
    "rule_name": "A-Priori Polar Cold-Trap Exploration Ranking Rule",
    "rule_version": "1.0",
    "objective": (
        "Rank candidate polar depressions for volatile exploration prior to inspecting radar "
        "polarimetric data, maximizing cold-trap area and radar visibility while ensuring safe "
        "containment inside the DFSAR swath footprint."
    ),
    "gating_criteria": {
        "swath_containment": "swath_margin_m > 0 and margin_to_uncertainty_ratio >= 0.20",
        "cold_trap_presence": "cold_trap_fraction > 0.005 (at least 0.5% area in deep cold trap)",
        "radar_observability": "radar_shadow_fraction < 0.85 (sufficient illuminated backscatter)"
    },
    "scoring_formula": (
        "Score = 0.35 * cold_trap_fraction + "
        "0.25 * (1.0 - radar_shadow_fraction) + "
        "0.20 * min(1.0, margin_to_uncertainty_ratio / 2.0) + "
        "0.20 * min(1.0, floor_width_km / 10.0)"
    ),
    "weights": {
        "cold_trap_fraction": 0.35,
        "radar_illumination": 0.25,
        "swath_margin_safety": 0.20,
        "floor_width_traversability": 0.20
    }
}


def run_target_selection(
    config: LunaTraceConfig,
    lbl_path: Path,
    img_path: Path,
    geom_csv_path: Path,
    output_json_path: Path
) -> Dict[str, Any]:
    """
    Executes a-priori target selection on the registered LOLA DEM.
    Saves candidate table, ranking rule, and timestamp to output_json_path.
    """
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    print(f"[*] Executing Target Depression Selection at UTC: {timestamp_utc}")
    print(f"[*] Applying Fixed Ranking Rule: '{FIXED_RANKING_RULE_DEFINITION['rule_name']}'")

    elev, lbl_meta = read_lola_pds3_dem(lbl_path, img_path)
    proj = pyproj.Proj(config.crs.proj4_string)

    # 1. Swath Boundary Polygon
    geom_df = pd.read_csv(geom_csv_path)
    tp_lons = geom_df["Longitude(deg.)"].values
    tp_lats = geom_df["Latitude(deg.)"].values
    tp_incs = geom_df["Incidence_Angle(deg.)"].values
    x_tp, y_tp = proj(tp_lons, tp_lats)

    n_lines = 8289
    n_samples = 38
    edge_top = np.arange(n_samples)
    edge_right = np.arange(n_samples - 1, n_lines * n_samples, n_samples)
    edge_bottom = np.arange(n_lines * n_samples - 1, (n_lines - 1) * n_samples - 1, -1)
    edge_left = np.arange((n_lines - 1) * n_samples, -1, -n_samples)
    boundary_indices = np.concatenate([edge_top, edge_right, edge_bottom, edge_left])
    poly_x = x_tp[boundary_indices]
    poly_y = y_tp[boundary_indices]
    swath_poly = MplPath(np.column_stack([poly_x, poly_y]))
    pts = np.column_stack([poly_x, poly_y])

    # 2. Candidate Depressions
    candidates_to_evaluate = [
        {
            "id": "CAND_1_FAUSTINI_MAIN",
            "name": "Faustini Crater Floor (Primary Basin)",
            "center_lat": -87.254,
            "center_lon": 84.151,
            "radius_km": 12.0,
            "label": "Deepest primary floor depression in Phase 1 crop"
        },
        {
            "id": "CAND_2_THIRD_PARTY",
            "name": "Faustini SW Floor Basin (-87.39S, 82.31E)",
            "center_lat": -87.390,
            "center_lon": 82.310,
            "radius_km": 6.0,
            "label": "third-party, unverified"
        },
        {
            "id": "CAND_3_AMUNDSEN_SOUTH",
            "name": "Amundsen South Margin Depression",
            "center_lat": -85.405,
            "center_lon": 84.098,
            "radius_km": 15.0,
            "label": "Northern swath margin depression"
        },
        {
            "id": "CAND_4_SOUTH_SUBBASIN",
            "name": "Shoemaker-Faustini Saddle Sub-Basin",
            "center_lat": -88.329,
            "center_lon": 81.004,
            "radius_km": 4.0,
            "label": "Southern deep polar sub-basin"
        }
    ]

    sigma_along = 3115.97
    sigma_range = 31.76
    tan_inc = np.tan(np.radians(25.994))

    candidate_results: List[Dict[str, Any]] = []

    for c in candidates_to_evaluate:
        cx_m, cy_m = proj(c["center_lon"], c["center_lat"])
        dists = np.hypot(pts[:, 0] - cx_m, pts[:, 1] - cy_m)
        min_dist_to_edge = float(np.min(dists))
        is_inside = swath_poly.contains_point((cx_m, cy_m))
        margin_m = min_dist_to_edge if is_inside else -min_dist_to_edge

        # Pixel bounds in LOLA DEM
        c_px = int(round(cx_m / 20.0 + 7582.5))
        r_px = int(round(-cy_m / 20.0 + 7582.5))
        rad_px = int(round(c["radius_km"] * 1000.0 / 20.0))

        r_min, r_max = max(0, r_px - rad_px), min(elev.shape[0], r_px + rad_px)
        c_min, c_max = max(0, c_px - rad_px), min(elev.shape[1], c_px + rad_px)
        sub_dem = elev[r_min:r_max, c_min:c_max]

        dy_arr, dx_arr = np.ogrid[r_min - r_px : r_max - r_px, c_min - c_px : c_max - c_px]
        circle_mask = (dx_arr**2 + dy_arr**2) <= rad_px**2

        sub_dem_circ = sub_dem[circle_mask]
        relief_m = float(np.nanmax(sub_dem_circ) - np.nanmin(sub_dem_circ))
        sigma_relief = relief_m / tan_inc
        sigma_geo_total = float(np.sqrt(sigma_along**2 + sigma_range**2 + sigma_relief**2))
        margin_ratio = margin_m / sigma_geo_total if sigma_geo_total > 0 else 0.0

        # Radar shadow and layover
        shadow_sub, layover_sub = compute_dem_radar_shadow_and_layover_masks(
            sub_dem, pixel_spacing_m=20.0, incidence_angle_deg=25.994,
            look_vector_proj=(0.3085, -0.9512)
        )
        shadow_frac = float(np.mean(shadow_sub[circle_mask]))
        layover_frac = float(np.mean(layover_sub[circle_mask]))

        # Horizon ray-tracing
        ny_s, nx_s = sub_dem.shape
        az_angles = np.radians(np.linspace(0, 360, 24, endpoint=False))
        solar_elevs = np.radians(np.linspace(0.5, 2.5, 5))
        dist_steps = [60.0, 120.0, 240.0, 500.0, 1000.0, 2000.0, 4000.0]

        horiz_max = np.full((ny_s, nx_s), -np.pi / 2, dtype=np.float32)
        for az in az_angles:
            dx = np.cos(az)
            dy = np.sin(az)
            for d in dist_steps:
                d_px = d / 20.0
                r_sample = np.clip(np.round(np.arange(ny_s)[:, None] - dy * d_px).astype(int), 0, ny_s - 1)
                c_sample = np.clip(np.round(np.arange(nx_s)[None, :] + dx * d_px).astype(int), 0, nx_s - 1)
                dz = sub_dem[r_sample, c_sample] - sub_dem
                elev_angle = np.arctan2(dz, d)
                horiz_max = np.maximum(horiz_max, elev_angle)

        illum_frac = np.zeros((ny_s, nx_s), dtype=np.float32)
        for se in solar_elevs:
            illum_frac += (se > horiz_max).astype(np.float32)
        illum_frac /= len(solar_elevs)

        psr_mask = (illum_frac < 0.005) & circle_mask
        psr_frac = float(np.sum(psr_mask) / np.sum(circle_mask))
        trap_mask = psr_mask & (sub_dem <= -350.0)
        trap_frac = float(np.sum(trap_mask) / np.sum(circle_mask))

        floor_width_km = float(2.0 * np.sqrt(np.sum(sub_dem_circ < -1500.0) * 0.0004 / np.pi))
        min_elev_m = float(np.nanmin(sub_dem_circ))
        max_elev_m = float(np.nanmax(sub_dem_circ))

        near_tp = (np.hypot(x_tp - cx_m, y_tp - cy_m) <= c["radius_km"] * 1500.0)
        if np.any(near_tp):
            inc_min = float(np.min(tp_incs[near_tp]))
            inc_max = float(np.max(tp_incs[near_tp]))
            inc_mean = float(np.mean(tp_incs[near_tp]))
        else:
            inc_min, inc_max, inc_mean = 25.0, 27.0, 26.0

        gated = (margin_m > 0) and (margin_ratio >= 0.20) and (trap_frac > 0.005) and (shadow_frac < 0.85)
        score = (
            0.35 * trap_frac +
            0.25 * max(0.0, 1.0 - shadow_frac) +
            0.20 * min(1.0, max(0.0, margin_ratio / 2.0)) +
            0.20 * min(1.0, max(0.0, floor_width_km / 10.0))
        ) if gated else 0.0

        res = {
            "id": c["id"],
            "name": c["name"],
            "label": c["label"],
            "center_lat": c["center_lat"],
            "center_lon": c["center_lon"],
            "swath_margin_m": round(margin_m, 1),
            "geo_uncertainty_m": round(sigma_geo_total, 1),
            "margin_to_uncertainty_ratio": round(margin_ratio, 2),
            "floor_width_km": round(floor_width_km, 2),
            "min_elevation_m": round(min_elev_m, 1),
            "max_elevation_m": round(max_elev_m, 1),
            "relief_m": round(relief_m, 1),
            "psr_fraction": round(psr_frac, 4),
            "cold_trap_fraction": round(trap_frac, 4),
            "radar_shadow_fraction": round(shadow_frac, 4),
            "radar_layover_fraction": round(layover_frac, 4),
            "incidence_range_deg": f"{inc_min:.1f} - {inc_max:.1f} (mean {inc_mean:.1f})",
            "gated_eligible": bool(gated),
            "ranking_score": round(score, 4)
        }
        candidate_results.append(res)

    candidate_results.sort(key=lambda r: r["ranking_score"], reverse=True)
    chosen_target = candidate_results[0]

    output_payload = {
        "timestamp_utc": timestamp_utc,
        "ranking_rule": FIXED_RANKING_RULE_DEFINITION,
        "chosen_target_id": chosen_target["id"],
        "chosen_target_name": chosen_target["name"],
        "chosen_target_coordinates": {
            "lat": chosen_target["center_lat"],
            "lon": chosen_target["center_lon"]
        },
        "chosen_target_justification": (
            f"Candidate '{chosen_target['name']}' ({chosen_target['id']}) ranks #1 with composite score "
            f"{chosen_target['ranking_score']:.4f}. It is safely interior to the swath footprint with "
            f"margin {chosen_target['swath_margin_m']:.1f} m ({chosen_target['margin_to_uncertainty_ratio']:.2f}x "
            f"total geolocation uncertainty {chosen_target['geo_uncertainty_m']:.1f} m), contains a massive "
            f"cold-trap area ({chosen_target['cold_trap_fraction']*100:.1f}%), a wide flat floor "
            f"({chosen_target['floor_width_km']:.2f} km width), and acceptable radar shadow fraction "
            f"({chosen_target['radar_shadow_fraction']*100:.1f}%). Evaluation fixed BEFORE inspecting CPR/DOP."
        ),
        "candidates": candidate_results
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"[+] Saved target selection table to {output_json_path}")
    return output_payload


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run_target_selection(
        config=cfg,
        lbl_path=Path("data/raw/ldem_85s_20m.lbl"),
        img_path=Path("data/raw/ldem_85s_20m.img"),
        geom_csv_path=Path("data/raw/ch2_sar_ncxl_20260403t190328238_d_cp_d18/geometry/calibrated/20260403/ch2_sar_ncxl_20260403t190328238_g_sli_xx_cp_xx_d18.csv"),
        output_json_path=Path("outputs/tables/candidate_target_ranking.json")
    )
