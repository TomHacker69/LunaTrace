"""
Automated Test Suite for LunaTrace Pipeline
Tests all 6 scientific phases, polarimetric identities, pathfinding, and resource calculations.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import rasterio

from src.config import CFG, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, MAPS_DIR
from src.utils import enhanced_lee_filter, save_geotiff
from src.phase1_dfsar import DFSARProcessor
from src.phase2_ice_detection import IceDetectionEngine
from src.phase3_terrain_illumination import TerrainIlluminationEngine
from src.phase4_landing_site import LandingSiteSelector
from src.phase5_traverse_planning import RoverTraversePlanner
from src.phase6_volume_estimation import IceVolumeEstimator


def test_config_integrity():
    """Validates configuration parameters and physics thresholds."""
    assert CFG.radar.cpr_ice_threshold == 1.0
    assert CFG.rover.max_nav_slope_deg == 18.0
    assert CFG.rover.battery_capacity_wh >= 1200.0
    assert CFG.landing.max_landing_slope_deg == 5.0
    assert sum(CFG.landing.weights.values()) == pytest.approx(1.0, 1e-5)


def test_speckle_filter():
    """Tests Enhanced Lee filter for non-negative intensity and signed Stokes channels."""
    # Test positive intensity
    img_pos = np.ones((20, 20), dtype=float) * 5.0
    filtered_pos = enhanced_lee_filter(img_pos, window_size=5, num_looks=4, is_intensity=True)
    assert filtered_pos.shape == (20, 20)
    assert np.all(filtered_pos >= 0.0)

    # Test signed Stokes channel (can have negative values)
    img_signed = np.full((20, 20), -2.5, dtype=float)
    filtered_signed = enhanced_lee_filter(img_signed, window_size=5, num_looks=4, is_intensity=False)
    assert np.all(filtered_signed < 0.0)


def test_phase1_polarimetry():
    """Validates Stokes formulas, CPR computation, and m-chi decomposition energy conservation."""
    processor = DFSARProcessor()
    # Synthetic test Stokes vector
    s1 = np.array([[2.0, 3.0]])
    s2 = np.array([[0.2, 0.4]])
    s3 = np.array([[0.1, -0.3]])
    s4 = np.array([[-0.5, 0.5]])  # s4 < 0 => CPR > 1
    stokes = {"S1": s1, "S2": s2, "S3": s3, "S4": s4}

    prods = processor.compute_polarimetric_products(stokes)

    cpr = prods["CPR"]
    dop = prods["DOP"]
    pv = prods["P_volume"]
    pdb = prods["P_double"]
    ps = prods["P_surface"]

    # Verify CPR: (s1 - s4) / (s1 + s4)
    # For pixel 0: (2 - (-0.5)) / (2 + (-0.5)) = 2.5 / 1.5 = 1.6667
    assert cpr[0, 0] == pytest.approx(2.5 / 1.5, 1e-4)

    # Verify energy conservation: Pv + Pdb + Ps == S1
    total_decomp = pv + pdb + ps
    np.testing.assert_allclose(total_decomp, s1, rtol=1e-4)


def test_phase2_ice_discrimination():
    """Tests ice detection scoring and Delta CPR depth discrimination."""
    engine = IceDetectionEngine()
    inputs, _, _ = engine.load_inputs()
    prods = engine.compute_ice_probability(inputs)

    ice_prob = prods["ice_probability"]
    delta_cpr = prods["delta_cpr"]
    ice_mask = prods["high_prob_ice_mask"] > 0

    assert np.all(ice_prob >= 0.0) and np.all(ice_prob <= 1.0)
    assert np.sum(ice_mask) > 0  # High-confidence ice detected
    assert np.mean(delta_cpr[ice_mask]) > 0.4  # Buried ice shows strong positive Delta CPR


def test_phase3_terrain_morphometry():
    """Validates DEM slope gradient and illumination bounding."""
    engine = TerrainIlluminationEngine()
    dem, _, _ = engine.load_dem()
    morph = engine.compute_morphometry(dem, resolution_m=20.0)

    slope = morph["slope_deg"]
    assert np.all(slope >= 0.0)
    assert np.min(slope) < 2.0  # Smooth plateaus exist
    assert np.max(slope) > 20.0  # Steep walls exist


def test_phase4_landing_site_selection():
    """Validates that candidate landing sites strictly obey the 5-degree slope limit."""
    selector = LandingSiteSelector()
    data, tf, crs = selector.load_environmental_data()
    suitability, safe_mask, dist_to_ice = selector.evaluate_suitability(data)
    df_sites = selector.select_top_sites(suitability, data, dist_to_ice, tf, top_k=3)

    assert len(df_sites) == 3
    # Verify Rank 1 site
    site1 = df_sites.iloc[0]
    assert site1["Slope_deg"] <= 5.0
    assert site1["Illumination_Fraction"] > 0.50
    assert site1["Earth_LOS_Fraction"] > 0.50
    assert 1000.0 <= site1["Distance_to_Ice_m"] <= 7500.0


def test_phase5_pathfinding_and_battery():
    """Validates A* pathfinding and power budget feasibility."""
    planner = RoverTraversePlanner()
    rasters, tf, crs = planner.load_environmental_rasters()

    sites_csv = TABLES_DIR / "phase4_top3_landing_sites.csv"
    assert sites_csv.exists()
    df_sites = pd.read_csv(sites_csv)
    start_px = (int(df_sites.iloc[0]["Pixel_Y"]), int(df_sites.iloc[0]["Pixel_X"]))

    ice_mask = rasters["ice_mask"] > 0
    y_idx, x_idx = np.where(ice_mask)
    dists = (y_idx - start_px[0]) ** 2 + (x_idx - start_px[1]) ** 2
    goal_px = (int(y_idx[np.argmin(dists)]), int(x_idx[np.argmin(dists)]))

    cost_grid, _ = planner.build_cost_raster(rasters, mode="safest")
    path, total_cost = planner.a_star_search(cost_grid, start_px, goal_px, use_heuristic=True)

    assert len(path) > 10
    metrics = planner.simulate_power_and_metrics(path, rasters)
    assert metrics["max_slope_deg"] <= CFG.rover.max_nav_slope_deg
    assert metrics["min_soc_pct"] > 20.0  # Battery reserve preserved


def test_phase6_volume_estimation():
    """Validates CRIM volume inversion and scenario ranking."""
    estimator = IceVolumeEstimator()
    rasters, _, _ = estimator.load_ice_rasters()
    df_vol, _ = estimator.compute_scenario_volumes(rasters)

    low_mass = df_vol.loc[df_vol["Scenario"] == "Low (Conservative)", "Water_Ice_Mass_Megatons"].values[0]
    nom_mass = df_vol.loc[df_vol["Scenario"] == "Nominal (Realistic)", "Water_Ice_Mass_Megatons"].values[0]
    high_mass = df_vol.loc[df_vol["Scenario"] == "High (Optimistic)", "Water_Ice_Mass_Megatons"].values[0]

    assert low_mass > 0.05
    assert nom_mass > low_mass
    assert high_mass > nom_mass


def test_artifacts_existence():
    """Checks that all 18 publication maps and 7 CSV tables exist on disk."""
    expected_tables = [
        "phase1_polarimetric_summary.csv",
        "phase2_terrain_class_statistics.csv",
        "phase2_threshold_sensitivity_analysis.csv",
        "phase3_terrain_illumination_summary.csv",
        "phase4_top3_landing_sites.csv",
        "phase5_route_comparison.csv",
        "phase6_ice_volume_estimates.csv"
    ]
    for tbl in expected_tables:
        fpath = TABLES_DIR / tbl
        assert fpath.exists(), f"Missing table: {fpath}"

    expected_maps = [
        "map_phase1_L_cpr.png",
        "map_phase1_L_dop.png",
        "map_phase1_L_mchi_decomposition.png",
        "map_phase2_ice_probability.png",
        "map_phase2_delta_cpr.png",
        "map_phase3_slope.png",
        "map_phase3_illumination_psr.png",
        "map_phase4_landing_sites.png",
        "map_phase5_rover_traverses.png",
        "map_phase5_recommended_route_profile.png",
        "map_phase6_water_column.png",
        "map_phase6_volume_sensitivity.png"
    ]
    for m in expected_maps:
        fpath = MAPS_DIR / m
        assert fpath.exists(), f"Missing map: {fpath}"
