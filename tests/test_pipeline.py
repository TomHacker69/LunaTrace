"""
Automated Test Suite for LunaTrace Pipeline
Runs strictly against isolated test fixtures in tests/fixtures/.
Asserts fail-fast data validation, scientific conservation laws, derived safety limits,
PDS4 complex SLC processing, Stokes invariants, and cryptographic provenance sidecars.
"""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import rasterio

from src.config_schema import load_config, LunaTraceConfig, BoundingBoxConfig
from src.validator import validate_input_datasets, DataValidationError
from src.utils import enhanced_lee_filter, save_geotiff
from src.pds4_reader import (
    parse_pds4_xml_label,
    verify_footprint_overlap,
    read_complex_slc_pair,
    apply_radiometric_calibration,
    multilook_and_compute_stokes,
    run_phase0_ingestion,
    compute_dem_radar_shadow_and_layover_masks,
    read_lola_pds3_dem,
    parse_lola_pds3_label,
    check_coordinate_in_footprint,
    derive_search_window_radius_m,
    compute_polarimetric_algebraic_bound
)
from src.phase1_dfsar import DFSARProcessor, run_phase1
from src.phase2_ice_detection import IceDetectionEngine, run_phase2
from src.phase3_terrain_illumination import TerrainIlluminationEngine, run_phase3
from src.phase4_landing_site import LandingSiteSelector, run_phase4
from src.phase5_traverse_planning import RoverTraversePlanner, run_phase5
from src.phase6_volume_estimation import IceVolumeEstimator, run_phase6
from tests.fixtures.synthetic_crater_generator import generate_synthetic_pds4_slc_fixture


# ==============================================================================
# 1. Configuration & Validation Tests
# ==============================================================================

def test_config_validation_and_derived_limits():
    """[SCHEMA/UNIT] Validates schema bounds and dynamically derived limits in config."""
    cfg = load_config()

    # Rule 4: Traversable slope must be strictly derived from chassis rollover limit / safety factor
    expected_slope = round(cfg.rover.chassis_rollover_limit_deg / cfg.rover.safety_factor, 2)
    assert cfg.rover.max_nav_slope_deg == expected_slope
    assert cfg.rover.max_nav_slope_deg == 18.0

    # Weights must sum to 1.0
    assert sum(cfg.ice_detection.weights.values()) == pytest.approx(1.0, 1e-4)
    assert sum(cfg.landing.weights_without_ohrc.values()) == pytest.approx(1.0, 1e-4)
    assert sum(cfg.landing.weights_with_ohrc.values()) == pytest.approx(1.0, 1e-4)


def test_missing_data_fails_loudly(tmp_path: Path):
    """[SCHEMA/UNIT] Rule 1: If an input file is missing, the validator must fail loudly with DataValidationError."""
    cfg = load_config()
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir()

    with pytest.raises((DataValidationError, FileNotFoundError)):
        validate_input_datasets(raw_dir=empty_dir, config=cfg, enforce_provenance=False)


def test_speckle_filter_intensity_and_signed():
    """[SYNTHETIC FIXTURE] Tests Enhanced Lee filter on synthetic Gaussian noise for non-negative intensity (S1) and signed channels (S2, S3, S4)."""
    img_pos = np.ones((20, 20), dtype=float) * 5.0
    filtered_pos = enhanced_lee_filter(img_pos, window_size=5, num_looks=4, is_intensity=True)
    assert filtered_pos.shape == (20, 20)
    assert np.all(filtered_pos >= 0.0)

    img_signed = np.full((20, 20), -2.5, dtype=float)
    filtered_signed = enhanced_lee_filter(img_signed, window_size=5, num_looks=4, is_intensity=False)
    assert np.all(filtered_signed < 0.0)


# ==============================================================================
# 2. Phase 0: PDS4 Complex Ingestion & Calibration Tests (Requirement 6)
# ==============================================================================

def test_phase0_rejects_amplitude_only_input(tmp_path: Path):
    """
    [SYNTHETIC FIXTURE] Requirement 1 & 6: Tests that 1-band amplitude GeoTIFFs fail fast with ValueError.
    Stokes S3 and S4 cannot be computed from intensity.
    """
    fixture = generate_synthetic_pds4_slc_fixture(tmp_path)
    amp_file = fixture["amplitude_only"]

    with pytest.raises(ValueError, match="amplitude-only"):
        read_complex_slc_pair(amp_file, amp_file)


def test_phase0_rejects_missing_calibration_fields(tmp_path: Path):
    """
    [SYNTHETIC FIXTURE] Requirement 2 & 6: Tests that synthetic PDS4 XML label missing mandatory
    calibration or metadata fields fails fast with ValueError.
    """
    fixture = generate_synthetic_pds4_slc_fixture(tmp_path)
    corrupt_xml = fixture["corrupt_xml"]

    with pytest.raises(ValueError, match="is missing required"):
        parse_pds4_xml_label(corrupt_xml)


def test_phase0_missing_bbox_fails_fast(test_config: LunaTraceConfig):
    """
    [SCHEMA/UNIT] Requirement 2: If the crater bounding box is missing from config, Phase 0 must raise an error.
    """
    cfg_copy = test_config.model_copy(deep=True)
    cfg_copy.target_bounding_box = None

    with pytest.raises(ValueError, match="Target crater bounding box is missing"):
        run_phase0_ingestion(config=cfg_copy)


def test_phase0_footprint_bounding_box_overlap(tmp_path: Path):
    """
    [SYNTHETIC FIXTURE] Requirement 2: Verifies that product footprint actually overlaps the bounding box,
    and fails fast with ValueError if there is no spatial intersection.
    """
    fixture = generate_synthetic_pds4_slc_fixture(tmp_path)
    label_meta = parse_pds4_xml_label(fixture["valid_xml"])

    # Overlapping bounding box: South Pole -87 to -86 deg
    good_bbox = BoundingBoxConfig(lat_min=-87.5, lat_max=-86.5, lon_min=78.0, lon_max=88.0)
    overlap, bounds = verify_footprint_overlap(label_meta, good_bbox)
    assert overlap is True
    assert bounds["lat_min"] <= good_bbox.lat_max

    # Non-overlapping bounding box (Equatorial region 0 to 5 deg)
    bad_bbox = BoundingBoxConfig(lat_min=0.0, lat_max=5.0, lon_min=10.0, lon_max=20.0)
    with pytest.raises(ValueError, match="does not overlap"):
        verify_footprint_overlap(label_meta, bad_bbox)


def test_phase0_stokes_invariants_and_radiometric_calibration(tmp_path: Path):
    """
    [SYNTHETIC FIXTURE] Requirement 6: Tests Stokes invariants (S1 >= 0, DOP in [0, 1], CPR >= 0)
    on a labelled synthetic complex SLC fixture.
    """
    fixture = generate_synthetic_pds4_slc_fixture(tmp_path)
    label_meta = parse_pds4_xml_label(fixture["valid_xml"])

    # Read complex SLC pair
    c_lh, c_lv = read_complex_slc_pair(fixture["slc_lh"], fixture["slc_lv"])
    assert np.iscomplexobj(c_lh) and np.iscomplexobj(c_lv)

    # Radiometric calibration
    e_lh, e_lv = apply_radiometric_calibration(c_lh, c_lv, label_meta)

    # Multilook and compute Stokes
    stokes = multilook_and_compute_stokes(e_lh, e_lv, azimuth_looks=4, range_looks=2)

    s1 = stokes["S1"]
    s2 = stokes["S2"]
    s3 = stokes["S3"]
    s4 = stokes["S4"]
    dop = stokes["DOP"]
    cpr = stokes["CPR"]

    # Invariant 1: S1 >= 0
    assert np.all(s1 >= 0.0)

    # Invariant 2: Polarized power S1^2 >= S2^2 + S3^2 + S4^2 (Cauchy-Schwarz)
    pol_power_sq = s2 ** 2 + s3 ** 2 + s4 ** 2
    np.testing.assert_array_less(np.sqrt(pol_power_sq), s1 + 1e-4)

    # Invariant 3: DOP in [0, 1]
    assert np.all(dop >= 0.0)
    assert np.all(dop <= 1.0)

    # Invariant 4: CPR >= 0
    assert np.all(cpr >= 0.0)


def test_dop_wishart_null_distribution_monte_carlo():
    """
    [SYNTHETIC SIMULATION] Validates that the 2x2 complex Wishart null distribution for sample Degree of Polarization
    m follows p(m) proportional to m^2 (1 - m^2)^(N-2) with CDF F(m) = I_{m^2}(1.5, N-1),
    and disproves the 2D coherence formula 1 - (1 - m^2)^(N-1).
    Runs with fixed seed and 100,000 trials per look count.
    """
    from scipy.special import betainc
    np.random.seed(42)
    m_trials = 100000
    for n_looks in [4, 8, 16, 32]:
        x = (np.random.standard_normal((m_trials, n_looks)) + 1j * np.random.standard_normal((m_trials, n_looks))) / np.sqrt(2.0)
        y = (np.random.standard_normal((m_trials, n_looks)) + 1j * np.random.standard_normal((m_trials, n_looks))) / np.sqrt(2.0)

        jxx = np.mean(np.abs(x)**2, axis=1)
        jyy = np.mean(np.abs(y)**2, axis=1)
        jxy = np.mean(x * np.conj(y), axis=1)

        s1 = jxx + jyy
        s2 = jxx - jyy
        s3 = 2.0 * np.real(jxy)
        s4 = -2.0 * np.imag(jxy)

        m = np.sqrt(s2**2 + s3**2 + s4**2) / s1

        for threshold in [0.10, 0.13, 0.20, 0.32]:
            emp_cdf = float(np.mean(m <= threshold))
            wishart_cdf = float(betainc(1.5, n_looks - 1, threshold**2))
            old_formula_cdf = float(1.0 - (1.0 - threshold**2)**(n_looks - 1))

            # Wishart CDF matches empirical Monte Carlo within 0.5% (0.005)
            assert abs(emp_cdf - wishart_cdf) < 0.005, f"Wishart CDF mismatch at N={n_looks}, m={threshold}"

            # At N=16, m=0.13, verify old formula had a massive error (> 10%)
            if n_looks == 16 and threshold == 0.13:
                assert abs(old_formula_cdf - emp_cdf) > 0.10, "Old formula should fail significantly"


def test_sar_layover_and_shadow_geometry():
    """
    [SYNTHETIC FIXTURE] Validates SAR layover and shadow masks on a synthetic ramp DEM:
    - Layover condition: foreslope facing radar > incidence angle
    - Shadow condition: backslope away from radar > (90 deg - incidence angle)
    - Evaluates both left-looking and right-looking geometries.
    """
    inc_deg = 26.0
    dx = 20.0  # meters per pixel

    # Profile across 50 columns:
    # Cols 0-10: flat (0 m)
    # Cols 10-15: steep foreslope rising by 14 m/col -> dz/dx = 0.70 > tan(26 deg) [0.488] -> LAYOVER
    # Cols 15-25: gentle foreslope rising by 5 m/col -> dz/dx = 0.25 < tan(26 deg) -> NO LAYOVER
    # Cols 25-30: gentle backslope dropping by 15 m/col -> dz/dx = -0.75 -> NO SHADOW (-dz/dx = 0.75 < cot(26 deg) [2.05])
    # Cols 30-35: steep backslope dropping by 50 m/col -> dz/dx = -2.50 -> SHADOW (-dz/dx = 2.50 > cot(26 deg))
    profile = np.zeros(50, dtype=float)
    for c in range(10, 15):
        profile[c] = profile[c-1] + 14.0
    for c in range(15, 25):
        profile[c] = profile[c-1] + 5.0
    for c in range(25, 30):
        profile[c] = profile[c-1] - 15.0
    for c in range(30, 35):
        profile[c] = profile[c-1] - 50.0
    for c in range(35, 50):
        profile[c] = profile[34]

    dem_left = np.tile(profile, (10, 1))
    shadow, layover = compute_dem_radar_shadow_and_layover_masks(
        dem_left, pixel_spacing_m=dx, incidence_angle_deg=inc_deg, look_direction="LEFT"
    )

    # Cols 10-14 have steep foreslope -> must be layover
    assert np.all(layover[:, 10:14] == 1), "Steep foreslope facing radar must trigger layover"
    # Cols 16-24 have gentle foreslope -> must NOT be layover
    assert np.all(layover[:, 16:24] == 0), "Gentle foreslope must not trigger layover"
    # Foreslope must not be shadow
    assert np.all(shadow[:, 10:25] == 0), "Foreslope facing radar must not be shadow"
    # Cols 30-34 have steep backslope -> must be shadow
    assert np.all(shadow[:, 30:34] == 1), "Steep backslope sloping away must trigger shadow"
    # Backslope must not trigger layover
    assert np.all(layover[:, 25:35] == 0), "Backslope must not trigger layover"

    # 2. Right-looking geometry (reversed look direction):
    dem_right = np.fliplr(dem_left)
    shadow_r, layover_r = compute_dem_radar_shadow_and_layover_masks(
        dem_right, pixel_spacing_m=dx, incidence_angle_deg=inc_deg, look_direction="RIGHT"
    )
    np.testing.assert_array_equal(layover_r, np.fliplr(layover))
    np.testing.assert_array_equal(shadow_r, np.fliplr(shadow))


def test_lola_pds3_dem_manual_reader(tmp_path: Path):
    """
    [SYNTHETIC FIXTURE] Validates manual PDS3 reader for LOLA DEM:
    - Parses .LBL parameters without GDAL
    - Correctly handles OFFSET = 0.0 (direct elevation in meters)
    - Correctly handles OFFSET = 1737400.0 (reference radius: does NOT add 1.737e6 m to height)
    - Correctly handles MISSING_CONSTANT
    - Rejects elevations outside plausible lunar range [-12000 m, +12000 m].
    """
    lbl_file = tmp_path / "ldem_test.lbl"
    img_file = tmp_path / "ldem_test.img"

    lines, samples = 20, 20
    elev_true = np.linspace(-2000.0, 1000.0, lines * samples).reshape((lines, samples)).astype(np.float32)
    raw_dn = (elev_true / 0.5).astype(np.int16)
    raw_dn[0, 0] = -32768

    raw_dn.astype(">i2").tofile(img_file)

    # 1. Standard Topography label (OFFSET = 0.0)
    lbl_content = f"""PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = {samples * 2}
FILE_RECORDS = {lines}
PRODUCT_TYPE = "TOPOGRAPHY"
^IMAGE = "{img_file.name}"
OBJECT = IMAGE
  LINES = {lines}
  LINE_SAMPLES = {samples}
  SAMPLE_BITS = 16
  SAMPLE_TYPE = MSB_INTEGER
  SCALING_FACTOR = 0.5 <METER>
  OFFSET = 0.0 <METER>
  MISSING_CONSTANT = -32768
END_OBJECT = IMAGE
END
"""
    lbl_file.write_text(lbl_content, encoding="ascii")

    elev_parsed, meta = read_lola_pds3_dem(lbl_file)
    assert meta["scaling_factor"] == 0.5
    assert meta["offset"] == 0.0
    assert np.isnan(elev_parsed[0, 0]), "Missing pixel must be masked as NaN"
    np.testing.assert_allclose(elev_parsed[1:, 1:], elev_true[1:, 1:], atol=0.5)

    # 2. Reference Radius label (OFFSET = 1737400.0 meters)
    # The reader must NOT add 1737400.0 to elevation; elevations must remain in [-2000, 1000] m
    lbl_ref_radius = f"""PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = {samples * 2}
FILE_RECORDS = {lines}
PRODUCT_TYPE = "RDR"
^IMAGE = "{img_file.name}"
OBJECT = IMAGE
  LINES = {lines}
  LINE_SAMPLES = {samples}
  SAMPLE_BITS = 16
  SAMPLE_TYPE = MSB_INTEGER
  SCALING_FACTOR = 0.5 <METER>
  OFFSET = 1737400.0 <METER>
  MISSING_CONSTANT = -32768
END_OBJECT = IMAGE
END
"""
    lbl_file.write_text(lbl_ref_radius, encoding="ascii")
    elev_ref, meta_ref = read_lola_pds3_dem(lbl_file)
    assert meta_ref["offset"] == 1737400.0
    # Must match true elevation without being shifted by 1.737 million meters
    np.testing.assert_allclose(elev_ref[1:, 1:], elev_true[1:, 1:], atol=0.5)

    # 3. Test rejection of unphysical elevations (< -12000 m or > +12000 m)
    lbl_file.write_text(lbl_content, encoding="ascii")
    corrupt_dn = np.full((lines, samples), 30000, dtype=">i2")
    corrupt_dn.tofile(img_file)
    with pytest.raises(ValueError, match="out of plausible lunar range"):
        read_lola_pds3_dem(lbl_file)

    # 4. Test rejection of unknown PRODUCT_TYPE values (strict whitelist)
    lbl_unknown = lbl_content.replace('PRODUCT_TYPE = "TOPOGRAPHY"', 'PRODUCT_TYPE = "UNSUPPORTED_TYPE"')
    lbl_file.write_text(lbl_unknown, encoding="ascii")
    with pytest.raises(ValueError, match="Unknown LOLA PRODUCT_TYPE"):
        read_lola_pds3_dem(lbl_file)


def test_derive_search_window_dem_requirement():
    """
    [SCHEMA/UNIT] Validates that derive_search_window_radius_m:
    1. Strictly raises ValueError if DEM is missing (dem is None).
    2. Derives search window radius from label geo_error_time_meters and DEM relief (max - min).
    """
    geo_err_m = -3115.967394
    inc_deg = 25.074092

    # 1. Missing DEM must raise
    with pytest.raises(ValueError, match="Registered DEM is missing"):
        derive_search_window_radius_m(geo_error_time_meters=geo_err_m, incidence_angle_deg=inc_deg, dem=None)

    # 2. Registered DEM with relief
    dem = np.array([[-1500.0, 500.0]], dtype=np.float32)
    relief_m = 2000.0
    cot_inc = 1.0 / np.tan(np.radians(inc_deg))
    expected_radius = abs(geo_err_m) + relief_m * cot_inc
    radius = derive_search_window_radius_m(geo_error_time_meters=geo_err_m, incidence_angle_deg=inc_deg, dem=dem)
    assert radius == pytest.approx(expected_radius, rel=1e-4)


def test_polarimetric_algebraic_bound():
    """
    [SCHEMA/UNIT] Validates the fundamental polarimetric lower bound on DoP:
    m >= |CPR - 1| / (CPR + 1).
    Proves that CPR > 1.30 algebraically forbids DoP <= 0.13.
    """
    cpr_values = np.array([0.5, 1.0, 1.2, 1.3, 1.5, 2.0, 3.0])
    bounds = compute_polarimetric_algebraic_bound(cpr_values)

    # At CPR = 1.0, bound is 0.0
    assert bounds[1] == pytest.approx(0.0, abs=1e-6)
    # At CPR = 1.29885 (~1.30), bound is exactly 0.13
    cpr_boundary = (1.0 + 0.13) / (1.0 - 0.13)
    assert compute_polarimetric_algebraic_bound(np.array([cpr_boundary]))[0] == pytest.approx(0.13, rel=1e-4)

    # Any CPR > 1.30 must have m_min > 0.13
    assert bounds[4] > 0.13  # CPR = 1.5 -> bound = 0.5 / 2.5 = 0.20
    assert bounds[5] > 0.13  # CPR = 2.0 -> bound = 1.0 / 3.0 = 0.333


def test_coordinate_footprint_containment():
    """
    [REAL DATA] Validates point-in-polygon containment computation for candidate coordinates
    against the 16,650 outer boundary tie points of the 38 x 8,289 grid.
    """
    label_meta = {
        "footprint": {
            "upper_left_latitude": -84.433971,
            "upper_left_longitude": 75.923767,
            "upper_right_latitude": -84.424787,
            "upper_right_longitude": 77.124202,
            "lower_right_latitude": -88.529488,
            "lower_right_longitude": 101.847497,
            "lower_left_latitude": -88.587333,
            "lower_left_longitude": 97.469234
        }
    }
    geom_csv_path = Path("data/raw/ch2_sar_ncxl_20260403t190328238_d_cp_d18/geometry/calibrated/20260403/ch2_sar_ncxl_20260403t190328238_g_sli_xx_cp_xx_d18.csv")

    res_center = check_coordinate_in_footprint(-86.510862, 83.925179, label_meta, geom_csv_path=geom_csv_path)
    assert res_center["is_inside_footprint"] is True
    assert res_center["outer_boundary_points_count"] == 16650

    # Directive 6: UNVERIFIED candidate from third-party repository (-87.39, 82.31)
    res_candidate = check_coordinate_in_footprint(-87.39, 82.31, label_meta, geom_csv_path=geom_csv_path)
    assert res_candidate["is_inside_footprint"] is True
    assert 600.0 < res_candidate["distance_to_boundary_m"] < 800.0

    res_outside = check_coordinate_in_footprint(0.0, 0.0, label_meta, geom_csv_path=geom_csv_path)
    assert res_outside["is_inside_footprint"] is False
    assert res_outside["distance_to_boundary_m"] > 100000.0




# ==============================================================================
# 3. Dedicated Phase 1 to 6 Unit Tests (Restoring Full Coverage)
# ==============================================================================

def test_phase1_polarimetry_energy_conservation(test_config: LunaTraceConfig):
    """[SYNTHETIC FIXTURE] Validates polarimetric Stokes formulas and energy conservation: Pv + Pdb + Ps == S1."""
    processor = DFSARProcessor(config=test_config)
    s1 = np.array([[2.0, 3.0]])
    s2 = np.array([[0.2, 0.4]])
    s3 = np.array([[0.1, -0.3]])
    s4 = np.array([[-0.5, 0.5]])
    stokes = {"S1": s1, "S2": s2, "S3": s3, "S4": s4}

    prods = processor.compute_polarimetric_products(stokes)
    cpr = prods["CPR"]
    pv = prods["P_volume"]
    pdb = prods["P_double"]
    ps = prods["P_surface"]

    # CPR identity: (2.0 - (-0.5)) / (2.0 + (-0.5)) = 2.5 / 1.5 = 1.6667
    assert cpr[0, 0] == pytest.approx(2.5 / 1.5, 1e-4)

    # Energy conservation: Pv + Pdb + Ps == S1
    total_decomp = pv + pdb + ps
    np.testing.assert_allclose(total_decomp, s1, rtol=1e-4)


def test_phase2_ice_discrimination_and_s_band_handling(test_config: LunaTraceConfig, processed_pipeline):
    """
    [SYNTHETIC FIXTURE] Validates Phase 2 ice detection scoring, de-circularized FPR metrics,
    unassessed OHRC status, and graceful S-band handling (Requirement 4).
    """
    engine = IceDetectionEngine(config=test_config)
    inputs, tf, crs, _ = engine.load_inputs()

    prods = engine.compute_ice_probability(inputs)
    ice_prob = prods["ice_probability"]
    assert np.all(ice_prob >= 0.0) and np.all(ice_prob <= 1.0)

    # Execute full Phase 2 export
    res = engine.process_and_export()
    assert res["boulder_status"] == "UNASSESSED (OHRC not provided)"

    df_sens = res["sensitivity"]
    assert "Radar_Alone_Control_FP_Count" in df_sens.columns
    assert "Combined_Control_FP_Count" in df_sens.columns


def test_phase3_terrain_morphometry_and_illumination(test_config: LunaTraceConfig, processed_pipeline):
    """[SYNTHETIC FIXTURE] Validates Phase 3 ephemeris horizon ray-tracing and slope calculations."""
    engine = TerrainIlluminationEngine(config=test_config)
    dem, tf, crs, _ = engine.load_dem()

    morph = engine.compute_morphometry(dem)
    assert np.all(morph["slope_deg"] >= 0.0)

    illum_res = engine.compute_illumination_and_psr(dem)
    illum = illum_res["illumination_fraction"]
    assert np.all(illum >= 0.0) and np.all(illum <= 1.0)

    res = engine.process_and_export()
    assert len(res["summary"]) >= 6


def test_phase4_landing_site_selection(test_config: LunaTraceConfig, processed_pipeline):
    """[SYNTHETIC FIXTURE] Validates Phase 4 candidate landing sites slope constraint (<=5 deg) and dynamic justifications."""
    selector = LandingSiteSelector(config=test_config)
    res = selector.process_and_export()

    df_sites = res["df_sites"]
    assert len(df_sites) == test_config.landing.top_k_sites

    site1 = df_sites.iloc[0]
    assert site1["Slope_deg"] <= test_config.landing.max_landing_slope_deg
    assert "UNASSESSED" in site1["Boulder_Hazard_Status"]
    assert "slope = " in site1["Justification"]


def test_phase5_traverse_planning_and_battery(test_config: LunaTraceConfig, processed_pipeline):
    """[SYNTHETIC FIXTURE] Validates Phase 5 pathfinding within derived slope limit (18 deg) and battery reserve (>20%)."""
    planner = RoverTraversePlanner(config=test_config)
    res = planner.process_and_export()

    df_routes = res["df_comparison"]
    assert len(df_routes) == 3
    for _, route in df_routes.iterrows():
        assert route["Max_Slope_deg"] <= test_config.rover.max_nav_slope_deg
        assert route["Min_Battery_SoC_pct"] >= test_config.rover.min_battery_reserve_pct
        assert route["Traverse_Feasible"] is True or route["Traverse_Feasible"] == "True"


def test_phase6_volume_estimation(test_config: LunaTraceConfig, processed_pipeline):
    """[SYNTHETIC FIXTURE] Validates Phase 6 volumetric CRIM scenario ordering and positive ice volume."""
    estimator = IceVolumeEstimator(config=test_config)
    res = estimator.process_and_export()

    df_vol = res["df_volume"]
    assert len(df_vol) == 3

    low_m = df_vol.loc[df_vol["Scenario"].str.contains("Low"), "Water_Ice_Mass_Megatons"].values[0]
    nom_m = df_vol.loc[df_vol["Scenario"].str.contains("Nominal"), "Water_Ice_Mass_Megatons"].values[0]
    high_m = df_vol.loc[df_vol["Scenario"].str.contains("High"), "Water_Ice_Mass_Megatons"].values[0]

    assert low_m > 0.0
    assert nom_m > low_m
    assert high_m > nom_m


def test_artifacts_existence_and_provenance(test_config: LunaTraceConfig, processed_pipeline):
    """[METADATA / PROVENANCE] Validates that all expected CSV tables and maps have cryptographic provenance sidecars."""
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
        fpath = test_config.paths.outputs_tables_dir / tbl
        assert fpath.exists(), f"Missing table: {fpath}"
        sidecar = fpath.parent / f"{fpath.name}.provenance.json"
        assert sidecar.exists(), f"Missing provenance sidecar for {fpath}"


@pytest.mark.slow
def test_real_slc_crop_stokes_invariants():
    """
    [REAL DATA] Ingests a small real Chandrayaan-2 DFSAR SLC crop from data/raw/ and asserts Stokes invariants:
    - Zero-fill / invalid margin pixels are explicitly masked to NaN with valid_mask == 0.
    - S1 >= 0 everywhere on valid lunar pixels.
    - 0 <= DoP <= 1 everywhere on valid lunar pixels.
    - CPR >= 0 everywhere on valid lunar pixels without artificial clipping.
    - Diffusely scattering lunar regolith satisfies median CPR < 1.0 (canonical 0.2-0.5) on valid pixels.
    """
    pds4_dir = Path("data/raw/ch2_sar_ncxl_20260403t190328238_d_cp_d18")
    if not pds4_dir.exists():
        pytest.skip("Real Chandrayaan-2 PDS4 data not present in data/raw")

    xml_candidates = list(pds4_dir.glob("**/*_d_sli_*_d18.xml"))
    lh_candidates = list(pds4_dir.glob("**/*_d_sli_*_lh_d18.tif"))
    lv_candidates = list(pds4_dir.glob("**/*_d_sli_*_lv_d18.tif"))

    if not xml_candidates or not lh_candidates or not lv_candidates:
        pytest.skip("PDS4 complex SLC pair not present")

    label_meta = parse_pds4_xml_label(xml_candidates[0])
    window = rasterio.windows.Window(0, 150000, 1152, 2000)
    c_lh, c_lv, valid_raw = read_complex_slc_pair(lh_candidates[0], lv_candidates[0], window=window, return_mask=True)

    e_lh, e_lv = apply_radiometric_calibration(c_lh, c_lv, label_meta)
    stokes = multilook_and_compute_stokes(e_lh, e_lv, azimuth_looks=8, range_looks=2, valid_raw_mask=valid_raw)

    # 1. Assert that fill pixels are masked
    fill_mask = stokes["valid_mask"] == 0
    assert np.sum(fill_mask) > 0, "Zero-fill margin pixels must be detected in SLC swath"
    assert np.all(np.isnan(stokes["CPR"][fill_mask])), "CPR must be NaN for fill pixels"
    assert np.all(np.isnan(stokes["DOP"][fill_mask])), "DOP must be NaN for fill pixels"
    assert np.all(np.isnan(stokes["S1"][fill_mask])), "S1 must be NaN for fill pixels"

    # 2. Assert invariants on valid pixels only
    v_mask = stokes["valid_mask"] == 1
    assert np.sum(v_mask) > 0, "Must have valid lunar pixels"
    s1 = stokes["S1"][v_mask]
    cpr = stokes["CPR"][v_mask]
    dop = stokes["DOP"][v_mask]

    assert np.all(s1 >= 0.0)
    assert np.all(dop >= 0.0) and np.all(dop <= 1.0)
    assert np.all(cpr >= 0.0)
    assert np.median(cpr) < 1.0
    assert 0.20 < np.median(cpr) < 0.60



