"""
LunaTrace Synthetic Crater Test Generator (TEST FIXTURE ONLY)
Used solely for automated unit testing and algorithmic validation within tests/.
Never executed in production runs or on real data.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from src.utils import save_geotiff
from src.config import CFG


def generate_synthetic_test_crater(
    output_dir: Path,
    grid_size: int = 500,
    resolution_m: float = 20.0,
    seed: int = 42
) -> Tuple[Dict[str, Path], rasterio.Affine]:
    """
    Generates a physically consistent synthetic lunar south pole test dataset covering a 10 km x 10 km area.
    Strictly for automated test suites inside tests/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    half_width = (grid_size * resolution_m) / 2.0
    x = np.linspace(-half_width, half_width, grid_size)
    y = np.linspace(half_width, -half_width, grid_size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)

    # 1. Topography: 8 km diameter crater centered at origin
    crater_radius = 4000.0  # 4 km radius
    rim_height = 220.0      # Uplifted rim
    floor_depth = -650.0    # Complex polar crater bowl depth

    elev = np.zeros_like(r)
    in_crater = r < crater_radius
    elev[in_crater] = floor_depth * (1.0 - (r[in_crater] / crater_radius) ** 2)

    # Ingress ramp corridor at ENE azimuth ~18 deg
    angle = np.arctan2(yy, xx)
    corridor_azimuth = np.radians(18.0)
    angular_dist = np.abs(np.arctan2(np.sin(angle - corridor_azimuth), np.cos(angle - corridor_azimuth)))
    ramp_width_rad = np.radians(22.0)

    ramp_r_min = 800.0
    ramp_r_max = crater_radius * 1.25
    in_ramp_radial = (r >= ramp_r_min) & (r <= ramp_r_max)
    ramp_profile = floor_depth + (r - ramp_r_min) * (rim_height + 50.0 - floor_depth) / (ramp_r_max - ramp_r_min)

    w = np.clip(np.cos((angular_dist / ramp_width_rad) * (np.pi / 2.0)), 0.0, 1.0) ** 2
    elev[in_ramp_radial] = (1.0 - w[in_ramp_radial]) * elev[in_ramp_radial] + w[in_ramp_radial] * ramp_profile[in_ramp_radial]

    rim_factor = np.exp(-((r - crater_radius) ** 2) / (2 * (800.0 ** 2)))
    elev += rim_height * rim_factor * (1.0 - w * 0.90)

    raw_roughness = (
        np.random.normal(0, 30.0, size=elev.shape) +
        35.0 * np.sin(xx / 400.0) * np.cos(yy / 400.0)
    )
    roughness = gaussian_filter(raw_roughness, sigma=4.0)
    elev += roughness

    # 2. Illumination & Permanently Shadowed Region
    illumination = np.clip((xx * 0.4 + yy * 0.4 - elev * 0.15) / 1000.0, 0.0, 1.0)
    illumination[r < (crater_radius * 0.85)] = 0.0
    rim_sunlit = (r > crater_radius * 0.9) & (r < crater_radius * 1.3) & (elev > 200)
    illumination[rim_sunlit] = np.random.uniform(0.65, 0.92, size=np.sum(rim_sunlit))

    doubly_shadowed = (r < (crater_radius * 0.45)) & (elev < -350.0)

    # 3. Hybrid Polarimetric Radar Response
    ice_core_zone = doubly_shadowed & (np.random.uniform(0, 1, size=elev.shape) > 0.25)
    ice_mask = gaussian_filter(ice_core_zone.astype(float), sigma=8) > 0.35
    rocky_ejecta = (r > crater_radius * 0.95) & (r < crater_radius * 1.35) & (roughness > 15.0)

    s1_base = np.random.gamma(shape=4.0, scale=0.08, size=elev.shape)
    s1_base = np.maximum(s1_base, 1.0e-4)
    s1_base[rocky_ejecta] *= 2.8
    s1_base[ice_mask] *= 3.5

    cpr_l_target = np.random.uniform(0.35, 0.55, size=elev.shape)
    cpr_l_target[rocky_ejecta] = np.random.uniform(1.10, 1.55, size=np.sum(rocky_ejecta))
    cpr_l_target[ice_mask] = np.random.uniform(1.35, 2.20, size=np.sum(ice_mask))

    cpr_s_target = np.random.uniform(0.32, 0.52, size=elev.shape)
    cpr_s_target[rocky_ejecta] = np.random.uniform(1.05, 1.50, size=np.sum(rocky_ejecta))
    cpr_s_target[ice_mask] = np.random.uniform(0.70, 1.15, size=np.sum(ice_mask))

    def synthesize_stokes(s1_arr, cpr_arr, is_ice, is_rock):
        s4 = s1_arr * (1.0 - cpr_arr) / (1.0 + cpr_arr)
        dop_target = np.random.uniform(0.65, 0.85, size=s1_arr.shape)
        dop_target[is_rock] = np.random.uniform(0.30, 0.50, size=np.sum(is_rock))
        dop_target[is_ice] = np.random.uniform(0.04, 0.11, size=np.sum(is_ice))

        m_s1 = dop_target * s1_arr
        rem_sq = np.maximum(m_s1 ** 2 - s4 ** 2, 0.0)
        rem = np.sqrt(rem_sq)

        phase = np.random.uniform(-np.pi, np.pi, size=s1_arr.shape)
        s2 = rem * np.cos(phase)
        s3 = rem * np.sin(phase)
        return s1_arr, s2, s3, s4

    l_s1, l_s2, l_s3, l_s4 = synthesize_stokes(s1_base * 1.1, cpr_l_target, ice_mask, rocky_ejecta)
    s_s1, s_s2, s_s3, s_s4 = synthesize_stokes(s1_base * 0.9, cpr_s_target, np.zeros_like(ice_mask, dtype=bool), rocky_ejecta)

    transform = from_origin(-half_width, half_width, resolution_m, resolution_m)

    files = {
        "dem": output_dir / "lola_dem_20m.tif",
        "illumination": output_dir / "crater_illumination.tif",
        "dfsar_L_S1": output_dir / "dfsar_L_S1.tif",
        "dfsar_L_S2": output_dir / "dfsar_L_S2.tif",
        "dfsar_L_S3": output_dir / "dfsar_L_S3.tif",
        "dfsar_L_S4": output_dir / "dfsar_L_S4.tif",
        "dfsar_S_S1": output_dir / "dfsar_S_S1.tif",
        "dfsar_S_S2": output_dir / "dfsar_S_S2.tif",
        "dfsar_S_S3": output_dir / "dfsar_S_S3.tif",
        "dfsar_S_S4": output_dir / "dfsar_S_S4.tif"
    }

    proj = CFG.crs.proj4_string
    save_geotiff(elev, files["dem"], transform, crs=proj)
    save_geotiff(illumination, files["illumination"], transform, crs=proj)

    save_geotiff(l_s1, files["dfsar_L_S1"], transform, crs=proj)
    save_geotiff(l_s2, files["dfsar_L_S2"], transform, crs=proj)
    save_geotiff(l_s3, files["dfsar_L_S3"], transform, crs=proj)
    save_geotiff(l_s4, files["dfsar_L_S4"], transform, crs=proj)

    save_geotiff(s_s1, files["dfsar_S_S1"], transform, crs=proj)
    save_geotiff(s_s2, files["dfsar_S_S2"], transform, crs=proj)
    save_geotiff(s_s3, files["dfsar_S_S3"], transform, crs=proj)
    save_geotiff(s_s4, files["dfsar_S_S4"], transform, crs=proj)

    return files, transform


def generate_synthetic_pds4_slc_fixture(output_dir: Path) -> Dict[str, Path]:
    """
    Generates labelled synthetic PDS4 XML labels, complex SLC files, and geometry CSV.
    Used strictly in tests/ for Phase 0 validation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid PDS4 XML label
    valid_xml = output_dir / "synthetic_d_sli_xx_cp_xx_d18.xml"
    valid_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational>
    <Observation_Area>
        <Mission_Area>
            <Product_Parameters>
                <frequency_band>L</frequency_band>
                <incidence_angle unit="deg">25.994</incidence_angle>
                <output_line_spacing unit="m">0.4889</output_line_spacing>
                <output_pixel_spacing unit="m">1.7987</output_pixel_spacing>
                <calibration_constant>70.308868</calibration_constant>
                <semi_major_radius unit="m">1737400.0</semi_major_radius>
                <semi_minor_radius unit="m">1737400.0</semi_minor_radius>
                <polarization_info>
                    <polarization>LH</polarization>
                    <bias_real>3.08159</bias_real>
                    <bias_imag>-1.18385</bias_imag>
                    <standard_deviation_real>10.55</standard_deviation_real>
                    <standard_deviation_imag>10.42</standard_deviation_imag>
                    <gain_imbalance>1.0127</gain_imbalance>
                    <phase_orthogonality>0.8765</phase_orthogonality>
                </polarization_info>
                <polarization_info>
                    <polarization>LV</polarization>
                    <bias_real>1.18489</bias_real>
                    <bias_imag>3.41592</bias_imag>
                    <standard_deviation_real>9.10</standard_deviation_real>
                    <standard_deviation_imag>9.11</standard_deviation_imag>
                    <gain_imbalance>0.9990</gain_imbalance>
                    <phase_orthogonality>0.4772</phase_orthogonality>
                </polarization_info>
            </Product_Parameters>
            <Geometry_Parameters>
                <upper_left_latitude unit="deg">-86.0</upper_left_latitude>
                <upper_left_longitude unit="deg">75.0</upper_left_longitude>
                <upper_right_latitude unit="deg">-86.0</upper_right_latitude>
                <upper_right_longitude unit="deg">90.0</upper_right_longitude>
                <lower_left_latitude unit="deg">-88.5</lower_left_latitude>
                <lower_left_longitude unit="deg">75.0</lower_left_longitude>
                <lower_right_latitude unit="deg">-88.5</lower_right_latitude>
                <lower_right_longitude unit="deg">90.0</lower_right_longitude>
                <centre_latitude unit="deg">-87.25</centre_latitude>
                <centre_longitude unit="deg">82.5</centre_longitude>
            </Geometry_Parameters>
        </Mission_Area>
    </Observation_Area>
</Product_Observational>"""
    with open(valid_xml, "w", encoding="utf-8") as f:
        f.write(valid_xml_content)

    # 2. Corrupt XML label (missing calibration_constant and polarization_info)
    corrupt_xml = output_dir / "corrupt_d_sli_missing_cal.xml"
    corrupt_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational>
    <Observation_Area>
        <Mission_Area>
            <Product_Parameters>
                <frequency_band>L</frequency_band>
                <incidence_angle unit="deg">25.994</incidence_angle>
            </Product_Parameters>
        </Mission_Area>
    </Observation_Area>
</Product_Observational>"""
    with open(corrupt_xml, "w", encoding="utf-8") as f:
        f.write(corrupt_xml_content)

    # 3. 2-Band Complex SLC pair (Real + Imaginary)
    np.random.seed(42)
    ny, nx = 64, 32
    real_lh = (np.random.randn(ny, nx) * 50.0 + 3.08159).astype(np.float32)
    imag_lh = (np.random.randn(ny, nx) * 50.0 - 1.18385).astype(np.float32)
    real_lv = (np.random.randn(ny, nx) * 45.0 + 1.18489).astype(np.float32)
    imag_lv = (np.random.randn(ny, nx) * 45.0 + 3.41592).astype(np.float32)

    slc_lh_path = output_dir / "synthetic_d_sli_xx_cp_lh_d18.tif"
    slc_lv_path = output_dir / "synthetic_d_sli_xx_cp_lv_d18.tif"

    import rasterio
    from rasterio.transform import Affine
    tf = Affine.identity()

    with rasterio.open(slc_lh_path, "w", driver="GTiff", height=ny, width=nx, count=2, dtype="float32", transform=tf) as dst:
        dst.write(real_lh, 1)
        dst.write(imag_lh, 2)

    with rasterio.open(slc_lv_path, "w", driver="GTiff", height=ny, width=nx, count=2, dtype="float32", transform=tf) as dst:
        dst.write(real_lv, 1)
        dst.write(imag_lv, 2)

    # 4. Amplitude-only 1-band raster (to test rejection)
    amplitude_path = output_dir / "amplitude_only_d_gri_lh.tif"
    with rasterio.open(amplitude_path, "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32", transform=tf) as dst:
        dst.write(np.abs(real_lh), 1)

    # 5. Geometry CSV
    geom_csv = output_dir / "synthetic_g_sli_xx_cp_xx_d18.csv"
    with open(geom_csv, "w", encoding="utf-8") as f:
        f.write("Latitude(deg.),Longitude(deg.),Slant_Range(m),Incidence_Angle(deg.)\n")
        f.write("-86.5,78.5,117000.0,25.5\n")
        f.write("-86.5,85.5,117500.0,25.8\n")
        f.write("-87.5,78.5,117000.0,25.5\n")
        f.write("-87.5,85.5,117500.0,25.8\n")

    return {
        "valid_xml": valid_xml,
        "corrupt_xml": corrupt_xml,
        "slc_lh": slc_lh_path,
        "slc_lv": slc_lv_path,
        "amplitude_only": amplitude_path,
        "geom_csv": geom_csv
    }

