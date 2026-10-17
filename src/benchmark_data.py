"""
Benchmark & Synthetic Lunar South Pole Dataset Generator
Simulates realistic Chandrayaan-2 DFSAR (L-band & S-band hybrid polarimetry),
DEM (LOLA proxy), and Illumination datasets over a lunar south polar crater.
"""

from pathlib import Path
import numpy as np
from affine import Affine
from rasterio.transform import from_origin

from src.config import CFG, RAW_DATA_DIR
from src.utils import save_geotiff


def generate_benchmark_crater_data(
    grid_size: int = 500,
    resolution_m: float = 20.0,
    seed: int = 42
) -> dict:
    """
    Generates a physically consistent synthetic lunar south pole dataset covering a 10 km x 10 km area.
    Simulates:
    1. DEM (Elevation with crater bowl, rim uplift, central mounds, and micro-roughness)
    2. Optical / Illumination map (sunlit rim, permanently shadowed floor, doubly shadowed core)
    3. Earth line-of-sight visibility
    4. L-band DFSAR Stokes vector (S1, S2, S3, S4)
    5. S-band DFSAR Stokes vector (S1, S2, S3, S4)
    """
    np.random.seed(seed)
    half_width = (grid_size * resolution_m) / 2.0
    x = np.linspace(-half_width, half_width, grid_size)
    y = np.linspace(half_width, -half_width, grid_size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    
    # 1. Topography: 8 km diameter crater centered at origin
    crater_radius = 4000.0  # 4 km radius
    rim_height = 450.0      # Uplifted rim
    floor_depth = -1800.0   # Deep polar crater bowl
    
    # Crater bowl profile (parabolic floor + uplifted rim + ejecta decay)
    elev = np.zeros_like(r)
    in_crater = r < crater_radius
    elev[in_crater] = floor_depth * (1.0 - (r[in_crater] / crater_radius) ** 2)
    
    # Uplifted rim and continuous ejecta blanket
    rim_factor = np.exp(-((r - crater_radius) ** 2) / (2 * (800.0 ** 2)))
    elev += rim_height * rim_factor
    
    # Random topographic roughness (craterlets and mounds)
    roughness = (
        np.random.normal(0, 15, size=elev.shape) +
        25.0 * np.sin(xx / 300.0) * np.cos(yy / 300.0)
    )
    elev += roughness
    
    # 2. Illumination & Permanently Shadowed Region (PSR)
    # Solar azimuth from top-right at low grazing angle (~1.5 degrees elevation)
    illumination = np.clip((xx * 0.4 + yy * 0.4 - elev * 0.15) / 1000.0, 0.0, 1.0)
    # Crater interior is permanently shadowed
    illumination[r < (crater_radius * 0.85)] = 0.0
    # Rim peaks receive intermittent/high illumination
    illumination[(r > crater_radius * 0.9) & (r < crater_radius * 1.3) & (elev > 200)] = np.random.uniform(0.65, 0.92, size=np.sum((r > crater_radius * 0.9) & (r < crater_radius * 1.3) & (elev > 200)))
    
    # Doubly Shadowed Core Zone (deepest floor with zero direct and secondary scattered light)
    doubly_shadowed = (r < (crater_radius * 0.45)) & (elev < -1200.0)
    
    # Earth Line-of-Sight Visibility (direct comms)
    # Earth librates around lunar North/South horizon; northern rims have higher LOS
    earth_los = np.clip((yy + half_width) / (2 * half_width) + (elev / 3000.0), 0.0, 1.0)
    earth_los[r < crater_radius * 0.8] = 0.0  # Floor blocked by high rim walls
    
    # Boulder & Roughness Hazard Distribution (high on steep rim walls and fresh ejecta)
    boulder_density = np.clip(
        0.15 * rim_factor +
        0.25 * (r > crater_radius * 0.85) * (r < crater_radius * 1.1) +
        np.random.exponential(0.04, size=elev.shape),
        0.0, 1.0
    )
    
    # 3. Simulate Hybrid Polarimetric Radar Response (DFSAR)
    # Define 3 Distinct Geologic Units:
    # A. Background highlands regolith: CPR ~ 0.35 - 0.55, DOP ~ 0.7 - 0.85
    # B. Rocky ejecta & steep rim walls: High CPR (1.1 - 1.5) due to blocky roughness in BOTH L and S
    # C. Subsurface Ice Deposits (confined in cold trap doubly-shadowed floor):
    #    - CBOE multiple volume scattering produces CPR > 1.2, DOP < 0.12
    #    - Subsurface lag: Ice is buried under ~0.5 m dry regolith.
    #      Therefore, L-band (penetration ~3.5m) senses ice strongly (CPR ~ 1.5 - 2.1).
    #      S-band (penetration ~1.2m) senses less ice (CPR ~ 0.8 - 1.1).
    #      Delta CPR (L - S) > 0.4 is a diagnostic signature of buried ice!
    
    ice_core_zone = doubly_shadowed & (np.random.uniform(0, 1, size=elev.shape) > 0.25)
    # Smooth ice patches
    from scipy.ndimage import gaussian_filter
    ice_mask = gaussian_filter(ice_core_zone.astype(float), sigma=8) > 0.35
    
    rocky_ejecta = (r > crater_radius * 0.95) & (r < crater_radius * 1.35) & (boulder_density > 0.25)
    
    # Base Stokes Parameters:
    # S1 (Total intensity): High for rough and ice, moderate for background
    s1_base = np.random.gamma(shape=4.0, scale=0.08, size=elev.shape)
    s1_base[rocky_ejecta] *= 2.8
    s1_base[ice_mask] *= 3.5
    
    # L-band Stokes synthesis
    # Target CPR: CPR = (S1 - S4) / (S1 + S4)  =>  S4 = S1 * (1 - CPR) / (1 + CPR)
    cpr_l_target = np.random.uniform(0.35, 0.55, size=elev.shape)
    cpr_l_target[rocky_ejecta] = np.random.uniform(1.10, 1.55, size=np.sum(rocky_ejecta))
    cpr_l_target[ice_mask] = np.random.uniform(1.35, 2.20, size=np.sum(ice_mask))  # Strong ice signature
    
    # S-band Stokes synthesis
    cpr_s_target = np.random.uniform(0.32, 0.52, size=elev.shape)
    cpr_s_target[rocky_ejecta] = np.random.uniform(1.05, 1.50, size=np.sum(rocky_ejecta))  # Still high CPR (rocks)
    cpr_s_target[ice_mask] = np.random.uniform(0.70, 1.15, size=np.sum(ice_mask))  # Lower CPR (ice is buried!)
    
    def synthesize_stokes(s1_arr, cpr_arr, is_ice, is_rock):
        s4 = s1_arr * (1.0 - cpr_arr) / (1.0 + cpr_arr)
        
        # Target DOP: m = sqrt(S2^2 + S3^2 + S4^2) / S1
        dop_target = np.random.uniform(0.65, 0.85, size=s1_arr.shape)
        dop_target[is_rock] = np.random.uniform(0.30, 0.50, size=np.sum(is_rock))
        dop_target[is_ice] = np.random.uniform(0.04, 0.11, size=np.sum(is_ice))  # Highly depolarized
        
        m_s1 = dop_target * s1_arr
        rem_sq = np.maximum(m_s1 ** 2 - s4 ** 2, 0.0)
        rem = np.sqrt(rem_sq)
        
        # Synthesize S2 and S3 with random polarimetric phase delta
        phase = np.random.uniform(-np.pi, np.pi, size=s1_arr.shape)
        s2 = rem * np.cos(phase)
        s3 = rem * np.sin(phase)
        return s1_arr, s2, s3, s4
    
    l_s1, l_s2, l_s3, l_s4 = synthesize_stokes(s1_base * 1.1, cpr_l_target, ice_mask, rocky_ejecta)
    s_s1, s_s2, s_s3, s_s4 = synthesize_stokes(s1_base * 0.9, cpr_s_target, np.zeros_like(ice_mask, dtype=bool), rocky_ejecta)
    
    # Construct affine geotransform (stereographic centered at crater)
    # top-left corner: (-half_width, half_width)
    transform = from_origin(-half_width, half_width, resolution_m, resolution_m)
    
    # Save all raw benchmark rasters
    files = {
        "dem": RAW_DATA_DIR / "lola_dem_20m.tif",
        "illumination": RAW_DATA_DIR / "crater_illumination.tif",
        "earth_los": RAW_DATA_DIR / "crater_earth_los.tif",
        "boulders": RAW_DATA_DIR / "crater_boulder_density.tif",
        "l_s1": RAW_DATA_DIR / "dfsar_L_S1.tif",
        "l_s2": RAW_DATA_DIR / "dfsar_L_S2.tif",
        "l_s3": RAW_DATA_DIR / "dfsar_L_S3.tif",
        "l_s4": RAW_DATA_DIR / "dfsar_L_S4.tif",
        "s_s1": RAW_DATA_DIR / "dfsar_S_S1.tif",
        "s_s2": RAW_DATA_DIR / "dfsar_S_S2.tif",
        "s_s3": RAW_DATA_DIR / "dfsar_S_S3.tif",
        "s_s4": RAW_DATA_DIR / "dfsar_S_S4.tif",
        "ground_truth_ice": RAW_DATA_DIR / "ground_truth_ice_mask.tif"
    }
    
    save_geotiff(elev, files["dem"], transform)
    save_geotiff(illumination, files["illumination"], transform)
    save_geotiff(earth_los, files["earth_los"], transform)
    save_geotiff(boulder_density, files["boulders"], transform)
    
    save_geotiff(l_s1, files["l_s1"], transform)
    save_geotiff(l_s2, files["l_s2"], transform)
    save_geotiff(l_s3, files["l_s3"], transform)
    save_geotiff(l_s4, files["l_s4"], transform)
    
    save_geotiff(s_s1, files["s_s1"], transform)
    save_geotiff(s_s2, files["s_s2"], transform)
    save_geotiff(s_s3, files["s_s3"], transform)
    save_geotiff(s_s4, files["s_s4"], transform)
    
    save_geotiff(ice_mask.astype(np.float32), files["ground_truth_ice"], transform)
    
    return files, transform
