"""
LunaTrace Master Configuration
Centralized configuration parameters for ISRO Problem Statement 8.
All physical constants, radar parameters, terrain thresholds, landing weights,
rover specifications, and volume estimation scenarios are strictly defined here.
Zero hardcoding permitted in operational phase modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MAPS_DIR = OUTPUTS_DIR / "maps"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MAPS_DIR, TABLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class RadarConfig:
    """Chandrayaan-2 DFSAR polarimetric processing parameters."""
    # Frequencies & Wavelengths
    freq_l_band_ghz: float = 1.25          # L-band frequency
    wavelength_l_m: float = 0.24           # L-band wavelength (~24 cm)
    freq_s_band_ghz: float = 2.50          # S-band frequency
    wavelength_s_m: float = 0.12           # S-band wavelength (~12 cm)
    
    # Speckle filtering (Enhanced Lee filter / Multilook)
    filter_type: str = "enhanced_lee"      # Options: 'enhanced_lee', 'boxcar'
    filter_window_size: int = 5            # 5x5 spatial window
    num_looks: int = 4                     # Multilook factor
    
    # Polarimetric Thresholds
    cpr_ice_threshold: float = 1.0         # CPR > 1.0 indicative of Coherent Backscatter Oppositions Effect (CBOE)
    dop_ice_max: float = 0.32              # Calibrated upper limit on DoP for depolarized volume scattering
    cpr_rock_threshold: float = 1.0        # Surface roughness threshold
    min_valid_stokes1: float = 1e-6        # Threshold to avoid division by zero in shadow or missing data
    cpr_min_clip: float = 0.0              # Lower bound for CPR calculation
    cpr_max_clip: float = 5.0              # Upper numerical bound for CPR raster output
    dop_min_clip: float = 0.0              # Lower bound for DoP calculation
    dop_max_clip: float = 1.0              # Upper bound for DoP calculation
    sin_2chi_min_clip: float = -1.0        # Lower bound for Raney m-chi sin(2*chi)
    sin_2chi_max_clip: float = 1.0        # Upper bound for Raney m-chi sin(2*chi)
    cpr_plot_vmin: float = 0.2
    cpr_plot_vmax: float = 2.0
    dop_plot_vmin: float = 0.0
    dop_plot_vmax: float = 1.0


@dataclass
class IceDetectionConfig:
    """Multi-Discriminator Ice Detection Parameters (Phase 2)."""
    # Scoring weights (Sum = 1.0)
    weight_pol: float = 0.40               # Weight for polarimetric volume response
    weight_freq: float = 0.35              # Weight for Delta CPR (L minus S) depth profiling
    weight_trap: float = 0.25              # Weight for cold-trap thermal stability prior
    roughness_penalty_scale: float = 0.85  # Multiplier for terrain hazard penalty
    
    # Sigmoidal response parameters
    cpr_sigmoid_steepness: float = 4.0
    dop_sigmoid_steepness: float = 15.0
    
    # Delta CPR depth thresholds
    delta_cpr_min_threshold: float = 0.20   # Minimum Delta CPR for buried ice indicator
    delta_cpr_scale: float = 0.50           # Normalization range
    delta_cpr_clip_min: float = -1.0
    delta_cpr_clip_max: float = 3.0
    delta_cpr_plot_vmin: float = -0.5
    delta_cpr_plot_vmax: float = 1.2
    
    # Thermal cold-trap stability priors
    cold_trap_illum_threshold: float = 0.10 # Direct solar illumination cutoff for stability
    cold_trap_depth_m: float = -350.0       # Elevation depression threshold
    weight_trap_illum: float = 0.60
    weight_trap_depth: float = 0.40
    trap_depth_normalization_m: float = 1000.0
    
    # Topographic hazard parameters
    hazard_slope_baseline_deg: float = 8.0
    hazard_slope_scale_deg: float = 10.0
    hazard_boulder_baseline: float = 0.18
    hazard_boulder_scale: float = 0.25
    weight_rough_slope: float = 0.60
    weight_rough_boulder: float = 0.40
    
    # High-confidence classification cutoffs
    high_prob_ice_cutoff: float = 0.50
    trap_confidence_cutoff: float = 0.40
    
    # Control region & rocky ejecta parameters
    control_illum_min: float = 0.40
    control_boulders_min: float = 0.20
    rocky_delta_cpr_max: float = 0.25
    bg_regolith_illum_min: float = 0.30
    bg_regolith_boulders_max: float = 0.15
    
    # Sensitivity analysis sweeps
    sensitivity_cpr_thresholds: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    sensitivity_dop_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.30, 0.35])


@dataclass
class TerrainConfig:
    """DEM and Topographic Analysis Parameters (Phase 3)."""
    # Projection: Lunar South Pole Stereographic (IAU 2000 Moon, EPSG:3032)
    projection_proj4: str = (
        "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
    )
    grid_resolution_m: float = 20.0        # 20 m per pixel for regional crater analysis
    extent_km: float = 10.0                # Half-width extent of analyzed scene (+-10 km)
    scale_bar_length_km: float = 4.0       # Length of cartographic scale bar
    km_per_deg_lat: float = 30.2           # Kilometers per degree latitude on the Moon
    max_regional_slope_plot: float = 30.0  # Max slope for colormap visualization
    contour_levels: List[float] = field(default_factory=lambda: [-1500.0, -1000.0, -500.0, 0.0, 500.0])
    slope_contour_levels: List[float] = field(default_factory=lambda: [5.0, 15.0])
    
    # Crater region parameters (Faustini / Shoemaker / Shackleton doubly shadowed floor)
    target_crater_name: str = "Faustini_Floor"
    center_lat: float = -87.3
    center_lon: float = 77.0
    
    # Morphometry moving window
    rms_window_size: int = 5               # 5x5 moving window (100 m baseline)
    
    # Horizon ray-tracing parameters
    num_horizon_azimuths: int = 16         # 16 azimuthal directions (0 to 360 deg)
    solar_elevation_grazing_deg: float = 1.5 # Mean solar grazing angle at lunar south pole
    ray_step_distances_m: List[float] = field(
        default_factory=lambda: [60.0, 120.0, 240.0, 500.0, 1000.0, 2000.0, 4000.0]
    )
    
    # Topographic & shadow thresholds
    crater_depression_threshold_m: float = -200.0
    deep_floor_shadow_threshold_m: float = -350.0
    psr_illumination_cutoff: float = 0.005
    sky_view_factor_cutoff: float = 0.86   # SVF threshold for doubly shadowed cold traps
    
    # Earth visibility ray-tracing
    earth_elevation_mean_deg: float = 3.5  # Mean Earth elevation above northern horizon
    earth_ray_distances_m: List[float] = field(
        default_factory=lambda: [100.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0]
    )


@dataclass
class LandingSiteConfig:
    """Multi-Criteria Landing Site Evaluation (Phase 4)."""
    max_landing_slope_deg: float = 5.0     # Lander constraint: slopes strictly <= 5 degrees
    max_boulder_density: float = 0.30      # Lander hazard: boulder density < 0.30
    min_illumination_fraction: float = 0.20 # Power survival: strictly outside PSR (> 20% sun)
    min_crater_distance_m: float = 1000.0  # Safe clearance from rim drop-off (1 km)
    max_crater_distance_m: float = 7500.0  # Traverse viability range to target ice deposits (7.5 km)
    optimal_ice_distance_m: float = 2000.0 # Ideal standoff distance for rover ingress
    top_k_sites: int = 3                   # Number of candidate sites to extract and rank
    spatial_separation_pixels: int = 25    # 500 m separation between candidate sites
    roughness_percentile: float = 95.0     # Percentile cutoff for roughness normalization
    
    # Multi-Criteria Weights (Sum = 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "slope": 0.30,                     # Lower slope is safer
        "roughness": 0.20,                 # Smooth terrain
        "illumination": 0.25,              # High solar power for lander
        "earth_los": 0.15,                 # Direct Earth line-of-sight for comms
        "ice_proximity": 0.10              # Proximity to ice candidate target
    })


@dataclass
class RoverConfig:
    """Rover Traverse Planning Parameters (Phase 5)."""
    max_nav_slope_deg: float = 18.0        # Max traversable slope for rover chassis (Apollo LRV / VIPER limit)
    max_boulder_hazard: float = 0.40       # Hard no-go boulder cutoff
    speed_m_per_s: float = 0.08            # Rover traverse speed (~288 m/h, VIPER shadow dash spec)
    battery_capacity_wh: float = 1800.0    # 1.8 kWh capacity (VIPER-class polar rover)
    baseline_power_w: float = 65.0         # Avionics and thermal baseload
    traction_power_w_per_deg: float = 4.0  # Extra motor power per degree of incline
    solar_charge_rate_w: float = 110.0     # Power generation under full illumination
    min_battery_reserve_pct: float = 20.0  # Mandatory safety reserve threshold
    max_shadow_time_hours: float = 18.0    # Hard cutoff for operating inside PSR on battery alone
    
    # Pathfinding cost weighting per mode
    slope_scale_shortest: float = 5.0
    weight_slope_shortest: float = 0.5
    weight_rough_shortest: float = 0.3
    
    slope_scale_safest: float = 4.0
    weight_slope_safest: float = 1.8
    weight_rough_safest: float = 3.0
    weight_boulder_safest: float = 2.5
    
    slope_scale_illuminated: float = 3.5
    weight_slope_illuminated: float = 1.2
    weight_rough_illuminated: float = 1.5
    weight_shadow_illuminated: float = 12.0


@dataclass
class VolumeScenarioConfig:
    """Parameters for a specific volume resource scenario."""
    base_fraction: float
    effective_depth_m: float
    dry_lag_m: float
    description: str


@dataclass
class VolumeConfig:
    """Subsurface Ice Dielectric & Volume Estimation (Phase 6)."""
    regolith_depth_max_m: float = 5.0      # Top 0-5 m regolith column
    effective_l_depth_m: float = 3.5       # Effective radar penetration at L-band
    effective_s_depth_m: float = 1.2       # Effective radar penetration at S-band
    ice_density_kg_m3: float = 920.0       # Density of crystalline water ice (0.92 t/m3)
    max_pore_saturation: float = 0.60      # Physical pore filling upper limit
    cpr_scale_range: float = 1.0           # CPR normalization scaling range
    cpr_norm_max_factor: float = 1.5       # Max normalized CPR factor
    nominal_lag_m: float = 0.5             # Nominal dry sublimation lag thickness
    
    # Sensitivity curve parameters
    sensitivity_fractions_min: float = 0.02
    sensitivity_fractions_max: float = 0.35
    sensitivity_fractions_steps: int = 50
    sensitivity_depths: List[float] = field(default_factory=lambda: [1.5, 2.5, 3.5, 5.0])
    
    # Regolith & Ice Dielectric properties
    eps_regolith_real: float = 2.70        # Typical highland anorthosite dielectric permittivity
    eps_ice_real: float = 3.15             # Water ice permittivity at ~40-100 K
    loss_tangent_dry_regolith: float = 0.0035 # Tan delta of dry low-Ti anorthosite
    loss_tangent_ice: float = 0.0001       # Tan delta of pure cryogenic water ice
    bulk_porosity: float = 0.42            # Subsurface regolith porosity
    
    # 3 Distinct Geological Scenarios
    scenarios: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Low (Conservative)": {
            "base_fraction": 0.05,
            "eff_depth_m": 2.0,
            "dry_lag_m": 0.8,
            "active_depth_m": 1.2,
            "description": "Sparse disseminated pore ice; thick 0.8m dry lag; shallow penetration depth (2.0m)"
        },
        "Nominal (Realistic)": {
            "base_fraction": 0.15,
            "eff_depth_m": 3.5,
            "dry_lag_m": 0.5,
            "active_depth_m": 3.0,
            "description": "Heterogeneous ice-regolith breccia; 0.5m dry lag; L-band penetration depth (3.5m)"
        },
        "High (Optimistic)": {
            "base_fraction": 0.30,
            "eff_depth_m": 5.0,
            "dry_lag_m": 0.3,
            "active_depth_m": 4.7,
            "description": "Segregated ice lenses / paleo-cometary deposits; full 0-5m column extrapolated"
        }
    })


@dataclass
class PipelineConfig:
    """Master pipeline configuration combining all modules."""
    random_seed: int = 42
    radar: RadarConfig = field(default_factory=RadarConfig)
    ice_detection: IceDetectionConfig = field(default_factory=IceDetectionConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    landing: LandingSiteConfig = field(default_factory=LandingSiteConfig)
    rover: RoverConfig = field(default_factory=RoverConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)


# Global default configuration instance
CFG = PipelineConfig()
