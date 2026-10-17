"""
LunaTrace Configuration
Centralized configuration parameters for ISRO Problem Statement 8.
All thresholds, radar constants, rover specs, and path variables are configured here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

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
    
    # Speckle filtering (Enhanced Lee filter / Boxcar)
    filter_type: str = "enhanced_lee"      # Options: 'enhanced_lee', 'boxcar', 'median'
    filter_window_size: int = 5            # e.g., 5x5 or 7x7
    num_looks: int = 4                     # Multilook factor
    
    # Polarimetric Thresholds
    cpr_ice_threshold: float = 1.0         # CPR > 1.0 indicative of Coherent Backscatter Oppositions Effect (CBOE)
    dop_ice_max: float = 0.13              # DOP < 0.13 for strongly depolarized / multiple scattering
    cpr_rock_threshold: float = 1.0        # High CPR on sunlit/exterior crater rim attributed to surface roughness
    min_valid_stokes1: float = 1e-6        # Threshold to avoid division by zero in shadow or missing data


@dataclass
class TerrainConfig:
    """DEM and Topographic Analysis Parameters."""
    # Projection: Lunar South Pole Stereographic (IAU 2000 Moon, EPSG:3032 or custom WKT)
    projection_proj4: str = (
        "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
    )
    grid_resolution_m: float = 20.0        # 20 m per pixel for regional crater analysis
    
    # Crater region parameters (Default focus: Faustini / Shoemaker / Shackleton doubly shadowed floor)
    target_crater_name: str = "Faustini_Floor"
    center_lat: float = -87.3
    center_lon: float = 77.0
    
    # Terrain thresholds
    slope_critical_deg: float = 15.0       # Steep slopes disqualify ice stability / rover safety
    boulder_hazard_threshold: float = 0.4  # Proxy roughness threshold (0 to 1 scale)


@dataclass
class LandingSiteConfig:
    """Multi-Criteria Landing Site Evaluation."""
    max_landing_slope_deg: float = 5.0     # Lander constraint: slopes strictly < 5 degrees
    min_crater_distance_m: float = 1000.0  # Safe clearance from rim drop-off
    max_crater_distance_m: float = 8000.0  # Traverse viability range to target ice deposits
    
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
    """Rover Traverse Planning Parameters."""
    max_nav_slope_deg: float = 18.0        # Max traversable slope for rover chassis (Apollo LRV slope limit)
    speed_m_per_s: float = 0.08            # Rover traverse speed (~288 m/h, VIPER shadow dash spec)
    battery_capacity_wh: float = 1800.0    # 1.8 kWh capacity (VIPER-class polar rover)
    baseline_power_w: float = 65.0         # Avionics and thermal baseload
    traction_power_w_per_deg: float = 4.0  # Extra motor power per degree of incline
    solar_charge_rate_w: float = 110.0     # Power generation under full illumination
    max_shadow_time_hours: float = 5.0     # Hard cutoff for operating inside PSR on battery alone


@dataclass
class VolumeConfig:
    """Subsurface Ice Dielectric & Volume Estimation."""
    regolith_depth_max_m: float = 5.0      # Top ~5 m regolith column
    effective_l_depth_m: float = 3.5       # Effective radar penetration at L-band (~10-15 wavelengths in dry regolith)
    effective_s_depth_m: float = 1.2       # Effective radar penetration at S-band (~10 wavelengths)
    
    # Regolith & Ice Dielectric properties
    eps_regolith_real: float = 2.7         # Typical lunar regolith dielectric permittivity
    eps_ice_real: float = 3.15             # Water ice permittivity at ~40-100 K
    loss_tangent_dry_regolith: float = 0.004 # Tan delta of low-Ti/Fe highland anorthosite
    
    # Volume volumetric fraction bounds (Low, Nominal, High scenarios)
    ice_fraction_scenarios: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.05,                       # 5% pure equivalent pore filling
        "nominal": 0.15,                   # 15% ice-regolith breccia
        "high": 0.30                       # 30% segregated ice lenses
    })


@dataclass
class PipelineConfig:
    """Master pipeline configuration combining all modules."""
    random_seed: int = 42
    radar: RadarConfig = field(default_factory=RadarConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    landing: LandingSiteConfig = field(default_factory=LandingSiteConfig)
    rover: RoverConfig = field(default_factory=RoverConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)


# Global default configuration instance
CFG = PipelineConfig()
