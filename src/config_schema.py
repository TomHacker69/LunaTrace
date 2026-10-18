"""
LunaTrace Pydantic Configuration Schema
Validates config.yaml and enforces typing, bounds, derived fields, and structural integrity.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field, computed_field, model_validator


class PathConfig(BaseModel):
    raw_data_dir: Path
    processed_data_dir: Path
    outputs_maps_dir: Path
    outputs_tables_dir: Path
    provenance_file: Path


class CRSConfig(BaseModel):
    proj4_string: str
    expected_epsg: int = 3032


class NumericsConfig(BaseModel):
    eps: float = Field(gt=0, default=1.0e-10)
    float_precision: str = "float32"


class RadarConfig(BaseModel):
    filter_type: str = "enhanced_lee"
    filter_window_size: int = Field(gt=0, default=5)
    num_looks: int = Field(gt=0, default=4)
    min_valid_stokes1: float = Field(gt=0, default=1.0e-6)
    cpr_min_clip: float = 0.0
    cpr_max_clip: float = Field(gt=0, default=5.0)
    dop_min_clip: float = 0.0
    dop_max_clip: float = Field(gt=0, default=1.0)
    sin_2chi_min_clip: float = -1.0
    sin_2chi_max_clip: float = 1.0
    cpr_ice_threshold: float = Field(gt=0, default=1.0)
    dop_ice_max: float = Field(gt=0, default=0.32)
    cpr_rock_threshold: float = Field(gt=0, default=1.0)


class IceDetectionConfig(BaseModel):
    weights: Dict[str, float]
    roughness_penalty_scale: float = Field(ge=0, le=1, default=0.85)
    cpr_sigmoid_steepness: float = Field(gt=0, default=4.0)
    dop_sigmoid_steepness: float = Field(gt=0, default=15.0)
    delta_cpr_min_threshold: float = 0.20
    delta_cpr_scale: float = Field(gt=0, default=0.50)
    delta_cpr_clip_min: float = -1.0
    delta_cpr_clip_max: float = 3.0
    cold_trap_illum_threshold: float = Field(ge=0, default=0.10)
    cold_trap_depth_m: float = -350.0
    weight_trap_illum: float = 0.60
    weight_trap_depth: float = 0.40
    trap_depth_normalization_m: float = Field(gt=0, default=1000.0)
    hazard_slope_baseline_deg: float = 8.0
    hazard_slope_scale_deg: float = Field(gt=0, default=10.0)
    weight_rough_slope: float = 1.0
    high_prob_ice_cutoff: float = Field(ge=0, le=1, default=0.50)
    trap_confidence_cutoff: float = Field(ge=0, le=1, default=0.40)
    control_illum_min: float = 0.40
    control_min_elevation_m: float = 0.0
    control_min_slope_deg: float = 10.0
    rocky_delta_cpr_max: float = 0.25
    bg_regolith_illum_min: float = 0.30
    bg_regolith_max_slope_deg: float = 8.0
    sensitivity_cpr_thresholds: List[float] = [0.8, 1.0, 1.2]
    sensitivity_dop_thresholds: List[float] = [0.25, 0.30, 0.35]

    @model_validator(mode="after")
    def validate_weights(self):
        w_sum = sum(self.weights.values())
        if abs(w_sum - 1.0) > 1e-4:
            raise ValueError(f"Ice detection weights must sum to 1.0, got {w_sum}")
        return self


class TerrainConfig(BaseModel):
    grid_resolution_m: float = Field(gt=0, default=20.0)
    extent_km: float = Field(gt=0, default=10.0)
    target_crater_name: str = "Faustini_Floor"
    center_lat: float = -87.3
    center_lon: float = 77.0
    rms_window_size: int = Field(gt=0, default=5)
    contour_levels: List[float] = [-1500.0, -1000.0, -500.0, 0.0, 500.0]
    slope_contour_levels: List[float] = [5.0, 15.0]


class IlluminationConfig(BaseModel):
    azimuth_count: int = Field(gt=0, default=72)
    solar_elevation_min_deg: float = Field(ge=0, default=0.5)
    solar_elevation_max_deg: float = Field(gt=0, default=2.5)
    solar_elevation_steps: int = Field(gt=0, default=5)
    ray_step_distances_m: List[float]
    psr_illumination_cutoff: float = Field(ge=0, default=0.005)
    sky_view_factor_cutoff: float = Field(ge=0, le=1, default=0.86)
    deep_floor_shadow_threshold_m: float = -350.0
    earth_elevation_mean_deg: float = Field(gt=0, default=3.5)
    earth_ray_distances_m: List[float]


class LandingConfig(BaseModel):
    max_landing_slope_deg: float = Field(gt=0, default=5.0)
    min_illumination_fraction: float = Field(ge=0, le=1, default=0.20)
    min_crater_distance_m: float = Field(ge=0, default=1000.0)
    max_crater_distance_m: float = Field(gt=0, default=7500.0)
    optimal_ice_distance_m: float = Field(gt=0, default=2000.0)
    top_k_sites: int = Field(gt=0, default=3)
    spatial_separation_pixels: int = Field(gt=0, default=25)
    roughness_percentile: float = Field(gt=0, le=100, default=95.0)
    weights_without_ohrc: Dict[str, float]
    weights_with_ohrc: Dict[str, float]

    @model_validator(mode="after")
    def validate_weights(self):
        w1 = sum(self.weights_without_ohrc.values())
        w2 = sum(self.weights_with_ohrc.values())
        if abs(w1 - 1.0) > 1e-4 or abs(w2 - 1.0) > 1e-4:
            raise ValueError(f"Landing weights must sum to 1.0; got w_without={w1}, w_with={w2}")
        return self


class RoverConfig(BaseModel):
    chassis_rollover_limit_deg: float = Field(gt=0, default=22.5)
    safety_factor: float = Field(gt=1.0, default=1.25)
    speed_m_per_s: float = Field(gt=0, default=0.08)
    battery_capacity_wh: float = Field(gt=0, default=1800.0)
    baseline_power_w: float = Field(gt=0, default=65.0)
    traction_power_w_per_deg: float = Field(gt=0, default=4.0)
    solar_charge_rate_w: float = Field(gt=0, default=110.0)
    min_battery_reserve_pct: float = Field(ge=0, le=100, default=20.0)
    max_shadow_time_hours: float = Field(gt=0, default=18.0)
    slope_scale_shortest: float = 5.0
    weight_slope_shortest: float = 0.5
    weight_rough_shortest: float = 0.3
    slope_scale_safest: float = 4.0
    weight_slope_safest: float = 1.8
    weight_rough_safest: float = 3.0
    slope_scale_illuminated: float = 3.5
    weight_slope_illuminated: float = 1.2
    weight_rough_illuminated: float = 1.5
    weight_shadow_illuminated: float = 12.0

    @computed_field
    @property
    def max_nav_slope_deg(self) -> float:
        """
        Traversable slope limit strictly derived from chassis rollover limit
        divided by engineering safety factor (Rule 4).
        """
        return round(self.chassis_rollover_limit_deg / self.safety_factor, 2)


class VolumeScenario(BaseModel):
    base_fraction: float = Field(gt=0, le=1)
    eff_depth_m: float = Field(gt=0)
    dry_lag_m: float = Field(ge=0)
    active_depth_m: float = Field(gt=0)
    description: str


class VolumeConfig(BaseModel):
    regolith_depth_max_m: float = Field(gt=0, default=5.0)
    cpr_scale_range: float = Field(gt=0, default=1.0)
    cpr_norm_max_factor: float = Field(gt=0, default=1.5)
    nominal_lag_m: float = Field(ge=0, default=0.5)
    sensitivity_fractions_min: float = 0.02
    sensitivity_fractions_max: float = 0.35
    sensitivity_fractions_steps: int = 50
    sensitivity_depths: List[float] = [1.5, 2.5, 3.5, 5.0]
    scenarios: Dict[str, VolumeScenario]


class VisualizationConfig(BaseModel):
    dpi: int = 300
    scale_bar_length_km: float = 4.0
    cpr_plot_vmin: float = 0.2
    cpr_plot_vmax: float = 2.0
    dop_plot_vmin: float = 0.0
    dop_plot_vmax: float = 1.0
    delta_cpr_plot_vmin: float = -0.5
    delta_cpr_plot_vmax: float = 1.2
    max_regional_slope_plot: float = 30.0
    default_vmin_percentile: float = 2.0
    default_vmax_percentile: float = 98.0
    profile_slope_ylim: float = 16.0
    profile_soc_ylim: float = 105.0
class BoundingBoxConfig(BaseModel):
    lat_min: float = Field(ge=-90.0, le=90.0)
    lat_max: float = Field(ge=-90.0, le=90.0)
    lon_min: float = Field(ge=-180.0, le=360.0)
    lon_max: float = Field(ge=-180.0, le=360.0)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be strictly less than lat_max ({self.lat_max})")
        return self


class PDS4IngestionConfig(BaseModel):
    azimuth_looks: int = Field(gt=0, default=8)
    range_looks: int = Field(gt=0, default=2)
    target_resolution_m: float = Field(gt=0, default=20.0)
    geocoding_interpolation: str = "bilinear"
    snr_near_noise_threshold_db: float = Field(default=3.0)
    monte_carlo_trials: int = Field(gt=0, default=100000)
    monte_carlo_random_seed: int = Field(default=42)


class DEMRegistrationConfig(BaseModel):
    min_correlation_rho: float = Field(default=0.4, ge=0.0, le=1.0)


class LunaTraceConfig(BaseModel):
    paths: PathConfig
    crs: CRSConfig
    numerics: NumericsConfig
    radar: RadarConfig
    ice_detection: IceDetectionConfig
    terrain: TerrainConfig
    illumination: IlluminationConfig
    landing: LandingConfig
    rover: RoverConfig
    volume: VolumeConfig
    visualization: VisualizationConfig
    target_bounding_box: Optional[BoundingBoxConfig] = None
    pds4_ingestion: PDS4IngestionConfig = Field(default_factory=PDS4IngestionConfig)
    dem_registration: DEMRegistrationConfig = Field(default_factory=DEMRegistrationConfig)


def load_config(config_path: Optional[Path] = None) -> LunaTraceConfig:
    """
    Loads, parses, and strictly validates config.yaml using Pydantic.
    Fails fast if config is missing, malformed, or violates schema bounds.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    if not isinstance(raw_dict, dict):
        raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary mapping")

    # Resolve relative paths relative to repository root
    base_dir = config_path.parent
    paths_dict = raw_dict.get("paths", {})
    for k, v in paths_dict.items():
        p = Path(v)
        if not p.is_absolute():
            paths_dict[k] = base_dir / p
    raw_dict["paths"] = paths_dict

    # Instantiate and validate through Pydantic
    validated_config = LunaTraceConfig(**raw_dict)
    return validated_config
