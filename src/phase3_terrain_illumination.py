"""
Phase 3: Terrain and Illumination Analysis
Derives slope, aspect, RMS roughness, horizon-based solar illumination modeling,
permanently shadowed regions (PSRs), doubly shadowed zones, and Earth line-of-sight visibility.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import rasterio

from src.config import CFG
from src.config_schema import LunaTraceConfig
from src.constants import LUNAR_KM_PER_DEG_LAT
from src.utils import save_geotiff, plot_cartographic_map
from src.provenance import write_provenance_sidecar


class TerrainIlluminationEngine:
    """
    Topographic Morphometry and Solar/Earth Horizon Visibility Engine.
    Processes LOLA DEMs and computes terrain metrics and solar/Earth illumination regimes.
    """

    def __init__(self, config: LunaTraceConfig = CFG):
        self.cfg = config
        self.terrain_cfg = config.terrain
        self.illum_cfg = config.illumination
        self.landing_cfg = config.landing
        self.rover_cfg = config.rover
        self.raw_dir = config.paths.raw_data_dir
        self.processed_dir = config.paths.processed_data_dir
        self.tables_dir = config.paths.outputs_tables_dir
        self.maps_dir = config.paths.outputs_maps_dir

    def load_dem(self) -> Tuple[np.ndarray, rasterio.Affine, str, Path]:
        """Loads LOLA DEM raster."""
        dem_path = self.raw_dir / "lola_dem_20m.tif"
        if not dem_path.exists():
            raise FileNotFoundError(f"Missing required DEM file: {dem_path}")

        with rasterio.open(dem_path) as src:
            if not src.crs:
                raise ValueError(f"DEM raster {dem_path} has no CRS defined in metadata!")
            dem = src.read(1).astype(np.float64)
            transform = src.transform
            crs = src.crs.to_string()

        return dem, transform, crs, dem_path

    def compute_morphometry(
        self,
        dem: np.ndarray,
        resolution_m: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Derives slope (degrees), aspect (azimuth 0-360 deg), and RMS topographic roughness.
        """
        if resolution_m is None:
            resolution_m = self.terrain_cfg.grid_resolution_m

        dy, dx = np.gradient(dem, resolution_m, resolution_m)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        slope_deg = np.degrees(slope_rad)

        # Aspect: direction of steepest descent (0 to 360 degrees clockwise from North)
        aspect_rad = np.arctan2(-dx, dy)
        aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0

        # Topographic RMS Roughness over a moving window
        mean_elev = uniform_filter(dem, size=self.terrain_cfg.rms_window_size)
        mean_sq_elev = uniform_filter(dem ** 2, size=self.terrain_cfg.rms_window_size)
        rms_roughness = np.sqrt(np.maximum(mean_sq_elev - mean_elev ** 2, 0.0))

        # Topographic Curvature (Laplacian)
        d2y, _ = np.gradient(dy, resolution_m, resolution_m)
        _, d2x = np.gradient(dx, resolution_m, resolution_m)
        curvature = d2x + d2y

        return {
            "slope_deg": slope_deg.astype(np.float32),
            "aspect_deg": aspect_deg.astype(np.float32),
            "rms_roughness": rms_roughness.astype(np.float32),
            "curvature": curvature.astype(np.float32)
        }

    def compute_illumination_and_psr(
        self,
        dem: np.ndarray,
        resolution_m: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulates lunar South Pole solar illumination using a multi-angle horizon ray-tracing model.
        Samples azimuthal directions (e.g. 72 azimuths) across solar elevation ranges over the lunar cycle.
        Illumination fraction is purely computed from terrain line-of-sight without artificial capping.
        """
        if resolution_m is None:
            resolution_m = self.terrain_cfg.grid_resolution_m

        ny, nx = dem.shape
        num_azimuths = self.illum_cfg.azimuth_count
        azimuth_angles = np.linspace(0, 2 * np.pi, num_azimuths, endpoint=False)
        horizon_elevations = np.zeros((num_azimuths, ny, nx), dtype=np.float32)

        step_distances = np.array(self.illum_cfg.ray_step_distances_m, dtype=float)

        for i, az in enumerate(azimuth_angles):
            dx_step = np.cos(az)
            dy_step = np.sin(az)
            max_tan_angle = np.full_like(dem, -999.0)

            for dist in step_distances:
                px_offset = int(np.round((dist * dx_step) / resolution_m))
                py_offset = int(np.round((dist * dy_step) / resolution_m))

                dem_shifted = np.roll(np.roll(dem, py_offset, axis=0), px_offset, axis=1)
                elev_diff = dem_shifted - dem
                tan_angle = elev_diff / dist
                max_tan_angle = np.maximum(max_tan_angle, tan_angle)

            horizon_elevations[i] = np.degrees(np.arctan(np.maximum(max_tan_angle, -1.0)))

        # Solar elevation range over lunar precession cycle
        solar_elevations = np.linspace(
            self.illum_cfg.solar_elevation_min_deg,
            self.illum_cfg.solar_elevation_max_deg,
            self.illum_cfg.solar_elevation_steps
        )

        # Average unobstructed fraction across all sampled solar elevations and azimuth angles
        total_evaluations = len(solar_elevations) * num_azimuths
        illum_accumulator = np.zeros((ny, nx), dtype=np.float64)

        for s_elev in solar_elevations:
            unobstructed = (horizon_elevations < s_elev).astype(float)
            illum_accumulator += np.sum(unobstructed, axis=0)

        illumination_fraction = (illum_accumulator / total_evaluations).astype(np.float32)

        # Floor regions below deep shadow threshold receive zero direct sunlight
        illumination_fraction[dem < self.illum_cfg.deep_floor_shadow_threshold_m] = 0.0

        # Permanently Shadowed Region (PSR): strict illumination cutoff
        psr_mask = (illumination_fraction < self.illum_cfg.psr_illumination_cutoff).astype(np.uint8)

        # Sky-View Factor (SVF): Fraction of visible celestial hemisphere (0 to 1)
        mean_horizon_rad = np.radians(np.clip(np.mean(horizon_elevations, axis=0), 0.0, 89.0))
        sky_view_factor = np.clip(np.cos(mean_horizon_rad) ** 2, 0.0, 1.0).astype(np.float32)

        # Doubly Shadowed Zones: Deepest interior floor with minimal secondary scatter
        doubly_shadowed = (
            (psr_mask > 0) &
            (sky_view_factor < self.illum_cfg.sky_view_factor_cutoff) &
            (dem < self.illum_cfg.deep_floor_shadow_threshold_m)
        ).astype(np.uint8)

        return {
            "illumination_fraction": illumination_fraction,
            "psr_mask": psr_mask,
            "sky_view_factor": sky_view_factor,
            "doubly_shadowed": doubly_shadowed
        }

    def compute_earth_visibility(
        self,
        dem: np.ndarray,
        resolution_m: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes Earth Line-of-Sight (LOS) visibility fraction over the lunar libration cycle.
        """
        if resolution_m is None:
            resolution_m = self.terrain_cfg.grid_resolution_m

        step_distances = np.array(self.illum_cfg.earth_ray_distances_m, dtype=float)
        max_north_tan = np.full_like(dem, -999.0)

        for dist in step_distances:
            py_offset = int(np.round(dist / resolution_m))
            dem_shifted = np.roll(dem, -py_offset, axis=0)
            elev_diff = dem_shifted - dem
            max_north_tan = np.maximum(max_north_tan, elev_diff / dist)

        north_horizon_deg = np.degrees(np.arctan(np.maximum(max_north_tan, -1.0)))

        earth_elevation_mean = self.illum_cfg.earth_elevation_mean_deg
        earth_los = np.clip((earth_elevation_mean - north_horizon_deg) / earth_elevation_mean, 0.0, 1.0)
        earth_los[dem < self.illum_cfg.deep_floor_shadow_threshold_m] = 0.0

        return earth_los.astype(np.float32)

    def process_and_export(self) -> Dict[str, Any]:
        """Runs Phase 3 processing, exports GeoTIFFs, cartographic maps, and summary statistics."""
        dem, transform, crs, dem_path = self.load_dem()
        input_paths = [dem_path]

        # 1. Morphometry
        morph = self.compute_morphometry(dem, resolution_m=self.terrain_cfg.grid_resolution_m)
        slope = morph["slope_deg"]
        aspect = morph["aspect_deg"]
        roughness = morph["rms_roughness"]

        # 2. Illumination, PSR, and Doubly Shadowed Zones
        illum_dict = self.compute_illumination_and_psr(dem, resolution_m=self.terrain_cfg.grid_resolution_m)
        illum = illum_dict["illumination_fraction"]
        psr = illum_dict["psr_mask"]
        doubly_shadowed = illum_dict["doubly_shadowed"]
        svf = illum_dict["sky_view_factor"]

        # 3. Earth Visibility
        earth_los = self.compute_earth_visibility(dem, resolution_m=self.terrain_cfg.grid_resolution_m)

        # Save Rasters with Provenance Sidecars
        raster_outputs = {
            "terrain_slope_deg.tif": slope,
            "terrain_aspect_deg.tif": aspect,
            "terrain_rms_roughness_m.tif": roughness,
            "terrain_illumination_fraction.tif": illum,
            "terrain_psr_mask.tif": psr,
            "terrain_doubly_shadowed_mask.tif": doubly_shadowed,
            "terrain_earth_los_visibility.tif": earth_los
        }

        for fname, arr in raster_outputs.items():
            out_p = self.processed_dir / fname
            save_geotiff(arr, out_p, transform, crs=crs)
            write_provenance_sidecar(out_p, input_paths, self.cfg.model_dump(), {"raster": fname})

        # Cartographic Map 1: Topographic Slope with Critical 5 deg & 15 deg Contours
        map_slope = plot_cartographic_map(
            data=slope,
            title="LOLA Topographic Slope (Degrees) with Hazard Thresholds",
            output_filename="map_phase3_slope.png",
            cmap="magma",
            vmin=0.0,
            vmax=self.cfg.visualization.max_regional_slope_plot,
            cbar_label="Slope (Degrees)",
            overlay_contour=slope,
            contour_levels=self.terrain_cfg.slope_contour_levels
        )
        write_provenance_sidecar(map_slope, input_paths, self.cfg.model_dump(), {"map": "slope"})

        # Cartographic Map 2: Solar Illumination Fraction & PSR Outlines
        map_illum = plot_cartographic_map(
            data=illum,
            title="Solar Illumination Fraction & Permanently Shadowed Regions (PSR)",
            output_filename="map_phase3_illumination_psr.png",
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Annual Average Illumination Fraction",
            overlay_contour=dem,
            contour_levels=self.terrain_cfg.contour_levels
        )
        write_provenance_sidecar(map_illum, input_paths, self.cfg.model_dump(), {"map": "illumination"})

        # Cartographic Map 3: Earth Line-of-Sight Visibility
        map_earth = plot_cartographic_map(
            data=earth_los,
            title="Earth Direct Line-of-Sight Visibility (Lunar Libration Cycle)",
            output_filename="map_phase3_earth_visibility.png",
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Earth LOS Visibility Fraction (0 = Occluded, 1 = Continuous Comms)"
        )
        write_provenance_sidecar(map_earth, input_paths, self.cfg.model_dump(), {"map": "earth_los"})

        # Cartographic Map 4: Doubly Shadowed Core Cold-Trap Map
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(9, 8), dpi=self.cfg.visualization.dpi)
        shadow_zones = np.zeros_like(psr, dtype=np.uint8)
        shadow_zones[psr > 0] = 1
        shadow_zones[doubly_shadowed > 0] = 2

        zone_cmap = ListedColormap(["#dcdcdc", "#3182bd", "#081d58"])
        ext = self.terrain_cfg.extent_km
        ax.imshow(shadow_zones, extent=[-ext, ext, -ext, ext], origin="upper", cmap=zone_cmap)
        ax.set_title("Permanently Shadowed Regions (PSR) & Doubly Shadowed Zones", fontsize=12, fontweight="bold")
        ax.set_xlabel("East-West Distance (km)")
        ax.set_ylabel("North-South Distance (km)")
        ax.grid(color="gray", linestyle="--", alpha=0.3)

        legend_elements = [
            Patch(facecolor="#dcdcdc", label="Sunlit / Intermittent Rim"),
            Patch(facecolor="#3182bd", label="Permanently Shadowed Region (PSR)"),
            Patch(facecolor="#081d58", label="Doubly Shadowed Core (<110 K Cold Trap)")
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
        map_shadow_zones = self.maps_dir / "map_phase3_shadow_zones.png"
        fig.tight_layout()
        fig.savefig(map_shadow_zones, bbox_inches="tight")
        plt.close(fig)
        write_provenance_sidecar(map_shadow_zones, input_paths, self.cfg.model_dump(), {"map": "shadow_zones"})

        # Compile Summary Table
        pixel_area_km2 = (self.terrain_cfg.grid_resolution_m / 1000.0) ** 2
        total_area_km2 = dem.size * pixel_area_km2

        max_land_slope = self.landing_cfg.max_landing_slope_deg
        max_rov_slope = self.rover_cfg.max_nav_slope_deg

        summary_data = [
            {"Metric": "Total Analyzed Regional Area", "Value": f"{total_area_km2:.1f} km²"},
            {"Metric": "Permanently Shadowed Region (PSR) Area", "Value": f"{np.sum(psr > 0) * pixel_area_km2:.2f} km² ({100.0 * np.sum(psr > 0) / dem.size:.1f}%)"},
            {"Metric": "Doubly Shadowed Core Zone Area", "Value": f"{np.sum(doubly_shadowed > 0) * pixel_area_km2:.2f} km² ({100.0 * np.sum(doubly_shadowed > 0) / dem.size:.1f}%)"},
            {"Metric": "Mean Regional Slope", "Value": f"{np.mean(slope):.2f}° (Std: {np.std(slope):.2f}°)"},
            {"Metric": f"Safe Landing Terrain Area (Slope < {max_land_slope:.0f}°)", "Value": f"{np.sum(slope < max_land_slope) * pixel_area_km2:.2f} km² ({100.0 * np.sum(slope < max_land_slope) / dem.size:.1f}%)"},
            {"Metric": f"Traversable Rover Area (Slope < {max_rov_slope:.0f}°)", "Value": f"{np.sum(slope < max_rov_slope) * pixel_area_km2:.2f} km² ({100.0 * np.sum(slope < max_rov_slope) / dem.size:.1f}%)"},
            {"Metric": "High Illumination Rim Area (Fraction > 0.6)", "Value": f"{np.sum(illum > 0.6) * pixel_area_km2:.2f} km² ({100.0 * np.sum(illum > 0.6) / dem.size:.1f}%)"},
            {"Metric": "Direct Earth Visibility Area (LOS > 0.5)", "Value": f"{np.sum(earth_los > 0.5) * pixel_area_km2:.2f} km² ({100.0 * np.sum(earth_los > 0.5) / dem.size:.1f}%)"}
        ]
        df_summary = pd.DataFrame(summary_data)
        summary_csv = self.tables_dir / "phase3_terrain_illumination_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        write_provenance_sidecar(summary_csv, input_paths, self.cfg.model_dump(), {"table": "terrain_summary"})

        return {
            "summary": df_summary,
            "maps": [map_slope, map_illum, map_earth, map_shadow_zones]
        }


def run_phase3(config: LunaTraceConfig = CFG):
    print("[*] Running Phase 3: Terrain and Illumination Analysis...")
    engine = TerrainIlluminationEngine(config=config)
    results = engine.process_and_export()
    print("[+] Phase 3 completed successfully!")
    print("\n--- Terrain and Illumination Summary ---")
    print(results["summary"].to_string(index=False))
    return results
