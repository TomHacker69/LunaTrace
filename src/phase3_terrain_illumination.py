"""
Phase 3: Terrain and Illumination Analysis
Derives slope, aspect, RMS roughness, horizon-based illumination modeling,
permanently shadowed regions (PSRs), doubly shadowed zones, and Earth line-of-sight visibility.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, generic_filter
import rasterio

from src.config import CFG, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, MAPS_DIR
from src.utils import save_geotiff, plot_cartographic_map


class TerrainIlluminationEngine:
    """
    Topographic Morphometry and Solar/Earth Horizon Visibility Engine.
    Processes LOLA DEMs and computes terrain metrics and solar/Earth illumination regimes.
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.terrain_cfg = config.terrain
        self.landing_cfg = config.landing
        self.rover_cfg = config.rover

    def load_dem(self) -> Tuple[np.ndarray, rasterio.Affine, str]:
        """Loads LOLA DEM raster."""
        dem_path = RAW_DATA_DIR / "lola_dem_20m.tif"
        if not dem_path.exists():
            raise FileNotFoundError(f"Missing DEM file: {dem_path}")

        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4

        return dem, transform, crs

    def compute_morphometry(
        self,
        dem: np.ndarray,
        resolution_m: float = 20.0
    ) -> Dict[str, np.ndarray]:
        """
        Derives slope (degrees), aspect (azimuth 0-360 deg), and RMS topographic roughness.
        """
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

        # Topographic Curvature (Laplacian: second derivative for convex rims / concave bowls)
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
        resolution_m: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulates lunar South Pole solar illumination using azimuthal horizon ray-tracing.
        Models grazing solar elevation angles over the lunar precession cycle.
        Identifies Permanently Shadowed Regions (PSRs) and Doubly Shadowed Core Zones.
        """
        if resolution_m is None:
            resolution_m = self.terrain_cfg.grid_resolution_m

        ny, nx = dem.shape
        half_w = (nx * resolution_m) / 2.0
        x = np.linspace(-half_w, half_w, nx)
        y = np.linspace(half_w, -half_w, ny)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx ** 2 + yy ** 2)

        # Multi-angle horizon profiling across azimuth directions
        num_azimuths = self.terrain_cfg.num_horizon_azimuths
        azimuth_angles = np.linspace(0, 2 * np.pi, num_azimuths, endpoint=False)
        horizon_elevations = np.zeros((num_azimuths, ny, nx), dtype=np.float32)

        # Ray-march radially outward to determine topographic horizon angle
        step_distances = np.array(self.terrain_cfg.ray_step_distances_m, dtype=float)
        
        for i, az in enumerate(azimuth_angles):
            dx_step = np.cos(az)
            dy_step = np.sin(az)
            max_tan_angle = np.full_like(dem, -999.0)

            for dist in step_distances:
                px_offset = int(np.round((dist * dx_step) / resolution_m))
                py_offset = int(np.round((dist * dy_step) / resolution_m))

                # Shifted terrain grid
                dem_shifted = np.roll(np.roll(dem, py_offset, axis=0), px_offset, axis=1)
                elev_diff = dem_shifted - dem
                tan_angle = elev_diff / dist
                max_tan_angle = np.maximum(max_tan_angle, tan_angle)

            horizon_elevations[i] = np.degrees(np.arctan(np.maximum(max_tan_angle, -1.0)))

        # Average illumination fraction: percentage of azimuths where the horizon is below grazing sun
        unobstructed = horizon_elevations < self.terrain_cfg.solar_elevation_grazing_deg
        illumination_fraction = np.mean(unobstructed.astype(float), axis=0)

        # Crater floor masking: deep depressions surrounded by high rims receive zero direct light
        crater_mask = dem < self.terrain_cfg.crater_depression_threshold_m
        illumination_fraction[crater_mask] = np.clip(illumination_fraction[crater_mask] * 0.1, 0.0, 1.0)
        illumination_fraction[dem < self.terrain_cfg.deep_floor_shadow_threshold_m] = 0.0  # Deep floor is totally dark

        # Permanently Shadowed Region (PSR): Illumination fraction is strictly below cutoff
        psr_mask = illumination_fraction < self.terrain_cfg.psr_illumination_cutoff

        # Sky-View Factor (SVF): Fraction of visible celestial hemisphere (0 to 1)
        mean_horizon_rad = np.radians(np.clip(np.mean(horizon_elevations, axis=0), 0.0, 89.0))
        sky_view_factor = np.clip(np.cos(mean_horizon_rad) ** 2, 0.0, 1.0)

        # Doubly Shadowed Zones: Deepest interior floor
        doubly_shadowed = psr_mask & (sky_view_factor < self.terrain_cfg.sky_view_factor_cutoff) & (dem < self.terrain_cfg.deep_floor_shadow_threshold_m)

        return {
            "illumination_fraction": illumination_fraction.astype(np.float32),
            "psr_mask": psr_mask.astype(np.uint8),
            "sky_view_factor": sky_view_factor.astype(np.float32),
            "doubly_shadowed": doubly_shadowed.astype(np.uint8)
        }

    def compute_earth_visibility(
        self,
        dem: np.ndarray,
        resolution_m: float = None
    ) -> np.ndarray:
        """
        Computes Earth Line-of-Sight (LOS) visibility fraction over the lunar libration cycle.
        From the lunar south pole, Earth oscillates between 0 deg and ~7 deg above the northern horizon.
        """
        if resolution_m is None:
            resolution_m = self.terrain_cfg.grid_resolution_m

        ny, nx = dem.shape
        # Northern horizon ray casting (towards +Y direction, azimuth ~ 0 deg)
        step_distances = np.array(self.terrain_cfg.earth_ray_distances_m, dtype=float)
        max_north_tan = np.full_like(dem, -999.0)

        for dist in step_distances:
            py_offset = int(np.round(dist / resolution_m))
            dem_shifted = np.roll(dem, -py_offset, axis=0)  # Looking North (+Y)
            elev_diff = dem_shifted - dem
            max_north_tan = np.maximum(max_north_tan, elev_diff / dist)

        north_horizon_deg = np.degrees(np.arctan(np.maximum(max_north_tan, -1.0)))

        # Earth libration model
        earth_elevation_mean = self.terrain_cfg.earth_elevation_mean_deg
        earth_los = np.clip((earth_elevation_mean - north_horizon_deg) / earth_elevation_mean, 0.0, 1.0)
        # Crater bowl floor has zero line of sight to Earth due to tall northern crater walls
        earth_los[dem < self.terrain_cfg.crater_depression_threshold_m] = 0.0

        return earth_los.astype(np.float32)

    def process_and_export(self) -> Dict[str, Any]:
        """Runs Phase 3 processing, exports GeoTIFFs, cartographic maps, and summary statistics."""
        dem, transform, crs = self.load_dem()

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

        # Save Rasters
        save_geotiff(slope, PROCESSED_DATA_DIR / "terrain_slope_deg.tif", transform, crs=crs)
        save_geotiff(aspect, PROCESSED_DATA_DIR / "terrain_aspect_deg.tif", transform, crs=crs)
        save_geotiff(roughness, PROCESSED_DATA_DIR / "terrain_rms_roughness_m.tif", transform, crs=crs)
        save_geotiff(illum, PROCESSED_DATA_DIR / "terrain_illumination_fraction.tif", transform, crs=crs)
        save_geotiff(psr, PROCESSED_DATA_DIR / "terrain_psr_mask.tif", transform, crs=crs)
        save_geotiff(doubly_shadowed, PROCESSED_DATA_DIR / "terrain_doubly_shadowed_mask.tif", transform, crs=crs)
        save_geotiff(earth_los, PROCESSED_DATA_DIR / "terrain_earth_los_visibility.tif", transform, crs=crs)

        # Cartographic Map 1: Topographic Slope with Critical 5 deg & 15 deg Contours
        map_slope = plot_cartographic_map(
            data=slope,
            title="LOLA Topographic Slope (Degrees) with Hazard Thresholds",
            output_filename="map_phase3_slope.png",
            cmap="magma",
            vmin=0.0,
            vmax=self.terrain_cfg.max_regional_slope_plot,
            cbar_label="Slope (Degrees)",
            overlay_contour=slope,
            contour_levels=self.terrain_cfg.slope_contour_levels
        )

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

        # Cartographic Map 3: Earth Line-of-Sight Visibility for Direct Communications
        map_earth = plot_cartographic_map(
            data=earth_los,
            title="Earth Direct Line-of-Sight Visibility (Lunar Libration Cycle)",
            output_filename="map_phase3_earth_visibility.png",
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Earth LOS Visibility Fraction (0 = Occluded, 1 = Continuous Comms)"
        )

        # Cartographic Map 4: Doubly Shadowed Core Cold-Trap Map
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
        # 0: Sunlit/Intermittent, 1: Singly Shadowed PSR, 2: Doubly Shadowed Core
        shadow_zones = np.zeros_like(psr, dtype=np.uint8)
        shadow_zones[psr > 0] = 1
        shadow_zones[doubly_shadowed > 0] = 2

        zone_cmap = ListedColormap(["#dcdcdc", "#3182bd", "#081d58"])  # Light gray, medium blue, dark navy
        ext = self.terrain_cfg.extent_km
        ax.imshow(shadow_zones, extent=[-ext, ext, -ext, ext], origin="upper", cmap=zone_cmap)
        ax.set_title("Permanently Shadowed Regions (PSR) & Doubly Shadowed Zones", fontsize=12, fontweight="bold")
        ax.set_xlabel("East-West Distance (km)")
        ax.set_ylabel("North-South Distance (km)")
        ax.grid(color="gray", linestyle="--", alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#dcdcdc", label="Sunlit / Intermittent Rim"),
            Patch(facecolor="#3182bd", label="Permanently Shadowed Region (PSR)"),
            Patch(facecolor="#081d58", label="Doubly Shadowed Core (<110 K Cold Trap)")
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
        map_shadow_zones = MAPS_DIR / "map_phase3_shadow_zones.png"
        fig.tight_layout()
        fig.savefig(map_shadow_zones, bbox_inches="tight")
        plt.close(fig)

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
        summary_csv = TABLES_DIR / "phase3_terrain_illumination_summary.csv"
        df_summary.to_csv(summary_csv, index=False)

        return {
            "summary": df_summary,
            "maps": [map_slope, map_illum, map_earth, map_shadow_zones]
        }


def run_phase3():
    print("[*] Running Phase 3: Terrain and Illumination Analysis...")
    engine = TerrainIlluminationEngine()
    results = engine.process_and_export()
    print("[+] Phase 3 completed successfully!")
    print("\n--- Terrain and Illumination Summary ---")
    print(results["summary"].to_string(index=False))
    return results


if __name__ == "__main__":
    run_phase3()
