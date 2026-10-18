"""
Phase 4: Landing Site Selection
Weighted Multi-Criteria Evaluation (MCE), engineering safety constraints,
ranking of candidate landing sites with dynamic justification and cryptographic provenance sidecars.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import rasterio

from src.config import CFG
from src.config_schema import LunaTraceConfig
from src.constants import LUNAR_KM_PER_DEG_LAT
from src.utils import save_geotiff, plot_cartographic_map
from src.provenance import write_provenance_sidecar


class LandingSiteSelector:
    """
    Landing Site Multi-Criteria Evaluation (MCE) Engine.
    Combines slope, roughness, illumination, Earth LOS, and distance to ice deposits
    under engineering landing safety constraints.
    """

    def __init__(self, config: LunaTraceConfig = CFG):
        self.cfg = config
        self.landing_cfg = config.landing
        self.terrain_cfg = config.terrain
        self.raw_dir = config.paths.raw_data_dir
        self.processed_dir = config.paths.processed_data_dir
        self.tables_dir = config.paths.outputs_tables_dir
        self.maps_dir = config.paths.outputs_maps_dir

    def load_environmental_data(self) -> Tuple[Dict[str, Any], rasterio.Affine, str, List[Path]]:
        """Loads processed terrain, radar ice probability, and illumination rasters."""
        data = {}
        meta_transform = None
        meta_crs = None
        loaded_paths = []

        paths = {
            "dem": self.raw_dir / "lola_dem_20m.tif",
            "slope": self.processed_dir / "terrain_slope_deg.tif",
            "roughness": self.processed_dir / "terrain_rms_roughness_m.tif",
            "illumination": self.processed_dir / "terrain_illumination_fraction.tif",
            "earth_los": self.processed_dir / "terrain_earth_los_visibility.tif",
            "psr_mask": self.processed_dir / "terrain_psr_mask.tif",
            "ice_mask": self.processed_dir / "ice_candidate_mask.tif",
            "ice_prob": self.processed_dir / "ice_probability_map.tif"
        }

        for k, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required raster for Phase 4: {p}")
            loaded_paths.append(p)
            with rasterio.open(p) as src:
                if not src.crs:
                    raise ValueError(f"Raster {p} has no CRS defined in metadata!")
                data[k] = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string()

        # Check for optional OHRC boulder raster
        ohrc_path = self.raw_dir / "ohrc_boulder_density.tif"
        if ohrc_path.exists():
            with rasterio.open(ohrc_path) as src:
                data["boulders"] = src.read(1).astype(np.float64)
            data["boulder_status"] = "ASSESSED (Chandrayaan-2 OHRC)"
            loaded_paths.append(ohrc_path)
        else:
            data["boulders"] = None
            data["boulder_status"] = "UNASSESSED (OHRC not provided)"

        return data, meta_transform, meta_crs, loaded_paths

    def compute_distance_to_ice(
        self,
        ice_mask: np.ndarray,
        resolution_m: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes Euclidean distance in meters from every pixel to the nearest high-probability ice deposit.
        """
        if resolution_m is None:
            resolution_m = self.terrain_cfg.grid_resolution_m

        inverted_mask = (ice_mask == 0).astype(np.uint8)
        dist_pixels = distance_transform_edt(inverted_mask)
        dist_meters = dist_pixels * resolution_m
        return dist_meters.astype(np.float32)

    def evaluate_suitability(
        self,
        data: Dict[str, Any],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes Weighted Multi-Criteria Evaluation (MCE) with hard safety constraints.
        If OHRC boulder data is missing, boulder-based scoring is explicitly disabled,
        and weights renormalize over slope, roughness, illumination, and Earth LOS.
        """
        slope = data["slope"]
        roughness = data["roughness"]
        illum = data["illumination"]
        earth_los = data["earth_los"]
        ice_mask = data["ice_mask"]

        dist_to_ice = self.compute_distance_to_ice(ice_mask, resolution_m=self.terrain_cfg.grid_resolution_m)

        # 1. Hard Engineering Safety Constraints
        valid_slope = slope <= self.landing_cfg.max_landing_slope_deg
        valid_illum = illum > self.landing_cfg.min_illumination_fraction
        valid_distance = (
            (dist_to_ice >= self.landing_cfg.min_crater_distance_m) &
            (dist_to_ice <= self.landing_cfg.max_crater_distance_m)
        )

        if data["boulders"] is not None:
            valid_boulders = data["boulders"] < 0.30
            safe_landing_mask = valid_slope & valid_boulders & valid_illum & valid_distance
        else:
            # Boulder constraint is unassessed without OHRC; safe mask strictly uses verified terrain constraints
            safe_landing_mask = valid_slope & valid_illum & valid_distance

        # 2. Normalized Criterion Scores (0 to 1, higher is better)
        c_slope = np.clip(1.0 - (slope / self.landing_cfg.max_landing_slope_deg), 0.0, 1.0)

        eps = self.cfg.numerics.eps
        pct = self.landing_cfg.roughness_percentile
        max_r = np.percentile(roughness, pct) + eps
        c_roughness = np.clip(1.0 - (roughness / max_r), 0.0, 1.0)

        c_illum = np.clip(illum, 0.0, 1.0)
        c_earth = np.clip(earth_los, 0.0, 1.0)

        # Proximity score
        opt_dist = self.landing_cfg.optimal_ice_distance_m
        dist_range = self.landing_cfg.max_crater_distance_m - self.landing_cfg.min_crater_distance_m
        c_ice_dist = np.clip(1.0 - np.abs(dist_to_ice - opt_dist) / dist_range, 0.0, 1.0)

        # 3. Weighted Multi-Criteria Overlay (MCO)
        if custom_weights is not None:
            weights = custom_weights
        elif data["boulders"] is not None:
            weights = self.landing_cfg.weights_with_ohrc
        else:
            weights = self.landing_cfg.weights_without_ohrc

        w_sum = sum(weights.values())
        norm_weights = {k: v / w_sum for k, v in weights.items()}

        raw_score = (
            norm_weights["slope"] * c_slope +
            norm_weights["roughness"] * c_roughness +
            norm_weights["illumination"] * c_illum +
            norm_weights["earth_los"] * c_earth
        )

        if "boulders" in norm_weights and data["boulders"] is not None:
            c_boulders = np.clip(1.0 - (data["boulders"] / 0.30), 0.0, 1.0)
            raw_score += norm_weights["boulders"] * c_boulders

        suitability_score = np.where(safe_landing_mask, raw_score, 0.0)
        return suitability_score.astype(np.float32), safe_landing_mask.astype(np.uint8), dist_to_ice

    def select_top_sites(
        self,
        suitability: np.ndarray,
        data: Dict[str, Any],
        dist_to_ice: np.ndarray,
        transform: rasterio.Affine,
        top_k: Optional[int] = None,
        min_separation_pixels: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extracts top-ranked candidate landing sites with spatial separation and dynamic justification.
        """
        if top_k is None:
            top_k = self.landing_cfg.top_k_sites
        if min_separation_pixels is None:
            min_separation_pixels = self.landing_cfg.spatial_separation_pixels

        res = self.terrain_cfg.grid_resolution_m
        ny, nx = suitability.shape
        half_w = (nx * res) / 2.0

        score_copy = suitability.copy()
        sites = []

        for rank in range(1, top_k + 1):
            max_idx = np.argmax(score_copy)
            py, px = np.unravel_index(max_idx, score_copy.shape)
            best_score = float(score_copy[py, px])

            if best_score <= 0.0:
                break

            x_km = (-half_w + px * res) / 1000.0
            y_km = (half_w - py * res) / 1000.0

            km_deg = LUNAR_KM_PER_DEG_LAT
            lat = self.terrain_cfg.center_lat + (y_km / km_deg)
            lon = self.terrain_cfg.center_lon + (x_km / (km_deg * np.cos(np.radians(abs(lat)))))

            slp = float(data["slope"][py, px])
            rgh = float(data["roughness"][py, px])
            ilm = float(data["illumination"][py, px])
            los = float(data["earth_los"][py, px])
            dst = float(dist_to_ice[py, px])
            elv = float(data["dem"][py, px])

            # Dynamic justification generated strictly from computed values (Rule 4)
            justification = (
                f"Rank {rank} candidate: slope = {slp:.2f} deg, "
                f"solar illumination = {ilm * 100:.1f}%, "
                f"Earth LOS = {los * 100:.1f}%, "
                f"standoff to ice target = {dst:.0f} m, "
                f"elevation = {elv:.1f} m. "
                f"Boulder hazard: {data['boulder_status']}."
            )

            site_info = {
                "Rank": rank,
                "Site_Name": f"Site_{rank}",
                "Pixel_Y": int(py),
                "Pixel_X": int(px),
                "X_km": round(float(x_km), 2),
                "Y_km": round(float(y_km), 2),
                "Latitude": round(float(lat), 4),
                "Longitude": round(float(lon), 4),
                "Composite_Score": round(best_score, 4),
                "Slope_deg": round(slp, 2),
                "RMS_Roughness_m": round(rgh, 2),
                "Illumination_Fraction": round(ilm, 3),
                "Earth_LOS_Fraction": round(los, 3),
                "Distance_to_Ice_m": round(dst, 1),
                "Elevation_m": round(elv, 1),
                "Boulder_Hazard_Status": data["boulder_status"],
                "Justification": justification
            }
            sites.append(site_info)

            # Spatial separation suppression
            y_min, y_max = max(0, py - min_separation_pixels), min(ny, py + min_separation_pixels + 1)
            x_min, x_max = max(0, px - min_separation_pixels), min(nx, px + min_separation_pixels + 1)
            score_copy[y_min:y_max, x_min:x_max] = 0.0

        return pd.DataFrame(sites)

    def process_and_export(self) -> Dict[str, Any]:
        """Runs Phase 4 landing site evaluation and exports maps and tables with provenance sidecars."""
        data, transform, crs, input_paths = self.load_environmental_data()

        # Evaluate suitability
        suitability, safe_mask, dist_to_ice = self.evaluate_suitability(data)

        # Extract top 3 sites
        df_sites = self.select_top_sites(suitability, data, dist_to_ice, transform, top_k=self.landing_cfg.top_k_sites)

        # Save rasters with provenance sidecars
        suit_path = self.processed_dir / "landing_suitability_score.tif"
        mask_path = self.processed_dir / "landing_candidate_mask.tif"
        save_geotiff(suitability, suit_path, transform, crs=crs)
        save_geotiff(safe_mask, mask_path, transform, crs=crs)

        write_provenance_sidecar(suit_path, input_paths, self.cfg.model_dump(), {"boulder_status": data["boulder_status"]})
        write_provenance_sidecar(mask_path, input_paths, self.cfg.model_dump(), {"boulder_status": data["boulder_status"]})

        # Export CSV table
        table_path = self.tables_dir / "phase4_top3_landing_sites.csv"
        df_sites.to_csv(table_path, index=False)
        write_provenance_sidecar(table_path, input_paths, self.cfg.model_dump(), {"boulder_status": data["boulder_status"]})

        # Cartographic Map
        points_to_plot = [
            (row["X_km"], row["Y_km"], f"{row['Site_Name']} (Rank {row['Rank']})", "*", "gold" if row["Rank"] == 1 else "cyan")
            for _, row in df_sites.iterrows()
        ]

        ice_mask = data["ice_mask"] > 0
        res = self.terrain_cfg.grid_resolution_m
        ny, nx = ice_mask.shape
        half_w = (nx * res) / 2.0
        ice_y, ice_x = np.where(ice_mask)
        if len(ice_y) > 0:
            ice_x_km = (-half_w + np.mean(ice_x) * res) / 1000.0
            ice_y_km = (half_w - np.mean(ice_y) * res) / 1000.0
            points_to_plot.append((ice_x_km, ice_y_km, "Target Ice Deposit", "X", "#00d2ff"))

        map_landing = plot_cartographic_map(
            data=suitability,
            title="Landing Site Suitability Overlay & Candidate Sites",
            output_filename="map_phase4_landing_sites.png",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Multi-Criteria Landing Suitability Score (0 to 1)",
            overlay_contour=data["dem"],
            contour_levels=self.terrain_cfg.contour_levels,
            points_to_plot=points_to_plot
        )
        write_provenance_sidecar(map_landing, input_paths, self.cfg.model_dump(), {"map": "landing_sites"})

        return {
            "df_sites": df_sites,
            "map_path": map_landing,
            "suitability": suitability,
            "boulder_status": data["boulder_status"]
        }


def run_phase4(config: LunaTraceConfig = CFG):
    print("[*] Running Phase 4: Landing Site Multi-Criteria Selection...")
    selector = LandingSiteSelector(config=config)
    results = selector.process_and_export()
    print(f"[+] Boulder Hazard Status: {results['boulder_status']}")
    print("[+] Phase 4 completed successfully!")
    print("\n--- Top 3 Ranked Landing Sites ---")
    cols_to_print = ["Rank", "Site_Name", "X_km", "Y_km", "Composite_Score", "Slope_deg", "Illumination_Fraction", "Earth_LOS_Fraction", "Distance_to_Ice_m", "Boulder_Hazard_Status"]
    print(results["df_sites"][cols_to_print].to_string(index=False))
    print("\n--- Candidate Justification ---")
    for _, row in results["df_sites"].iterrows():
        print(f"[{row['Site_Name']}]: {row['Justification']}")
    return results
