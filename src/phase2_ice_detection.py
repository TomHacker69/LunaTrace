"""
Phase 2: Subsurface Ice Detection Framework
Multi-discriminator scoring, CPR ambiguity resolution (ice vs. rocky ejecta),
dual-frequency depth profiling (Delta CPR), and sensitivity analysis.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import rasterio

from src.config import CFG, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, MAPS_DIR
from src.utils import save_geotiff, plot_cartographic_map


class IceDetectionEngine:
    """
    Quantitative Lunar Ice Detection and Characterization Engine.
    Combines radar polarimetry (L and S band), topographic slopes, illumination priors,
    and control region discriminators to compute a continuous Ice Probability Map (0 to 1).
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.radar_cfg = config.radar
        self.ice_cfg = config.ice_detection
        self.terrain_cfg = config.terrain

    def load_inputs(self) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, str]:
        """Loads all required Phase 1 products and environmental rasters."""
        inputs = {}
        meta_transform = None
        meta_crs = None

        paths = {
            "cpr_l": PROCESSED_DATA_DIR / "dfsar_L_CPR.tif",
            "dop_l": PROCESSED_DATA_DIR / "dfsar_L_DOP.tif",
            "pv_l": PROCESSED_DATA_DIR / "dfsar_L_P_volume.tif",
            "s1_l": PROCESSED_DATA_DIR / "dfsar_L_S1.tif",
            "cpr_s": PROCESSED_DATA_DIR / "dfsar_S_CPR.tif",
            "dop_s": PROCESSED_DATA_DIR / "dfsar_S_DOP.tif",
            "dem": RAW_DATA_DIR / "lola_dem_20m.tif",
            "illumination": RAW_DATA_DIR / "crater_illumination.tif",
            "boulders": RAW_DATA_DIR / "crater_boulder_density.tif"
        }

        for key, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required raster: {p}")
            with rasterio.open(p) as src:
                inputs[key] = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4

        # Compute slope in degrees from DEM
        dem = inputs["dem"]
        res = self.terrain_cfg.grid_resolution_m
        dy, dx = np.gradient(dem, res, res)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        inputs["slope_deg"] = np.degrees(slope_rad)

        return inputs, meta_transform, meta_crs

    def define_geological_regions(
        self,
        inputs: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Delineates key geomorphologic zones:
        1. Crater interior permanently shadowed cold-trap (PSR)
        2. Exterior rough rocky ejecta blanket (control region)
        3. Background normal highland regolith
        """
        dem = inputs["dem"]
        illum = inputs["illumination"]
        boulders = inputs["boulders"]
        slope = inputs["slope_deg"]

        # Interior cold trap: zero illumination and low elevation bowl
        psr_mask = (illum < self.terrain_cfg.psr_illumination_cutoff) & (dem < self.ice_cfg.cold_trap_depth_m)

        # Exterior control region: Sunlit ejecta blanket outside rim with boulders
        control_rocky_ejecta = (illum > self.ice_cfg.control_illum_min) & (boulders > self.ice_cfg.control_boulders_min) & (dem > 0.0)

        # Background highland regolith: smooth, illuminated, low boulder count
        background_regolith = (
            (illum > self.ice_cfg.bg_regolith_illum_min) &
            (boulders < self.ice_cfg.bg_regolith_boulders_max) &
            (slope < self.ice_cfg.hazard_slope_baseline_deg)
        )

        return psr_mask, control_rocky_ejecta, background_regolith

    def compute_ice_probability(
        self,
        inputs: Dict[str, np.ndarray],
        cpr_threshold: float = None,
        dop_threshold: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculates continuous ice probability (0 to 1) using a multi-factor scoring model:
        1. Polarimetric Score (CPR_L > 1.0 and low DOP relative to theoretical minimum)
        2. Dual-Frequency Depth Score (CPR_L - CPR_S)
        3. Thermal Cold Trap Stability Prior (PSR and elevation)
        4. Roughness & Slope Hazard Penalty
        """
        if cpr_threshold is None:
            cpr_threshold = self.radar_cfg.cpr_ice_threshold
        if dop_threshold is None:
            dop_threshold = self.radar_cfg.dop_ice_max

        cpr_l = inputs["cpr_l"]
        dop_l = inputs["dop_l"]
        cpr_s = inputs["cpr_s"]
        slope = inputs["slope_deg"]
        boulders = inputs["boulders"]
        illum = inputs["illumination"]
        dem = inputs["dem"]

        # 1. Dual-frequency depth discriminator: Delta CPR (L - S)
        # Deep ice shows CPR_L >> CPR_S; surface rocks show CPR_L ~ CPR_S
        delta_cpr = np.clip(cpr_l - cpr_s, self.ice_cfg.delta_cpr_clip_min, self.ice_cfg.delta_cpr_clip_max)

        # 2. Polarimetric Ice Candidate Score (S_pol)
        # Accounts for the fundamental radar bound: m >= |CPR - 1| / (CPR + 1)
        cpr_score = 1.0 / (1.0 + np.exp(-self.ice_cfg.cpr_sigmoid_steepness * (cpr_l - cpr_threshold)))
        dop_score = 1.0 / (1.0 + np.exp(self.ice_cfg.dop_sigmoid_steepness * (dop_l - dop_threshold)))
        s_pol = cpr_score * dop_score

        # 3. Dual-Frequency Differential Score (S_freq)
        # Strong positive Delta CPR indicates buried dielectric ice layer
        s_freq = np.clip((delta_cpr - self.ice_cfg.delta_cpr_min_threshold) / self.ice_cfg.delta_cpr_scale, 0.0, 1.0)

        # 4. Thermal & Cold-Trap Prior (S_trap)
        s_trap_illum = np.clip((self.ice_cfg.cold_trap_illum_threshold - illum) / self.ice_cfg.cold_trap_illum_threshold, 0.0, 1.0)
        deep_floor = np.clip((-dem + self.ice_cfg.cold_trap_depth_m) / self.ice_cfg.trap_depth_normalization_m, 0.0, 1.0)
        s_trap = np.clip(self.ice_cfg.weight_trap_illum * s_trap_illum + self.ice_cfg.weight_trap_depth * deep_floor, 0.0, 1.0)

        # 5. Roughness & Slope Penalty (P_rough)
        p_slope = np.clip((slope - self.ice_cfg.hazard_slope_baseline_deg) / self.ice_cfg.hazard_slope_scale_deg, 0.0, 1.0)
        p_boulder = np.clip((boulders - self.ice_cfg.hazard_boulder_baseline) / self.ice_cfg.hazard_boulder_scale, 0.0, 1.0)
        p_rough = np.clip(self.ice_cfg.weight_rough_slope * p_slope + self.ice_cfg.weight_rough_boulder * p_boulder, 0.0, 1.0)

        # 6. Combined Continuous Ice Probability (0 to 1)
        raw_prob = (
            self.ice_cfg.weight_pol * s_pol +
            self.ice_cfg.weight_freq * s_freq +
            self.ice_cfg.weight_trap * s_trap
        ) * (1.0 - self.ice_cfg.roughness_penalty_scale * p_rough)
        ice_probability = np.clip(raw_prob, 0.0, 1.0)

        # Candidate Classification Mask (High Confidence Ice)
        high_prob_ice = (ice_probability >= self.ice_cfg.high_prob_ice_cutoff) & (s_trap > self.ice_cfg.trap_confidence_cutoff)

        # Ejecta / Rocky False Positive Mask (High CPR in exterior or low Delta CPR)
        rocky_terrain = (cpr_l > cpr_threshold) & (delta_cpr < self.ice_cfg.rocky_delta_cpr_max) & (boulders > self.ice_cfg.control_boulders_min)

        return {
            "ice_probability": ice_probability.astype(np.float32),
            "delta_cpr": delta_cpr.astype(np.float32),
            "high_prob_ice_mask": high_prob_ice.astype(np.uint8),
            "rocky_terrain_mask": rocky_terrain.astype(np.uint8),
            "s_pol": s_pol.astype(np.float32),
            "s_freq": s_freq.astype(np.float32),
            "s_trap": s_trap.astype(np.float32)
        }

    def run_sensitivity_analysis(
        self,
        inputs: Dict[str, np.ndarray],
        control_mask: np.ndarray,
        cpr_thresholds: list = None,
        dop_thresholds: list = None
    ) -> pd.DataFrame:
        """
        Sensitivity analysis of detection performance across threshold perturbations.
        Monitors detected ice area (km2) and false-positive rate in the control ejecta region.
        """
        if cpr_thresholds is None:
            cpr_thresholds = self.ice_cfg.sensitivity_cpr_thresholds
        if dop_thresholds is None:
            dop_thresholds = self.ice_cfg.sensitivity_dop_thresholds

        pixel_area_km2 = (self.terrain_cfg.grid_resolution_m / 1000.0) ** 2
        results = []

        for cpr_th in cpr_thresholds:
            for dop_th in dop_thresholds:
                prods = self.compute_ice_probability(inputs, cpr_threshold=cpr_th, dop_threshold=dop_th)
                ice_mask = prods["high_prob_ice_mask"] > 0
                
                # Ice detected inside candidate zone
                ice_pixels = np.sum(ice_mask)
                ice_area_km2 = ice_pixels * pixel_area_km2

                # False positives in control rocky ejecta region
                false_positives = np.sum(ice_mask & control_mask)
                control_total = np.maximum(np.sum(control_mask), 1)
                fpr = 100.0 * false_positives / control_total

                results.append({
                    "CPR_Threshold": cpr_th,
                    "DOP_Threshold": dop_th,
                    "Detected_Ice_Pixels": int(ice_pixels),
                    "Detected_Ice_Area_km2": round(float(ice_area_km2), 3),
                    "Control_False_Positives": int(false_positives),
                    "Control_False_Positive_Rate_pct": round(float(fpr), 2)
                })

        return pd.DataFrame(results)

    def process_and_export(self) -> Dict[str, Any]:
        """Runs Phase 2 end-to-end, exports GeoTIFFs, maps, and CSV tables."""
        inputs, transform, crs = self.load_inputs()
        psr_mask, control_mask, bg_mask = self.define_geological_regions(inputs)

        # Baseline detection
        prods = self.compute_ice_probability(
            inputs,
            cpr_threshold=self.radar_cfg.cpr_ice_threshold,
            dop_threshold=self.radar_cfg.dop_ice_max
        )

        # Save rasters
        ice_prob_path = PROCESSED_DATA_DIR / "ice_probability_map.tif"
        delta_cpr_path = PROCESSED_DATA_DIR / "delta_cpr_L_minus_S.tif"
        ice_mask_path = PROCESSED_DATA_DIR / "ice_candidate_mask.tif"

        save_geotiff(prods["ice_probability"], ice_prob_path, transform, crs=crs)
        save_geotiff(prods["delta_cpr"], delta_cpr_path, transform, crs=crs)
        save_geotiff(prods["high_prob_ice_mask"], ice_mask_path, transform, crs=crs)

        # Cartographic Map 1: Ice Probability Map
        map_ice_prob = plot_cartographic_map(
            data=prods["ice_probability"],
            title="Lunar South Pole Subsurface Ice Probability Map (0 to 1)",
            output_filename="map_phase2_ice_probability.png",
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Ice Probability (Multi-Discriminator Score)",
            overlay_contour=inputs["dem"],
            contour_levels=self.terrain_cfg.contour_levels
        )

        # Cartographic Map 2: Delta CPR (L-band minus S-band)
        map_delta_cpr = plot_cartographic_map(
            data=prods["delta_cpr"],
            title="DFSAR Dual-Frequency Differential CPR (L-band - S-band)",
            output_filename="map_phase2_delta_cpr.png",
            cmap="RdYlBu_r",
            vmin=self.ice_cfg.delta_cpr_plot_vmin,
            vmax=self.ice_cfg.delta_cpr_plot_vmax,
            cbar_label="Delta CPR (L - S) [Buried Ice Indicator]",
            overlay_contour=inputs["dem"],
            contour_levels=self.terrain_cfg.contour_levels
        )

        # Cartographic Map 3: Geological Classification Overlay
        # 0: Background, 1: Rocky Ejecta / Rim, 2: High Probability Ice
        classified = np.zeros_like(prods["ice_probability"], dtype=np.uint8)
        classified[prods["rocky_terrain_mask"] > 0] = 1
        classified[prods["high_prob_ice_mask"] > 0] = 2

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
        custom_cmap = ListedColormap(["#2b2b2b", "#d95f02", "#00d2ff"])  # Gray, Orange, Cyan
        ext = self.terrain_cfg.extent_km
        cax = ax.imshow(classified, extent=[-ext, ext, -ext, ext], origin="upper", cmap=custom_cmap)
        ax.set_title("Geological Discrimination: Ice vs. Rocky Ejecta vs. Regolith", fontsize=12, fontweight="bold")
        ax.set_xlabel("East-West Distance (km)")
        ax.set_ylabel("North-South Distance (km)")
        ax.grid(color="gray", linestyle="--", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2b2b2b", label="Background Regolith"),
            Patch(facecolor="#d95f02", label="Rocky Ejecta / Rim (False Positive CPR)"),
            Patch(facecolor="#00d2ff", label="Subsurface Ice Deposit (High Probability)")
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
        class_map_path = MAPS_DIR / "map_phase2_geological_classification.png"
        fig.tight_layout()
        fig.savefig(class_map_path, bbox_inches="tight")
        plt.close(fig)

        # Compile Regional Class Statistics Table
        pixel_area_km2 = (self.terrain_cfg.grid_resolution_m / 1000.0) ** 2
        ice_mask = prods["high_prob_ice_mask"] > 0
        rock_mask = prods["rocky_terrain_mask"] > 0
        bg_mask_eff = ~(ice_mask | rock_mask)

        def get_stats(m):
            return {
                "Area_km2": round(float(np.sum(m) * pixel_area_km2), 2),
                "Fraction_pct": round(float(100.0 * np.sum(m) / m.size), 2),
                "Mean_CPR_L": round(float(np.mean(inputs["cpr_l"][m])), 3),
                "Mean_CPR_S": round(float(np.mean(inputs["cpr_s"][m])), 3),
                "Mean_Delta_CPR": round(float(np.mean(prods["delta_cpr"][m])), 3),
                "Mean_DOP_L": round(float(np.mean(inputs["dop_l"][m])), 3)
            }

        class_stats = pd.DataFrame([
            {"Terrain_Class": "Subsurface Ice Deposit", **get_stats(ice_mask)},
            {"Terrain_Class": "Rocky Ejecta / Rim", **get_stats(rock_mask)},
            {"Terrain_Class": "Background Regolith", **get_stats(bg_mask_eff)}
        ])
        stats_path = TABLES_DIR / "phase2_terrain_class_statistics.csv"
        class_stats.to_csv(stats_path, index=False)

        # Run Sensitivity Analysis
        df_sens = self.run_sensitivity_analysis(inputs, control_mask)
        sens_path = TABLES_DIR / "phase2_threshold_sensitivity_analysis.csv"
        df_sens.to_csv(sens_path, index=False)

        return {
            "class_stats": class_stats,
            "sensitivity": df_sens,
            "maps": [map_ice_prob, map_delta_cpr, class_map_path]
        }


def run_phase2():
    print("[*] Running Phase 2: Ice Detection and Discrimination Framework...")
    engine = IceDetectionEngine()
    results = engine.process_and_export()
    print("[+] Phase 2 completed successfully!")
    print("\n--- Terrain Class Statistics ---")
    print(results["class_stats"].to_string(index=False))
    print("\n--- Sensitivity Analysis ---")
    print(results["sensitivity"].to_string(index=False))
    return results


if __name__ == "__main__":
    run_phase2()
