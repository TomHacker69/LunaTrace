"""
Phase 2: Subsurface Ice Detection Framework
Multi-discriminator scoring, CPR ambiguity resolution (ice vs. rocky ejecta),
dual-frequency depth profiling (Delta CPR), and sensitivity analysis.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
import rasterio

from src.config import CFG
from src.config_schema import LunaTraceConfig
from src.utils import save_geotiff, plot_cartographic_map
from src.provenance import write_provenance_sidecar


class IceDetectionEngine:
    """
    Quantitative Lunar Ice Detection and Characterization Engine.
    Combines radar polarimetry (L and S band), topographic slopes, illumination priors,
    and control region discriminators to compute a continuous Ice Probability Map (0 to 1).
    """

    def __init__(self, config: LunaTraceConfig = CFG):
        self.cfg = config
        self.radar_cfg = config.radar
        self.ice_cfg = config.ice_detection
        self.terrain_cfg = config.terrain
        self.raw_dir = config.paths.raw_data_dir
        self.processed_dir = config.paths.processed_data_dir
        self.tables_dir = config.paths.outputs_tables_dir
        self.maps_dir = config.paths.outputs_maps_dir

    def load_inputs(self) -> Tuple[Dict[str, Any], rasterio.Affine, str, List[Path]]:
        """
        Loads all required Phase 1 products and environmental rasters.
        Fails loudly if required rasters or CRS metadata are missing.
        Explicitly flags whether Chandrayaan-2 OHRC boulder data is assessed.
        """
        inputs = {}
        meta_transform = None
        meta_crs = None
        loaded_paths = []

        required_paths = {
            "cpr_l": self.processed_dir / "dfsar_L_CPR.tif",
            "dop_l": self.processed_dir / "dfsar_L_DOP.tif",
            "pv_l": self.processed_dir / "dfsar_L_P_volume.tif",
            "s1_l": self.processed_dir / "dfsar_L_S1.tif",
            "dem": self.raw_dir / "lola_dem_20m.tif",
            "illumination": self.raw_dir / "crater_illumination.tif"
        }

        for key, p in required_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required raster for Phase 2: {p}")
            loaded_paths.append(p)
            with rasterio.open(p) as src:
                if not src.crs:
                    raise ValueError(f"Raster {p} has no CRS defined in metadata!")
                inputs[key] = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string()

        # Check for optional S-band DFSAR product (Directive 4)
        cpr_s_path = self.processed_dir / "dfsar_S_CPR.tif"
        dop_s_path = self.processed_dir / "dfsar_S_DOP.tif"
        if cpr_s_path.exists() and dop_s_path.exists():
            with rasterio.open(cpr_s_path) as src:
                inputs["cpr_s"] = src.read(1).astype(np.float64)
            with rasterio.open(dop_s_path) as src:
                inputs["dop_s"] = src.read(1).astype(np.float64)
            inputs["s_band_available"] = True
            inputs["s_band_status"] = "AVAILABLE"
            loaded_paths.extend([cpr_s_path, dop_s_path])
        else:
            inputs["cpr_s"] = None
            inputs["dop_s"] = None
            inputs["s_band_available"] = False
            inputs["s_band_status"] = "S-band unavailable"

        # Check for optional Chandrayaan-2 OHRC boulder density
        ohrc_path = self.raw_dir / "ohrc_boulder_density.tif"
        if ohrc_path.exists():
            with rasterio.open(ohrc_path) as src:
                inputs["boulders"] = src.read(1).astype(np.float64)
                inputs["boulder_status"] = "ASSESSED (Chandrayaan-2 OHRC)"
            loaded_paths.append(ohrc_path)
        else:
            inputs["boulders"] = None
            inputs["boulder_status"] = "UNASSESSED (OHRC not provided)"

        # Compute slope in degrees from DEM
        dem = inputs["dem"]
        res = self.terrain_cfg.grid_resolution_m
        dy, dx = np.gradient(dem, res, res)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        inputs["slope_deg"] = np.degrees(slope_rad)

        return inputs, meta_transform, meta_crs, loaded_paths

    def define_geological_regions(
        self,
        inputs: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Delineates key geomorphologic zones:
        1. Crater interior permanently shadowed cold-trap (PSR)
        2. Exterior rough rocky ejecta blanket (control region)
        3. Background normal highland regolith
        """
        dem = inputs["dem"]
        illum = inputs["illumination"]
        slope = inputs["slope_deg"]

        # Interior cold trap: low illumination and depressed elevation bowl
        psr_mask = (illum < self.cfg.illumination.psr_illumination_cutoff) & (dem < self.ice_cfg.cold_trap_depth_m)

        # Exterior control region: Sunlit elevated crater rim ejecta
        control_rocky_ejecta = (
            (illum > self.ice_cfg.control_illum_min) &
            (dem > self.ice_cfg.control_min_elevation_m) &
            (slope > self.ice_cfg.control_min_slope_deg)
        )

        # Background highland regolith: smooth, illuminated
        background_regolith = (
            (illum > self.ice_cfg.bg_regolith_illum_min) &
            (slope < self.ice_cfg.bg_regolith_max_slope_deg)
        )

        return psr_mask, control_rocky_ejecta, background_regolith

    def compute_ice_probability(
        self,
        inputs: Dict[str, Any],
        cpr_threshold: Optional[float] = None,
        dop_threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculates continuous ice probability (0 to 1) using a multi-factor scoring model:
        1. Polarimetric Score (CPR_L > 1.0 and low DOP relative to theoretical minimum)
        2. Dual-Frequency Depth Score (CPR_L - CPR_S)
        3. Thermal Cold Trap Stability Prior (PSR and elevation)
        4. Roughness Hazard Penalty
        """
        if cpr_threshold is None:
            cpr_threshold = self.radar_cfg.cpr_ice_threshold
        if dop_threshold is None:
            dop_threshold = self.radar_cfg.dop_ice_max

        cpr_l = inputs["cpr_l"]
        dop_l = inputs["dop_l"]
        cpr_s = inputs["cpr_s"]
        slope = inputs["slope_deg"]
        illum = inputs["illumination"]
        dem = inputs["dem"]
        s_band_avail = inputs.get("s_band_available", True) and cpr_s is not None

        # 1. Dual-frequency depth discriminator: Delta CPR (L - S)
        if s_band_avail:
            delta_cpr = np.clip(cpr_l - cpr_s, self.ice_cfg.delta_cpr_clip_min, self.ice_cfg.delta_cpr_clip_max)
            s_freq = np.clip((delta_cpr - self.ice_cfg.delta_cpr_min_threshold) / self.ice_cfg.delta_cpr_scale, 0.0, 1.0)
        else:
            delta_cpr = np.full_like(cpr_l, np.nan)
            s_freq = np.zeros_like(cpr_l)

        # 2. Polarimetric Ice Candidate Score (S_pol)
        cpr_score = 1.0 / (1.0 + np.exp(-self.ice_cfg.cpr_sigmoid_steepness * (cpr_l - cpr_threshold)))
        dop_score = 1.0 / (1.0 + np.exp(self.ice_cfg.dop_sigmoid_steepness * (dop_l - dop_threshold)))
        s_pol = cpr_score * dop_score

        # 3. Thermal & Cold-Trap Prior (S_trap)
        s_trap_illum = np.clip((self.ice_cfg.cold_trap_illum_threshold - illum) / self.ice_cfg.cold_trap_illum_threshold, 0.0, 1.0)
        deep_floor = np.clip((-dem + self.ice_cfg.cold_trap_depth_m) / self.ice_cfg.trap_depth_normalization_m, 0.0, 1.0)
        s_trap = np.clip(self.ice_cfg.weight_trap_illum * s_trap_illum + self.ice_cfg.weight_trap_depth * deep_floor, 0.0, 1.0)

        # 4. Topographic Hazard Penalty (P_rough)
        p_slope = np.clip((slope - self.ice_cfg.hazard_slope_baseline_deg) / self.ice_cfg.hazard_slope_scale_deg, 0.0, 1.0)

        if inputs["boulders"] is not None:
            # When OHRC boulder data is provided, incorporate boulder hazard
            boulders = inputs["boulders"]
            p_boulder = np.clip(boulders / 0.40, 0.0, 1.0)
            p_rough = np.clip(0.6 * p_slope + 0.4 * p_boulder, 0.0, 1.0)
        else:
            # Without OHRC, roughness hazard is strictly topographic slope
            p_rough = p_slope

        # 5. Combined Continuous Ice Probability (0 to 1)
        if s_band_avail:
            w_pol = self.ice_cfg.weights["polarimetric"]
            w_freq = self.ice_cfg.weights["dual_frequency"]
            w_trap = self.ice_cfg.weights["cold_trap"]
            total_w = w_pol + w_freq + w_trap
            w_pol, w_freq, w_trap = w_pol / total_w, w_freq / total_w, w_trap / total_w
            raw_prob = (w_pol * s_pol + w_freq * s_freq + w_trap * s_trap) * (1.0 - self.ice_cfg.roughness_penalty_scale * p_rough)
            rocky_terrain = (cpr_l > cpr_threshold) & (delta_cpr < self.ice_cfg.rocky_delta_cpr_max)
        else:
            # When S-band is unavailable, disable dual-frequency scoring and normalize polarimetric + cold trap weights
            w_pol = self.ice_cfg.weights["polarimetric"]
            w_trap = self.ice_cfg.weights["cold_trap"]
            total_w = w_pol + w_trap
            w_pol_adj = w_pol / total_w
            w_trap_adj = w_trap / total_w
            raw_prob = (w_pol_adj * s_pol + w_trap_adj * s_trap) * (1.0 - self.ice_cfg.roughness_penalty_scale * p_rough)
            # Without S-band, rocky false positive mask uses control region sunlit elevation condition
            rocky_terrain = (cpr_l > cpr_threshold) & (illum > self.ice_cfg.control_illum_min)

        ice_probability = np.clip(raw_prob, 0.0, 1.0)

        # Candidate Classification Mask
        high_prob_ice = (ice_probability >= self.ice_cfg.high_prob_ice_cutoff) & (s_trap > self.ice_cfg.trap_confidence_cutoff)

        # Ejecta / Rocky False Positive Mask (High CPR in exterior or low Delta CPR)
        rocky_terrain = (cpr_l > cpr_threshold) & (delta_cpr < self.ice_cfg.rocky_delta_cpr_max)

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
        inputs: Dict[str, Any],
        control_mask: np.ndarray,
        cpr_thresholds: Optional[List[float]] = None,
        dop_thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Sensitivity analysis evaluating detection performance and reporting
        both radar-alone false alarms and combined framework performance in the control region.
        """
        if cpr_thresholds is None:
            cpr_thresholds = self.ice_cfg.sensitivity_cpr_thresholds
        if dop_thresholds is None:
            dop_thresholds = self.ice_cfg.sensitivity_dop_thresholds

        pixel_area_km2 = (self.terrain_cfg.grid_resolution_m / 1000.0) ** 2
        control_total = int(np.maximum(np.sum(control_mask), 1))
        results = []

        for cpr_th in cpr_thresholds:
            for dop_th in dop_thresholds:
                prods = self.compute_ice_probability(inputs, cpr_threshold=cpr_th, dop_threshold=dop_th)
                ice_mask = prods["high_prob_ice_mask"] > 0
                ice_pixels = int(np.sum(ice_mask))
                ice_area_km2 = ice_pixels * pixel_area_km2

                # 1. Radar-Alone False Positives in Control Region (demonstrates CPR ambiguity)
                cpr_l = inputs["cpr_l"]
                dop_l = inputs["dop_l"]
                radar_alone_mask = (cpr_l > cpr_th) & (dop_l < dop_th)
                radar_fp_count = int(np.sum(radar_alone_mask & control_mask))
                radar_fpr = 100.0 * radar_fp_count / control_total

                # 2. Combined Framework False Positives (after Delta CPR & Thermal Trap gating)
                combined_fp_count = int(np.sum(ice_mask & control_mask))
                combined_fpr = 100.0 * combined_fp_count / control_total

                results.append({
                    "CPR_Threshold": cpr_th,
                    "DOP_Threshold": dop_th,
                    "Detected_Ice_Pixels": ice_pixels,
                    "Detected_Ice_Area_km2": round(float(ice_area_km2), 3),
                    "Radar_Alone_Control_FP_Count": radar_fp_count,
                    "Radar_Alone_Control_FPR_pct": round(float(radar_fpr), 2),
                    "Combined_Control_FP_Count": combined_fp_count,
                    "Combined_Control_FPR_pct": round(float(combined_fpr), 2)
                })

        return pd.DataFrame(results)

    def process_and_export(self) -> Dict[str, Any]:
        """Runs Phase 2 end-to-end, exports GeoTIFFs, maps, and CSV tables with provenance sidecars."""
        inputs, transform, crs, input_paths = self.load_inputs()
        psr_mask, control_mask, bg_mask = self.define_geological_regions(inputs)

        # Baseline detection
        prods = self.compute_ice_probability(
            inputs,
            cpr_threshold=self.radar_cfg.cpr_ice_threshold,
            dop_threshold=self.radar_cfg.dop_ice_max
        )

        # Save rasters with provenance sidecars
        ice_prob_path = self.processed_dir / "ice_probability_map.tif"
        delta_cpr_path = self.processed_dir / "delta_cpr_L_minus_S.tif"
        ice_mask_path = self.processed_dir / "ice_candidate_mask.tif"

        save_geotiff(prods["ice_probability"], ice_prob_path, transform, crs=crs)
        save_geotiff(prods["delta_cpr"], delta_cpr_path, transform, crs=crs)
        save_geotiff(prods["high_prob_ice_mask"], ice_mask_path, transform, crs=crs)

        for p_out in [ice_prob_path, delta_cpr_path, ice_mask_path]:
            write_provenance_sidecar(
                output_path=p_out,
                input_paths=input_paths,
                config_dict=self.cfg.model_dump(),
                metadata={"boulder_status": inputs["boulder_status"]}
            )

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
        write_provenance_sidecar(map_ice_prob, input_paths, self.cfg.model_dump(), {"map": "ice_probability"})

        # Cartographic Map 2: Delta CPR (if S-band available)
        if inputs["s_band_available"]:
            map_delta_cpr = plot_cartographic_map(
                data=prods["delta_cpr"],
                title="DFSAR Dual-Frequency Differential CPR (L-band - S-band)",
                output_filename="map_phase2_delta_cpr.png",
                cmap="RdYlBu_r",
                vmin=self.cfg.visualization.delta_cpr_plot_vmin,
                vmax=self.cfg.visualization.delta_cpr_plot_vmax,
                cbar_label="Delta CPR (L - S) [Buried Ice Indicator]",
                overlay_contour=inputs["dem"],
                contour_levels=self.terrain_cfg.contour_levels
            )
            write_provenance_sidecar(map_delta_cpr, input_paths, self.cfg.model_dump(), {"map": "delta_cpr"})
        else:
            map_delta_cpr = None

        # Cartographic Map 3: Geological Classification Overlay
        classified = np.zeros_like(prods["ice_probability"], dtype=np.uint8)
        classified[prods["rocky_terrain_mask"] > 0] = 1
        classified[prods["high_prob_ice_mask"] > 0] = 2

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(9, 8), dpi=self.cfg.visualization.dpi)
        custom_cmap = ListedColormap(["#2b2b2b", "#d95f02", "#00d2ff"])
        ext = self.terrain_cfg.extent_km
        cax = ax.imshow(classified, extent=[-ext, ext, -ext, ext], origin="upper", cmap=custom_cmap)
        ax.set_title("Geological Discrimination: Ice vs. Rocky Ejecta vs. Regolith", fontsize=12, fontweight="bold")
        ax.set_xlabel("East-West Distance (km)")
        ax.set_ylabel("North-South Distance (km)")
        ax.grid(color="gray", linestyle="--", alpha=0.3)

        legend_elements = [
            Patch(facecolor="#2b2b2b", label="Background Regolith"),
            Patch(facecolor="#d95f02", label="Rocky Ejecta / Rim (High CPR Ambiguity)"),
            Patch(facecolor="#00d2ff", label="Subsurface Ice Deposit (High Confidence)")
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
        class_map_path = self.maps_dir / "map_phase2_geological_classification.png"
        fig.tight_layout()
        fig.savefig(class_map_path, bbox_inches="tight")
        plt.close(fig)
        write_provenance_sidecar(class_map_path, input_paths, self.cfg.model_dump(), {"map": "geological_classification"})

        # Compile Regional Class Statistics Table
        pixel_area_km2 = (self.terrain_cfg.grid_resolution_m / 1000.0) ** 2
        ice_mask = prods["high_prob_ice_mask"] > 0
        rock_mask = prods["rocky_terrain_mask"] > 0
        bg_mask_eff = ~(ice_mask | rock_mask)

        def get_stats(m):
            if np.sum(m) == 0:
                mean_cpr_s = "S-band unavailable" if not inputs["s_band_available"] else 0.0
                mean_delta = "S-band unavailable" if not inputs["s_band_available"] else 0.0
                return {
                    "Area_km2": 0.0,
                    "Fraction_pct": 0.0,
                    "Mean_CPR_L": 0.0,
                    "Mean_CPR_S": mean_cpr_s,
                    "Mean_Delta_CPR": mean_delta,
                    "Mean_DOP_L": 0.0
                }
            if inputs["s_band_available"] and inputs["cpr_s"] is not None:
                mean_cpr_s = round(float(np.mean(inputs["cpr_s"][m])), 3)
                mean_delta = round(float(np.mean(prods["delta_cpr"][m])), 3)
            else:
                mean_cpr_s = "S-band unavailable"
                mean_delta = "S-band unavailable"

            return {
                "Area_km2": round(float(np.sum(m) * pixel_area_km2), 2),
                "Fraction_pct": round(float(100.0 * np.sum(m) / m.size), 2),
                "Mean_CPR_L": round(float(np.mean(inputs["cpr_l"][m])), 3),
                "Mean_CPR_S": mean_cpr_s,
                "Mean_Delta_CPR": mean_delta,
                "Mean_DOP_L": round(float(np.mean(inputs["dop_l"][m])), 3)
            }

        class_stats = pd.DataFrame([
            {"Terrain_Class": "Subsurface Ice Deposit", **get_stats(ice_mask)},
            {"Terrain_Class": "Rocky Ejecta / Rim", **get_stats(rock_mask)},
            {"Terrain_Class": "Background Regolith", **get_stats(bg_mask_eff)}
        ])
        stats_path = self.tables_dir / "phase2_terrain_class_statistics.csv"
        class_stats.to_csv(stats_path, index=False)
        write_provenance_sidecar(stats_path, input_paths, self.cfg.model_dump(), {"table": "class_stats"})

        # Run Sensitivity Analysis
        df_sens = self.run_sensitivity_analysis(inputs, control_mask)
        sens_path = self.tables_dir / "phase2_threshold_sensitivity_analysis.csv"
        df_sens.to_csv(sens_path, index=False)
        write_provenance_sidecar(sens_path, input_paths, self.cfg.model_dump(), {"table": "sensitivity_analysis"})

        return {
            "class_stats": class_stats,
            "sensitivity": df_sens,
            "boulder_status": inputs["boulder_status"],
            "maps": [map_ice_prob, map_delta_cpr, class_map_path]
        }


def run_phase2(config: LunaTraceConfig = CFG):
    print("[*] Running Phase 2: Ice Detection and Discrimination Framework...")
    engine = IceDetectionEngine(config=config)
    results = engine.process_and_export()
    print(f"[+] Boulder Hazard Status: {results['boulder_status']}")
    print("[+] Phase 2 completed successfully!")
    print("\n--- Terrain Class Statistics ---")
    print(results["class_stats"].to_string(index=False))
    print("\n--- Sensitivity Analysis ---")
    print(results["sensitivity"].to_string(index=False))
    return results
