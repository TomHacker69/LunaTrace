"""
Phase 6: Subsurface Ice Volume Estimation
Dielectric mixing models (CRIM / Looyenga), radar penetration depth column (0 to 5 m),
Low/Nominal/High uncertainty scenarios, water mass computation, and sensitivity curve.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from src.config import CFG, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, MAPS_DIR
from src.utils import save_geotiff, plot_cartographic_map


class IceVolumeEstimator:
    """
    Subsurface Ice Dielectric Inversion and Volumetric Resource Estimation Engine.
    Translates dual-frequency polarimetric radar signatures and cold-trap probabilities
    into volumetric ice deposits with rigorous uncertainty bounds.
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.volume_cfg = config.volume
        self.radar_cfg = config.radar
        self.terrain_cfg = config.terrain

    def load_ice_rasters(self) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, str]:
        """Loads ice probability, CPR, and environmental rasters."""
        rasters = {}
        meta_transform = None
        meta_crs = None

        paths = {
            "ice_prob": PROCESSED_DATA_DIR / "ice_probability_map.tif",
            "ice_mask": PROCESSED_DATA_DIR / "ice_candidate_mask.tif",
            "cpr_l": PROCESSED_DATA_DIR / "dfsar_L_CPR.tif",
            "cpr_s": PROCESSED_DATA_DIR / "dfsar_S_CPR.tif",
            "delta_cpr": PROCESSED_DATA_DIR / "delta_cpr_L_minus_S.tif",
            "dem": RAW_DATA_DIR / "lola_dem_20m.tif"
        }

        for k, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required raster for Phase 6: {p}")
            with rasterio.open(p) as src:
                rasters[k] = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4

        return rasters, meta_transform, meta_crs

    def compute_local_ice_fraction(
        self,
        cpr_l: np.ndarray,
        ice_prob: np.ndarray,
        base_fraction: float = 0.15
    ) -> np.ndarray:
        """
        Inverts radar response to volumetric ice fraction using the Complex Refractive Index Model (CRIM)
        and polarimetric CBOE enhancement scaling.
        f_ice(x, y) = base_fraction * P_ice(x, y) * [(CPR_L - 1.0) / (2.0 - 1.0)]
        """
        cpr_norm = np.clip((cpr_l - self.radar_cfg.cpr_ice_threshold) / self.volume_cfg.cpr_scale_range, 0.0, self.volume_cfg.cpr_norm_max_factor)
        # Weight by continuous multi-discriminator ice probability
        ice_fraction = base_fraction * ice_prob * cpr_norm
        return np.clip(ice_fraction, 0.0, self.volume_cfg.max_pore_saturation)

    def compute_scenario_volumes(
        self,
        rasters: Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Computes Low, Nominal, and High subsurface ice volume and mass estimates in the top 0-5 m regolith column.
        """
        cpr_l = rasters["cpr_l"]
        ice_prob = rasters["ice_prob"]
        ice_mask = rasters["ice_mask"] > 0

        res_m = self.terrain_cfg.grid_resolution_m
        pixel_area_m2 = res_m * res_m
        ice_density_kg_m3 = self.volume_cfg.ice_density_kg_m3
        scenarios = self.volume_cfg.scenarios

        results = []
        fraction_maps = {}

        for name, params in scenarios.items():
            base_f = params["base_fraction"]
            act_depth = params["active_depth_m"]

            # Compute pixel-wise ice fraction
            f_map = self.compute_local_ice_fraction(cpr_l, ice_prob, base_fraction=base_f)
            # Mask to high-probability ice candidate pixels
            f_map_masked = np.where(ice_mask, f_map, 0.0)
            fraction_maps[name] = f_map_masked

            # Equivalent pure water ice column per pixel (meters of pure water equivalent)
            pure_ice_col_m = f_map_masked * act_depth

            # Total Volume of pure water ice (m3)
            vol_pure_ice_m3 = float(np.sum(pure_ice_col_m) * pixel_area_m2)
            vol_pure_ice_km3 = vol_pure_ice_m3 / 1e9

            # Total Mass of water ice (metric tons and Megatons)
            mass_kg = vol_pure_ice_m3 * ice_density_kg_m3
            mass_tons = mass_kg / 1000.0
            mass_megatons = mass_tons / 1e6

            # Total Bulk Regolith-Ice Matrix Volume (m3)
            bulk_matrix_m3 = float(np.sum(ice_mask) * pixel_area_m2 * act_depth)

            results.append({
                "Scenario": name,
                "Assumed_Base_Ice_Fraction_pct": round(base_f * 100, 1),
                "Effective_Column_Depth_m": params["eff_depth_m"],
                "Dry_Lag_Thickness_m": params["dry_lag_m"],
                "Active_Ice_Bearing_Depth_m": act_depth,
                "Mean_Detected_Ice_Fraction_pct": round(float(np.mean(f_map_masked[ice_mask])) * 100, 2),
                "Pure_Ice_Volume_m3": int(vol_pure_ice_m3),
                "Pure_Ice_Volume_km3": round(vol_pure_ice_km3, 4),
                "Water_Ice_Mass_Metric_Tons": int(mass_tons),
                "Water_Ice_Mass_Megatons": round(mass_megatons, 2),
                "Bulk_Ore_Volume_m3": int(bulk_matrix_m3),
                "Geological_Assumptions": params["description"]
            })

        df_volume = pd.DataFrame(results)
        return df_volume, fraction_maps

    def plot_sensitivity_curves(
        self,
        rasters: Dict[str, np.ndarray],
        output_filename: str = "map_phase6_volume_sensitivity.png"
    ) -> Path:
        """
        Generates sensitivity curves of total water ice mass (Megatons)
        as a function of assumed volumetric ice fraction (2% to 35%)
        across multiple column penetration depths (1.5 m, 3.5 m, 5.0 m).
        """
        cpr_l = rasters["cpr_l"]
        ice_prob = rasters["ice_prob"]
        ice_mask = rasters["ice_mask"] > 0
        pixel_area_m2 = (self.terrain_cfg.grid_resolution_m) ** 2
        ice_density_kg_m3 = self.volume_cfg.ice_density_kg_m3

        fractions_range = np.linspace(
            self.volume_cfg.sensitivity_fractions_min,
            self.volume_cfg.sensitivity_fractions_max,
            self.volume_cfg.sensitivity_fractions_steps
        )
        depth_options = self.volume_cfg.sensitivity_depths

        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        colors = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]

        for depth, color in zip(depth_options, colors):
            masses_megatons = []
            for bf in fractions_range:
                f_map = self.compute_local_ice_fraction(cpr_l, ice_prob, base_fraction=bf)
                pure_col = np.where(ice_mask, f_map * max(depth - self.volume_cfg.nominal_lag_m, 0.0), 0.0)
                vol_m3 = np.sum(pure_col) * pixel_area_m2
                mass_mt = (vol_m3 * ice_density_kg_m3) / 1e9
                masses_megatons.append(mass_mt)

            ax.plot(fractions_range * 100, masses_megatons, label=f"Column Depth = {depth} m", color=color, linewidth=2.4)

        # Dynamically highlight configured scenarios
        scen_colors = {"Low (Conservative)": "#2b83ba", "Nominal (Realistic)": "#fdae61", "High (Optimistic)": "#d7191c"}
        for s_name, s_cfg in self.volume_cfg.scenarios.items():
            s_f = s_cfg["base_fraction"]
            s_d = s_cfg["active_depth_m"]
            f_map = self.compute_local_ice_fraction(cpr_l, ice_prob, base_fraction=s_f)
            pure_col = np.where(ice_mask, f_map * s_d, 0.0)
            s_mass = (np.sum(pure_col) * pixel_area_m2 * ice_density_kg_m3) / 1e9
            ax.scatter([s_f * 100], [s_mass], color=scen_colors.get(s_name, "black"), s=100, zorder=6,
                       label=f"{s_name.split()[0]} Scenario ({s_f*100:.0f}%, {s_cfg['eff_depth_m']}m)")

        ax.set_title("Subsurface Water Ice Resource Sensitivity to Dielectric Parameters & Depth", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Volumetric Ice Fraction in Regolith Matrix (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Total Estimated Water Ice Mass (Megatons, Mt)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

        out_path = MAPS_DIR / output_filename
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def process_and_export(self) -> Dict[str, Any]:
        """Runs Phase 6 volume estimation, exports rasters, tables, and sensitivity plots."""
        rasters, transform, crs = self.load_ice_rasters()

        df_volume, fraction_maps = self.compute_scenario_volumes(rasters)

        # Save CSV Table
        table_path = TABLES_DIR / "phase6_ice_volume_estimates.csv"
        df_volume.to_csv(table_path, index=False)

        # Save Nominal Water Equivalent Column Raster (m of pure water)
        nominal_fraction = fraction_maps["Nominal (Realistic)"]
        nominal_depth_col = nominal_fraction * 3.0  # 3.0 m active column
        save_geotiff(nominal_depth_col, PROCESSED_DATA_DIR / "nominal_water_column_m.tif", transform, crs=crs)

        # Cartographic Map: Spatial Distribution of Pure Water Column
        map_water_col = plot_cartographic_map(
            data=nominal_depth_col,
            title="Nominal Subsurface Water Ice Column Depth (m Equivalent Pure H2O)",
            output_filename="map_phase6_water_column.png",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Equivalent Pure Water Column (Meters)",
            overlay_contour=rasters["dem"],
            contour_levels=[-1500, -1000, -500, 0, 500]
        )

        # Sensitivity Plot
        sens_plot_path = self.plot_sensitivity_curves(rasters)

        return {
            "df_volume": df_volume,
            "map_water_col": map_water_col,
            "sens_plot_path": sens_plot_path
        }


def run_phase6():
    print("[*] Running Phase 6: Subsurface Ice Volume Estimation & Resource Analysis...")
    estimator = IceVolumeEstimator()
    results = estimator.process_and_export()
    print("[+] Phase 6 completed successfully!")
    print("\n--- Subsurface Ice Volume & Mass Estimates (0-5 m Column) ---")
    cols = ["Scenario", "Assumed_Base_Ice_Fraction_pct", "Effective_Column_Depth_m", "Mean_Detected_Ice_Fraction_pct", "Water_Ice_Mass_Megatons", "Geological_Assumptions"]
    print(results["df_volume"][cols].to_string(index=False))
    return results


if __name__ == "__main__":
    run_phase6()
