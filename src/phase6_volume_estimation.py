"""
Phase 6: Subsurface Ice Volume Estimation
Dielectric mixing models (CRIM), radar penetration depth column (0 to 5 m),
Low/Nominal/High uncertainty scenarios, water mass computation, sensitivity analysis, and cryptographic provenance sidecars.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from src.config import CFG
from src.config_schema import LunaTraceConfig
from src.constants import (
    ICE_DENSITY_KG_M3,
    EPS_REGOLITH_REAL,
    EPS_ICE_REAL,
    LOSS_TANGENT_DRY_REGOLITH,
    LOSS_TANGENT_ICE,
    BULK_REGOLITH_POROSITY,
    M3_TO_KM3,
    KG_TO_METRIC_TON,
    METRIC_TON_TO_MEGATON
)
from src.utils import save_geotiff, plot_cartographic_map
from src.provenance import write_provenance_sidecar


class IceVolumeEstimator:
    """
    Subsurface Ice Dielectric Inversion and Volumetric Resource Estimation Engine.
    Translates dual-frequency polarimetric radar signatures and cold-trap probabilities
    into volumetric ice deposits with rigorous uncertainty bounds.
    """

    def __init__(self, config: LunaTraceConfig = CFG):
        self.cfg = config
        self.volume_cfg = config.volume
        self.radar_cfg = config.radar
        self.terrain_cfg = config.terrain
        self.raw_dir = config.paths.raw_data_dir
        self.processed_dir = config.paths.processed_data_dir
        self.tables_dir = config.paths.outputs_tables_dir
        self.maps_dir = config.paths.outputs_maps_dir

    def load_ice_rasters(self) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, str, List[Path]]:
        """Loads ice probability, CPR, and environmental rasters."""
        rasters = {}
        meta_transform = None
        meta_crs = None
        loaded_paths = []

        paths = {
            "ice_prob": self.processed_dir / "ice_probability_map.tif",
            "ice_mask": self.processed_dir / "ice_candidate_mask.tif",
            "cpr_l": self.processed_dir / "dfsar_L_CPR.tif",
            "cpr_s": self.processed_dir / "dfsar_S_CPR.tif",
            "delta_cpr": self.processed_dir / "delta_cpr_L_minus_S.tif",
            "dem": self.raw_dir / "lola_dem_20m.tif"
        }

        for k, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required raster for Phase 6: {p}")
            loaded_paths.append(p)
            with rasterio.open(p) as src:
                if not src.crs:
                    raise ValueError(f"Raster {p} has no CRS defined in metadata!")
                rasters[k] = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string()

        return rasters, meta_transform, meta_crs, loaded_paths

    def compute_local_ice_fraction(
        self,
        cpr_l: np.ndarray,
        ice_prob: np.ndarray,
        base_fraction: float
    ) -> np.ndarray:
        """
        Inverts radar response to volumetric ice fraction using the Complex Refractive Index Model (CRIM)
        and polarimetric CBOE enhancement scaling.
        f_ice(x, y) = base_fraction * P_ice(x, y) * [(CPR_L - 1.0) / cpr_scale_range]
        """
        cpr_norm = np.clip(
            (cpr_l - self.radar_cfg.cpr_ice_threshold) / self.volume_cfg.cpr_scale_range,
            0.0,
            self.volume_cfg.cpr_norm_max_factor
        )
        ice_fraction = base_fraction * ice_prob * cpr_norm
        return np.clip(ice_fraction, 0.0, 0.60)  # Max physical pore volume saturation

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
        ice_density_kg_m3 = ICE_DENSITY_KG_M3
        scenarios = self.volume_cfg.scenarios

        results = []
        fraction_maps = {}

        for name, scen in scenarios.items():
            base_f = scen.base_fraction
            act_depth = scen.active_depth_m

            # Compute pixel-wise ice fraction
            f_map = self.compute_local_ice_fraction(cpr_l, ice_prob, base_fraction=base_f)
            f_map_masked = np.where(ice_mask, f_map, 0.0)
            fraction_maps[name] = f_map_masked

            pure_ice_col_m = f_map_masked * act_depth

            vol_pure_ice_m3 = float(np.sum(pure_ice_col_m) * pixel_area_m2)
            vol_pure_ice_km3 = vol_pure_ice_m3 * M3_TO_KM3

            mass_kg = vol_pure_ice_m3 * ice_density_kg_m3
            mass_tons = mass_kg * KG_TO_METRIC_TON
            mass_megatons = mass_tons * METRIC_TON_TO_MEGATON

            bulk_matrix_m3 = float(np.sum(ice_mask) * pixel_area_m2 * act_depth)

            results.append({
                "Scenario": name,
                "Assumed_Base_Ice_Fraction_pct": round(base_f * 100, 1),
                "Effective_Column_Depth_m": scen.eff_depth_m,
                "Dry_Lag_Thickness_m": scen.dry_lag_m,
                "Active_Ice_Bearing_Depth_m": act_depth,
                "Mean_Detected_Ice_Fraction_pct": round(float(np.mean(f_map_masked[ice_mask])) * 100, 2),
                "Pure_Ice_Volume_m3": int(vol_pure_ice_m3),
                "Pure_Ice_Volume_km3": round(vol_pure_ice_km3, 4),
                "Water_Ice_Mass_Metric_Tons": int(mass_tons),
                "Water_Ice_Mass_Megatons": round(mass_megatons, 2),
                "Bulk_Ore_Volume_m3": int(bulk_matrix_m3),
                "Geological_Assumptions": scen.description
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
        as a function of assumed volumetric ice fraction across multiple column penetration depths.
        """
        cpr_l = rasters["cpr_l"]
        ice_prob = rasters["ice_prob"]
        ice_mask = rasters["ice_mask"] > 0
        pixel_area_m2 = (self.terrain_cfg.grid_resolution_m) ** 2
        ice_density_kg_m3 = ICE_DENSITY_KG_M3

        fractions_range = np.linspace(
            self.volume_cfg.sensitivity_fractions_min,
            self.volume_cfg.sensitivity_fractions_max,
            self.volume_cfg.sensitivity_fractions_steps
        )
        depth_options = self.volume_cfg.sensitivity_depths

        fig, ax = plt.subplots(figsize=(9, 6), dpi=self.cfg.visualization.dpi)
        colors = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]

        for depth, color in zip(depth_options, colors):
            masses_megatons = []
            for bf in fractions_range:
                f_map = self.compute_local_ice_fraction(cpr_l, ice_prob, base_fraction=bf)
                pure_col = np.where(ice_mask, f_map * max(depth - self.volume_cfg.nominal_lag_m, 0.0), 0.0)
                vol_m3 = np.sum(pure_col) * pixel_area_m2
                mass_mt = (vol_m3 * ice_density_kg_m3) * METRIC_TON_TO_MEGATON * KG_TO_METRIC_TON
                masses_megatons.append(mass_mt)

            ax.plot(fractions_range * 100, masses_megatons, label=f"Column Depth = {depth} m", color=color, linewidth=2.4)

        # Highlight configured scenarios dynamically without dictionary fallback
        scen_colors = {
            "Low (Conservative)": "#2b83ba",
            "Nominal (Realistic)": "#fdae61",
            "High (Optimistic)": "#d7191c"
        }

        for s_name, s_cfg in self.volume_cfg.scenarios.items():
            s_f = s_cfg.base_fraction
            s_d = s_cfg.active_depth_m
            f_map = self.compute_local_ice_fraction(cpr_l, ice_prob, base_fraction=s_f)
            pure_col = np.where(ice_mask, f_map * s_d, 0.0)
            s_mass = (np.sum(pure_col) * pixel_area_m2 * ice_density_kg_m3) * METRIC_TON_TO_MEGATON * KG_TO_METRIC_TON
            ax.scatter(
                [s_f * 100],
                [s_mass],
                color=scen_colors[s_name],
                s=100,
                zorder=6,
                label=f"{s_name.split()[0]} Scenario ({s_f * 100:.0f}%, {s_cfg.eff_depth_m}m)"
            )

        ax.set_title("Subsurface Water Ice Resource Sensitivity to Dielectric Parameters & Depth", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Volumetric Ice Fraction in Regolith Matrix (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Total Estimated Water Ice Mass (Megatons, Mt)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

        out_path = self.maps_dir / output_filename
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def process_and_export(self) -> Dict[str, Any]:
        """Runs Phase 6 volumetric estimation and exports maps and tables with provenance sidecars."""
        rasters, transform, crs, input_paths = self.load_ice_rasters()

        # Compute Low, Nominal, High volumes
        df_volume, fraction_maps = self.compute_scenario_volumes(rasters)

        # Export CSV table
        table_path = self.tables_dir / "phase6_ice_volume_estimates.csv"
        df_volume.to_csv(table_path, index=False)
        write_provenance_sidecar(table_path, input_paths, self.cfg.model_dump(), {"table": "volume_estimates"})

        # Export Nominal Ice Column Raster
        nom_map = fraction_maps["Nominal (Realistic)"]
        nom_depth_m = self.volume_cfg.scenarios["Nominal (Realistic)"].active_depth_m
        pure_water_column_m = nom_map * nom_depth_m

        water_raster_path = self.processed_dir / "nominal_water_column_m.tif"
        save_geotiff(pure_water_column_m, water_raster_path, transform, crs=crs)
        write_provenance_sidecar(water_raster_path, input_paths, self.cfg.model_dump(), {"scenario": "Nominal"})

        # Cartographic Map 1: Nominal Equivalent Water Ice Column
        map_water_col = plot_cartographic_map(
            data=pure_water_column_m,
            title="Nominal Equivalent Water Ice Column (Meters of Pure H2O Equivalent)",
            output_filename="map_phase6_water_column.png",
            cmap="Blues",
            vmin=0.0,
            vmax=round(float(np.max(pure_water_column_m)), 2) + 0.05,
            cbar_label="Equivalent Pure Water Column (Meters)",
            overlay_contour=rasters["dem"],
            contour_levels=self.terrain_cfg.contour_levels
        )
        write_provenance_sidecar(map_water_col, input_paths, self.cfg.model_dump(), {"map": "water_column"})

        # Cartographic Map 2: Sensitivity Curves
        map_sensitivity = self.plot_sensitivity_curves(rasters, output_filename="map_phase6_volume_sensitivity.png")
        write_provenance_sidecar(map_sensitivity, input_paths, self.cfg.model_dump(), {"map": "sensitivity_curve"})

        return {
            "df_volume": df_volume,
            "map_water_col": map_water_col,
            "map_sensitivity": map_sensitivity,
            "fraction_maps": fraction_maps
        }


def run_phase6(config: LunaTraceConfig = CFG):
    print("[*] Running Phase 6: Subsurface Ice Volume Estimation & Resource Analysis...")
    estimator = IceVolumeEstimator(config=config)
    results = estimator.process_and_export()
    print("[+] Phase 6 completed successfully!")
    print("\n--- Subsurface Ice Volume & Mass Estimates (0-5 m Column) ---")
    cols = ["Scenario", "Assumed_Base_Ice_Fraction_pct", "Effective_Column_Depth_m", "Mean_Detected_Ice_Fraction_pct", "Water_Ice_Mass_Megatons", "Geological_Assumptions"]
    print(results["df_volume"][cols].to_string(index=False))
    return results
