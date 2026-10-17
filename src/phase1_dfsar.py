"""
Phase 1: DFSAR Polarimetric Processing Pipeline
Computes Stokes parameters, CPR, DOP, and Raney m-chi decomposition
for Chandrayaan-2 DFSAR L-band and S-band hybrid-polarimetric data.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import rasterio

from src.config import CFG, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, MAPS_DIR
from src.utils import enhanced_lee_filter, save_geotiff, plot_cartographic_map


class DFSARProcessor:
    """
    Chandrayaan-2 Dual Frequency SAR (DFSAR) Polarimetric Engine.
    Handles L-band and S-band hybrid polarimetric data (Right Circular Transmit, Linear H & V Receive).
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.radar_cfg = config.radar
        self.terrain_cfg = config.terrain

    def load_stokes_band(
        self,
        band: str = "L",
        apply_filter: bool = True
    ) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, str]:
        """
        Loads Stokes parameter rasters (S1, S2, S3, S4) for the given band ('L' or 'S').
        Applies speckle filtering to suppress multiplicative radar noise.
        """
        band = band.upper()
        prefix = f"dfsar_{band}_"
        stokes = {}
        meta_transform = None
        meta_crs = None

        for param in ["S1", "S2", "S3", "S4"]:
            fpath = RAW_DATA_DIR / f"{prefix}{param}.tif"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required radar file: {fpath}")

            with rasterio.open(fpath) as src:
                data = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4

            if apply_filter:
                data = enhanced_lee_filter(
                    data,
                    window_size=self.radar_cfg.filter_window_size,
                    num_looks=self.radar_cfg.num_looks,
                    is_intensity=(param == "S1")
                )
            stokes[param] = data

        return stokes, meta_transform, meta_crs

    def compute_polarimetric_products(
        self,
        stokes: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Computes Circular Polarization Ratio (CPR), Degree of Polarization (DOP),
        and Raney m-chi decomposition (Volume, Double-bounce, Surface scattering).
        """
        s1 = np.maximum(stokes["S1"], self.radar_cfg.min_valid_stokes1)
        s2 = stokes["S2"]
        s3 = stokes["S3"]
        s4 = stokes["S4"]

        # 1. Circular Polarization Ratio (CPR)
        # CPR = (S1 - S4) / (S1 + S4) = P_SC / P_OC
        denom_cpr = np.maximum(s1 + s4, 1e-10)
        cpr = np.clip((s1 - s4) / denom_cpr, self.radar_cfg.cpr_min_clip, self.radar_cfg.cpr_max_clip)

        # 2. Degree of Polarization (DOP, m)
        # m = sqrt(S2^2 + S3^2 + S4^2) / S1
        pol_intensity = np.sqrt(s2 ** 2 + s3 ** 2 + s4 ** 2)
        dop = np.clip(pol_intensity / s1, self.radar_cfg.dop_min_clip, self.radar_cfg.dop_max_clip)

        # 3. Raney m-chi Decomposition
        # Ellipticity parameter: sin(2*chi) = -S4 / (m * S1)
        denom_chi = np.maximum(dop * s1, 1e-10)
        sin_2chi = np.clip(-s4 / denom_chi, self.radar_cfg.sin_2chi_min_clip, self.radar_cfg.sin_2chi_max_clip)

        # Volume scattering: unpolarized backscatter
        p_volume = np.maximum(s1 * (1.0 - dop), 0.0)

        # Double-bounce (even-bounce dihedral)
        p_double = np.maximum(0.5 * s1 * dop * (1.0 + sin_2chi), 0.0)

        # Surface scattering (odd-bounce single specular)
        p_surface = np.maximum(0.5 * s1 * dop * (1.0 - sin_2chi), 0.0)

        return {
            "CPR": cpr.astype(np.float32),
            "DOP": dop.astype(np.float32),
            "P_volume": p_volume.astype(np.float32),
            "P_double": p_double.astype(np.float32),
            "P_surface": p_surface.astype(np.float32),
            "S1": s1.astype(np.float32)
        }

    def process_and_export(self, band: str = "L") -> Dict[str, Path]:
        """
        Executes end-to-end processing for a single DFSAR band, saving GeoTIFFs,
        generating cartographic figures, and recording summary statistics.
        """
        band = band.upper()
        stokes, transform, crs = self.load_stokes_band(band=band)
        products = self.compute_polarimetric_products(stokes)

        output_files = {}
        for name, arr in products.items():
            out_path = PROCESSED_DATA_DIR / f"dfsar_{band}_{name}.tif"
            save_geotiff(arr, out_path, transform, crs=crs)
            output_files[name] = out_path

        # Generate Cartographic Maps
        # 1. CPR Map
        cpr_map_path = plot_cartographic_map(
            data=products["CPR"],
            title=f"Chandrayaan-2 DFSAR {band}-Band Circular Polarization Ratio (CPR)",
            output_filename=f"map_phase1_{band}_cpr.png",
            cmap="inferno",
            vmin=self.radar_cfg.cpr_plot_vmin,
            vmax=self.radar_cfg.cpr_plot_vmax,
            cbar_label="CPR (Same-Sense / Opposite-Sense)"
        )
        output_files["map_cpr"] = cpr_map_path

        # 2. DOP Map
        dop_map_path = plot_cartographic_map(
            data=products["DOP"],
            title=f"Chandrayaan-2 DFSAR {band}-Band Degree of Polarization (DoP)",
            output_filename=f"map_phase1_{band}_dop.png",
            cmap="plasma_r",
            vmin=self.radar_cfg.dop_plot_vmin,
            vmax=self.radar_cfg.dop_plot_vmax,
            cbar_label="Degree of Polarization (m)"
        )
        output_files["map_dop"] = dop_map_path

        # 3. Raney m-chi Decomposition RGB Composite Map
        # Red = Double Bounce, Green = Volume, Blue = Surface
        pv = products["P_volume"]
        pdb = products["P_double"]
        ps = products["P_surface"]

        norm_r = np.clip(pdb / (np.percentile(pdb, 98) + 1e-6), 0, 1)
        norm_g = np.clip(pv / (np.percentile(pv, 98) + 1e-6), 0, 1)
        norm_b = np.clip(ps / (np.percentile(ps, 98) + 1e-6), 0, 1)
        rgb = np.stack([norm_r, norm_g, norm_b], axis=-1)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
        ext = self.terrain_cfg.extent_km
        ax.imshow(rgb, extent=[-ext, ext, -ext, ext], origin="upper")
        ax.set_title(f"DFSAR {band}-Band m-chi Decomposition (R: Double, G: Volume, B: Surface)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("East-West Distance (km)")
        ax.set_ylabel("North-South Distance (km)")
        ax.grid(color="gray", linestyle="--", alpha=0.3)
        rgb_out = MAPS_DIR / f"map_phase1_{band}_mchi_decomposition.png"
        fig.tight_layout()
        fig.savefig(rgb_out, bbox_inches="tight")
        plt.close(fig)
        output_files["map_mchi"] = rgb_out

        return output_files, products


def run_phase1() -> pd.DataFrame:
    """Runs Phase 1 for both L-band and S-band, compiles polarimetric statistics."""
    processor = DFSARProcessor()
    summary_rows = []

    for band in ["L", "S"]:
        print(f"[*] Processing Chandrayaan-2 DFSAR {band}-band...")
        out_files, prods = processor.process_and_export(band=band)
        
        cpr = prods["CPR"]
        dop = prods["DOP"]
        pv = prods["P_volume"]
        ps = prods["P_surface"]
        pdb = prods["P_double"]

        summary_rows.append({
            "Band": band,
            "Mean_CPR": float(np.mean(cpr)),
            "Std_CPR": float(np.std(cpr)),
            "Median_CPR": float(np.median(cpr)),
            "Percent_CPR_gt_1": float(100.0 * np.sum(cpr > processor.radar_cfg.cpr_ice_threshold) / cpr.size),
            "Mean_DOP": float(np.mean(dop)),
            "Percent_DOP_lt_013": float(100.0 * np.sum(dop < processor.radar_cfg.dop_ice_max) / dop.size),
            "Mean_Vol_Fraction": float(np.mean(pv / (pv + ps + pdb + 1e-6))),
            "Mean_Surface_Fraction": float(np.mean(ps / (pv + ps + pdb + 1e-6))),
            "Mean_Double_Fraction": float(np.mean(pdb / (pv + ps + pdb + 1e-6)))
        })

    df_summary = pd.DataFrame(summary_rows)
    table_path = TABLES_DIR / "phase1_polarimetric_summary.csv"
    df_summary.to_csv(table_path, index=False)
    print(f"[+] Polarimetric summary written to {table_path}")
    return df_summary


if __name__ == "__main__":
    df = run_phase1()
    print(df.to_string())
