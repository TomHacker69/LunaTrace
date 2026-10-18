"""
Phase 1: DFSAR Polarimetric Processing Pipeline
Computes Stokes parameters, CPR, DOP, and Raney m-chi decomposition
for Chandrayaan-2 DFSAR L-band and S-band hybrid-polarimetric data.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import rasterio

from src.config import CFG
from src.constants import (
    DFSAR_L_BAND_FREQ_GHZ,
    DFSAR_S_BAND_FREQ_GHZ,
    DFSAR_L_BAND_WAVELENGTH_M,
    DFSAR_S_BAND_WAVELENGTH_M
)
from src.utils import enhanced_lee_filter, save_geotiff, plot_cartographic_map
from src.provenance import write_provenance_sidecar


class DFSARProcessor:
    """
    Chandrayaan-2 Dual Frequency SAR (DFSAR) Polarimetric Engine.
    Handles L-band and S-band hybrid polarimetric data (Right Circular Transmit, Linear H & V Receive).
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.radar_cfg = config.radar
        self.terrain_cfg = config.terrain
        self.raw_dir = config.paths.raw_data_dir
        self.processed_dir = config.paths.processed_data_dir
        self.tables_dir = config.paths.outputs_tables_dir
        self.maps_dir = config.paths.outputs_maps_dir

    def load_stokes_band(
        self,
        band: str = "L",
        apply_filter: bool = True
    ) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, str, List[Path]]:
        """
        Loads Stokes parameter rasters (S1, S2, S3, S4) for the given band ('L' or 'S').
        Applies speckle filtering to suppress multiplicative radar noise.
        Fails loudly if files are missing or CRS is unreferenced.
        """
        band = band.upper()
        prefix = f"dfsar_{band}_"
        stokes = {}
        meta_transform = None
        meta_crs = None
        loaded_paths = []

        for param in ["S1", "S2", "S3", "S4"]:
            fpath = self.raw_dir / f"{prefix}{param}.tif"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required radar dataset: {fpath}")

            loaded_paths.append(fpath)
            with rasterio.open(fpath) as src:
                if not src.crs:
                    raise ValueError(f"Raster {fpath} has no CRS defined in metadata!")
                data = src.read(1).astype(np.float64)
                if meta_transform is None:
                    meta_transform = src.transform
                    meta_crs = src.crs.to_string()

            if apply_filter:
                data = enhanced_lee_filter(
                    data,
                    window_size=self.radar_cfg.filter_window_size,
                    num_looks=self.radar_cfg.num_looks,
                    is_intensity=(param == "S1")
                )
            stokes[param] = data

        return stokes, meta_transform, meta_crs, loaded_paths

    def compute_polarimetric_products(
        self,
        stokes: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Computes Circular Polarization Ratio (CPR), Degree of Polarization (DOP),
        and Raney m-chi decomposition (Volume, Double-bounce, Surface scattering).
        """
        eps = self.cfg.numerics.eps
        s1 = np.maximum(stokes["S1"], self.radar_cfg.min_valid_stokes1)
        s2 = stokes["S2"]
        s3 = stokes["S3"]
        s4 = stokes["S4"]

        # 1. Circular Polarization Ratio (CPR)
        denom_cpr = np.maximum(s1 + s4, eps)
        cpr = np.clip((s1 - s4) / denom_cpr, self.radar_cfg.cpr_min_clip, self.radar_cfg.cpr_max_clip)

        # 2. Degree of Polarization (DOP, m)
        pol_intensity = np.sqrt(s2 ** 2 + s3 ** 2 + s4 ** 2)
        dop = np.clip(pol_intensity / s1, self.radar_cfg.dop_min_clip, self.radar_cfg.dop_max_clip)

        # 3. Raney m-chi Decomposition
        denom_chi = np.maximum(dop * s1, eps)
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

    def process_and_export(self, band: str = "L") -> Tuple[Dict[str, Path], Dict[str, np.ndarray]]:
        """
        Executes end-to-end processing for a single DFSAR band, saving GeoTIFFs,
        generating cartographic figures, and generating cryptographic provenance sidecars.
        """
        band = band.upper()
        stokes, transform, crs, input_paths = self.load_stokes_band(band=band)
        products = self.compute_polarimetric_products(stokes)

        output_files = {}
        for name, arr in products.items():
            out_path = self.processed_dir / f"dfsar_{band}_{name}.tif"
            save_geotiff(arr, out_path, transform, crs=crs)
            write_provenance_sidecar(
                output_path=out_path,
                input_paths=input_paths,
                config_dict=self.cfg.model_dump(),
                metadata={"band": band, "product": name}
            )
            output_files[name] = out_path

        # Generate Cartographic Maps
        # 1. CPR Map
        cpr_map_path = plot_cartographic_map(
            data=products["CPR"],
            title=f"Chandrayaan-2 DFSAR {band}-Band Circular Polarization Ratio (CPR)",
            output_filename=f"map_phase1_{band}_cpr.png",
            cmap="inferno",
            vmin=self.cfg.visualization.cpr_plot_vmin,
            vmax=self.cfg.visualization.cpr_plot_vmax,
            cbar_label="CPR (Same-Sense / Opposite-Sense)"
        )
        write_provenance_sidecar(
            output_path=cpr_map_path,
            input_paths=input_paths,
            config_dict=self.cfg.model_dump(),
            metadata={"band": band, "map": "CPR"}
        )
        output_files["map_cpr"] = cpr_map_path

        # 2. DOP Map
        dop_map_path = plot_cartographic_map(
            data=products["DOP"],
            title=f"Chandrayaan-2 DFSAR {band}-Band Degree of Polarization (DoP)",
            output_filename=f"map_phase1_{band}_dop.png",
            cmap="plasma_r",
            vmin=self.cfg.visualization.dop_plot_vmin,
            vmax=self.cfg.visualization.dop_plot_vmax,
            cbar_label="Degree of Polarization (m)"
        )
        write_provenance_sidecar(
            output_path=dop_map_path,
            input_paths=input_paths,
            config_dict=self.cfg.model_dump(),
            metadata={"band": band, "map": "DOP"}
        )
        output_files["map_dop"] = dop_map_path

        # 3. Raney m-chi Decomposition RGB Composite Map
        pv = products["P_volume"]
        pdb = products["P_double"]
        ps = products["P_surface"]

        eps = self.cfg.numerics.eps
        pct = self.cfg.visualization.default_vmax_percentile
        norm_r = np.clip(pdb / (np.percentile(pdb, pct) + eps), 0, 1)
        norm_g = np.clip(pv / (np.percentile(pv, pct) + eps), 0, 1)
        norm_b = np.clip(ps / (np.percentile(ps, pct) + eps), 0, 1)
        rgb = np.stack([norm_r, norm_g, norm_b], axis=-1)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8), dpi=self.cfg.visualization.dpi)
        ext = self.terrain_cfg.extent_km
        ax.imshow(rgb, extent=[-ext, ext, -ext, ext], origin="upper")
        ax.set_title(f"DFSAR {band}-Band m-chi Decomposition (R: Double, G: Volume, B: Surface)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("East-West Distance (km)")
        ax.set_ylabel("North-South Distance (km)")
        ax.grid(color="gray", linestyle="--", alpha=0.3)
        rgb_out = self.maps_dir / f"map_phase1_{band}_mchi_decomposition.png"
        fig.tight_layout()
        fig.savefig(rgb_out, bbox_inches="tight")
        plt.close(fig)

        write_provenance_sidecar(
            output_path=rgb_out,
            input_paths=input_paths,
            config_dict=self.cfg.model_dump(),
            metadata={"band": band, "map": "mchi_decomposition"}
        )
        output_files["map_mchi"] = rgb_out

        return output_files, products


def run_phase1(config: LunaTraceConfig = CFG) -> pd.DataFrame:
    """Runs Phase 1 for both L-band and S-band, compiles polarimetric statistics with provenance sidecars."""
    processor = DFSARProcessor(config=config)
    summary_rows = []
    all_inputs = []

    for band in ["L", "S"]:
        print(f"[*] Processing Chandrayaan-2 DFSAR {band}-band...")
        out_files, prods = processor.process_and_export(band=band)
        
        cpr = prods["CPR"]
        dop = prods["DOP"]
        pv = prods["P_volume"]
        ps = prods["P_surface"]
        pdb = prods["P_double"]

        for param in ["S1", "S2", "S3", "S4"]:
            all_inputs.append(processor.raw_dir / f"dfsar_{band}_{param}.tif")

        summary_rows.append({
            "Band": band,
            "Mean_CPR": float(np.mean(cpr)),
            "Std_CPR": float(np.std(cpr)),
            "Median_CPR": float(np.median(cpr)),
            "Percent_CPR_gt_1": float(100.0 * np.sum(cpr > processor.radar_cfg.cpr_ice_threshold) / cpr.size),
            "Mean_DOP": float(np.mean(dop)),
            "Percent_DOP_lt_032": float(100.0 * np.sum(dop < processor.radar_cfg.dop_ice_max) / dop.size),
            "Mean_Vol_Fraction": float(np.mean(pv / (pv + ps + pdb + processor.cfg.numerics.eps))),
            "Mean_Surface_Fraction": float(np.mean(ps / (pv + ps + pdb + processor.cfg.numerics.eps))),
            "Mean_Double_Fraction": float(np.mean(pdb / (pv + ps + pdb + processor.cfg.numerics.eps)))
        })

    df_summary = pd.DataFrame(summary_rows)
    table_path = processor.tables_dir / "phase1_polarimetric_summary.csv"
    df_summary.to_csv(table_path, index=False)
    write_provenance_sidecar(
        output_path=table_path,
        input_paths=all_inputs,
        config_dict=config.model_dump(),
        metadata={"phase": 1, "description": "Polarimetric summary statistics"}
    )
    print(f"[+] Polarimetric summary written to {table_path}")
    return df_summary
