"""
LunaTrace Primary Data Validation Layer
Validates presence, provenance, CRS, resolution, extent overlap, and physical value ranges.
Fails loudly with descriptive error messages upon any anomaly.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import rasterio

from src.config_schema import LunaTraceConfig


class DataValidationError(Exception):
    """Raised when input datasets fail spatial, physical, or provenance validation."""
    pass


def validate_input_datasets(
    raw_dir: Path,
    config: LunaTraceConfig,
    provenance_doc: Optional[Path] = None,
    enforce_provenance: bool = True
) -> Dict[str, Any]:
    """
    Validates input datasets prior to execution.
    Fails fast if files are missing, CRS mismatches, extents do not overlap,
    or physical values violate lunar boundaries.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    # Core required DFSAR and DEM rasters
    required_files = [
        "dfsar_L_S1.tif", "dfsar_L_S2.tif", "dfsar_L_S3.tif", "dfsar_L_S4.tif",
        "dfsar_S_S1.tif", "dfsar_S_S2.tif", "dfsar_S_S3.tif", "dfsar_S_S4.tif",
        "lola_dem_20m.tif", "crater_illumination.tif"
    ]

    missing = [f for f in required_files if not (raw_dir / f).exists()]
    if missing:
        raise DataValidationError(
            f"Missing required primary mission dataset(s) in {raw_dir}:\n" +
            "\n".join(f"  - {f}" for f in missing)
        )

    # Optional OHRC boulder assessment check
    ohrc_file = raw_dir / "ohrc_boulder_density.tif"
    boulder_assessed = ohrc_file.exists()

    # Enforce mission provenance against DATA_PROVENANCE.md
    if enforce_provenance:
        if provenance_doc is None:
            provenance_doc = raw_dir.parent.parent / "DATA_PROVENANCE.md"
        if not provenance_doc.exists():
            raise DataValidationError(f"Required provenance registry not found: {provenance_doc}")

        with open(provenance_doc, "r", encoding="utf-8") as f:
            prov_text = f.read()

        # Check that present files are acknowledged in registry
        for fname in required_files:
            if fname not in prov_text:
                raise DataValidationError(
                    f"File '{fname}' in {raw_dir} is not an authorized dataset recorded in {provenance_doc.name}!"
                )

    # Metadata & Geometry Validation
    base_transform = None
    base_shape = None
    base_bounds = None

    for fname in required_files:
        fpath = raw_dir / fname
        with rasterio.open(fpath) as src:
            # 1. CRS Check
            if not src.crs:
                raise DataValidationError(f"Dataset {fname} has no CRS defined in metadata!")
            
            # Check projection units are meters
            crs_wkt = src.crs.to_wkt().lower()
            if "metre" not in crs_wkt and "meter" not in crs_wkt and "+units=m" not in config.crs.proj4_string:
                raise DataValidationError(f"Dataset {fname} CRS does not use metric units: {src.crs}")

            # 2. Resolution Check
            res_x, res_y = abs(src.transform.a), abs(src.transform.e)
            expected_res = config.terrain.grid_resolution_m
            if abs(res_x - expected_res) > 0.05 or abs(res_y - expected_res) > 0.05:
                raise DataValidationError(
                    f"Dataset {fname} pixel resolution ({res_x:.2f} m, {res_y:.2f} m) does not match "
                    f"expected grid resolution ({expected_res:.2f} m)!"
                )

            # 3. Spatial Extent & Grid Alignment
            if base_transform is None:
                base_transform = src.transform
                base_shape = (src.height, src.width)
                base_bounds = src.bounds
            else:
                if (src.height, src.width) != base_shape:
                    raise DataValidationError(
                        f"Grid dimension mismatch in {fname}: expected {base_shape}, got {(src.height, src.width)}"
                    )
                if abs(src.bounds.left - base_bounds.left) > 1.0 or abs(src.bounds.top - base_bounds.top) > 1.0:
                    raise DataValidationError(
                        f"Spatial extent mismatch in {fname}: bounding box {src.bounds} does not align with {base_bounds}"
                    )

            # 4. Physical Value Ranges
            data_sample = src.read(1)
            if np.all(np.isnan(data_sample)):
                raise DataValidationError(f"Dataset {fname} contains only NaN values!")

            if "S1" in fname and np.any(data_sample < 0.0):
                neg_count = int(np.sum(data_sample < 0.0))
                raise DataValidationError(
                    f"Radar Stokes S1 intensity in {fname} contains {neg_count} negative pixels! "
                    "Physical radar intensity cannot be negative."
                )

            if "dem" in fname:
                min_elev, max_elev = float(np.nanmin(data_sample)), float(np.nanmax(data_sample))
                if min_elev < -9200.0 or max_elev > 11000.0:
                    raise DataValidationError(
                        f"DEM elevation range [{min_elev}, {max_elev}] in {fname} exceeds physical lunar extremes (-9100m to +10800m)!"
                    )

            if "illumination" in fname:
                min_i, max_i = float(np.nanmin(data_sample)), float(np.nanmax(data_sample))
                if min_i < -0.01 or max_i > 1.01:
                    raise DataValidationError(
                        f"Illumination values in {fname} out of bounds: [{min_i}, {max_i}]. Expected [0.0, 1.0]."
                    )

    return {
        "valid": True,
        "grid_shape": base_shape,
        "resolution_m": config.terrain.grid_resolution_m,
        "boulder_assessed": boulder_assessed,
        "total_rasters_validated": len(required_files) + (1 if boulder_assessed else 0)
    }
