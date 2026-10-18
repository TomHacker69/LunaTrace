"""
LunaTrace Geospatial Utilities
Raster I/O, cartographic visualization, and Lee speckle filtering.
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy.ndimage import uniform_filter

# Windows DLL helper
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = [p for p in sys.path if "site-packages" in p]
    for sp in site_packages:
        libs_dir = os.path.join(sp, "rasterio.libs")
        if os.path.isdir(libs_dir):
            try:
                os.add_dll_directory(libs_dir)
            except (FileNotFoundError, OSError):
                pass

import rasterio
from affine import Affine

from src.config import CFG, MAPS_DIR


def enhanced_lee_filter(
    image: np.ndarray,
    window_size: int = 5,
    num_looks: int = 4,
    is_intensity: bool = True
) -> np.ndarray:
    """
    Enhanced Lee speckle filter for radar imagery.
    Handles both non-negative intensity (S1) and signed polarimetric Stokes channels (S2, S3, S4).
    Citation: Lee, J. S. (1980). Digital image enhancement and noise filtering by use of local statistics.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 2(2), 165-168.
    """
    img = image.astype(np.float64)
    if is_intensity:
        img = np.maximum(img, CFG.numerics.eps)
        mean_val = uniform_filter(img, size=window_size)
        mean_sqr = uniform_filter(img ** 2, size=window_size)
        var_val = np.maximum(mean_sqr - mean_val ** 2, 0.0)

        cu = 1.0 / np.sqrt(num_looks)
        ci = np.sqrt(var_val) / (mean_val + CFG.numerics.eps)
        cu_max = np.sqrt(1.0 + 2.0 / num_looks)

        weight = np.clip((ci - cu) / (cu_max - cu + CFG.numerics.eps), 0.0, 1.0)
        filtered = mean_val + weight * (img - mean_val)
    else:
        # Multilook spatial averaging for signed Stokes channels (S2, S3, S4)
        filtered = uniform_filter(img, size=window_size)

    return filtered


def save_geotiff(
    data: np.ndarray,
    output_path: Path,
    transform: Affine,
    crs: str = CFG.crs.proj4_string,
    nodata: Optional[float] = -9999.0
) -> None:
    """
    Saves a 2D numpy array as a georeferenced GeoTIFF raster.
    Strictly validates input: fails if unexpected Infs exist.
    """
    if np.any(np.isinf(data)):
        raise FloatingPointError(f"Cannot save raster to {output_path}: array contains infinite values!")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape

    out_dtype = np.float32 if data.dtype in [np.float32, np.float64] else data.dtype

    actual_nodata = nodata
    if actual_nodata is not None and np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        if not (info.min <= actual_nodata <= info.max):
            actual_nodata = None

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=out_dtype,
        crs=crs,
        transform=transform,
        nodata=actual_nodata,
        compress="lzw"
    ) as dst:
        dst.write(data.astype(out_dtype), 1)


def plot_cartographic_map(
    data: np.ndarray,
    title: str,
    output_filename: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "",
    extent_km: Optional[Tuple[float, float, float, float]] = None,
    overlay_contour: Optional[np.ndarray] = None,
    contour_levels: Optional[list] = None,
    points_to_plot: Optional[list] = None,
    traverses_to_plot: Optional[list] = None
) -> Path:
    """
    Renders publication-standard lunar maps with scale bar, coordinates, North arrow, and high-contrast colorbars.
    """
    if extent_km is None:
        ext = CFG.terrain.extent_km
        extent_km = (-ext, ext, -ext, ext)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=CFG.visualization.dpi)

    default_vmin = np.nanpercentile(data, CFG.visualization.default_vmin_percentile)
    default_vmax = np.nanpercentile(data, CFG.visualization.default_vmax_percentile)

    norm = Normalize(vmin=vmin if vmin is not None else default_vmin,
                     vmax=vmax if vmax is not None else default_vmax)

    im = ax.imshow(
        data,
        extent=[extent_km[0], extent_km[1], extent_km[2], extent_km[3]],
        origin="upper",
        cmap=cmap,
        norm=norm
    )

    if overlay_contour is not None and contour_levels is not None:
        ax.contour(
            overlay_contour,
            levels=contour_levels,
            extent=[extent_km[0], extent_km[1], extent_km[2], extent_km[3]],
            origin="upper",
            colors="white",
            linewidths=0.7,
            alpha=0.6
        )

    if points_to_plot:
        for pt_x, pt_y, pt_label, pt_marker, pt_col in points_to_plot:
            ax.scatter(pt_x, pt_y, color=pt_col, marker=pt_marker, s=90, edgecolors="black", zorder=6, label=pt_label)
            ax.annotate(f" {pt_label}", (pt_x, pt_y), color="white", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    if traverses_to_plot:
        for tx, ty, tlabel, tcol, tstyle in traverses_to_plot:
            ax.plot(tx, ty, color=tcol, linestyle=tstyle, linewidth=2.5, label=tlabel, zorder=5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("East-West Distance from Crater Center (km)", fontsize=10)
    ax.set_ylabel("North-South Distance from Crater Center (km)", fontsize=10)

    # Scale bar
    bar_length_km = CFG.visualization.scale_bar_length_km
    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_length_km,
        f"{int(bar_length_km)} km",
        "lower left",
        pad=0.6,
        color="white",
        frameon=True,
        size_vertical=0.25,
        fontproperties=fm.FontProperties(size=9, weight="bold")
    )
    ax.add_artist(scalebar)

    # North arrow
    ax.annotate(
        "N",
        xy=(0.95, 0.95),
        xytext=(0.95, 0.86),
        xycoords="axes fraction",
        arrowprops=dict(facecolor="white", edgecolor="black", width=2, headwidth=8),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10, fontweight="bold")

    if points_to_plot or traverses_to_plot:
        ax.legend(loc="upper left", framealpha=0.85, fontsize=8)

    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    out_file = MAPS_DIR / output_filename
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
