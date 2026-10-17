"""
LunaTrace Geospatial Utilities
Raster I/O, cartographic visualization, speckle filtering, and benchmark lunar dataset generation.
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy.ndimage import uniform_filter

# Windows DLL helper
if sys.platform == "win32":
    site_packages = [p for p in sys.path if "site-packages" in p]
    for sp in site_packages:
        libs_dir = os.path.join(sp, "rasterio.libs")
        if os.path.isdir(libs_dir):
            try:
                os.add_dll_directory(libs_dir)
            except Exception:
                pass

import rasterio
from rasterio.transform import from_origin
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
    """
    img = image.astype(np.float64)
    if is_intensity:
        img = np.maximum(img, 1e-10)
        mean_val = uniform_filter(img, size=window_size)
        mean_sqr = uniform_filter(img ** 2, size=window_size)
        var_val = np.maximum(mean_sqr - mean_val ** 2, 0.0)
        
        cu = 1.0 / np.sqrt(num_looks)
        ci = np.sqrt(var_val) / (mean_val + 1e-10)
        cu_max = np.sqrt(1.0 + 2.0 / num_looks)
        
        weight = np.clip((ci - cu) / (cu_max - cu + 1e-10), 0.0, 1.0)
        filtered = mean_val + weight * (img - mean_val)
    else:
        # Multilook spatial spatial averaging for signed Stokes channels (S2, S3, S4)
        filtered = uniform_filter(img, size=window_size)
        
    return filtered


def save_geotiff(
    data: np.ndarray,
    output_path: Path,
    transform: Affine,
    crs: str = CFG.terrain.projection_proj4,
    nodata: float = -9999.0
) -> None:
    """Saves a 2D numpy array as a georeferenced GeoTIFF raster."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype if data.dtype in [np.float32, np.float64] else np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw"
    ) as dst:
        clean_data = np.nan_to_num(data, nan=nodata)
        dst.write(clean_data.astype(np.float32), 1)


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
    points_to_plot: Optional[list] = None,  # List of tuples: (x, y, label, marker, color)
    traverses_to_plot: Optional[list] = None # List of tuples: (coords_x, coords_y, label, color, style)
) -> Path:
    """
    Renders publication-standard lunar maps with scale bar, coordinates, North arrow, and high-contrast colorbars.
    """
    if extent_km is None:
        ext = CFG.terrain.extent_km
        extent_km = (-ext, ext, -ext, ext)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    
    norm = Normalize(vmin=vmin if vmin is not None else np.nanpercentile(data, 2),
                     vmax=vmax if vmax is not None else np.nanpercentile(data, 98))
    
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
            
    # Labels & Title
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("East-West Distance from Crater Center (km)", fontsize=10)
    ax.set_ylabel("North-South Distance from Crater Center (km)", fontsize=10)
    
    # Scale Bar (AnchoredSizeBar)
    bar_length_km = CFG.terrain.scale_bar_length_km
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
    
    # North Arrow
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
    
    # Colorbar
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
