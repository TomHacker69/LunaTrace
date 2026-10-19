"""
LunaTrace Polarimetric Cluster Null Baseline Simulation
Implements spatially-varying 2x2 complex Wishart speckle simulation around local mean covariance,
and empirical spatial permutation diagnostic baseline. Evaluates chance cluster distributions
against observed DFSAR slant-range candidate clusters.
"""

from pathlib import Path
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter

from src.config_schema import load_config, LunaTraceConfig
from src.pds4_reader import (
    parse_pds4_xml_label,
    read_complex_slc_pair,
    apply_radiometric_calibration,
    multilook_and_compute_stokes,
    derive_crop_from_bounding_box
)
import rasterio.windows


def estimate_spatially_varying_covariance(
    p_lh: np.ndarray,
    p_lv: np.ndarray,
    cross_real: np.ndarray,
    cross_imag: np.ndarray,
    valid_mask: np.ndarray,
    smoothing_window: int = 51
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates the spatially varying local mean 2x2 complex covariance matrix
    over a large boxcar smoothing window (e.g. 31x31, 51x51, 101x101).
    """
    v_float = valid_mask.astype(np.float32)
    count = uniform_filter(v_float, size=smoothing_window)
    denom = np.maximum(count, 1e-6)

    c11_sum = uniform_filter(np.where(valid_mask, p_lh, 0.0).astype(np.float32), size=smoothing_window)
    c22_sum = uniform_filter(np.where(valid_mask, p_lv, 0.0).astype(np.float32), size=smoothing_window)
    cr_sum = uniform_filter(np.where(valid_mask, cross_real, 0.0).astype(np.float32), size=smoothing_window)
    ci_sum = uniform_filter(np.where(valid_mask, cross_imag, 0.0).astype(np.float32), size=smoothing_window)

    c11_mean = c11_sum / denom
    c22_mean = c22_sum / denom
    cr_mean = cr_sum / denom
    ci_mean = ci_sum / denom

    return c11_mean, c22_mean, cr_mean, ci_mean


def simulate_wishart_field_around_covariance(
    c11_mean: np.ndarray,
    c22_mean: np.ndarray,
    cr_mean: np.ndarray,
    ci_mean: np.ndarray,
    valid_mask: np.ndarray,
    n_cell: float = 4.21,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates a 2D field of 2x2 complex Wishart distributed matrices with N_cell looks
    around the spatially varying local mean covariance using Bartlett decomposition.
    """
    if rng is None:
        rng = np.random.default_rng()

    shape = c11_mean.shape
    eps = 1e-10

    # Cholesky factor of local mean covariance L = [[l11, 0], [l21, l22]]
    l11 = np.sqrt(np.maximum(c11_mean, eps))
    l21_r = cr_mean / l11
    l21_i = -ci_mean / l11
    l21_mag_sq = l21_r**2 + l21_i**2
    l22 = np.sqrt(np.maximum(c22_mean - l21_mag_sq, eps))

    # Bartlett decomposition for Wishart A ~ W_2(N, I)
    a11 = rng.gamma(shape=n_cell, scale=1.0, size=shape).astype(np.float32)
    a22_base = rng.gamma(shape=max(n_cell - 1.0, 0.1), scale=1.0, size=shape).astype(np.float32)
    zr = rng.normal(0, np.sqrt(0.5), size=shape).astype(np.float32)
    zi = rng.normal(0, np.sqrt(0.5), size=shape).astype(np.float32)
    a22 = a22_base + (zr**2 + zi**2)
    sqrt_a11 = np.sqrt(a11)
    a12_r = sqrt_a11 * zr
    a12_i = sqrt_a11 * zi

    # W = (1/N) * L * A * L^H
    inv_n = 1.0 / n_cell
    w11 = inv_n * (l11**2 * a11)
    w22 = inv_n * (l21_mag_sq * a11 + l22**2 * a22 + 2.0 * l22 * (l21_r * a12_r + l21_i * a12_i))
    w12_r = inv_n * (l11 * (a12_r * l22 + a11 * l21_r))
    w12_i = inv_n * (l11 * (a12_i * l22 - a11 * l21_i))

    # Mask invalid margin
    w11 = np.where(valid_mask, w11, 0.0).astype(np.float32)
    w22 = np.where(valid_mask, w22, 0.0).astype(np.float32)
    w12_r = np.where(valid_mask, w12_r, 0.0).astype(np.float32)
    w12_i = np.where(valid_mask, w12_i, 0.0).astype(np.float32)

    return w11, w22, w12_r, w12_i


def compute_windowed_cpr_dop(
    p_lh: np.ndarray,
    p_lv: np.ndarray,
    cross_real: np.ndarray,
    cross_imag: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes CPR and Degree of Polarization (DoP) from smoothed covariance
    over an identical window_size x window_size boxcar.
    """
    v_float = valid_mask.astype(np.float32)
    count = uniform_filter(v_float, size=window_size)
    valid_win = valid_mask & (count > 0.5)
    denom_count = np.maximum(count, 1e-6)

    s1_w = (uniform_filter(p_lh, size=window_size) + uniform_filter(p_lv, size=window_size)) / denom_count
    s2_w = (uniform_filter(p_lh, size=window_size) - uniform_filter(p_lv, size=window_size)) / denom_count
    s3_w = (2.0 * uniform_filter(cross_real, size=window_size)) / denom_count
    s4_w = (-2.0 * uniform_filter(cross_imag, size=window_size)) / denom_count

    denom_cpr = np.where(valid_win & ((s1_w + s4_w) > 1e-10), s1_w + s4_w, 1.0)
    cpr_w = np.where(valid_win, (s1_w - s4_w) / denom_cpr, np.nan)

    pol_pow_w = np.sqrt(np.maximum(s2_w**2 + s3_w**2 + s4_w**2, 0.0))
    dop_w = np.where(valid_win & (s1_w > 1e-10), pol_pow_w / np.maximum(s1_w, 1e-10), np.nan)

    return cpr_w, dop_w


def run_spatially_varying_simulation_suite(
    p_lh: np.ndarray,
    p_lv: np.ndarray,
    cross_real: np.ndarray,
    cross_imag: np.ndarray,
    valid_mask: np.ndarray,
    config: LunaTraceConfig,
    smoothing_windows: Optional[List[int]] = None,
    mc_runs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes the spatially varying null baseline over configured smoothing window sizes
    and Monte Carlo runs. Compares chance cluster distribution with observed clusters.
    """
    if smoothing_windows is None:
        smoothing_windows = config.cluster_baseline.smoothing_windows
    if mc_runs is None:
        mc_runs = config.cluster_baseline.monte_carlo_runs

    base_seed = config.cluster_baseline.random_seed
    n_cell = config.cluster_baseline.n_cell
    cpr_thresh = config.cluster_baseline.cpr_threshold
    dop_thresh = config.cluster_baseline.dop_threshold

    results: Dict[str, Any] = {}

    for L in smoothing_windows:
        print(f"\n[*] Evaluating Spatially Varying Null (Smoothing Window L = {L}x{L}, {mc_runs} MC Runs)...")
        c11_m, c22_m, cr_m, ci_m = estimate_spatially_varying_covariance(
            p_lh, p_lv, cross_real, cross_imag, valid_mask, smoothing_window=L
        )

        counts_7: List[int] = []
        max_sz_7: List[int] = []
        all_sz_7: List[int] = []

        counts_9: List[int] = []
        max_sz_9: List[int] = []
        all_sz_9: List[int] = []

        for run_idx in range(mc_runs):
            rng = np.random.default_rng(base_seed + run_idx * 1000 + L * 10)
            w11, w22, w12_r, w12_i = simulate_wishart_field_around_covariance(
                c11_m, c22_m, cr_m, ci_m, valid_mask, n_cell=n_cell, rng=rng
            )

            # Evaluate 7x7
            cpr_7, dop_7 = compute_windowed_cpr_dop(w11, w22, w12_r, w12_i, valid_mask, window_size=7)
            cand_7 = valid_mask & (cpr_7 > cpr_thresh) & (dop_7 <= dop_thresh)
            lbl_7, n7 = ndi.label(cand_7, structure=np.ones((3, 3), dtype=int))
            if n7 > 0:
                sz7 = np.bincount(lbl_7.ravel())[1:]
                counts_7.append(n7)
                max_sz_7.append(int(np.max(sz7)))
                all_sz_7.extend(sz7.tolist())
            else:
                counts_7.append(0)
                max_sz_7.append(0)

            # Evaluate 9x9
            cpr_9, dop_9 = compute_windowed_cpr_dop(w11, w22, w12_r, w12_i, valid_mask, window_size=9)
            cand_9 = valid_mask & (cpr_9 > cpr_thresh) & (dop_9 <= dop_thresh)
            lbl_9, n9 = ndi.label(cand_9, structure=np.ones((3, 3), dtype=int))
            if n9 > 0:
                sz9 = np.bincount(lbl_9.ravel())[1:]
                counts_9.append(n9)
                max_sz_9.append(int(np.max(sz9)))
                all_sz_9.extend(sz9.tolist())
            else:
                counts_9.append(0)
                max_sz_9.append(0)

        for w_size, c_list, m_list, a_list in [(7, counts_7, max_sz_7, all_sz_7), (9, counts_9, max_sz_9, all_sz_9)]:
            p99_max_sz = float(np.percentile(m_list, 99)) if m_list else 0.0
            key = f"L{L}_W{w_size}"
            results[key] = {
                "smoothing_window": L,
                "boxcar_window": w_size,
                "mc_runs": mc_runs,
                "mean_cluster_count": float(np.mean(c_list)),
                "std_cluster_count": float(np.std(c_list)),
                "p99_max_cluster_size": p99_max_sz,
                "max_chance_cluster_size_observed": int(np.max(m_list)) if m_list else 0,
                "all_chance_sizes_summary": {
                    "min": int(np.min(a_list)) if a_list else 0,
                    "median": float(np.median(a_list)) if a_list else 0.0,
                    "mean": float(np.mean(a_list)) if a_list else 0.0,
                    "max": int(np.max(a_list)) if a_list else 0
                }
            }

    return results


def run_shuffle_diagnostic(
    p_lh: np.ndarray,
    p_lv: np.ndarray,
    cross_real: np.ndarray,
    cross_imag: np.ndarray,
    valid_mask: np.ndarray,
    config: LunaTraceConfig,
    mc_runs: int = 5
) -> Dict[str, Any]:
    """
    Diagnostic Baseline: Evaluates candidate clusters under spatial permutation
    (scrambling) of native cells across valid pixels.
    """
    valid_idx = np.where(valid_mask)
    n_v = len(valid_idx[0])
    base_seed = config.cluster_baseline.random_seed
    cpr_thresh = config.cluster_baseline.cpr_threshold
    dop_thresh = config.cluster_baseline.dop_threshold

    diag_results: Dict[str, Any] = {}

    for w_size in [7, 9]:
        counts = []
        max_sizes = []
        for r in range(mc_runs):
            rng = np.random.default_rng(base_seed + r)
            perm = rng.permutation(n_v)

            p_lh_shuf = np.zeros_like(p_lh)
            p_lv_shuf = np.zeros_like(p_lv)
            cr_shuf = np.zeros_like(cross_real)
            ci_shuf = np.zeros_like(cross_imag)

            p_lh_shuf[valid_idx] = p_lh[valid_idx][perm]
            p_lv_shuf[valid_idx] = p_lv[valid_idx][perm]
            cr_shuf[valid_idx] = cross_real[valid_idx][perm]
            ci_shuf[valid_idx] = cross_imag[valid_idx][perm]

            cpr_w, dop_w = compute_windowed_cpr_dop(
                p_lh_shuf, p_lv_shuf, cr_shuf, ci_shuf, valid_mask, window_size=w_size
            )
            cand_mask = valid_mask & (cpr_w > cpr_thresh) & (dop_w <= dop_thresh)
            lbl, n_clust = ndi.label(cand_mask, structure=np.ones((3, 3), dtype=int))
            counts.append(n_clust)
            if n_clust > 0:
                sz = np.bincount(lbl.ravel())[1:]
                max_sizes.append(int(np.max(sz)))
            else:
                max_sizes.append(0)

        diag_results[f"W{w_size}"] = {
            "mean_count": float(np.mean(counts)),
            "std_count": float(np.std(counts)),
            "max_size": int(np.max(max_sizes)) if max_sizes else 0
        }

    return diag_results
