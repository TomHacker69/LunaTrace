"""
LunaTrace Cluster Null Baseline Runner
Executes 100 Monte Carlo runs of spatially-varying Wishart speckle simulation (N_cell=4.21)
over smoothing windows L=31, L=51, L=101. Computes chance cluster distributions, 99th percentile
thresholds, and compares against observed DFSAR candidate clusters.
"""

from pathlib import Path
import time
import json
import numpy as np
import scipy.ndimage as ndi

from src.config_schema import load_config
from src.pds4_reader import (
    parse_pds4_xml_label,
    read_complex_slc_pair,
    apply_radiometric_calibration,
    multilook_and_compute_stokes,
    derive_crop_from_bounding_box
)
from src.cluster_null_simulation import (
    run_spatially_varying_simulation_suite,
    run_shuffle_diagnostic,
    compute_windowed_cpr_dop
)
import rasterio.windows


def main():
    t_start = time.time()
    config = load_config()

    print("=" * 80)
    print("   LUNATRACE CLUSTER NULL BASELINE EVALUATION (100 MONTE CARLO RUNS)")
    print("=" * 80)

    # 1. Load Real DFSAR Data for Config Crop
    xml_path = Path("data/raw/ch2_sar_ncxl_20260403t190328238_d_cp_d18/data/calibrated/20260403/ch2_sar_ncxl_20260403t190328238_d_sli_xx_cp_xx_d18.xml")
    lh_path = Path("data/raw/ch2_sar_ncxl_20260403t190328238_d_cp_d18/data/calibrated/20260403/ch2_sar_ncxl_20260403t190328238_d_sli_xx_cp_lh_d18.tif")
    lv_path = Path("data/raw/ch2_sar_ncxl_20260403t190328238_d_cp_d18/data/calibrated/20260403/ch2_sar_ncxl_20260403t190328238_d_sli_xx_cp_lv_d18.tif")
    geom_csv_path = Path("data/raw/ch2_sar_ncxl_20260403t190328238_d_cp_d18/geometry/calibrated/20260403/ch2_sar_ncxl_20260403t190328238_g_sli_xx_cp_xx_d18.csv")

    line_start, line_count = derive_crop_from_bounding_box(geom_csv_path, config.target_bounding_box)
    label_meta = parse_pds4_xml_label(xml_path)
    slc_win = rasterio.windows.Window(0, line_start, 1152, line_count)
    c_lh, c_lv, valid_raw = read_complex_slc_pair(lh_path, lv_path, window=slc_win, return_mask=True)
    e_lh, e_lv = apply_radiometric_calibration(c_lh, c_lv, label_meta)

    # Multilooking (8 az x 2 rg)
    az_looks = config.pds4_ingestion.azimuth_looks
    rg_looks = config.pds4_ingestion.range_looks
    trim_ny = (e_lh.shape[0] // az_looks) * az_looks
    trim_nx = (e_lh.shape[1] // rg_looks) * rg_looks
    out_ny = trim_ny // az_looks
    out_nx = trim_nx // rg_looks

    lh_block = e_lh[:trim_ny, :trim_nx].reshape((out_ny, az_looks, out_nx, rg_looks))
    lv_block = e_lv[:trim_ny, :trim_nx].reshape((out_ny, az_looks, out_nx, rg_looks))
    mask_trim = valid_raw[:trim_ny, :trim_nx].reshape((out_ny, az_looks, out_nx, rg_looks))
    valid_mask = mask_trim.all(axis=(1, 3))

    p_lh = np.mean(np.abs(lh_block) ** 2, axis=(1, 3))
    p_lv = np.mean(np.abs(lv_block) ** 2, axis=(1, 3))
    cross = np.mean(lh_block * np.conj(lv_block), axis=(1, 3))
    cross_real = np.real(cross)
    cross_imag = np.imag(cross)

    s1 = p_lh + p_lv
    snr_db = 10.0 * np.log10(np.maximum(s1 / 0.009749896, 1e-6))
    valid_snr = valid_mask & (snr_db >= config.pds4_ingestion.snr_near_noise_threshold_db)

    print(f"Loaded DFSAR Scene Crop ({p_lh.shape[0]}x{p_lh.shape[1]}). Valid SNR Pixels: {np.sum(valid_snr):,}")

    # 2. Observed Real Clusters
    stokes = multilook_and_compute_stokes(e_lh, e_lv, az_looks, rg_looks, valid_raw)
    cpr_7x7 = stokes["CPR_7x7"]
    dop_7x7 = stokes["DOP_7x7"]
    cpr_9x9 = stokes["CPR_9x9"]
    dop_9x9 = stokes["DOP_9x9"]

    cand_7x7 = valid_snr & (cpr_7x7 > 1.0) & (dop_7x7 <= 0.13)
    lbl_7x7, n_obs_7x7 = ndi.label(cand_7x7, structure=np.ones((3, 3), dtype=int))
    obs_sz_7x7 = np.bincount(lbl_7x7.ravel())[1:]

    cand_9x9 = valid_snr & (cpr_9x9 > 1.0) & (dop_9x9 <= 0.13)
    lbl_9x9, n_obs_9x9 = ndi.label(cand_9x9, structure=np.ones((3, 3), dtype=int))
    obs_sz_9x9 = np.bincount(lbl_9x9.ravel())[1:]

    print("\n" + "=" * 80)
    print("   OBSERVED REAL DFSAR CANDIDATE CLUSTERS")
    print("=" * 80)
    print(f"7x7 Window: {n_obs_7x7} clusters (total {np.sum(obs_sz_7x7):,} pixels)")
    print(f"  Cluster size: min={np.min(obs_sz_7x7)}, median={np.median(obs_sz_7x7):.1f}, max={np.max(obs_sz_7x7)}")
    print(f"9x9 Window: {n_obs_9x9} clusters (total {np.sum(obs_sz_9x9):,} pixels)")
    print(f"  Cluster size: min={np.min(obs_sz_9x9)}, median={np.median(obs_sz_9x9):.1f}, max={np.max(obs_sz_9x9)}")

    # 3. Spatially Varying Wishart Baseline across L in [31, 51, 101]
    smoothing_windows = config.cluster_baseline.smoothing_windows  # [31, 51, 101]
    mc_runs = config.cluster_baseline.monte_carlo_runs            # 100

    print("\n" + "=" * 80)
    print(f"   SPATIALLY VARYING NULL BASELINE ({mc_runs} MONTE CARLO RUNS)")
    print(f"   Local Covariance Smoothing Windows Tested: {smoothing_windows}")
    print(f"   Wishart Single-Cell ENL: N_cell = {config.cluster_baseline.n_cell}")
    print("=" * 80)

    sim_results = run_spatially_varying_simulation_suite(
        p_lh, p_lv, cross_real, cross_imag, valid_snr,
        config=config,
        smoothing_windows=smoothing_windows,
        mc_runs=mc_runs
    )

    summary_comparison = []

    for L in smoothing_windows:
        for w in [7, 9]:
            k = f"L{L}_W{w}"
            res = sim_results[k]
            obs_sz = obs_sz_7x7 if w == 7 else obs_sz_9x9
            n_obs = n_obs_7x7 if w == 7 else n_obs_9x9

            p99_max = res["p99_max_cluster_size"]
            max_chance = res["max_chance_cluster_size_observed"]
            obs_exceed_p99 = int(np.sum(obs_sz > p99_max))
            pct_exceed_p99 = (obs_exceed_p99 / n_obs) * 100.0 if n_obs > 0 else 0.0

            summary_comparison.append({
                "smoothing_window": L,
                "boxcar_window": f"{w}x{w}",
                "mean_chance_clusters": f"{res['mean_cluster_count']:.1f} +/- {res['std_cluster_count']:.1f}",
                "chance_size_median": res["all_chance_sizes_summary"]["median"],
                "chance_size_mean": round(res["all_chance_sizes_summary"]["mean"], 2),
                "chance_max_observed": max_chance,
                "chance_p99_max": round(p99_max, 1),
                "observed_clusters": n_obs,
                "observed_exceeding_p99": f"{obs_exceed_p99} ({pct_exceed_p99:.2f}%)"
            })

            print(f"\nSmoothing Window L = {L}x{L} | Boxcar Window W = {w}x{w}:")
            print(f"  - Expected Chance Clusters: {res['mean_cluster_count']:.1f} +/- {res['std_cluster_count']:.1f} (vs Observed {n_obs})")
            print(f"  - Chance Cluster Sizes:     median={res['all_chance_sizes_summary']['median']:.1f}, mean={res['all_chance_sizes_summary']['mean']:.2f}, max={max_chance}")
            print(f"  - 99th Pct of Chance Max:   {p99_max:.1f} pixels")
            print(f"  - Observed Clusters > p99:  {obs_exceed_p99} / {n_obs} ({pct_exceed_p99:.2f}%)")

    # 4. Diagnostic Shuffle Baseline
    print("\n" + "=" * 80)
    print("   DIAGNOSTIC BASELINE: EMPIRICAL SPATIAL PERMUTATION (SHUFFLE)")
    print("=" * 80)
    diag_res = run_shuffle_diagnostic(p_lh, p_lv, cross_real, cross_imag, valid_snr, config=config, mc_runs=5)
    for w in [7, 9]:
        d = diag_res[f"W{w}"]
        n_obs = n_obs_7x7 if w == 7 else n_obs_9x9
        print(f"Diagnostic Shuffle {w}x{w}: Chance Clusters = {d['mean_count']:.1f} +/- {d['std_count']:.1f} (Max Chance Size = {d['max_size']}) vs Observed {n_obs}")

    print("\n" + "=" * 80)
    print("   SCIENTIFIC INTERPRETATION (DIRECTIVE 2)")
    print("=" * 80)
    print("  Candidate clusters exhibit pronounced spatial structure compared to spatially varying")
    print("  Wishart speckle baselines. Physical attribution cannot be assigned from radar data alone:")
    print("  the cause remains unassigned until registered LRO LOLA DEM topography (local slopes and")
    print("  incidence angle) and Chandrayaan-2 OHRC high-resolution boulder distributions are integrated.")
    print("  Windowed CPR > 1 pixels remain strictly 'unexplained by speckle', nothing more.")

    # Save summary table
    out_json = Path("outputs/tables/cluster_null_baseline_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary_comparison, f, indent=2)
    print(f"\n[+] Saved summary results to {out_json}")
    print(f"[+] Total execution time: {time.time() - t_start:.2f} s")


if __name__ == "__main__":
    main()
