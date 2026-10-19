"""
Tests for Polarimetric Cluster Null Baseline Simulation
Verifies Bartlett Wishart decomposition, spatially varying covariance estimation,
and reproducibility under configured seeds.
"""

import pytest
import numpy as np
from src.config_schema import load_config
from src.cluster_null_simulation import (
    estimate_spatially_varying_covariance,
    simulate_wishart_field_around_covariance,
    compute_windowed_cpr_dop,
    run_spatially_varying_simulation_suite
)


def test_bartlett_wishart_covariance_conservation():
    """Verifies that Bartlett decomposition preserves local mean 2x2 covariance."""
    rng = np.random.default_rng(42)
    ny, nx = 200, 100
    n_cell = 4.21

    c11 = np.full((ny, nx), 0.030, dtype=np.float32)
    c22 = np.full((ny, nx), 0.025, dtype=np.float32)
    cr = np.full((ny, nx), 0.004, dtype=np.float32)
    ci = np.full((ny, nx), -0.010, dtype=np.float32)
    vmask = np.ones((ny, nx), dtype=bool)

    # Average over 50 realizations
    w11_acc = np.zeros((ny, nx), dtype=np.float64)
    w22_acc = np.zeros((ny, nx), dtype=np.float64)
    wr_acc = np.zeros((ny, nx), dtype=np.float64)
    wi_acc = np.zeros((ny, nx), dtype=np.float64)
    n_runs = 50

    for _ in range(n_runs):
        w11, w22, wr, wi = simulate_wishart_field_around_covariance(
            c11, c22, cr, ci, vmask, n_cell=n_cell, rng=rng
        )
        w11_acc += w11
        w22_acc += w22
        wr_acc += wr
        wi_acc += wi

    mean_w11 = float(np.mean(w11_acc / n_runs))
    mean_w22 = float(np.mean(w22_acc / n_runs))
    mean_wr = float(np.mean(wr_acc / n_runs))
    mean_wi = float(np.mean(wi_acc / n_runs))

    assert np.isclose(mean_w11, 0.030, rtol=0.05)
    assert np.isclose(mean_w22, 0.025, rtol=0.05)
    assert np.isclose(mean_wr, 0.004, atol=0.002)
    assert np.isclose(mean_wi, -0.010, atol=0.002)


def test_spatially_varying_simulation_reproducibility():
    """Verifies that running with identical seeds produces identical cluster baseline metrics."""
    config = load_config()
    ny, nx = 150, 80
    vmask = np.ones((ny, nx), dtype=bool)
    rng = np.random.default_rng(123)

    p_lh = rng.uniform(0.01, 0.05, size=(ny, nx)).astype(np.float32)
    p_lv = rng.uniform(0.01, 0.05, size=(ny, nx)).astype(np.float32)
    cr = rng.uniform(-0.005, 0.005, size=(ny, nx)).astype(np.float32)
    ci = rng.uniform(-0.01, -0.001, size=(ny, nx)).astype(np.float32)

    res1 = run_spatially_varying_simulation_suite(
        p_lh, p_lv, cr, ci, vmask, config=config, smoothing_windows=[31], mc_runs=3
    )
    res2 = run_spatially_varying_simulation_suite(
        p_lh, p_lv, cr, ci, vmask, config=config, smoothing_windows=[31], mc_runs=3
    )

    assert res1["L31_W7"]["mean_cluster_count"] == res2["L31_W7"]["mean_cluster_count"]
    assert res1["L31_W7"]["p99_max_cluster_size"] == res2["L31_W7"]["p99_max_cluster_size"]
