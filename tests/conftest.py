"""
Pytest configuration and session fixtures for LunaTrace.
Generates isolated test fixtures inside tests/fixtures/data.
"""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pytest
from src.config_schema import load_config, LunaTraceConfig
from tests.fixtures.synthetic_crater_generator import generate_synthetic_test_crater


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Creates an isolated temporary directory with synthetic test fixtures."""
    fixtures_dir = tmp_path_factory.mktemp("luna_test_fixtures")
    raw_dir = fixtures_dir / "raw"
    processed_dir = fixtures_dir / "processed"
    maps_dir = fixtures_dir / "maps"
    tables_dir = fixtures_dir / "tables"

    for d in [raw_dir, processed_dir, maps_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate synthetic crater in test raw dir
    generate_synthetic_test_crater(output_dir=raw_dir)
    return fixtures_dir


@pytest.fixture(scope="session")
def test_config(test_data_dir: Path) -> LunaTraceConfig:
    """Loads a configuration pointing exclusively to test fixtures."""
    cfg = load_config()
    # Override paths to test directory
    cfg.paths.raw_data_dir = test_data_dir / "raw"
    cfg.paths.processed_data_dir = test_data_dir / "processed"
    cfg.paths.outputs_maps_dir = test_data_dir / "maps"
    cfg.paths.outputs_tables_dir = test_data_dir / "tables"
    return cfg


@pytest.fixture(scope="session")
def processed_pipeline(test_config: LunaTraceConfig):
    """Executes phases 1 to 6 on test fixtures once per test session to populate processed rasters."""
    from src.phase1_dfsar import run_phase1
    from src.phase2_ice_detection import run_phase2
    from src.phase3_terrain_illumination import run_phase3
    from src.phase4_landing_site import run_phase4
    from src.phase5_traverse_planning import run_phase5
    from src.phase6_volume_estimation import run_phase6

    p1 = run_phase1(test_config)
    p2 = run_phase2(test_config)
    p3 = run_phase3(test_config)
    p4 = run_phase4(test_config)
    p5 = run_phase5(test_config)
    p6 = run_phase6(test_config)

    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6
    }

