"""
LunaTrace Master Pipeline Runner
Executes all 6 phases of the Chandrayaan-2 Radar Geospatial Pipeline.
ISRO Problem Statement 8.
Fails fast with clear exceptions if data or parameters are missing or invalid.
"""

import argparse
import sys
import time
from pathlib import Path

from src.config_schema import load_config
from src.validator import validate_input_datasets, DataValidationError
from src.phase1_dfsar import run_phase1
from src.phase2_ice_detection import run_phase2
from src.phase3_terrain_illumination import run_phase3
from src.phase4_landing_site import run_phase4
from src.phase5_traverse_planning import run_phase5
from src.phase6_volume_estimation import run_phase6


def run_pipeline(config_path: Path = None):
    start_time = time.time()
    config = load_config(config_path)

    print("=" * 80)
    print("   LunaTrace: Chandrayaan-2 DFSAR Subsurface Ice & Traverse Pipeline")
    print("   Lunar South Pole Cold-Trap Exploration (ISRO Problem Statement 8)")
    print("=" * 80)

    # 1. Primary Data Validation Layer
    print("\n>>> PRE-FLIGHT: Validating Primary Mission Datasets...")
    try:
        val_summary = validate_input_datasets(
            raw_dir=config.paths.raw_data_dir,
            config=config,
            provenance_doc=config.paths.provenance_file,
            enforce_provenance=True
        )
        print(f"[+] Validation Passed: {val_summary['total_rasters_validated']} rasters verified.")
        boulder_info = "Detected (OHRC)" if val_summary["boulder_assessed"] else "Not provided (UNASSESSED)"
        print(f"[+] Chandrayaan-2 OHRC Boulder Assessment: {boulder_info}")
    except (DataValidationError, FileNotFoundError) as e:
        print(f"\n[!] DATA VALIDATION FAILED:\n{e}\n")
        print("Please supply verified Chandrayaan-2 DFSAR and LOLA datasets into data/raw/")
        print("as documented in DATA_PROVENANCE.md.")
        sys.exit(1)

    # Phase 0: PDS4 Ingestion & Complex SLC Calibration (if PDS4 labels are detected)
    pds4_xmls = list(config.paths.raw_data_dir.rglob("*.xml"))
    slc_tifs = list(config.paths.raw_data_dir.rglob("*_sli_*.tif"))
    if pds4_xmls and slc_tifs:
        print("\n>>> PHASE 0: Chandrayaan-2 PDS4 Raw Ingestion & SLC Complex Calibration")
        from src.pds4_reader import run_phase0_ingestion
        run_phase0_ingestion(config=config)

    print("\n>>> PHASE 1: DFSAR Polarimetric Processing")
    run_phase1(config=config)

    print("\n>>> PHASE 2: Subsurface Ice Detection & Multi-Discriminator Framework")
    run_phase2(config=config)

    print("\n>>> PHASE 3: Topographic Morphometry & Illumination Analysis")
    run_phase3(config=config)

    print("\n>>> PHASE 4: Safe Landing Site Multi-Criteria Selection")
    run_phase4(config=config)

    print("\n>>> PHASE 5: Rover Traverse Planning & Battery Energy Simulation")
    run_phase5(config=config)

    print("\n>>> PHASE 6: Subsurface Ice Volumetric Resource Estimation")
    run_phase6(config=config)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"   [+] ALL 6 PHASES COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print(f"   All maps generated in: {config.paths.outputs_maps_dir}")
    print(f"   All CSV tables generated in: {config.paths.outputs_tables_dir}")
    print("   Cryptographic SHA-256 provenance sidecars (.provenance.json) created for all outputs.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LunaTrace Master Pipeline Runner")
    parser.add_argument("--config", type=Path, default=None, help="Path to custom config.yaml")
    args = parser.parse_args()
    run_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
