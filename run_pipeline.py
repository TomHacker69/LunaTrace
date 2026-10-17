"""
LunaTrace Master Pipeline Runner
Executes all 6 phases of the Chandrayaan-2 Radar Geospatial Pipeline
ISRO Problem Statement 8
"""

import time
from pathlib import Path
import pandas as pd

from src.phase1_dfsar import run_phase1
from src.phase2_ice_detection import run_phase2
from src.phase3_terrain_illumination import run_phase3
from src.phase4_landing_site import run_phase4
from src.phase5_traverse_planning import run_phase5
from src.phase6_volume_estimation import run_phase6


def main():
    start_time = time.time()
    print("=" * 80)
    print("   LunaTrace: Chandrayaan-2 DFSAR Subsurface Ice & Traverse Pipeline")
    print("   Lunar South Pole Cold-Trap Exploration (ISRO Problem Statement 8)")
    print("=" * 80)

    print("\n>>> PHASE 1: DFSAR Polarimetric Processing")
    run_phase1()

    print("\n>>> PHASE 2: Subsurface Ice Detection & Multi-Discriminator Framework")
    run_phase2()

    print("\n>>> PHASE 3: Topographic Morphometry & Illumination Analysis")
    run_phase3()

    print("\n>>> PHASE 4: Safe Landing Site Multi-Criteria Selection")
    run_phase4()

    print("\n>>> PHASE 5: Rover Traverse Planning & Battery Energy Simulation")
    run_phase5()

    print("\n>>> PHASE 6: Subsurface Ice Volumetric Resource Estimation")
    run_phase6()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"   [+] ALL 6 PHASES COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("   All maps generated in: outputs/maps/")
    print("   All CSV tables generated in: outputs/tables/")
    print("=" * 80)


if __name__ == "__main__":
    main()
