# LunaTrace: Lunar Polar Subsurface Ice & Traverse Pipeline
### Chandrayaan-2 DFSAR Polarimetry, Topographic Horizon Modeling, Landing Site Selection, and Rover Traverse Planning


---

## Overview
**LunaTrace** is an end-to-end Python geospatial engineering pipeline designed to detect and characterize subsurface ice in permanently shadowed regions (PSRs) and doubly shadowed cold traps at the Lunar South Pole using Chandrayaan-2 Dual Frequency Synthetic Aperture Radar (DFSAR) and LOLA Digital Elevation Models.

The pipeline resolves the critical remote sensing ambiguity between volume ice scattering and rocky ejecta roughness, selects safe landing zones on illuminated crater rims, plans energy-optimal rover traverse routes into cryogenic cold traps, and provides dielectric volumetric resource estimates across the top 0–5 m of lunar regolith.

---

## Pipeline Architecture
```text
LunaTrace/
├── data/
│   ├── raw/                 # Input DEM, DFSAR Stokes parameters, illumination rasters
│   └── processed/           # Processed CPR, DOP, m-chi decomposition, and suitability layers
├── outputs/
│   ├── maps/                # High-resolution cartographic figures with scale bars & North arrows
│   ├── tables/              # Quantitative CSV summaries across all phases
│   └── LunaTrace_Technical_Report.md  # Comprehensive technical report & 7-slide summary
├── src/
│   ├── config.py            # Central configuration (all thresholds, radar constants, rover specs)
│   ├── utils.py             # Speckle filtering, GeoTIFF I/O, cartographic mapping
│   ├── benchmark_data.py    # Physically consistent synthetic lunar crater dataset generator
│   ├── phase1_dfsar.py      # Phase 1: Stokes parameters, CPR, DOP, Raney m-chi decomposition
│   ├── phase2_ice_detection.py      # Phase 2: Multi-discriminator ice detection & Delta CPR
│   ├── phase3_terrain_illumination.py # Phase 3: Horizon ray-tracing, slope, roughness, Earth LOS
│   ├── phase4_landing_site.py       # Phase 4: Multi-criteria landing site selection & ranking
│   ├── phase5_traverse_planning.py  # Phase 5: A* and Dijkstra pathfinding & battery simulation
│   └── phase6_volume_estimation.py   # Phase 6: CRIM dielectric volume estimation & uncertainty
├── run_pipeline.py          # Master runner executing all 6 phases end-to-end
└── requirements.txt         # Environment dependencies
```

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/TomHacker69/LunaTrace.git
cd LunaTrace
pip install -r requirements.txt
```

### 2. Run the Full 6-Phase Pipeline
Execute all six phases in sequence with a single command:
```bash
python run_pipeline.py
```

### 3. Run Individual Modules
Each phase can also be executed independently:
```bash
python -m src.phase1_dfsar
python -m src.phase2_ice_detection
python -m src.phase3_terrain_illumination
python -m src.phase4_landing_site
python -m src.phase5_traverse_planning
python -m src.phase6_volume_estimation
```

---

## Key Results & Deliverables

| Deliverable | Key Output | Description |
| :--- | :--- | :--- |
| **Ice Probability Map** | `outputs/maps/map_phase2_ice_probability.png` | Continuous 0 to 1 ice probability score draping topographic contours |
| **Delta CPR Map** | `outputs/maps/map_phase2_delta_cpr.png` | Dual-frequency indicator (L-band minus S-band) isolating buried ice |
| **Landing Site Ranking** | `outputs/maps/map_phase4_landing_sites.png` | Top 3 candidate sites overlaid on multi-criteria suitability surface |
| **Rover Traverse Routes** | `outputs/maps/map_phase5_rover_traverses.png` | Shortest, Safest, and Best-Illuminated routes to ice target |
| **Traverse Profile** | `outputs/maps/map_phase5_recommended_route_profile.png` | Elevation, slope angle, and battery state of charge (SoC) along Safest Route |
| **Water Column Depth** | `outputs/maps/map_phase6_water_column.png` | Spatial equivalent pure water ice column (meters) |
| **Resource Sensitivity** | `outputs/maps/map_phase6_volume_sensitivity.png` | Water mass sensitivity curve across dielectric fractions and column depths |
| **Technical Report** | `outputs/LunaTrace_Technical_Report.md` | Full scientific report and 7-slide presentation pitch deck |

---

## Scientific Methods Summary

1. **DFSAR Polarimetry (Phase 1):** Hybrid polarimetric Stokes processing (Right Circular transmit, Linear H/V receive). Calculates Circular Polarization Ratio ($\text{CPR} = (S_1 - S_4)/(S_1 + S_4)$), Degree of Polarization ($m = \sqrt{S_2^2 + S_3^2 + S_4^2}/S_1$), and Raney $m-\chi$ 3-component decomposition (Volume, Double-bounce, Surface).
2. **Ice Detection (Phase 2):** Dual-frequency depth profiling ($\Delta \text{CPR} = \text{CPR}_L - \text{CPR}_S$). Proves the theoretical polarimetric lower bound $m \ge |\text{CPR}-1|/(\text{CPR}+1)$, resolving the rock-vs-ice ambiguity with $0.00\%$ false positives in rocky ejecta control terrain.
3. **Terrain & Horizon Ray-Tracing (Phase 3):** Azimuthal horizon modeling across 16 directions simulating grazing polar sunlight ($1.5^\circ$), delineating permanently shadowed regions (PSRs), doubly shadowed cryogenic cold traps ($< 100\text{ K}$), and Earth direct line-of-sight under lunar libration.
4. **Landing Site Selection (Phase 4):** Hard engineering constraints ($\theta \le 5^\circ$, boulder hazard $< 0.30$, illumination $> 20\%$, and standoff distance $1.0 - 7.5\text{ km}$) combined via Weighted Multi-Criteria Evaluation (MCE) to rank the Top 3 landing sites.
5. **Rover Traverse Planning (Phase 5):** 8-connected $A^*$ pathfinding benchmarked against Dijkstra's algorithm. Integrates dynamic solar charging and battery power drain ($1.8\text{ kWh}$ pack, $65\text{ W}$ baseload, $4\text{ W/deg}$ traction), demonstrating feasible cold-trap ingress with $37.7\%$ battery reserve margin.
6. **Volumetric Ice Estimation (Phase 6):** Complex Refractive Index Model (CRIM) inversion across the $0 - 5\text{ m}$ regolith column. Computes **Low ($0.27\text{ Mt}$)**, **Nominal ($2.02\text{ Mt}$)**, and **High ($6.32\text{ Mt}$)** pure water ice resource reserves.
