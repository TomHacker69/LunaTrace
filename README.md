# LunaTrace: Lunar Polar Subsurface Ice & Traverse Pipeline
### Chandrayaan-2 DFSAR Polarimetry, Topographic Horizon-Ephemeris Modeling, Landing Site Selection, and Rover Traverse Planning
**ISRO Hackathon Problem Statement 8**

---

## Overview
**LunaTrace** is an end-to-end scientific remote sensing and mission planning pipeline designed to detect and quantify subsurface water ice deposits in permanently shadowed regions (PSRs) and cryogenic cold traps at the Lunar South Pole using **Chandrayaan-2 Dual-Frequency Synthetic Aperture Radar (DFSAR)** and **Lunar Reconnaissance Orbiter (LRO) LOLA Digital Elevation Models**.

The pipeline resolves the fundamental radar ambiguity between volumetric ice scattering and surface rock roughness, identifies safe landing zones on illuminated crater rims, plans energy-optimal rover traverse routes into cold traps, and provides dielectric volumetric resource estimates across the top 0–5 m of lunar regolith.

---

## Core Scientific Architectural Principles

1. **Zero Hardcoded Logic:**
   - Every threshold, weight, spatial resolution, and solar parameter is defined in `config.yaml` and validated strictly on startup using Pydantic (`src/config_schema.py`).
   - Physical constants (permittivities, densities, radii, conversion factors) are strictly isolated in `src/constants.py` with peer-reviewed literature citations (*Archinal et al. 2018, Margot et al. 1999, Olhoeft 1975, Fa & Wieczorek 2012, Pettinelli et al. 2015*).

2. **Strict Provenance & Data Integrity:**
   - `data/raw/` contains only verified real mission products. The pipeline automatically halts if unverified or unauthorized datasets are detected.
   - Authorized datasets and SHA-256 verification rules are cataloged in `DATA_PROVENANCE.md`.
   - Every generated raster (`.tif`) and table (`.csv`) is accompanied by a cryptographic `.provenance.json` sidecar capturing input file hashes, software environment versions, git commit hash, and parameter snapshots.

3. **No Synthetic Fallbacks in Production:**
   - Synthetic benchmark generators are quarantined strictly within `tests/fixtures/` and are never run on real missions.
   - If optional datasets (such as Chandrayaan-2 OHRC 0.25 m optical imagery) are absent, the pipeline does not fall back to rough proxies; it explicitly marks the boulder hazard layer as `"UNASSESSED (OHRC not provided)"` and preserves topographic RMS roughness as an independent metric.

4. **Horizon-Ephemeris Illumination & Derived Engineering Margins:**
   - Solar illumination is calculated via a multi-angle horizon ray-tracing ephemeris model across **72 azimuthal directions** ($5^\circ$ spacing) and solar elevations over the lunar precession cycle ($0.5^\circ - 2.5^\circ$). No artificial capping or arbitrary depression factors are applied; 100% illumination appears only if genuinely unobstructed.
   - Rover traversable slope limits are strictly derived from mechanical parameters: $\theta_{\text{nav}} = \theta_{\text{rollover}} / \text{SF} = 22.5^\circ / 1.25 = 18.0^\circ$.

---

## Repository Structure

```text
LunaTrace/
├── config.yaml              # Schema-validated central configuration
├── DATA_PROVENANCE.md       # Mission dataset registry & SHA-256 verification
├── AUDIT.md                 # Pre-refactoring scientific & compliance audit
├── requirements.txt         # Python environment dependencies
├── run_pipeline.py          # Master CLI runner executing all 6 phases
├── app.py                   # Interactive Streamlit mission planning dashboard
├── data/
│   ├── raw/                 # Real Chandrayaan-2 DFSAR and LOLA GeoTIFFs (verified)
│   └── processed/           # Processed polarimetric, illumination, and suitability rasters
├── outputs/
│   ├── maps/                # High-resolution cartographic figures
│   └── tables/              # Quantitative CSV summaries with .provenance.json sidecars
├── src/
│   ├── constants.py         # Cited physical and planetary constants
│   ├── config_schema.py     # Pydantic v2 configuration validation
│   ├── config.py            # Configuration loader bridge
│   ├── validator.py         # Pre-flight raster CRS, extent, and provenance verification
│   ├── provenance.py        # Cryptographic sidecar generator (SHA-256)
│   ├── utils.py             # Enhanced Lee filter, GeoTIFF I/O, cartographic plotting
│   ├── phase1_dfsar.py      # Stokes polarimetry, CPR, DoP, m-chi decomposition
│   ├── phase2_ice_detection.py # Subsurface ice discrimination & de-circularized FPR
│   ├── phase3_terrain_illumination.py # Horizon ephemeris ray-tracing & PSR mapping
│   ├── phase4_landing_site.py # Multi-criteria landing site selection & ranking
│   ├── phase5_traverse_planning.py # A* and Dijkstra traverse planning & power modeling
│   └── phase6_volume_estimation.py # Volumetric CRIM ice modeling & resource curves
└── tests/
    ├── conftest.py          # Isolated test fixture session setup
    ├── test_pipeline.py     # Automated test suite (config, filters, physics, E2E)
    └── fixtures/
        └── synthetic_crater_generator.py # Synthetic test crater (TEST ONLY)
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

### 2. Automated Test Suite
Run the test suite against isolated fixtures in `tests/`:
```bash
python -m pytest tests/test_pipeline.py -v
```

### 3. Launch the Interactive Mission Dashboard
Start the Streamlit interactive dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The dashboard features:
- Interactive 3D terrain surface rendering.
- Polarimetric CPR and $m-\chi$ RGB decomposition visualization.
- Sensitivity sweep charts with de-circularized control false-positive rates.
- Multi-criteria candidate landing sites with dynamic engineering justifications.
- Rover traverse pathfinding maps with battery State-of-Charge (SoC) profiles.
- Volumetric CRIM scenario comparisons and uncertainty envelopes.
- Cryptographic `.provenance.json` sidecar inspector.

### 4. Run the Full Pipeline from CLI
To run all six phases end-to-end on mission data:
```bash
python run_pipeline.py
```

*Note: If `data/raw/` does not yet contain real verified mission rasters, `run_pipeline.py` halts loudly with instructions on required files.*

---

## Scientific Phases Summary

1. **DFSAR Polarimetry (Phase 1):** Hybrid polarimetric Stokes processing (Right Circular transmit, Linear H/V receive). Calculates Circular Polarization Ratio ($\text{CPR} = (S_1 - S_4)/(S_1 + S_4)$), Degree of Polarization ($m = \sqrt{S_2^2 + S_3^2 + S_4^2}/S_1$), and Raney $m-\chi$ decomposition with Enhanced Lee speckle filtering.
2. **Ice Detection (Phase 2):** Dual-frequency depth profiling ($\Delta \text{CPR} = \text{CPR}_L - \text{CPR}_S$). Tests radar-alone rejection power alongside the combined thermal cold trap model to de-circularize false-positive assessment. Boulder hazard is flagged explicitly as unassessed when OHRC is absent.
3. **Terrain & Ephemeris Horizon Modeling (Phase 3):** Azimuthal ray-tracing across 72 directions over the lunar precession cycle, delineating permanently shadowed regions (PSRs), doubly shadowed zones, and direct Earth visibility without artificial clipping.
4. **Landing Site Selection (Phase 4):** Hard engineering constraints ($\theta \le 5^\circ$, illumination $\ge 20\%$, ice standoff $1.0 - 7.5\text{ km}$) combined via Multi-Criteria Evaluation (MCE) with dynamically synthesized justifications.
5. **Rover Traverse Planning (Phase 5):** 8-connected $A^*$ pathfinding benchmarked against Dijkstra's algorithm. Adheres to derived chassis rollover safety limits ($\theta_{\text{nav}} \le 18.0^\circ$) and solar charging energy balances ($1.8\text{ kWh}$ battery, $20\%$ minimum reserve floor).
6. **Volumetric Ice Estimation (Phase 6):** Complex Refractive Index Model (CRIM) inversion across $0 - 5\text{ m}$ regolith column across Low, Nominal, and High geological scenarios using peer-reviewed dielectric constants.

---
*LunaTrace v2.0 | Pure Real Data Architecture | Developed for ISRO Hackathon PS-8*
