# LunaTrace Repository Audit (ISRO Problem Statement 8)
**Date:** October 18, 2026  
**Auditor:** Senior Scientific Software Engineer & Planetary Remote Sensing Specialist  
**Repository:** `TomHacker69/LunaTrace`  
**Status:** Pre-Refactoring Audit Report (Task 1 Complete)

---

## Executive Summary

An exhaustive scientific and software engineering audit of the `LunaTrace` codebase was conducted in accordance with the five core compliance rules:
1. **Zero fallbacks** (fail loudly on missing/corrupt inputs).
2. **Zero hardcoding** (all parameters, paths, thresholds, and plot settings in a schema-validated `config.yaml`; physical constants isolated with citations in `src/constants.py`).
3. **Zero silent exception swallowing** (elimination of bare `except:` and `except Exception: pass`).
4. **Zero fabricated results** (all numbers in outputs, dashboard, and README dynamically computed).
5. **Real-data demarcation** (synthetic generators strictly quarantined inside `tests/`; clear flagging and handling for unavailable datasets like Chandrayaan-2 OHRC).

Below is the itemized inventory of all findings across the codebase with exact file paths, line numbers, analysis, and proposed refactoring solutions.

---

## 1. Magic Numbers, Inline Thresholds, Weights, Paths & CRS Strings

| File | Line(s) | Finding / Magic Value | Issue Analysis | Proposed Fix |
| :--- | :--- | :--- | :--- | :--- |
| `src/__init__.py` | 13 | `"rasterio.libs"` | Hardcoded directory string inline. | Move to a system/platform configuration helper or eliminate if standard wheels are used. |
| `src/config.py` | 14-20 | Hardcoded `Path` definitions (`data/raw`, `outputs/maps`, etc.) | Directory layout hardcoded in Python file instead of `config.yaml`. | Migrate paths to `config.yaml` validated via Pydantic model (`PathConfig`). |
| `src/config.py` | 94-96 | `"+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"` | Inline PROJ.4 projection string. | Move to `config.yaml` (`crs.proj4_string`) with citation to IAU 2000 Moon Ellipsoid / EPSG:3032 in `constants.py`. |
| `src/config.py` | 178, 182-186 | `920.0`, `2.70`, `3.15`, `0.0035`, `0.0001`, `0.42` | Physical constants (ice density, dielectric permittivity, loss tangents) defined in config dataclass without formal bibliographic citations. | Move all physical constants to `src/constants.py` with literature citations (e.g., Olhoeft 1974, Carrier et al. 1991, Fa & Wieczorek 2012, Pettinelli et al. 2015). |
| `src/phase1_dfsar.py` | 80, 90 | `1e-10` | Hardcoded epsilon inline to prevent zero-division. | Move numerical stability epsilon to `config.yaml` (`numerics.eps`). |
| `src/phase1_dfsar.py` | 157-159 | `98`, `1e-6` | Hardcoded 98th percentile clip and epsilon for RGB stretch. | Move visualization percentiles and stretch limits to `config.yaml` (`visualization.mchi_percentile`). |
| `src/phase2_ice_detection.py` | 85 | `0.0` | `dem > 0.0` hardcoded elevation boundary for control region. | Define `control_region_min_elevation_m` in `config.yaml`. |
| `src/phase2_ice_detection.py` | 134 | `dem < self.ice_cfg.cold_trap_depth_m` | Sign inversion and threshold inline. | Centralize cold trap depth normalization in `config.yaml`. |
| `src/phase3_terrain_illumination.py` | 44 | `resolution_m: float = 20.0` | Default parameter value hardcoded in method signature. | Change signature to `resolution_m: Optional[float] = None`, defaulting strictly from validated config. |
| `src/phase3_terrain_illumination.py` | 125 | `* 0.1` | Depression illumination attenuation factor (`0.1`) hardcoded inline. | Move to `config.yaml` (`terrain.crater_floor_scatter_attenuation`). |
| `src/phase3_terrain_illumination.py` | 132 | `89.0` | Maximum horizon clamping angle (`89.0°`) hardcoded inline. | Move to `config.yaml` (`terrain.max_horizon_elevation_deg`). |
| `src/phase3_terrain_illumination.py` | 168 | `-1.0` | Tangent clamping limit (`-1.0`) hardcoded inline. | Move to `config.yaml` (`terrain.min_horizon_tan`). |
| `src/phase3_terrain_illumination.py` | 283, 284 | `illum > 0.6`, `earth_los > 0.5` | Thresholds for summary metrics (`0.6`, `0.5`) hardcoded inline. | Add `summary_high_illum_cutoff: 0.6` and `summary_earth_los_cutoff: 0.5` to `config.yaml`. |
| `src/phase4_landing_site.py` | 62 | `resolution_m: float = 20.0` | Method signature inline default. | Default strictly from validated config. |
| `src/phase4_landing_site.py` | 206-209 | Hardcoded strings with text numbers: `<2.5 deg`, `~75%`, `2.2 km` | Justification text has hardcoded numbers that differ from computed site values (`0.09°`, `100%`, `2440 m`). | Dynamically format justification strings using the computed table attributes (`f"{row.Slope_deg:.2f}°"`). |
| `src/phase5_traverse_planning.py` | 207 | `soc_history = [100.0]` | Initial Battery State of Charge (`100.0%`) hardcoded inline. | Move `initial_battery_soc_pct` to `config.yaml`. |
| `src/phase5_traverse_planning.py` | 404, 413 | `ax2.set_ylim(0, 16)`, `ax3.set_ylim(0, 105)` | Profile plot axis limits (`16`, `105`) hardcoded inline. | Move plot limits to `config.yaml` (`visualization.profile_slope_ylim`, etc.). |
| `src/phase6_volume_estimation.py` | 61 | `base_fraction: float = 0.15` | Default argument in method signature. | Sourced strictly from validated config scenario. |
| `src/phase6_volume_estimation.py` | 107, 111, 112 | `1e9`, `1000.0`, `1e6` | Unit conversion constants ($m^3 \to km^3$, $kg \to t$, $t \to Mt$) hardcoded inline. | Move unit conversion factors to `src/constants.py` with standard unit definitions. |
| `src/utils.py` | 38, 39 | `window_size: int = 5, num_looks: int = 4` | Method signature inline defaults. | Sourced strictly from validated config. |
| `src/utils.py` | 55 | `1.0 + 2.0 / num_looks` | Multiplicative speckle filter coefficient hardcoded inline. | Document Lee (1980) speckle formula in `constants.py` and cite source paper. |
| `src/utils.py` | 114 | `np.nanpercentile(data, 2)`, `np.nanpercentile(data, 98)` | Cartographic plot normalization percentiles (`2`, `98`) hardcoded inline. | Move to `config.yaml` (`visualization.default_vmin_percentile`, etc.). |

---

## 2. Silent Error Handling, Swallowed Exceptions & Fallback Defaults

| File | Line(s) | Finding | Issue Analysis | Proposed Fix |
| :--- | :--- | :--- | :--- | :--- |
| `src/__init__.py` | 16-19 | `try: os.add_dll_directory(libs_dir) except Exception: pass` | Swallows all exceptions without logging or re-raising. Violates Rule 3. | Inspect specifically for `FileNotFoundError`/`OSError`, log warning, or handle via deterministic platform checks. |
| `src/utils.py` | 24-27 | `try: os.add_dll_directory(libs_dir) except Exception: pass` | Duplicate bare `except Exception: pass` block in `utils.py`. | Eliminate duplication; handle centrally in system loader with explicit error logging. |
| `src/utils.py` | 90 | `clean_data = np.nan_to_num(data, nan=nodata)` | `np.nan_to_num` silently substitutes `NaN` / `Inf` with arbitrary `nodata` or zero values. Violates Rule 1 & 2. | Add pre-save raster validation: if unexpected NaNs or Infs exist in physical arrays, raise `FloatingPointError` with coordinates/band details. Explicit nodata masking must be controlled by schema. |
| `src/phase1_dfsar.py` | 52 | `meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4` | Fallback default: if input GeoTIFF lacks CRS metadata, it silently substitutes config CRS. | Fail loudly: if raster has no CRS or CRS does not match expected EPSG:3032 / IAU 2000 Moon, raise `ValueError(f"Raster {fpath} is missing CRS metadata!")`. |
| `src/phase2_ice_detection.py` | 55 | `meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4` | Silent CRS substitution fallback. | Validate raster CRS against config CRS; raise exception if mismatch or missing. |
| `src/phase3_terrain_illumination.py` | 37 | `crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4` | Silent CRS substitution fallback. | Validate raster CRS against config CRS; raise exception if mismatch or missing. |
| `src/phase4_landing_site.py` | 56 | `meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4` | Silent CRS substitution fallback. | Validate raster CRS against config CRS; raise exception if mismatch or missing. |
| `src/phase5_traverse_planning.py` | 53 | `meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4` | Silent CRS substitution fallback. | Validate raster CRS against config CRS; raise exception if mismatch or missing. |
| `src/phase5_traverse_planning.py` | 118-124 | `if cost_grid[goal_y, goal_x] == np.inf: ... distance_transform_edt(~passable) ... goal_y, goal_x = indices[...]` | Silent goal snapping: if requested target is inside an impassable obstacle, it silently alters the goal coordinates without notifying caller. | Raise `ValueError("Goal location is impassable according to safety limits.")` or return an explicit snapped status flag with warning if permitted by config. |
| `src/phase6_volume_estimation.py` | 53 | `meta_crs = src.crs.to_string() if src.crs else self.terrain_cfg.projection_proj4` | Silent CRS substitution fallback. | Validate raster CRS against config CRS; raise exception if mismatch or missing. |
| `src/phase6_volume_estimation.py` | 180 | `color=scen_colors.get(s_name, "black")` | `.get(s_name, "black")` fallback dictionary access. Violates Rule 2. | Strictly index dictionary: `scen_colors[s_name]`. If scenario is unknown, fail loudly with missing key error. |

---

## 3. Synthetic, Simulated, Random or Generated Data Outside Tests

| File | Line(s) | Finding | Issue Analysis | Proposed Fix |
| :--- | :--- | :--- | :--- | :--- |
| `src/benchmark_data.py` | Entire file (lines 1-199) | `generate_benchmark_crater_data()`, `np.random.normal`, `np.random.gamma`, `np.random.uniform`, saving directly into `data/raw/` | Generates synthetic DEM, DFSAR Stokes S1-S4, illumination, and boulders outside `tests/`. Violates Rule 5 ("Synthetic data is allowed ONLY inside tests folder, clearly labelled, for unit tests. Real data only for main run"). | Move `benchmark_data.py` to `tests/fixtures/synthetic_crater_generator.py`. Restrict its execution strictly to automated test fixtures. `data/raw` must be reserved for real mission data. |
| `data/raw/` | All 13 `.tif` files | `lola_dem_20m.tif`, `crater_illumination.tif`, `dfsar_L_S1..S4.tif`, `dfsar_S_S1..S4.tif`, `crater_boulder_density.tif`, `ground_truth_ice_mask.tif` | These files currently in `data/raw/` are synthetic outputs from `benchmark_data.py`, not raw PDS4/GeoTIFF mission products. | Clearly document data state in `DATA_PROVENANCE.md`. Provide real Chandrayaan-2 DFSAR PDS4 / LOLA GeoTIFF ingestion pipeline. For standalone offline demonstration, store synthetic test fixtures in `tests/data/` or provide a `--synthetic-test-mode` flag that is explicitly logged. |

---

## 4. Proxies Standing in for Real Data Not Clearly Flagged

| File | Line(s) | Finding | Issue Analysis | Proposed Fix |
| :--- | :--- | :--- | :--- | :--- |
| `src/benchmark_data.py` | 96-101 | `wall_slope_proxy = ...; boulder_density = ...` | Boulder density was simulated using an empirical slope-decay proxy and exponential noise, but is later treated as an independent sensor input in Phase 4 and Phase 5. | State explicitly that Chandrayaan-2 OHRC (0.25 m) is required for true boulder mapping. If OHRC is missing, mark boulder hazard as unassessed or derived from LOLA RMS roughness proxy with explicit caveat in metadata and UI. |
| `src/phase2_ice_detection.py` | 45 | `"boulders": RAW_DATA_DIR / "crater_boulder_density.tif"` | The pipeline assumes `crater_boulder_density.tif` exists as an independent observation. | Add validation: check whether OHRC boulder product exists. If absent, explicitly disable OHRC-dependent boulder scoring or fallback to Topographic Roughness with clear warning. |
| `src/phase3_terrain_illumination.py` | 80-127 | 16-azimuth static horizon ray-tracing as proxy for lunar year illumination | 16 discrete azimuthal rays on a local 10 km grid serve as a proxy for the continuous 18.6-year lunar precession cycle. Regional topography >10 km away (e.g., Malapert mountain peak) is omitted. | Clearly document the horizon ray-tracing model assumptions in `config.yaml` and README. Flag that illumination is a local horizon proxy, not an integrated SPICE/ephemeris simulation. |

---

## 5. Pre-Set / Artifact Output Values Investigation

### A. Control False-Positive Rate of Exactly 0.00%
- **Location:** `outputs/tables/phase2_threshold_sensitivity_analysis.csv`
- **Observed Value:** `Control_False_Positive_Rate_pct: 0.00` across all CPR/DoP thresholds ($0.8 - 1.2$, $0.25 - 0.35$).
- **Investigation:**
  In `phase2_ice_detection.py`:
  1. The control region is defined at line 85 as:
     $$\text{control\_rocky\_ejecta} = (\text{illum} > 0.40) \land (\text{boulders} > 0.20) \land (\text{dem} > 0.0)$$
  2. Ice candidates are classified at line 151 as:
     $$\text{high\_prob\_ice} = (\text{ice\_probability} \ge 0.50) \land (s_{\text{trap}} > 0.40)$$
  3. But the thermal trap score $s_{\text{trap}}$ (lines 133–135) is:
     $$s_{\text{trap}} = 0.6 \cdot \text{clip}\left(\frac{0.10 - \text{illum}}{0.10}, 0, 1\right) + 0.4 \cdot \text{clip}\left(\frac{-\text{dem} - 350}{1000}, 0, 1\right)$$
  4. In the control region, $\text{illum} > 0.40 \implies \frac{0.10 - 0.40}{0.10} < 0 \implies s_{\text{trap\_illum}} = 0.0$.
  5. Furthermore, $\text{dem} > 0.0 \implies -\text{dem} < 0 \implies \text{deep\_floor} = 0.0$.
  6. Therefore, $s_{\text{trap}} \equiv 0.000$ identically across the entire control region!
- **Scientific Conclusion:** **This is an artifact of circular logic.** Because $s_{\text{trap}} > 0.40$ is a mandatory gate for ice detection, and the control region was defined with illumination and elevation conditions where $s_{\text{trap}}$ is identically zero, false positives are mathematically impossible regardless of radar CPR and DoP values!
- **Proposed Fix:** Evaluate radar false positives in the control region using **radar discriminators alone** ($\text{CPR}_L > 1.0$ and $\text{DoP} < 0.32$ and $\Delta\text{CPR} > 0.2$), separating radar false-alarm rejection from spatial/thermal masking.

### B. Illumination & Earth LOS Fractions of Exactly 1.000 (100%)
- **Location:** `outputs/tables/phase4_top3_landing_sites.csv`
- **Observed Value:** `Site_1`: Illumination Fraction = `1.0`, Earth LOS Fraction = `1.0`.
- **Investigation:**
  In `phase3_terrain_illumination.py`:
  1. Northern ray-tracing calculates `north_horizon_deg` towards +Y. On the elevated northern rim plateau, the horizon angle is $\le 0^\circ$ (looking outward/downhill).
  2. `earth_los = np.clip((3.5 - north_horizon_deg) / 3.5, 0.0, 1.0)`. For $\text{north\_horizon\_deg} \le 0^\circ$, $(3.5 - 0)/3.5 = 1.0$, which clips to exactly `1.0`.
  3. For solar illumination, all 16 azimuthal directions from this local summit have horizon elevations below the grazing $1.5^\circ$ solar angle, resulting in $\text{mean}(1.0) = 1.000$.
- **Scientific Conclusion:** **This is a numerical clipping artifact.** At the Lunar South Pole (lat $-87.3^\circ$), maximum continuous solar illumination on the highest peaks (e.g., Shackleton connecting ridge, Malapert plateau) peaks at ~86–92% due to orbital inclination ($1.54^\circ$) and seasonal variations. An annual average of 100% is physically impossible on the Moon.
- **Proposed Fix:** Incorporate the lunar obliquity cycle or cap maximum achievable polar illumination at physical empirical bounds (e.g., max 86% annual average from Mazarico et al. 2011 / Gläser et al. 2014) in `constants.py`.

### C. Rover Route Max Slope Sitting Exactly on Limit (17.88° vs. 18.00°)
- **Location:** `outputs/tables/phase5_route_comparison.csv`
- **Observed Value:** `Max_Slope_deg: 17.88` across all three routes (Shortest, Safest, Best-Illuminated).
- **Investigation:**
  In `phase5_traverse_planning.py`:
  1. `obstacle_mask = (slope > 18.0) | (boulders > 0.40)`. Impassable cells receive `cost = np.inf`.
  2. The natural crater rim has walls sloping between $22^\circ$ and $34^\circ$.
  3. The ingress terrace ramp is the only viable corridor connecting the rim to the floor, but its steepest bottleneck cell has a slope of $17.88^\circ$.
  4. Because this bottleneck is the sole passage with $\text{slope} \le 18.0^\circ$, all three pathfinding variants are channeled through this exact passage, recording a maximum slope of $17.88^\circ$.
- **Scientific Conclusion:** **This is a real topological bottleneck, not a hardcoded bug.** However, operating within $0.12^\circ$ ($< 1\%$) of the absolute mechanical chassis rollover threshold violates NASA/ISRO safety margin standards (which require a safety factor $\ge 1.3\times$).
- **Proposed Fix:** Document this engineering constraint clearly; add an explicit clearance margin parameter in `config.yaml` (`rover.safety_factor_margin: 1.25`) so that pathfinding penalizes operating near the chassis threshold.

---

## 6. Architecture & Configuration Gap Analysis

1. **Missing `config.yaml`:**
   - Current configuration is a Python script (`src/config.py`).
   - Standard scientific pipeline architecture requires an external human-readable, schema-validated `config.yaml` with Pydantic validation (`src/config_schema.py`) to prevent silent type coercion or invalid values.
2. **Missing `src/constants.py`:**
   - Physical constants (e.g., regolith permittivity $\varepsilon_r = 2.70$, ice permittivity $\varepsilon_i = 3.15$, ice density $\rho = 920\text{ kg/m}^3$) are currently mixed inside dataclass defaults without bibliographic citations.
3. **Missing Data Validation Layer:**
   - There is no preliminary raster validation step checking whether bounding boxes, projections, pixel sizes, and nodata values across DFSAR, LOLA DEM, and optical rasters are mutually congruent.
4. **Missing Output Provenance Sidecars:**
   - GeoTIFF and CSV outputs currently lack `.json` metadata sidecars recording input file hashes (SHA-256), git commit hash, execution timestamp, and parameter snapshots.
5. **Hardcoded Numbers in `README.md`:**
   - Sections of the README contain hardcoded metrics (e.g., "$0.00\%$ false positives", "$0.27\text{ Mt}$", "$37.7\%$ battery reserve") that were manually typed rather than populated from pipeline outputs.

---

## 7. Action Plan for Refactoring (Awaiting User Approval)

Upon your confirmation, the refactoring will execute in the following modular steps:

1. **Step 1: Configuration & Physical Constants Isolation**
   - Create `config.yaml` with all paths, thresholds, weights, and tolerances.
   - Create `src/constants.py` with physical constants and literature citations.
   - Implement `src/config_schema.py` using Pydantic to validate `config.yaml` on load.
2. **Step 2: Strict Data Validation & Provenance Layer**
   - Implement `src/validator.py`: checks file presence, CRS identity, resolution alignment, spatial bounding box intersection, and value sanity.
   - Implement `src/provenance.py`: creates cryptographic SHA-256 sidecars (`.json`) for every generated raster and table.
3. **Step 3: Relocate Synthetic Benchmarking to Tests**
   - Move `src/benchmark_data.py` into `tests/fixtures/synthetic_crater_generator.py`.
   - Update tests to explicitly generate and clean up test fixtures in isolated temporary test directories.
   - Clearly delineate real input ingestion from test fixture generation.
4. **Step 4: De-Circularize Ice & Control Evaluation**
   - Refactor Phase 2 control evaluation to test radar discrimination power independently of spatial/elevation masking.
5. **Step 5: Interactive Application (`app.py`)**
   - Build Streamlit application with live parameter binding from `config.yaml`, cached execution, 2D/3D maps, and provenance inspection.
6. **Step 6: Dynamic Documentation Sync**
   - Update `README.md` to reference dynamically generated pipeline summaries.

---
*End of Audit Report. No code modifications have been made. Awaiting user review and authorization to proceed to Task 2.*
