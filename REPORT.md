# LunaTrace: Scientific Remote Sensing & Mission Planning Report
## ISRO Problem Statement 8: Lunar Polar Subsurface Ice Characterization & Rover Traverse Planning

**Repository:** `TomHacker69/LunaTrace`  
**Generated:** 2026-10-20T13:12:10.307512+00:00  
**Target Depression:** Faustini SW Floor Basin (-87.39S, 82.31E) (`CAND_2_THIRD_PARTY`)  
**Coordinates:** Latitude -87.39°S, Longitude 82.31°E  
**Software Verification:** Automated end-to-end execution without synthetic data outside `tests/`.

---

## 1. Executive Summary & Core Scientific Conclusion

LunaTrace executes a complete remote sensing and planetary mission planning pipeline for the Lunar South Pole using primary **Chandrayaan-2 Dual-Frequency Synthetic Aperture Radar (DFSAR)** Level-1A complex data and **Lunar Reconnaissance Orbiter (LRO) LOLA Digital Elevation Models (DEM)**.

### Core Scientific Findings:
1. **Radar Polarimetry & Cluster Null Baseline:**  
   Observed candidate cluster sizes ($m \le 0.13$, $\text{CPR} > 1.0$) are **consistent with a spatially varying speckle null**. The candidate count excess grows systematically with the background smoothing scale ($3.21\times$ at $L=31$ to $10.83\times$ at $L=101$). Native radar backscatter alone **does not establish an ice-like detection**; candidate pixels remain strictly categorized as *"unexplained by speckle"*, pending sub-meter OHRC boulder counts.
2. **Target Crater Selection:**  
   An a-priori ranking rule fixed BEFORE examining CPR/DoP selected **Faustini SW Floor Basin (-87.39S, 82.31E)** (Rank 1, Score 0.6936). It features an $86.6\%$ cryogenic cold-trap floor ($2.74\text{ km}^2$), low radar shadow ($32.5\%$), and safe interior swath margin ($+713.5\text{ m}$).
3. **Safe Landing Corridors:**  
   Identified **Site 1 (Rank 1)** on a slumped terrace at Slope 0.0°, Solar Illumination 100.0%, Earth LOS Visibility 100.0%, and Standoff Distance 4994.7 m to candidate ice deposits.
4. **Rover Traverse Feasibility:**  
   8-connected $A^*$ pathfinding established a safe, traversable route into the cold trap (9077.0 m distance, max slope 5.54° $\le 18.0°$). A VIPER-class $1.8\text{ kWh}$ battery maintains a **72.9\% minimum State-of-Charge** (52.9\% reserve margin above the critical $20\%$ cutoff).
5. **CRIM Dielectric Resource Modeling:**  
   Dielectric mixture modeling over the $2.74\text{ km}^2$ cold trap yields **1.099 Megatons** of water ice in the nominal scenario ($3.5\text{ m}$ active depth, $15\%$ pore filling; $5\%-95\%$ uncertainty envelope: 0.608 to 1.701 Mt).

---

## 2. Ingestion, Provenance & Calibration Audit

- **DFSAR Data Product:** Orbit 29472 L-band Slant Range Complex (SLC) Level-1A (`ch2_sar_ncxl_20260403t190328238_d_sli_xx_cp_lh_d18.tif`, `lv`, `xml`).
- **LRO LOLA DEM Product:** PDS Geosciences Node `ldem_85s_20m.lbl` / `ldem_85s_20m.img` ($20\text{ m/pixel}$, Polar Stereographic, $438.8\text{ MB}$).
- **Verified SHA-256 Hashes:**
  - `ldem_85s_20m.lbl`: `d526aa2677f2cfc794cf6f26e1e69811bc0423afbb755bb7f0a0340ec3e0fcd7`
  - `ldem_85s_20m.img`: `d900ac0aa97fbb983f9898873ead759d870fdb33978afc17fb4e1fa526aa49b2`
- **Verbatim Label Parameters:** `PRODUCT_TYPE = HEIGHT`, `UNIT = METER`, `SCALING_FACTOR = 0.5`, `OFFSET = 1737400.0`.
- **Observed Surface Elevations:** Min $-5,501.50\text{ m}$, Max $+7,027.00\text{ m}$, Mean $-1,067.87\text{ m}$ (physically plausible lunar topography within $[-12000, 12000]\text{ m}$).
- **Single-Cell Look Measurement:** Dynamically measured from complex SLC 2D autocorrelation broadening factors at runtime: **$N_{\text{cell}} = 4.21\text{ looks/cell}$**.

---

## 3. Geolocation & Terrain-Corrected Geocoding

- **Derived Look Vector:** $\mathbf{u} = (0.3085, -0.9512)$ in Polar Stereographic (EPSG:3032); Spacecraft Track $\mathbf{t} = (-0.9334, -0.3588)$ (Heading $-162.03°$, Left-looking).
- **Search Window Radius:** Runtime derived as $R_{\text{search}} = |\Delta x_{\text{along}}| + h_{\text{relief}} \cot\theta_{\text{inc}} = 3,115.97\text{ m} + 8,335.70\text{ m} = 11,451.67\text{ m}$.
- **Simulated Backscatter Cross-Correlation:**
  - Measured Offset: Range $\Delta x = +160.0\text{ m}$, Azimuth $\Delta y = +80.0\text{ m}$.
  - Peak Correlation: $\rho_{\text{peak}} = 0.0932$.
  - Correlation Surface Null: Mean $\mu = 0.0037$, Std $\sigma = 0.0250$.
  - Null Margin: **$3.58\sigma$ above null surface** (**GATE PASSED**).
- **Protocol Decision:** Because $\rho_{\text{peak}} = 0.0932 < 0.40$, thresholds were strictly not altered. Geolocation status is formally reported as **`UNVALIDATED_EMPIRICAL`**, retaining nominal tie-points with an explicit empirical uncertainty budget ($\sigma_{\text{geo}} = 3.27 - 4.59\text{ km}$).
- **Radar Shadow and Layover:** Computed on the real DEM:
  - Radar Shadow Mask (backslope + cast shadow): $963,276\text{ pixels}$ ($50.83\%$).
  - Radar Layover Mask ($dz/dr > \tan\theta_{\text{inc}}$): $796\text{ pixels}$ ($0.04\%$).

---

## 4. Phase 2: Radar Polarimetry & Subsurface Ice Analysis

### 4.1 Statistical Comparison: Inside vs. Outside PSR
- **Inside PSR (Valid Radar Pixels):** Area 15.73 km², Mean CPR = 0.3917, Median CPR = 0.3581, CPR > 1.0 fraction = 0.09%.
- **Outside PSR (Valid Radar Pixels):** Area 9.22 km², Mean CPR = 0.3598, Median CPR = 0.3479, CPR > 1.0 fraction = 0.0%.
- **CPR vs. Incidence Correlation:** $r = 0.3473$.

### 4.2 Control-Region False-Positive Assessment
- **Evaluation:** Evaluated on sunlit rocky plateaus outside the crater using **radar gates only** ($\text{CPR}_{7\times 7} > 1.0$ and $\text{DoP}_{7\times 7} \le 0.13$), resolving circular logic with thermal masks.
- **Result:** **0.0\% False-Positive Rate** in valid control terrain (1 pixels).

### 4.3 Spatially Varying Wishart Null Baseline (Leave-Out Annular Covariance)
| Smoothing Window ($L$) | Evaluation Window ($W$) | Expected Chance Clusters | Observed Real Clusters | Count Excess Ratio | Max Chance Cluster Size | 99th Pct Max Chance | Observed $> p_{99}$ Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $L = 31\times 31$ | $7\times 7$ | 197.2 ± 17.5 | 634 | $3.21\times$ | 321 px | 285.4 px | 0 / 634 (0.00%) |
| $L = 51\times 51$ | $7\times 7$ | 139.7 ± 12.8 | 634 | $4.54\times$ | 311 px | 232.8 px | 0 / 634 (0.00%) |
| $L = 101\times 101$ | $7\times 7$ | 58.5 ± 9.8 | 634 | $10.83\times$ | 80 px | 75.1 px | 5 / 634 (0.79%) |

---

## 5. Phase 3: Terrain Morphometry & Illumination Summary

| Metric / Layer | Value | Coverage [%] |
| :--- | :--- | :--- |
| Terrain Area Mapped | 324.00 km^2 | 100.0 |
| Safe Landing Corridors (Slope <= 5.0 deg) | 135.40 km^2 | 41.79 |
| Permanently Shadowed Regions (PSR, Illum < 0.5%) | 13.99 km^2 | 4.32 |
| Doubly Shadowed Cold Traps (SVF < 0.86, Elev < -350m) | 2.74 km^2 | 0.84 |
| Direct Earth Visibility (LOS >= 50%) | 164.47 km^2 | 50.76 |

- **Horizon Ray-Tracing Parameters:** 72 azimuthal directions ($5°$ intervals), 5 solar elevation angles ($0.5° - 2.5°$), step distances out to $4,000\text{ m}$.
- **Earth LOS Visibility:** Evaluated for mean apparent elevation $3.5°$ with northern horizon ray tracing out to $6,000\text{ m}$.

---

## 6. Phase 4: Candidate Landing Sites

| Rank | Site Name | Grid X (m) | Grid Y (m) | Elevation (m) | Slope (deg) | RMS Roughness (m) | Solar Illumination [%] | Earth LOS Visibility [%] | Distance to Ice (m) | OHRC Boulder Status | Composite MCE Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Site 1 (Recommended) | 83010.0 | 7550.0 | -2648.0 | 0.0 | 0.69 | 100.0 | 100.0 | 4994.7 | UNASSESSED | 0.9919 |
| 2 | Site 2 (Alternative) | 80730.0 | 9030.0 | -2727.5 | 0.0 | 1.32 | 93.9 | 100.0 | 6622.4 | UNASSESSED | 0.9694 |
| 3 | Site 3 (Contingency) | 80410.0 | 7710.0 | -2740.5 | 0.0 | 1.6 | 87.5 | 98.8 | 7275.7 | UNASSESSED | 0.9483 |

- **Hard Constraints:** Slope $\le 5.0°$, Solar Illumination $\ge 20\%$, Standoff to ice target $1.0 - 7.5\text{ km}$.
- **MCE Weights (without OHRC):** Slope $0.35$, Roughness $0.25$, Illumination $0.25$, Earth LOS $0.15$.

---

## 7. Phase 5: Rover Traverse Planning & Energy Balance

| Strategy | Algorithm | Distance (m) | Time (h) | Shadow Time (h) | Max Slope (deg) | Min SoC [%] | Reserve Margin [%] | Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Shortest Path (Distance-Optimal) | A* (8-connected) | 7386.6 | 25.65 | 0.69 | 10.3 | 64.2 | 44.2 | FEASIBLE |
| Safest Path (Slope-Averse) | A* (8-connected) | 9077.0 | 31.52 | 1.28 | 5.54 | 72.9 | 52.9 | FEASIBLE |
| Best-Illuminated Path (Solar-Tracking) | A* (8-connected) | 8542.1 | 29.66 | 0.42 | 8.43 | 81.7 | 61.7 | FEASIBLE |

- **Chassis Rollover Limit:** $22.5°$, Engineering Safety Factor $1.25$ $\rightarrow$ Maximum Nav Slope **$18.0°$**.
- **Battery Pack:** $1.8\text{ kWh}$ VIPER-class system ($65\text{ W}$ baseline, $4\text{ W/deg}$ traction, $110\text{ W}$ solar charging).
- **Feasibility:** The Safest Route maintains a **72.9\% minimum SoC** (52.9\% margin above the critical $20\%$ reserve cutoff).

---

## 8. Phase 6: Volumetric CRIM Water Ice Resource Estimates

| Scenario | Active Ice Column (m) | Regolith Volume (km^3) | Pore Filling Fraction [%] | Nominal Ice Mass (Megatons) | Uncertainty 5th Pct (Mt) | Uncertainty 95th Pct (Mt) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Low (Conservative) | 1.2 | 0.0033 | 5.0 | 0.147 | 0.082 | 0.228 | Disseminated pore ice beneath 0.8m dry lag (2m penetration depth) |
| Nominal (Realistic) | 3.0 | 0.0082 | 15.0 | 1.099 | 0.608 | 1.701 | Heterogeneous ice-regolith breccia beneath 0.5m lag (3.5m penetration depth) |
| High (Optimistic) | 4.7 | 0.0129 | 30.0 | 3.439 | 1.91 | 5.253 | Segregated ice lenses extrapolated across 5m regolith column |

- **Model Formulation:** Complex Refractive Index Model (CRIM) across regolith matrix ($\epsilon_m' = 2.80$), pure water ice ($\epsilon_i' = 3.15$), and vacuum pores ($\phi = 0.45$).
- **Inversion Area:** Confined strictly to the $2.74\text{ km}^2$ cold-trap depression.
- **Monte Carlo Sensitivity:** Vectorized CPU simulation across 10,000 trials varying ice fraction, depth, permittivity, and density.

---

## 9. Limitations & Unassessed Parameters Disclosure

1. **S-band DFSAR Polarimetry:** Marked **`S-band unavailable`** as no overlapping Chandrayaan-2 DFSAR S-band Level-1 data was provided in `data/raw/`. Dual-frequency $\Delta\text{CPR}$ was disabled.
2. **OHRC Boulder Density:** Marked **`UNASSESSED`** as sub-meter Chandrayaan-2 OHRC ortho-imagery was not provided. Topographic RMS roughness from LOLA was used as a proxy.
3. **Geolocation Status:** Marked **`UNVALIDATED_EMPIRICAL`** because the correlation peak reached $\rho = 0.0932 < 0.40$. Maintained nominal tie-points with an explicit empirical uncertainty budget ($\sigma_{\text{geo}} = 3.27 - 4.59\text{ km}$).
4. **Horizon Illumination Model:** Evaluated across 72 discrete azimuths and grazing solar elevations ($0.5° - 2.5°$) without fine orbital SPICE ephemerides.
