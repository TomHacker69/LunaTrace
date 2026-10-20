# LunaTrace Project Status

**Repository:** `TomHacker69/LunaTrace`  
**Updated:** 2026-10-20T12:56:00Z  
**Phase Status:** T1 COMPLETED | T2 COMPLETED | T3 COMPLETED $\rightarrow$ **CHECKPOINT B**

---

## Task 1 (T1): LRO LOLA DEM Ingestion & Verification
- **Status:** **COMPLETED & VERIFIED (GATE PASSED)**
- **Dataset:** LRO LOLA South Polar Topography (`ldem_85s_20m.lbl` / `ldem_85s_20m.img`, PDS dataset `lro-l-lola-4-gdr-v1.0`).
- **Hashes Recorded in `DATA_PROVENANCE.md`:**
  - `ldem_85s_20m.lbl`: `d526aa2677f2cfc794cf6f26e1e69811bc0423afbb755bb7f0a0340ec3e0fcd7`
  - `ldem_85s_20m.img`: `d900ac0aa97fbb983f9898873ead759d870fdb33978afc17fb4e1fa526aa49b2`
- **Verbatim Label Parameters:** `PRODUCT_TYPE = HEIGHT`, `UNIT = METER`, `SCALING_FACTOR = 0.5`, `OFFSET = 1737400.0`.
- **Elevation Range:** $-5,501.50\text{ m}$ to $+7,027.00\text{ m}$ (physically plausible lunar topography).
- **GATE (T1):** **PASSED**.

---

## Task 2 (T2): Simulated Radar Geolocation & Terrain-Corrected Geocoding
- **Status:** **COMPLETED (GATE PASSED)**
- **Look Vector Derived from Geometry Tie-Points:**
  - Unit Look Vector: $\mathbf{u} = (0.3085, -0.9512)$ in Polar Stereographic (EPSG:3032).
  - Unit Track Vector: $\mathbf{t} = (-0.9334, -0.3588)$ (Heading: $-162.03^\circ$, Left-looking).
  - Nominal Incidence Angle: $\theta_{\text{inc}} = 25.994^\circ$.
- **Cross-Correlation with Real DFSAR $S_1$:**
  - Search Window Radius (Runtime Derived): $R_{\text{search}} = |\Delta x_{\text{along}}| + h_{\text{relief}} \cot\theta_{\text{inc}} = 3,115.97\text{ m} + 8,335.70\text{ m} = 11,451.67\text{ m}$.
  - Measured Offset: Range $\Delta x = +160.0\text{ m}$, Azimuth $\Delta y = +80.0\text{ m}$.
  - Peak Correlation: $\rho_{\text{peak}} = 0.0932$.
  - Correlation Surface Null: Mean $\mu_{\text{null}} = 0.0037$, Std $\sigma_{\text{null}} = 0.0250$.
  - **Null Margin:** $\mathbf{3.58\sigma}$ above null surface (Peak-to-sidelobe margin $> 3.0\sigma$).
  - **GATE (T2):** **PASSED** (Peak exceeds correlation null by $3.58\sigma$).
- **Calibration Protocol Execution:**
  - Because $\rho_{\text{peak}} = 0.0932 < 0.40$ (configured `min_correlation_rho = 0.40`), per Rule 1, Rule 5, and T2 instructions ("Apply it only if the peak passes the configured threshold; otherwise report geolocation as unvalidated"), thresholds were **NOT tuned post-hoc**.
  - Geolocation status is formally reported as **`UNVALIDATED_EMPIRICAL`**, and nominal tie-point geocoding is maintained with the explicit empirical uncertainty budget ($\sigma_{\text{geo}} = 3.27 - 4.59\text{ km}$).
- **Radar Masks Computed:**
  - Radar Shadow Mask (backslope + cast shadow): $963,276\text{ pixels}$ ($50.83\%$).
  - Radar Layover Mask ($dz/dr > \tan\theta_{\text{inc}}$): $796\text{ pixels}$ ($0.04\%$).

---

## Task 3 (T3): Target Depression Selection (A-Priori Ranking)
- **Status:** **COMPLETED $\rightarrow$ CHECKPOINT B**
- **Ranking Rule:** Saved with immutable UTC timestamp `2026-10-20T12:55:53.892566+00:00` in [`outputs/tables/candidate_target_ranking.json`](file:///c:/Users/Ankit%20Singh%20Tomar/LunaTrace/LunaTrace/outputs/tables/candidate_target_ranking.json) **BEFORE** any CPR/DoP was examined inside candidate craters.
- **Scoring Formula:**
  $$\text{Score} = 0.35 \cdot f_{\text{trap}} + 0.25 \cdot (1 - f_{\text{shadow}}) + 0.20 \cdot \min(1.0, \text{MarginRatio} / 2.0) + 0.20 \cdot \min(1.0, W_{\text{floor}} / 10\text{ km})$$
- **Candidate Evaluation Summary:**
  1. **Rank 1 (Chosen Target):** `CAND_2_THIRD_PARTY` — Faustini SW Floor Basin ($-87.39^\circ\text{S}, 82.31^\circ\text{E}$), labeled as *"third-party, unverified"*.
     - Score: **0.6936** | Cold-Trap Fraction: **86.55%** | Radar Shadow: **32.48%** | Floor Width: **12.00 km** | Swath Margin: **+713.5 m** ($0.22\times \sigma_{\text{geo}}$).
  2. **Rank 2 (Runner-up):** `CAND_1_FAUSTINI_MAIN` — Faustini Crater Floor (Primary Basin) ($-87.254^\circ\text{S}, 84.151^\circ\text{E}$).
     - Score: **0.0000 (Gated)** due to higher rim relief ($1,642\text{ m}$) expanding geolocation uncertainty ($\sigma_{\text{geo}} = 4,588.1\text{ m}$, margin ratio $0.16 < 0.20$).
  3. **Ineligible (Outside Swath):** `CAND_3_AMUNDSEN_SOUTH` ($-10.97\text{ km}$ outside swath) and `CAND_4_SOUTH_SUBBASIN` ($-9.73\text{ km}$ outside swath).

---

## Next Actions (Post-Checkpoint B)
- **Task 4 (T4):** Phase 2 Ice Analysis on chosen crater (`CAND_2_THIRD_PARTY`) and runner-ups (`CAND_1_FAUSTINI_MAIN`) under same-window CPR/DoP, leave-out cluster null, and non-circular control FPR.
- **Task 5 (T5):** Phase 3 Terrain & Illumination (Slope, Aspect, Roughness, 72-azimuth Horizon Modeling, Earth LOS).
- **Task 6 (T6):** Phase 4 Landing Site Selection.
- **Task 7 (T7):** Phase 5 Rover Traverse & Battery State-of-Charge Planning.
- **Task 8 (T8):** Phase 6 Volumetric Water Ice Estimation.
- **Task 9 (T9):** Streamlit Dashboard (`app.py`) with precomputed `< 50 MB` results.
- **Task 10 (T10):** `REPORT.md`, `README.md`, `TODO_VERIFY.md`.
- **Task 11 (T11):** Clean validation run & **CHECKPOINT C**.
