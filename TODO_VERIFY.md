# LunaTrace: Verification Registry & External Assumptions (TODO_VERIFY)

**Repository:** `TomHacker69/LunaTrace`  
**Purpose:** Comprehensive registry of all external physical constants, literature citations, sensor specifications, and mission assumptions for independent scientific verification (Rule 4).

---

## 1. Physical Constants & Planetary Datums

| Parameter / Constant | Value | Literature Citation | Verification Status | Verification Guidance for User |
| :--- | :--- | :--- | :--- | :--- |
| **Lunar Mean Radius** | $1,737,400.0\\text{ m}$ ($1,737.4\\text{ km}$) | Archinal et al. (2018), *IAU/IAG Working Group on Cartographic Coordinates* | <span style="color:#10b981;">STANDARD IAU</span> | Check against PDS4 label `<isda:semi_major_radius>` (verbatim $1737400.0$). |
| **Speed of Light in Vacuum** | $299,792,458\\text{ m/s}$ | CODATA (2018) | <span style="color:#10b981;">EXACT SI</span> | Fundamental physical constant. |
| **Water Ice Density** | $920.0\\text{ kg/m}^3$ | Hobbs, P. V. (1974), *Ice Physics*, Clarendon Press | <span style="color:#f59e0b;">TODO_VERIFY</span> | Verify cryogenic temperature dependence at $T < 100\\text{ K}$ (density increases to $\\sim 930\\text{ kg/m}^3$). |
| **Lunar Highland Regolith Permittivity** | $\\varepsilon_{\\text{matrix}}' = 2.80 - j0.005$ | Olhoeft & Strangway (1975), *Earth Planet. Sci. Lett.*, 24(3) | <span style="color:#f59e0b;">TODO_VERIFY</span> | Measured on Apollo 16 anorthosite samples; verify frequency dependence at L-band ($1.25\\text{ GHz}$). |
| **Pure Water Ice Permittivity** | $\\varepsilon_{\\text{ice}}' = 3.15 - j0.0001$ | Warren & Brandt (2008), *J. Geophys. Res.*, 113(D14) | <span style="color:#f59e0b;">TODO_VERIFY</span> | Measured at microwave frequencies at $100\\text{ K}$. |
| **Bulk Regolith Porosity** | $\\phi = 0.45$ ($45\\%$) | Heiken et al. (1991), *Lunar Sourcebook*, Cambridge Univ. Press | <span style="color:#f59e0b;">TODO_VERIFY</span> | Highland surface regolith ($0 - 15\\text{ cm}$) has $\\phi \\approx 0.45 - 0.50$; compacted at depth ($> 1\\text{ m}$) to $\\sim 0.35$. |

---

## 2. Sensor Metadata & Instrument Parameters

| Sensor / Tag | Label Value | Source Document | Verification Guidance |
| :--- | :--- | :--- | :--- |
| **DFSAR Center Frequency** | $1.250\\times 10^9\\text{ Hz}$ ($1.25\\text{ GHz}$, $\\lambda = 23.98\\text{ cm}$) | PDS4 XML `<isda:radar_center_frequency>` | Check line 118 of `ch2_sar_..._xx_d18.xml`. |
| **Calibration Constant $K_{\\text{cal}}$** | $80.00\\text{ dB}$ | PDS4 XML `<isda:calibration_constant>` | Check line 136 of `ch2_sar_..._xx_d18.xml`. |
| **NESZ Floor Coefficient** | $9.749896\\times 10^{-3}$ ($-20.11\\text{ dB}$) | PDS4 XML `<isda:nes0_coeff_0>` | Check lines 91 and 102 of `ch2_sar_..._xx_d18.xml`. |
| **Along-Track Timing Residual** | $-3,115.967394\\text{ m}$ ($-2.026348\\text{ s}$) | PDS4 XML `<isda:geo_error_time_meters>` | Check lines 140-141 of `ch2_sar_..._xx_d18.xml`. |
| **Range Geometry Residual** | $+31.759268\\text{ m}$ | PDS4 XML `<isda:geo_error_range_meters>` | Check line 142 of `ch2_sar_..._xx_d18.xml`. |
| **LOLA Scaling Factor** | $0.5$ | PDS3 `.lbl` `SCALING_FACTOR` | Check line 58 of `ldem_85s_20m.lbl`. |
| **LOLA Offset (Reference Sphere)** | $1,737,400.0\\text{ m}$ | PDS3 `.lbl` `OFFSET` | Check line 59 of `ldem_85s_20m.lbl` (Do not add to height products). |

---

## 3. Mission Engineering & Rover Assumptions

| Assumption / Rule | Adopted Value | Baseline Reference | Verification Guidance |
| :--- | :--- | :--- | :--- |
| **Chassis Static Rollover Angle** | $22.5^\\circ$ | NASA VIPER Mobility Specification | <span style="color:#f59e0b;">TODO_VERIFY</span>: Verify with rover mechanical design for hackathon payload chassis. |
| **Engineering Safety Factor** | $1.25$ | Planetary Rover Navigation Standard | <span style="color:#f59e0b;">TODO_VERIFY</span>: $\\theta_{\\text{nav,max}} = 22.5^\\circ / 1.25 = 18.0^\\circ$. |
| **Rover Baseline Power Draw** | $65.0\\text{ W}$ | NASA VIPER Power Budget Baseline | <span style="color:#f59e0b;">TODO_VERIFY</span>: Check avionics + thermal heater baseline. |
| **Slope Traction Power Coefficient** | $4.0\\text{ W/deg}$ | Bekker Terramechanics Model | <span style="color:#f59e0b;">TODO_VERIFY</span>: Verify wheel slippage in polar highland regolith. |
| **Solar Array Generation** | $110.0\\text{ W}$ | 4-panel vertical solar mast | <span style="color:#f59e0b;">TODO_VERIFY</span>: Assumes grazing incidence ($1^\\circ - 2^\\circ$) at pole. |
| **Battery Storage Capacity** | $1,800\\text{ Wh}$ ($1.8\\text{ kWh}$) | VIPER Li-ion Secondary Battery Pack | <span style="color:#f59e0b;">TODO_VERIFY</span>: Verify allowable DoD (Depth of Discharge). |
| **Minimum Reserve Margin** | $20\\%$ ($360\\text{ Wh}$) | Mission Critical Power Cutoff | Hard floor preventing battery freeze. |
| **Maximum Shadow Duration** | $18.0\\text{ hours}$ | Cold-trap dash thermal survival limit | Without RTG or RHU, electronics freeze below $50\\text{ K}$. |

---

## 4. Scientific Hypotheses & Unassessed Claims

| Claim / Item | Status | Action Required |
| :--- | :--- | :--- |
| **Ice Detection via Radar Alone** | **NOT ESTABLISHED** | Confirmed: Native radar statistics alone are consistent with speckle null; do not assert ice without OHRC or ground truth. |
| **S-band Polarization Data** | **UNAVAILABLE** | S-band Level-1 data was not provided; dual-frequency $\\Delta\\text{CPR}$ depth inversion is disabled. |
| **Sub-meter Boulder Hazards** | **UNASSESSED** | Chandrayaan-2 OHRC was not provided; keep RMS roughness as separate proxy. |
| **Dem Correlation Peak ($0.0932$)** | **UNVALIDATED** | Geolocation is maintained with empirical uncertainty budget rather than applying unvalidated offset. |
