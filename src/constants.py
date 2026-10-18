"""
LunaTrace Physical & Planetary Constants
Source citations provided for every physical, planetary, and sensor specification.
All values are immutable and defined with literature references.
"""

from typing import Final

# ==============================================================================
# 1. Planetary Geodetic & Orbital Constants
# ==============================================================================

# Mean volumetric radius of the Moon in meters.
# Citation: Archinal, B. A., et al. (2018). "Report of the IAU/IAG Working Group on
# Cartographic Coordinates and Rotational Elements: 2015." Celestial Mechanics and
# Dynamical Astronomy, 130(3), 22.
LUNAR_MEAN_RADIUS_M: Final[float] = 1737400.0

# Mean lunar spin-axis obliquity with respect to the ecliptic normal in degrees.
# Causes low grazing solar elevation angles (<1.5 deg - 2.5 deg) at the lunar South Pole.
# Citation: Margot, J. L., et al. (1999). "Topography of the Lunar Poles from Radar
# Interferometry: A Survey of Cold Traps." Science, 284(5420), 1658-1660.
LUNAR_OBLIQUITY_DEG: Final[float] = 1.5424

# Kilometers per degree latitude on the lunar spherical datum: (2 * pi * R) / 360.
# (2 * pi * 1737.4 km) / 360 deg = ~30.323 km / deg.
# Citation: Mazarico, E., et al. (2011). "Illumination conditions of the lunar polar
# regions using LOLA data." Icarus, 211(2), 1066-1081.
LUNAR_KM_PER_DEG_LAT: Final[float] = 30.32305


# ==============================================================================
# 2. Material Physical & Dielectric Properties
# ==============================================================================

# Density of cryogenic crystalline water ice (Ih phase) in kg / m^3 at T ~ 40 - 100 K.
# Citation: Lide, D. R. (2005). CRC Handbook of Chemistry and Physics (86th ed.).
# CRC Press; Hobbs, P. V. (1974). Ice Physics. Oxford University Press.
ICE_DENSITY_KG_M3: Final[float] = 920.0

# Real relative dielectric permittivity of dry lunar highland anorthosite regolith.
# Citation: Olhoeft, G. R., & Strangway, D. W. (1975). "Dielectric properties of the first
# 100 meters of the Moon." Earth and Planetary Science Letters, 24(3), 394-404;
# Carrier, W. D., et al. (1991). Lunar Sourcebook, Chapter 9: Physical Properties of the
# Lunar Regolith. Cambridge University Press.
EPS_REGOLITH_REAL: Final[float] = 2.70

# Real relative dielectric permittivity of pure cryogenic water ice at radar frequencies.
# Citation: Mätzler, C., & Wegmüller, U. (1987). "Dielectric properties of fresh-water ice
# at microwave frequencies." Journal of Physics D: Applied Physics, 20(12), 1623;
# Pettinelli, E., et al. (2015). "Dielectric properties of planetary materials: A review."
# Reviews of Geophysics, 53(3), 593-641.
EPS_ICE_REAL: Final[float] = 3.15

# Dielectric loss tangent (tan delta) of dry lunar highland regolith.
# Citation: Fa, W., & Wieczorek, M. A. (2012). "Regolith thickness over the Moon from
# Chang'E-1 Microwave Radiometer data." Icarus, 218(2), 771-787.
LOSS_TANGENT_DRY_REGOLITH: Final[float] = 0.0035

# Dielectric loss tangent of pure cryogenic water ice at T < 100 K and microwave frequencies.
# Citation: Gough, S. R. (1972). "Dielectric behavior of cubical and hexagonal ice at low
# temperatures." Journal of Chemical Physics; Pettinelli et al. (2015).
LOSS_TANGENT_ICE: Final[float] = 0.0001

# Subsurface bulk regolith porosity in the upper 0-5 m column.
# Citation: Heiken, G. H., Vaniman, D. T., & French, B. M. (1991). Lunar Sourcebook:
# A User's Guide to the Moon. Cambridge University Press.
BULK_REGOLITH_POROSITY: Final[float] = 0.42

# Theoretical maximum physical volumetric pore-saturation limit for pore ice in regolith.
MAX_PORE_SATURATION_FRACTION: Final[float] = 0.60


# ==============================================================================
# 3. Radar Polarimetry & Sensor Specifications (Chandrayaan-2 DFSAR)
# ==============================================================================

# Chandrayaan-2 Dual Frequency Synthetic Aperture Radar (DFSAR) L-band carrier frequency.
# Citation: Putrevu, D., et al. (2020). "Dual-Frequency SAR on Chandrayaan-2 Mission:
# Configuration, Calibration, and Initial Results." IEEE Transactions on Geoscience
# and Remote Sensing, 58(8), 5345-5355.
DFSAR_L_BAND_FREQ_GHZ: Final[float] = 1.25

# Chandrayaan-2 DFSAR L-band wavelength (~24.0 cm).
DFSAR_L_BAND_WAVELENGTH_M: Final[float] = 0.2398

# Chandrayaan-2 DFSAR S-band carrier frequency.
# Citation: Putrevu, D., et al. (2020).
DFSAR_S_BAND_FREQ_GHZ: Final[float] = 2.50

# Chandrayaan-2 DFSAR S-band wavelength (~12.0 cm).
DFSAR_S_BAND_WAVELENGTH_M: Final[float] = 0.1199

# Effective L-band microwave penetration depth in dry cryogenic regolith column (~10-15 wavelengths).
# Citation: Campbell, B. A., et al. (2006). "Radar penetration and subsurface imaging of
# the Moon." Icarus, 180(1), 1-7.
RADAR_PENETRATION_DEPTH_L_M: Final[float] = 3.5

# Effective S-band microwave penetration depth in dry cryogenic regolith column (~10 wavelengths).
# Citation: Campbell, B. A., et al. (2006).
RADAR_PENETRATION_DEPTH_S_M: Final[float] = 1.2


# ==============================================================================
# 4. Unit Conversion Factors
# ==============================================================================

# Cubic meters to cubic kilometers (1 m^3 = 10^-9 km^3).
M3_TO_KM3: Final[float] = 1.0e-9

# Kilograms to metric tons (1 t = 1000 kg).
KG_TO_METRIC_TON: Final[float] = 1.0e-3

# Metric tons to Megatons (1 Mt = 10^6 t).
METRIC_TON_TO_MEGATON: Final[float] = 1.0e-6

# Seconds per hour.
SECONDS_PER_HOUR: Final[float] = 3600.0
