"""
LunaTrace: Chandrayaan-2 Radar & Optical Processing Pipeline
Subsurface Ice Detection, Landing Site Selection, and Rover Traverse Planning
Lunar South Pole (ISRO Problem Statement 8)
"""

import sys
import os

# Handle Windows DLL directory for rasterio / GDAL if needed
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = [p for p in sys.path if "site-packages" in p]
    for sp in site_packages:
        libs_dir = os.path.join(sp, "rasterio.libs")
        if os.path.isdir(libs_dir):
            try:
                os.add_dll_directory(libs_dir)
            except (FileNotFoundError, OSError):
                pass
