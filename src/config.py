"""
LunaTrace Master Configuration Bridge
Exposes validated configuration and standardized paths loaded strictly from config.yaml.
"""

from pathlib import Path
from src.config_schema import load_config, LunaTraceConfig

# Load validated configuration
CFG: LunaTraceConfig = load_config()

# Standardized paths for pipeline modules
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = CFG.paths.raw_data_dir
PROCESSED_DATA_DIR = CFG.paths.processed_data_dir
MAPS_DIR = CFG.paths.outputs_maps_dir
TABLES_DIR = CFG.paths.outputs_tables_dir

# Ensure output directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MAPS_DIR, TABLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)
