"""
LunaTrace Provenance Engine
Generates cryptographic SHA-256 sidecars (.provenance.json) for every pipeline output raster and table.
Records input data hashes, git commit, runtime environment, and active configuration.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot compute hash for missing file: {file_path}")
    
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, or 'UNKNOWN' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def get_software_versions() -> Dict[str, str]:
    """Captures key runtime library versions."""
    import rasterio
    import pydantic
    import scipy
    import pandas

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "pydantic": pydantic.__version__
    }


def write_provenance_sidecar(
    output_path: Path,
    input_paths: List[Path],
    config_dict: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Writes a JSON provenance sidecar file: <output_file>.provenance.json
    Fails fast if any input file is missing.
    """
    sidecar_path = output_path.parent / f"{output_path.name}.provenance.json"

    # Compute hashes for all existing inputs
    input_hashes = {}
    for p in input_paths:
        if p.exists():
            input_hashes[p.name] = compute_file_sha256(p)
        else:
            raise FileNotFoundError(f"Provenance generation failed: input {p} does not exist")

    # Output file hash
    output_hash = compute_file_sha256(output_path) if output_path.exists() else None

    provenance_record = {
        "output_file": output_path.name,
        "output_sha256": output_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "software_versions": get_software_versions(),
        "input_files": input_hashes,
        "config_snapshot": config_dict or {},
        "metadata": metadata or {}
    }

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(provenance_record, f, indent=2, default=str)

    return sidecar_path
