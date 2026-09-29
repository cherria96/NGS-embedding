"""Generic artifact-writing helpers (design-spec item 17's artifact table)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder, default=str)


def write_df(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)


def hash_file(path: Optional[str]) -> Optional[str]:
    if path is None or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_versions() -> dict:
    versions = {}
    for mod_name in ("numpy", "pandas", "scipy", "sklearn", "cvxpy", "yaml", "Bio", "matplotlib"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    return versions


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def build_run_manifest(config_dict: dict, input_hashes: dict, mode: str, run_id: str,
                        extra: Optional[dict] = None) -> dict:
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "timestamp_unix": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": config_dict.get("seed"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "dependency_versions": dependency_versions(),
        "input_hashes": input_hashes,
        "config": config_dict,
        "smoke": config_dict.get("smoke", False),
    }
    if extra:
        manifest.update(extra)
    return manifest
