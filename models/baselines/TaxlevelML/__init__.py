"""Taxonomic baselines, including the existing flat-module training scripts."""
from pathlib import Path
import sys

# Historical scripts and serialized estimators use bare sibling module names.
_directory = str(Path(__file__).resolve().parent)
if _directory not in sys.path:
    sys.path.insert(0, _directory)
