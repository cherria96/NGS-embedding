

# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import pytest

from datasets.datapreprocess.synthetic import generate_synthetic_dataset


@pytest.fixture
def small_synth():
    return generate_synthetic_dataset(n_sites=8, visits_per_site=4, p_arc=6, p_bac=7, seed=123,
                                       depth_arc=4000, depth_bac=6000, missing_visit_prob=0.1)


@pytest.fixture
def prefixed_counts(small_synth):
    a = small_synth.counts_arc.copy()
    a.columns = [f"ARC::{c}" for c in a.columns]
    b = small_synth.counts_bac.copy()
    b.columns = [f"BAC::{c}" for c in b.columns]
    return a, b
