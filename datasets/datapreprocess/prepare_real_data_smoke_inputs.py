#!/usr/bin/env python3
"""Derives fusionML-ready copies of the real QIIME2 count tables for
configs/real_smoke.yaml and configs/real_research_template.yaml.

Why this exists: `data/qiimeresult/{ARC,BAC}/table_filtered.tsv` column
headers are `{round}-{No}[-{site_code}]` (e.g. "1-1-BSSG"), while
`data/final/metadata.csv`'s `SampleID` is the canonical `{round}-{No}`
(e.g. "1-1") per README.md's documented ID convention. This script applies
the SAME round-No normalization `prepare_warning_targets_data.py` already
uses elsewhere in this repo (`round_no_key`: join the first two
"-"-separated tokens) and writes a renamed COPY -- it never edits the
checked-in QIIME tables in data/qiimeresult/, which CLAUDE.md documents as
the git-tracked source of truth.

Verified empirically (see session notes): this normalization gives a
perfect, collision-free match against metadata.csv for both domains
(ARC: 134/134, BAC: 140/140).

Run once (or after data/qiimeresult/*/table_filtered.tsv changes):
    conda activate torch
    python fusionML/prepare_real_data_smoke_inputs.py
"""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

import os

import pandas as pd

ROOT = str(_REPO_ROOT)
QIIME_DIR = os.path.join(ROOT, "datasets/raw", "qiimeresult")
OUT_DIR = os.path.join(ROOT, "models/fusionML", "real_data_cache")


def round_no_key(sample_col: str) -> str:
    parts = sample_col.split("-")
    return parts[0] + "-" + parts[1]


def rename_columns(domain: str) -> str:
    src = os.path.join(QIIME_DIR, domain, "table_filtered.tsv")
    df = pd.read_csv(src, sep="\t", skiprows=1, index_col=0)
    before = len(df.columns)
    df.columns = [round_no_key(c) for c in df.columns]
    dup = df.columns[df.columns.duplicated()].unique().tolist()
    if dup:
        raise SystemExit(f"{domain}: round-No normalization produced {len(dup)} duplicate sample IDs: {dup}")
    out_path = os.path.join(OUT_DIR, f"{domain}_table_filtered_roundno.tsv")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(out_path, sep="\t")
    print(f"{domain}: {before} raw columns -> {len(df.columns)} round-No-normalized sample IDs -> {out_path}")
    return out_path


if __name__ == "__main__":
    rename_columns("ARC")
    rename_columns("BAC")
