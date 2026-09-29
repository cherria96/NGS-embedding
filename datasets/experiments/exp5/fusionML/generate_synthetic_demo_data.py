#!/usr/bin/env python3
"""Writes the SYNTHETIC demonstration dataset used by configs/synthetic_demo.yaml.

Run once (or whenever you want a fresh synthetic draw):
    conda activate torch
    python fusionML/generate_synthetic_demo_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionML.synthetic import generate_synthetic_dataset, write_synthetic_dataset_to_dir

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
    ds = generate_synthetic_dataset(
        n_sites=12, visits_per_site=4, p_arc=8, p_bac=10, seed=20260917,
        depth_arc=4000, depth_bac=7000, missing_visit_prob=0.15,
    )
    paths = write_synthetic_dataset_to_dir(ds, out_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")
