"""Synthetic data generator for the end-to-end implementation demonstration
(design-spec item 18.10).

"Generate positive compositions for two separate domains, introduce known
zero-sum coefficients and repeated-site variation, convert to separate
simulated count assays, and execute a small nested grouped run. State
explicitly that this tests implementation, not the scientific hypothesis in
real AD data." Every artifact this module produces is labeled synthetic.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _random_binary_newick(tip_names: List[str], rng: np.random.Generator) -> str:
    """A random-topology binary tree with positive branch lengths, for tests
    and the synthetic demo -- not a claim about any real phylogeny."""
    nodes = [f"{name}:{rng.uniform(0.05, 1.0):.6f}" for name in tip_names]
    while len(nodes) > 1:
        rng.shuffle(nodes)
        a, b = nodes.pop(), nodes.pop()
        bl = rng.uniform(0.05, 1.0)
        nodes.append(f"({a},{b}):{bl:.6f}")
    return nodes[0] + ";"


@dataclasses.dataclass
class SyntheticDataset:
    counts_arc: pd.DataFrame  # samples x raw feature id (unprefixed), raw counts
    counts_bac: pd.DataFrame
    metadata: pd.DataFrame  # sample_id, site_id, target
    tree_arc_newick: str
    tree_bac_newick: str
    taxonomy_arc: pd.DataFrame  # Feature ID, Taxon
    taxonomy_bac: pd.DataFrame
    true_beta_a: np.ndarray
    true_beta_b: np.ndarray
    true_b0: float
    feature_ids_a: List[str]
    feature_ids_b: List[str]
    site_of_sample: Dict[str, str]


def generate_synthetic_dataset(
    n_sites: int = 10, visits_per_site: int = 4, p_arc: int = 8, p_bac: int = 10,
    seed: int = 20260917, depth_arc: int = 3000, depth_bac: int = 6000,
    site_effect_sd: float = 0.6, visit_noise_sd: float = 0.15, y_noise_sd: float = 0.05,
    missing_visit_prob: float = 0.15, task: str = "regression",
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)

    feature_ids_a = [f"synASV_A{i:03d}" for i in range(p_arc)]
    feature_ids_b = [f"synASV_B{i:03d}" for i in range(p_bac)]

    # Known zero-sum true coefficients (a handful of nonzero, rest zero).
    true_beta_a = np.zeros(p_arc)
    true_beta_a[0], true_beta_a[1] = 1.2, -1.2
    if p_arc > 3:
        true_beta_a[2], true_beta_a[3] = 0.6, -0.6
    true_beta_b = np.zeros(p_bac)
    true_beta_b[0], true_beta_b[1] = 0.9, -0.9
    if p_bac > 3:
        true_beta_b[2], true_beta_b[3] = -0.4, 0.4
    true_b0 = 0.0 if task == "classification" else 5.0

    site_ids = [f"SITE{i:02d}" for i in range(n_sites)]
    site_baseline_a = {s: rng.normal(0, site_effect_sd, size=p_arc) for s in site_ids}
    site_baseline_b = {s: rng.normal(0, site_effect_sd, size=p_bac) for s in site_ids}

    rows_meta, rows_counts_a, rows_counts_b = [], [], []
    site_of_sample: Dict[str, str] = {}
    sample_counter = 0
    for s in site_ids:
        n_visits = visits_per_site
        for v in range(n_visits):
            if v > 0 and rng.random() < missing_visit_prob:
                continue
            sample_counter += 1
            sample_id = f"SYN-{sample_counter:04d}"
            site_of_sample[sample_id] = s

            log_a = site_baseline_a[s] + rng.normal(0, visit_noise_sd, size=p_arc)
            log_b = site_baseline_b[s] + rng.normal(0, visit_noise_sd, size=p_bac)
            q_a = np.exp(log_a); q_a /= q_a.sum()
            q_b = np.exp(log_b); q_b /= q_b.sum()

            eta = true_b0 + float(np.log(q_a) @ true_beta_a) + float(np.log(q_b) @ true_beta_b)
            if task == "classification":
                prob = 1.0 / (1.0 + np.exp(-eta))
                y = float(rng.random() < prob)
            else:
                y = eta + rng.normal(0, y_noise_sd)

            counts_a = rng.multinomial(depth_arc, q_a)
            counts_b = rng.multinomial(depth_bac, q_b)

            rows_meta.append({"sample_id": sample_id, "site_id": s, "target": y})
            rows_counts_a.append(pd.Series(counts_a, index=feature_ids_a, name=sample_id))
            rows_counts_b.append(pd.Series(counts_b, index=feature_ids_b, name=sample_id))

    metadata = pd.DataFrame(rows_meta).set_index("sample_id", drop=False)
    counts_arc = pd.DataFrame(rows_counts_a)
    counts_bac = pd.DataFrame(rows_counts_b)

    tree_arc = _random_binary_newick(feature_ids_a, np.random.default_rng(seed + 1))
    tree_bac = _random_binary_newick(feature_ids_b, np.random.default_rng(seed + 2))

    tax_arc = pd.DataFrame({
        "Feature ID": feature_ids_a,
        "Taxon": [f"d__Archaea; p__SynP; c__SynC; o__SynO; f__SynF; g__SynGenusA{i % 3}" for i in range(p_arc)],
        "Confidence": 0.99,
    })
    tax_bac = pd.DataFrame({
        "Feature ID": feature_ids_b,
        "Taxon": [f"d__Bacteria; p__SynP; c__SynC; o__SynO; f__SynF; g__SynGenusB{i % 4}" for i in range(p_bac)],
        "Confidence": 0.99,
    })

    return SyntheticDataset(
        counts_arc=counts_arc, counts_bac=counts_bac, metadata=metadata,
        tree_arc_newick=tree_arc, tree_bac_newick=tree_bac,
        taxonomy_arc=tax_arc, taxonomy_bac=tax_bac,
        true_beta_a=true_beta_a, true_beta_b=true_beta_b, true_b0=true_b0,
        feature_ids_a=feature_ids_a, feature_ids_b=feature_ids_b, site_of_sample=site_of_sample,
    )


def write_synthetic_dataset_to_dir(ds: SyntheticDataset, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "arc_counts": os.path.join(out_dir, "SYNTHETIC_arc_counts.csv"),
        "bac_counts": os.path.join(out_dir, "SYNTHETIC_bac_counts.csv"),
        "metadata": os.path.join(out_dir, "SYNTHETIC_metadata.csv"),
        "arc_tree": os.path.join(out_dir, "SYNTHETIC_arc_tree.nwk"),
        "bac_tree": os.path.join(out_dir, "SYNTHETIC_bac_tree.nwk"),
        "arc_taxonomy": os.path.join(out_dir, "SYNTHETIC_arc_taxonomy.tsv"),
        "bac_taxonomy": os.path.join(out_dir, "SYNTHETIC_bac_taxonomy.tsv"),
    }
    a = ds.counts_arc.copy(); a.index.name = "sample_id"
    b = ds.counts_bac.copy(); b.index.name = "sample_id"
    a.to_csv(paths["arc_counts"])
    b.to_csv(paths["bac_counts"])
    meta_out = ds.metadata.rename(columns={"target": "methane_yield_SYNTHETIC"})
    meta_out.to_csv(paths["metadata"], index=False)
    with open(paths["arc_tree"], "w") as f:
        f.write(ds.tree_arc_newick)
    with open(paths["bac_tree"], "w") as f:
        f.write(ds.tree_bac_newick)
    ds.taxonomy_arc.to_csv(paths["arc_taxonomy"], sep="\t", index=False)
    ds.taxonomy_bac.to_csv(paths["bac_taxonomy"], sep="\t", index=False)
    return paths
