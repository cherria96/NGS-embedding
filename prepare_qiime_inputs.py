#!/usr/bin/env python3
"""
prepare_qiime_inputs.py -- reshape one data/qiimeresult/{ARC,BAC} directory into
the (counts CSV, taxonomy CSV) pair that Phylospec/reduce_features_for_tree_models.py
consumes.

Two shape mismatches vs. what that script expects, both handled here:

  * table_filtered.tsv is features-in-rows / samples-in-columns (QIIME2 biom-export
    convention), with a leading "# Constructed from biom file" comment line. The
    reduction script wants samples-in-rows, so this transposes it.

  * The per-domain taxonomy file is metadata.tsv, not metadata.csv -- tab-delimited,
    with a `#q2:types` QIIME2 type-declaration row that has to be dropped. This
    rewrites it as a plain two-column CSV (`Feature ID`, `Taxon`).

Taxon LINEAGE STRING PARSING (splitting on ';'/'|', stripping 'g__' prefixes,
recognising uncultured/unidentified/unassigned/etc as unclassified, padding
missing ranks) is intentionally NOT duplicated here -- reduce_features_for_tree_
models.load_taxonomy()/split_lineage() already implements it, and that should stay
the one place the logic lives rather than risking two parsers drifting apart.

Sample ids are reconciled to the `data/final/metadata.csv` SampleID convention
("1-11b-DGYSb" -> "1-11b") via round_no_key(), matching the join key already used
by prepare_warning_targets_data.py.

This script never drops a feature. A table feature with no matching taxonomy row
is reported (features_without_taxonomy_row) rather than silently dropped -- absence
of a taxonomy row is not evidence the feature doesn't belong on the tree, and
reduce_features_for_tree_models.py already treats an unmatched feature as fully
unclassified at every rank, which is the correct default.
"""
import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
QIIME_DIR = os.path.join(ROOT, "data", "qiimeresult")


def round_no_key(sample_col):
    """'1-11b-DGYSb' -> '1-11b' -- matches data/final/metadata.csv's SampleID."""
    parts = sample_col.split("-")
    return parts[0] + "-" + parts[1]


def load_feature_table(domain):
    """features x samples, raw read counts, columns renamed to SampleID keys."""
    path = os.path.join(QIIME_DIR, domain, "table_filtered.tsv")
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)
    df.columns = [round_no_key(c) for c in df.columns]
    return df


def load_taxonomy_csv(domain):
    """Feature ID, Taxon -- from metadata.tsv, dropping the '#q2:types' row."""
    path = os.path.join(QIIME_DIR, domain, "metadata.tsv")
    tx = pd.read_csv(path, sep="\t", skiprows=[1])
    return tx[["Feature ID", "Taxon"]]


def prepare_domain_inputs(domain, out_prefix):
    """Writes `<out_prefix>_abundance.csv` (SampleID-indexed raw counts, samples
    in rows) and `<out_prefix>_taxonomy.csv` (Feature ID, Taxon). Returns a
    coverage report dict; also prints it so a bare CLI run is self-documenting.
    """
    counts = load_feature_table(domain)  # features x samples
    tax = load_taxonomy_csv(domain)

    abundance = counts.T  # samples x features
    abundance.index.name = "SampleID"
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    abundance.to_csv(f"{out_prefix}_abundance.csv")
    tax.to_csv(f"{out_prefix}_taxonomy.csv", index=False)

    tax_ids = set(tax["Feature ID"])
    missing_tax = [f for f in counts.index if f not in tax_ids]
    dup_sample_cols = counts.columns[counts.columns.duplicated()].unique().tolist()
    report = {
        "domain": domain,
        "n_samples": int(abundance.shape[0]),
        "n_features": int(abundance.shape[1]),
        "total_reads": float(counts.values.sum()),
        "features_without_taxonomy_row": len(missing_tax),
        "features_without_taxonomy_row_examples": missing_tax[:5],
        "duplicate_sample_keys_after_round_no_key": dup_sample_cols,
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain", choices=["ARC", "BAC"])
    ap.add_argument("-o", "--out-prefix", required=True,
                     help="writes <out-prefix>_abundance.csv and <out-prefix>_taxonomy.csv")
    a = ap.parse_args()
    prepare_domain_inputs(a.domain, a.out_prefix)


if __name__ == "__main__":
    main()
