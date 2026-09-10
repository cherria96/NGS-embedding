"""
Merge the five per-flag warning datasets (acid_accumulation, acid_base_balance,
ammonia_toxicity, biogas_quality, buffer_capacity) into a single multi-label
train/test CSV per domain (ARC, BAC, merged).

Each per-flag folder shares the same SampleID set and the same feature columns
(only the "Group" label differs), so we take the features from one flag folder
and attach one 0/1 column per flag (1 = Warning, 0 = Normal).
"""
import os
import shutil
import pandas as pd

FLAGS = [
    "acid_accumulation",
    "acid_base_balance",
    "ammonia_toxicity",
    "biogas_quality",
    "buffer_capacity",
]

DOMAINS = {
    "ARC": "warnings_ARC",
    "BAC": "warnings_BAC",
    "merged": "warnings_merged",
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "example")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input_for_all_models")


def build_split(domain_folder, split):
    """split is 'train' or 'test'; returns a DataFrame: SampleID, features..., <flags>"""
    merged = None
    for flag in FLAGS:
        path = os.path.join(EXAMPLE_DIR, domain_folder, flag, f"example_{split}.csv")
        df = pd.read_csv(path)
        label = (df["Group"] == "Warning").astype(int).rename(flag)
        if merged is None:
            merged = df.drop(columns=["Group"])
        else:
            # sanity check: identical sample order and feature columns across flags
            assert list(df["SampleID"]) == list(merged["SampleID"]), \
                f"SampleID mismatch between flags in {domain_folder}/{split}"
        merged = pd.concat([merged, label], axis=1)
    return merged


def main():
    for domain_key, domain_folder in DOMAINS.items():
        out_domain_dir = os.path.join(OUT_DIR, f"Warnings_{domain_key}")
        os.makedirs(out_domain_dir, exist_ok=True)

        train_df = build_split(domain_folder, "train")
        test_df = build_split(domain_folder, "test")

        train_path = os.path.join(out_domain_dir, "train.csv")
        test_path = os.path.join(out_domain_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        tree_src = os.path.join(EXAMPLE_DIR, domain_folder, "phylogeny.nwk")
        tree_dst = os.path.join(out_domain_dir, "phylogeny.nwk")
        shutil.copyfile(tree_src, tree_dst)

        print(f"[{domain_key}] train={train_df.shape} test={test_df.shape} -> {out_domain_dir}")


if __name__ == "__main__":
    main()
