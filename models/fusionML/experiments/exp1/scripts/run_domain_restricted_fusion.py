"""Fit the domain-restricted and both-domain fusion arms for three warning flags.

Answers "where should the fusion penalty act?" -- ARC edges only, BAC edges
only, or both domains -- at a chosen taxonomic resolution, on exactly the
Experiment 1 QC protocol (134 samples, 35 site groups, 3 outer folds x 2
repeats, 2 inner folds, same seeds and candidate generation).

`--resolution` selects asv (the original ARC/BAC rooted trees) or one of the
monophyly-consistent rank trees under output/genus_tree/taxlevel_autorun/.
Each resolution writes its own run directory, so they can be compared
like-for-like instead of against the older 37-site smoke runs.

`fusion_arc_only` / `fusion_bac_only` keep BOTH domains as predictors; only
the named domain receives fusion edges, so the other domain's coefficients
are fit completely unfused. These are not single-assay models.

Both selection rules are exported. Unlike run_asv_fusion.py this run keys its
partition-fit cache on the family spec, because fusion_domains is a spec
property that the candidate tuple does not encode.

Writes a separate checkpointed run; preserves every existing run directory.
"""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))


import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fusionmlv2-exp1-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fusionmlv2-exp1-cache")
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from function import pipeline as pipe
from configs.fusionml import Config
from function.folds import make_site_grouped_folds, sample_ids_for_sites
from models.fusionML.groups import extract_groups, verify_group_reconstruction
from models.fusionML.search import Candidate, FAMILY_SPECS, generate_candidates, generate_interaction_candidates
from function.evaluate import per_sample_log_loss, select_threshold_via_inner_oof_f2
from function.investigation_20260921 import learning_curves as legacy

TARGETS = ["warning_ammonia_toxicity", "warning_acid_base_balance", "warning_biogas_quality",
           # One-sided alkaline half of warning_acid_base_balance (pH > 8.0). The
           # two-sided flag is non-monotone in pH and a log-contrast score cannot
           # represent it; this half can be represented.
           "warning_acid_high"]
# Targets that the saved genus QC experiment also ran, so their outer splits
# can be asserted identical to the reused comparison.
REFERENCE_TARGETS = ["warning_ammonia_toxicity", "warning_acid_base_balance"]
MODELS = ["fusion_arc_only", "fusion_bac_only", "shared_hybrid",
          "domain_adaptive_hybrid", "domain_adaptive_interaction"]
FUSION_DOMAINS = {m: FAMILY_SPECS[m].fusion_domains for m in MODELS}
RESOLUTIONS = ["asv", "genus", "family", "order", "class"]
RANK_TREES = ROOT / "datasets/processed/genus_tree/taxlevel_autorun"
OUTPUT = ROOT / "results/classification result/fusionML/real_results_domain_fusion_{resolution}_exp1_20260922"
KEYS = ["rows", "predictions", "search", "failures", "splits",
        "coefficients_and_groups", "domain_counts", "interactions", "filter_audit"]
INPUTS = {}
REFERENCE = None
RESOLUTION = "asv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(target, resolution):
    cfg = Config.from_yaml(str(ROOT / "configs/fusionML/real_smoke_classification_phylo.yaml"))
    cfg.raw["resolution"] = resolution
    cfg.raw["columns"]["target"] = target
    # A rank resolution aggregates on its own tree; asv uses the rooted trees
    # directly and must have no rank tree configured at all.
    cfg.raw["inputs"]["rank_trees"] = {} if resolution == "asv" else {
        resolution: {domain: str(RANK_TREES / resolution / f"{domain}_tree.nwk") for domain in ["arc", "bac"]}}
    for domain in ["arc", "bac"]:
        cfg.raw["inputs"][domain + "_tree"] = str(ROOT / f"datasets/raw/qiimeresult/{domain.upper()}/rooted-tree.nwk")
    cfg.raw["cv"]["families"] = list(MODELS)
    return cfg


def trees_and_tables(inp, resolution):
    """The trees the fusion graph is built on and the counts fed to it."""
    if resolution == "asv":
        return inp.tree_arc, inp.tree_bac, inp.counts_arc_asv, inp.counts_bac_asv
    arc, bac = pipe.tables_for_resolution(inp, resolution)
    return inp.rank_tree_arc, inp.rank_tree_bac, arc, bac


def init_worker(resolution="asv", targets=None):
    global INPUTS, REFERENCE, RESOLUTION
    INPUTS, RESOLUTION = {}, resolution
    targets = list(targets or TARGETS)
    labels = pd.read_csv(ROOT / "models/fusionML/experiments/exp1/data/investigation_20260921/learning_qc/predictions.csv")
    REFERENCE = pd.read_csv(ROOT / "models/fusionML/experiments/exp1/data/investigation_20260921/learning_qc/splits.csv")
    REFERENCE = REFERENCE[REFERENCE.fraction.eq(1)]
    for target in targets:
        cfg = make_config(target, resolution)
        inp = pipe.load_all_inputs(cfg)
        assert inp.rank_used == (None if resolution == "asv" else resolution)
        inp.metadata["site_id"] = inp.metadata.site_id.replace({"BSIb": "BSI", "DGYb": "DGY"})
        old = labels[labels.target.eq(REFERENCE_TARGETS[0]) & labels.fraction.eq(1)].drop_duplicates("sample_id").set_index("sample_id")
        assert set(inp.cohort_ids) == set(old.index)  # Same cohort for every flag.
        assert np.array_equal(inp.metadata.loc[old.index, "site_id"].to_numpy(), old.site_id.to_numpy())
        if target in REFERENCE_TARGETS:
            same = labels[labels.target.eq(target) & labels.fraction.eq(1)].drop_duplicates("sample_id").set_index("sample_id")
            assert np.array_equal(inp.metadata.loc[same.index, "target"].to_numpy(), same.y.to_numpy())
        INPUTS[target] = (cfg, inp)


class RestrictedWorkbench(legacy.Workbench):
    """Partition-fit cache keyed on the family spec, not only the candidate.

    fusion_arc_only/fusion_bac_only share candidate tuples with the hybrids
    but build a different edge set, so the model name must enter the key.
    """

    def fit_fusion(self, ids, model, candidate):
        key = (tuple(ids), model, candidate.key())
        if key not in self.fit_cache:
            seed = pipe.derive_seed(self.cfg["seed"], "matched_graph", *sorted(ids))
            pf = pipe.fit_partition(ids, FAMILY_SPECS[model], candidate, self.inp, RESOLUTION, self.cfg, seed)
            domains = FUSION_DOMAINS[model]
            if domains == "arc_only":
                assert not len(pf.edges_b_idx), "BAC received fusion edges in an ARC-restricted fit"
            if domains == "bac_only":
                assert not len(pf.edges_a_idx), "ARC received fusion edges in a BAC-restricted fit"
            tree_arc, tree_bac, _, _ = trees_and_tables(self.inp, RESOLUTION)
            for tree, features in [(tree_arc, pf.retained_a), (tree_bac, pf.retained_b)]:
                assert not tree.coverage(features)[1], "Retained features missing from supplied tree"
            self.fit_cache[key] = pf
        return self.fit_cache[key]

    def predict(self, train, test, model, candidate):
        pf = self.fit_fusion(train, model, candidate)
        pred = pipe.predict_with_partition_fit(pf, test, self.inp, RESOLUTION)
        good = pred.index[pred.scorable].tolist()
        stats = pipe.count_active(pf, 1e-6)
        stats.update(retained_taxa=len(pf.retained_a) + len(pf.retained_b),
                     retained_arc=len(pf.retained_a), retained_bac=len(pf.retained_b),
                     n_edges_arc=len(pf.edges_a_idx), n_edges_bac=len(pf.edges_b_idx),
                     equality_residual=max(pf.fit_result.equality_residual_a, pf.fit_result.equality_residual_b))
        return good, pred.loc[good, "prediction"].to_numpy(), stats, pf


def export_fit(pf, inp, tags, resolution):
    coefficients, domain_counts = [], []
    maximum_residual, reconstruction_error = 0., 0.
    _, _, counts_arc, counts_bac = trees_and_tables(inp, resolution)
    transformed = pf.jp.transform(counts_arc.loc[pf.kept_train_ids],
                                  counts_bac.loc[pf.kept_train_ids], pf.pseudocount)
    for domain, features, beta, indices, log_key in [
            ("ARC", pf.retained_a, pf.fit_result.beta_a, pf.edges_a_idx, "log_a"),
            ("BAC", pf.retained_b, pf.fit_result.beta_b, pf.edges_b_idx, "log_b")]:
        group = extract_groups(domain, features, beta, indices, 1e-6)
        reconstruction_error = max(reconstruction_error, verify_group_reconstruction(transformed[log_key], beta, features, group))
        for j, feature in enumerate(features):
            gid = group.group_of_feature[feature]
            maximum_residual = max(maximum_residual, abs(group.residuals[feature]))
            coefficients.append({**tags, "domain": domain, "feature_id": feature, "beta": float(beta[j]),
                                 "group_id": gid, "group_mean_beta": group.group_mean_beta[gid],
                                 "residual": group.residuals[feature], "active": bool(abs(beta[j]) > 1e-6)})
        domain_counts.append({**tags, "domain": domain, "retained_features": len(features),
                              "fused": bool(len(indices)), "n_edges": len(indices),
                              "active_groups": sum(abs(v) > 1e-6 for v in group.group_mean_beta.values()),
                              "all_groups": len(group.group_members)})
    interactions = []
    if pf.interaction_transform is not None:
        _, frame = pipe.interaction_artifacts_from_pf(pf, 1e-6)
        interactions = [dict(row, **tags) for row in frame.to_dict("records")]
    return coefficients, domain_counts, interactions, maximum_residual, reconstruction_error


def run_task(task):
    target, repeat, fold = task
    cfg, inp = INPUTS[target]
    sites = inp.metadata.loc[inp.cohort_ids, "site_id"].to_dict()
    outer = make_site_grouped_folds(sites, 3, cfg["seed"] + repeat)
    full = sorted(outer.train_sites_for_outer_fold(fold))
    # Reproduce the full-fraction ordering used by the saved genus experiment.
    rng = np.random.default_rng(pipe.derive_seed(cfg["seed"], "learning_sites", repeat, fold))
    train = sample_ids_for_sites(sites, list(rng.permutation(full)))
    test = sample_ids_for_sites(sites, outer.sites_in_fold(fold))
    if target in REFERENCE_TARGETS:
        reference = REFERENCE[(REFERENCE.target == target) & (REFERENCE.repeat == repeat) & (REFERENCE.outer_fold == fold)]
        assert set(train) == set(reference[reference.role.eq("train")].sample_id)
        assert set(test) == set(reference[reference.role.eq("test")].sample_id)
    assert set(sites[s] for s in train).isdisjoint(sites[s] for s in test)
    inner_map = {s: sites[s] for s in train}
    inner = make_site_grouped_folds(inner_map, 2, pipe.derive_seed(cfg["seed"], "matched_inner", repeat, fold, 1.0))
    splits = [(sample_ids_for_sites(inner_map, inner.train_sites_for_outer_fold(k)),
               sample_ids_for_sites(inner_map, inner.sites_in_fold(k))) for k in range(2)]
    tags = dict(target=target, repeat=repeat, outer_fold=fold, fraction=1.0, resolution=RESOLUTION,
                n_train=len(train), n_train_sites=len(full),
                n_train_positive=int(inp.metadata.loc[train, "target"].sum()),
                n_test_positive=int(inp.metadata.loc[test, "target"].sum()))
    result = {key: [] for key in KEYS}
    for role, ids in [("train", train), ("test", test)]:
        result["splits"].extend({**tags, "role": role, "sample_id": s, "site_id": sites[s]} for s in ids)
    for k, (tr, va) in enumerate(splits):
        assert set(sites[s] for s in tr).isdisjoint(sites[s] for s in va)
        for role, ids in [(f"inner{k}_train", tr), (f"inner{k}_validation", va)]:
            result["splits"].extend({**tags, "role": role, "sample_id": s, "site_id": sites[s]} for s in ids)
    wb = RestrictedWorkbench(cfg, inp, None)
    started = time.time()
    try:
        for model in MODELS:
            mtags = {**tags, "model": model, "fusion_domains": FUSION_DOMAINS[model]}
            # Identical seed derivation to learning_curves.candidate_sets, so the
            # both-domain families reproduce the saved ASV run's candidate grids.
            base = "domain_adaptive_hybrid" if model == "domain_adaptive_interaction" else model
            seed = pipe.derive_seed(cfg["seed"], "candidates", base, fold if repeat == 0 else f"{repeat}:{fold}")
            candidates = generate_candidates(FAMILY_SPECS[base], cfg["search"], seed)
            if model == "domain_adaptive_interaction":
                candidates = generate_interaction_candidates(candidates, cfg["interaction"])
            scores = []
            for candidate in candidates:
                yy, pp, ss, complexities = [], [], [], []
                try:
                    for tr, va in splits:
                        good, p, stats, _ = wb.predict(tr, va, model, candidate)
                        if len(good) / len(va) < .8:
                            raise ValueError("Validation coverage below 0.8")
                        yy.extend(inp.metadata.loc[good, "target"]); pp.extend(p); ss.extend(inp.metadata.loc[good, "site_id"])
                        complexities.append((stats["n_active_products"], stats["n_required_taxa"], stats["n_active_groups"])
                                            if model == "domain_adaptive_interaction" else (stats["n_active_groups"], stats["n_active_taxa"]))
                    losses = pd.Series(per_sample_log_loss(np.array(yy), np.array(pp))).groupby(np.asarray(ss)).mean().to_numpy()
                    scores.append(dict(candidate=candidate, loss=float(losses.mean()), site_losses=losses,
                                       complexity=tuple(np.mean(complexities, axis=0)), y=np.array(yy), p=np.array(pp), sites=ss))
                except Exception as exc:
                    result["failures"].append({**mtags, "candidate": str(candidate), "error": f"{type(exc).__name__}: {exc}"})
            if not scores:
                # Rare-event flags can leave an inner training partition single-class.
                for rule in ["one_se", "min_loss"]:
                    result["rows"].append({**mtags, "selection": rule, "status": "unavailable",
                                           "reason": "No valid candidate across all inner folds"})
                continue
            best = min(scores, key=lambda x: x["loss"])
            se = float(np.std(best["site_losses"]) / np.sqrt(len(best["site_losses"])))
            chosen = min((s for s in scores if s["loss"] <= best["loss"] + se),
                         key=lambda s: (s["complexity"], s["loss"], s["candidate"].key()))
            for score in scores:
                result["search"].append({**mtags, "candidate": str(score["candidate"]), "loss": score["loss"],
                                         "complexity": str(score["complexity"]),
                                         "selected_one_se": score is chosen, "selected_min_loss": score is best})
            for rule, selected in [("one_se", chosen), ("min_loss", best)]:
                fit_tags = {**mtags, "selection": rule}
                threshold = select_threshold_via_inner_oof_f2(selected["y"], selected["p"], selected["sites"])
                good, p, stats, pf = wb.predict(train, test, model, selected["candidate"])
                trpred = pipe.predict_with_partition_fit(pf, train, inp, RESOLUTION)
                trgood = trpred.index[trpred.scorable]
                train_metric = legacy.metric(inp.metadata.loc[trgood, "target"], trpred.loc[trgood, "prediction"],
                                             inp.metadata.loc[trgood, "site_id"], trpred.loc[trgood, "prediction"] >= threshold)
                test_metric = legacy.metric(inp.metadata.loc[good, "target"], p, inp.metadata.loc[good, "site_id"], p >= threshold)
                coef, domains, interactions, residual, reconstruction = export_fit(pf, inp, fit_tags, RESOLUTION)
                for key, values in [("coefficients_and_groups", coef), ("domain_counts", domains), ("interactions", interactions)]:
                    result[key].extend(values)
                result["rows"].append({**fit_tags, **stats, "status": "ok", "reason": "",
                                       "candidate": str(selected["candidate"]), "threshold": threshold,
                                       "coverage": len(good) / len(test), "inner_loss": selected["loss"], "inner_se": se,
                                       "maximum_residual": residual, "group_reconstruction_error": reconstruction,
                                       **{"train_" + k: v for k, v in train_metric.items()},
                                       **{"test_" + k: v for k, v in test_metric.items()}})
                result["predictions"].extend({**fit_tags, "sample_id": s, "site_id": sites[s],
                                              "y": float(inp.metadata.loc[s, "target"]), "p": float(v), "threshold": threshold}
                                             for s, v in zip(good, p))
        # Filter audit at the shared unfused reference point; identical across models.
        reference_fit = wb.fit_fusion(train, "shared_hybrid", Candidate(0, .1, 0, None, None))
        tree_arc, tree_bac, counts_arc, counts_bac = trees_and_tables(inp, RESOLUTION)
        for domain, tree, counts, features in [("ARC", tree_arc, counts_arc, reference_fit.retained_a),
                                               ("BAC", tree_bac, counts_bac, reference_fit.retained_b)]:
            mass = counts.loc[train, features].sum(axis=1) / counts.loc[train].sum(axis=1)
            result["filter_audit"].append({**tags, "domain": domain, "input_features": counts.shape[1],
                                           "retained_features": len(features), "mean_retained_mass": float(mass.mean()),
                                           "min_retained_mass": float(mass.min()),
                                           "tree_uncovered_retained": len(tree.coverage(features)[1])})
    finally:
        wb.close()
    result.update(seconds=time.time() - started, task=list(task))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resolution", choices=RESOLUTIONS, default="asv")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS,
                        help="Subset of flags to fit; the run directory records which.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(str(OUTPUT).format(resolution=args.resolution))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parts").mkdir(exist_ok=True)
    cfg = make_config(TARGETS[0], args.resolution)
    from function.snapshots import source_files
    paths = source_files(ROOT) + [Path(__file__), ROOT / "function/investigation_20260921/learning_curves.py",
                                                      ROOT / "configs/fusionML/real_smoke_classification_phylo.yaml"]
    paths += [Path(cfg.resolved_input_path(k)) for k in ["arc_tree", "bac_tree", "arc_counts", "bac_counts", "arc_taxonomy", "bac_taxonomy"]]
    paths += [Path(cfg.resolved_path(cfg["chemistry"]["workbook"]))]
    paths += [ROOT / "models/fusionML/experiments/exp1/data/investigation_20260921/learning_qc" / n for n in ["predictions.csv", "splits.csv"]]
    if args.resolution != "asv":
        rank = cfg["inputs"]["rank_trees"][args.resolution]
        paths += [Path(cfg.resolved_path(rank[d])) for d in ["arc", "bac"]]
        # load_rank_resolution derives the crosswalk from the tree path.
        paths += [Path(cfg.resolved_path(rank[d]).rsplit("_tree.nwk", 1)[0] + "_mapping.csv") for d in ["arc", "bac"]]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
    protocol = {"resolution": args.resolution, "targets": list(args.targets), "outer_folds": 3, "inner_folds": 2, "repeats": 2,
                "grouping": "BSIb→BSI, DGYb→DGY", "config": cfg.raw, "hashes": hashes,
                "models": MODELS, "fusion_domains": FUSION_DOMAINS,
                "question": "where the fusion penalty acts: ARC edges only, BAC edges only, or both domains",
                "selection": "both one-SE and minimum inner site-logloss; thresholds from inner OOF F2",
                "comparison": "same cohort, labels, seeds, candidate generation and outer/inner split rules as the saved ASV and genus QC experiments",
                "tree_paths": {d: str(ROOT / f"datasets/raw/qiimeresult/{d}/rooted-tree.nwk") for d in ["ARC", "BAC"]}}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    mp = args.output / "run_manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text())["fingerprint"] == fingerprint, "Run inputs changed; use a new output directory"
    else:
        mp.write_text(json.dumps({**protocol, "fingerprint": fingerprint, "started_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    tasks = [(target, repeat, fold) for target in args.targets for repeat in range(2) for fold in range(3)]
    name = lambda task: "_".join(map(str, task)) + ".json"
    todo = [task for task in tasks if not (args.output / "parts" / name(task)).exists()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.resolution, args.targets)) as pool:
        futures = {pool.submit(run_task, task): task for task in todo}
        for future in as_completed(futures):
            result = future.result()
            (args.output / "parts" / name(result["task"])).write_text(json.dumps(result, default=float))
            print("DONE", result["task"], round(result["seconds"], 1), "seconds;", len(result["rows"]), "fit records", flush=True)
    parts = [json.loads((args.output / "parts" / name(task)).read_text()) for task in tasks]
    for key in KEYS:
        records = [row for part in parts for row in part[key]]
        pd.DataFrame(records if records else [], columns=None if records else ["error"]).to_csv(args.output / (key + ".csv"), index=False)
    manifest = json.loads(mp.read_text())
    manifest.update(completed_at=datetime.now(timezone.utc).isoformat(), completed_tasks=len(parts),
                    output_hashes={p.name: sha(p) for p in args.output.glob("*.csv")})
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed domain-restricted fusion at resolution={args.resolution}:", args.output, flush=True)


if __name__ == "__main__":
    main()
