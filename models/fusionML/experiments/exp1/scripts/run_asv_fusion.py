"""Run real ASV-tree fusion on the same cohort and splits as Experiment 1.

Uses the actual fusionML estimator. Preserves all old runs and exports new
results into fusionML/real_results_asv_exp1_20260922 for subsequent copying.
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
from models.fusionML.search import Candidate, InteractionCandidate, FAMILY_SPECS
from function.evaluate import per_sample_log_loss, select_threshold_via_inner_oof_f2
from function.investigation_20260921 import learning_curves as legacy

TARGETS = ["warning_ammonia_toxicity", "warning_acid_base_balance"]
MODEL_MAP = {"genus_ridge": "asv_ridge", "genus_elastic_net": "asv_elastic_net",
             "phylogeny_only": "phylogeny_only", "abundance_only": "abundance_only",
             "shared_hybrid": "shared_hybrid", "domain_adaptive_hybrid": "domain_adaptive_hybrid",
             "domain_adaptive_interaction": "domain_adaptive_interaction", "adaptive_expanded": "adaptive_expanded"}
OUTPUT = ROOT / "results/classification result/fusionML/real_results_asv_exp1_20260922"
INPUTS = {}
REFERENCE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(target):
    cfg = Config.from_yaml(str(ROOT / "configs/fusionML/real_smoke_classification_phylo.yaml"))
    cfg.raw["resolution"] = "asv"
    cfg.raw["columns"]["target"] = target
    cfg.raw["inputs"]["rank_trees"] = {}
    for domain in ["arc", "bac"]:
        cfg.raw["inputs"][domain + "_tree"] = str(ROOT / f"datasets/raw/qiimeresult/{domain.upper()}/rooted-tree.nwk")
    cfg.raw["cv"]["families"] = list(MODEL_MAP.values())
    return cfg


def init_worker():
    global INPUTS, REFERENCE
    INPUTS = {}
    reference = pd.read_csv(ROOT / "models/fusionML/experiments/exp1/data/investigation_20260921/learning_qc/predictions.csv")
    REFERENCE = pd.read_csv(ROOT / "models/fusionML/experiments/exp1/data/investigation_20260921/learning_qc/splits.csv")
    REFERENCE = REFERENCE[REFERENCE.fraction.eq(1)]
    for target in TARGETS:
        cfg = make_config(target)
        inp = pipe.load_all_inputs(cfg)
        assert inp.rank_used is None  # No genus aggregation enters these fits.
        inp.metadata["site_id"] = inp.metadata.site_id.replace({"BSIb": "BSI", "DGYb": "DGY"})
        old = reference[(reference.target == target) & reference.fraction.eq(1)].drop_duplicates("sample_id").set_index("sample_id")
        assert set(inp.cohort_ids) == set(old.index)
        assert np.array_equal(inp.metadata.loc[old.index, "target"].to_numpy(), old.y.to_numpy())
        assert np.array_equal(inp.metadata.loc[old.index, "site_id"].to_numpy(), old.site_id.to_numpy())
        INPUTS[target] = (cfg, inp)


class ASVWorkbench(legacy.Workbench):
    def fit_fusion(self, ids, candidate):
        key = (tuple(ids), isinstance(candidate, InteractionCandidate), candidate.key())
        if key not in self.fit_cache:
            name = "domain_adaptive_interaction" if isinstance(candidate, InteractionCandidate) else "domain_adaptive_hybrid"
            seed = pipe.derive_seed(self.cfg["seed"], "matched_graph", *sorted(ids))
            pf = pipe.fit_partition(ids, FAMILY_SPECS[name], candidate, self.inp, "asv", self.cfg, seed)
            for tree, features in [(self.inp.tree_arc, pf.retained_a), (self.inp.tree_bac, pf.retained_b)]:
                assert not tree.coverage(features)[1], "Retained ASVs missing from supplied tree"
            self.fit_cache[key] = pf
        return self.fit_cache[key]

    def predict(self, train, test, candidate):
        pf = self.fit_fusion(train, candidate)
        pred = pipe.predict_with_partition_fit(pf, test, self.inp, "asv")
        good = pred.index[pred.scorable].tolist()
        stats = pipe.count_active(pf, 1e-6)
        stats.update(retained_taxa=len(pf.retained_a) + len(pf.retained_b),
                     retained_arc=len(pf.retained_a), retained_bac=len(pf.retained_b),
                     n_edges=len(pf.edges_a_idx) + len(pf.edges_b_idx),
                     equality_residual=max(pf.fit_result.equality_residual_a, pf.fit_result.equality_residual_b))
        return good, pred.loc[good, "prediction"].to_numpy(), stats, pf


def export_fit(pf, inp, tags):
    coefficients, edges, domain_counts = [], [], []
    maximum_residual, reconstruction_error = 0., 0.
    transformed = pf.jp.transform(inp.counts_arc_asv.loc[pf.kept_train_ids],
        inp.counts_bac_asv.loc[pf.kept_train_ids], pf.pseudocount)
    for domain, features, beta, graph, indices, log_key in [
        ("ARC", pf.retained_a, pf.fit_result.beta_a, pf.mixed_edges_a, pf.edges_a_idx, "log_a"),
        ("BAC", pf.retained_b, pf.fit_result.beta_b, pf.mixed_edges_b, pf.edges_b_idx, "log_b")]:
        group = extract_groups(domain, features, beta, indices, 1e-6)
        error = verify_group_reconstruction(transformed[log_key], beta, features, group)
        reconstruction_error = max(reconstruction_error, error)
        for j, feature in enumerate(features):
            gid = group.group_of_feature[feature]
            residual = group.residuals[feature]
            maximum_residual = max(maximum_residual, abs(residual))
            coefficients.append({**tags, "domain": domain, "feature_id": feature, "beta": beta[j],
                "group_id": gid, "group_mean_beta": group.group_mean_beta[gid], "residual": residual,
                "active": abs(beta[j]) > 1e-6})
        if len(graph):
            edges.extend([{**row, **tags, "domain": domain} for row in graph.to_dict("records")])
        domain_counts.append({**tags, "domain": domain, "retained_asvs": len(features),
            "active_groups": sum(abs(v) > 1e-6 for v in group.group_mean_beta.values()),
            "all_groups": len(group.group_members)})
    balances, interactions = pipe.interaction_artifacts_from_pf(pf, 1e-6) if pf.interaction_transform is not None else (pd.DataFrame(), pd.DataFrame())
    return coefficients, edges, domain_counts, [dict(row, **tags) for row in balances.to_dict("records")], [dict(row, **tags) for row in interactions.to_dict("records")], maximum_residual, reconstruction_error


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
    reference = REFERENCE[(REFERENCE.target == target) & (REFERENCE.repeat == repeat) & (REFERENCE.outer_fold == fold)]
    assert set(train) == set(reference[reference.role.eq("train")].sample_id)
    assert set(test) == set(reference[reference.role.eq("test")].sample_id)
    inner_map = {s: sites[s] for s in train}
    inner = make_site_grouped_folds(inner_map, 2, pipe.derive_seed(cfg["seed"], "matched_inner", repeat, fold, 1.0))
    splits = [(sample_ids_for_sites(inner_map, inner.train_sites_for_outer_fold(k)),
               sample_ids_for_sites(inner_map, inner.sites_in_fold(k))) for k in range(2)]
    tags = dict(target=target, repeat=repeat, outer_fold=fold, fraction=1.0, resolution="asv",
        n_train=len(train), n_train_sites=len(full), n_train_positive=int(inp.metadata.loc[train, "target"].sum()))
    result = {k: [] for k in ["rows", "predictions", "search", "failures", "splits", "coefficients_and_groups", "graph_edges", "domain_counts", "balances", "interactions", "regularization_paths", "filter_audit"]}
    for role, ids in [("train", train), ("test", test)]:
        result["splits"].extend([{**tags, "role": role, "sample_id": s, "site_id": sites[s]} for s in ids])
    for k, (tr, va) in enumerate(splits):
        assert set(sites[s] for s in tr).isdisjoint(sites[s] for s in va)
        for role, ids in [(f"inner{k}_train", tr), (f"inner{k}_validation", va)]:
            result["splits"].extend([{**tags, "role": role, "sample_id": s, "site_id": sites[s]} for s in ids])
    wb = ASVWorkbench(cfg, inp, None)
    started = time.time()
    try:
        sets = legacy.candidate_sets(cfg, fold, repeat, True)
        for old_name, name in MODEL_MAP.items():
            scores = []
            for candidate in sets[old_name]:
                yy, pp, ss, complexities = [], [], [], []
                try:
                    for tr, va in splits:
                        good, p, stats, _ = wb.predict(tr, va, candidate)
                        if len(good) / len(va) < .8:
                            raise ValueError("Validation coverage below 0.8")
                        yy.extend(inp.metadata.loc[good, "target"]); pp.extend(p); ss.extend(inp.metadata.loc[good, "site_id"])
                        complexities.append((stats["n_active_products"], stats["n_required_taxa"], stats["n_active_groups"])
                            if name == "domain_adaptive_interaction" else (stats["n_active_groups"], stats["n_active_taxa"]))
                    losses = pd.Series(per_sample_log_loss(np.array(yy), np.array(pp))).groupby(np.asarray(ss)).mean().to_numpy()
                    scores.append(dict(candidate=candidate, loss=losses.mean(), site_losses=losses,
                        complexity=tuple(np.mean(complexities, axis=0)), y=np.array(yy), p=np.array(pp), sites=ss))
                except Exception as exc:
                    result["failures"].append({**tags, "model": name, "candidate": str(candidate), "error": f"{type(exc).__name__}: {exc}"})
            if not scores:
                raise RuntimeError(f"No valid candidates for {task} {name}")
            best = min(scores, key=lambda x: x["loss"])
            se = np.std(best["site_losses"]) / np.sqrt(len(best["site_losses"]))
            chosen = min((s for s in scores if s["loss"] <= best["loss"] + se), key=lambda s: (s["complexity"], s["loss"], s["candidate"].key()))
            for score in scores:
                result["search"].append({**tags, "model": name, "candidate": str(score["candidate"]),
                    "loss": score["loss"], "complexity": str(score["complexity"]),
                    "selected_one_se": score is chosen, "selected_min_loss": score is best})
            for rule, selected in [("one_se", chosen), ("min_loss", best)]:
                fit_tags = {**tags, "model": name, "selection": rule}
                threshold = select_threshold_via_inner_oof_f2(selected["y"], selected["p"], selected["sites"])
                good, p, stats, pf = wb.predict(train, test, selected["candidate"])
                trpred = pipe.predict_with_partition_fit(pf, train, inp, "asv")
                trgood = trpred.index[trpred.scorable]
                train_metric = legacy.metric(inp.metadata.loc[trgood, "target"], trpred.loc[trgood, "prediction"], inp.metadata.loc[trgood, "site_id"], trpred.loc[trgood, "prediction"] >= threshold)
                test_metric = legacy.metric(inp.metadata.loc[good, "target"], p, inp.metadata.loc[good, "site_id"], p >= threshold)
                coef, edges, domains, balances, interactions, residual, reconstruction = export_fit(pf, inp, fit_tags)
                for key, values in [("coefficients_and_groups", coef), ("graph_edges", edges), ("domain_counts", domains), ("balances", balances), ("interactions", interactions)]:
                    result[key].extend(values)
                result["rows"].append({**fit_tags, **stats, "status": "ok", "candidate": str(selected["candidate"]),
                    "threshold": threshold, "coverage": len(good) / len(test), "inner_loss": selected["loss"],
                    "inner_se": se, "maximum_residual": residual, "group_reconstruction_error": reconstruction,
                    **{"train_" + k: v for k, v in train_metric.items()}, **{"test_" + k: v for k, v in test_metric.items()}})
                result["predictions"].extend([{**fit_tags, "sample_id": s, "site_id": sites[s], "y": float(inp.metadata.loc[s, "target"]), "p": float(v), "threshold": threshold} for s, v in zip(good, p)])
        ridge = wb.fit_fusion(train, Candidate(0, .1, 0, None, None))
        for domain, counts, features in [("ARC", inp.counts_arc_asv, ridge.retained_a), ("BAC", inp.counts_bac_asv, ridge.retained_b)]:
            mass = counts.loc[train, features].sum(axis=1) / counts.loc[train].sum(axis=1)
            result["filter_audit"].append({**tags, "domain": domain, "input_asvs": counts.shape[1],
                "retained_asvs": len(features), "mean_retained_mass": mass.mean(), "min_retained_mass": mass.min(),
                "tree_uncovered_retained": len((inp.tree_arc if domain == "ARC" else inp.tree_bac).coverage(features)[1])})
        # Fixed-penalty curves on the same QC splits, not test-selected tuning.
        for graph, alpha in [("phylogeny", 1.), ("abundance", 0.), ("shared", .5)]:
            for strength in [0., .01, .1, .3, 1., 3., 10., 30., 100.]:
                candidate = Candidate(0, .1, strength, alpha if strength else None, alpha if strength else None)
                good, p, stats, _ = wb.predict(train, test, candidate)
                metric = legacy.metric(inp.metadata.loc[good, "target"], p, inp.metadata.loc[good, "site_id"])
                result["regularization_paths"].append({**tags, **stats, "graph": graph, "lambda_f": strength,
                    "test_logloss": metric["logloss"], "test_auc": metric["auc"], "coverage": len(good) / len(test)})
    finally:
        wb.close()
    result.update(seconds=time.time() - started, task=list(task))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parts").mkdir(exist_ok=True)
    cfg = make_config(TARGETS[0])
    from function.snapshots import source_files
    paths = source_files(ROOT) + [Path(__file__), ROOT / "function/investigation_20260921/learning_curves.py",
        ROOT / "configs/fusionML/real_smoke_classification_phylo.yaml"]
    paths += [Path(cfg.resolved_input_path(k)) for k in ["arc_tree", "bac_tree", "arc_counts", "bac_counts", "arc_taxonomy", "bac_taxonomy"]]
    paths += [Path(cfg.resolved_path(cfg["chemistry"]["workbook"]))]
    paths += [ROOT / "models/fusionML/experiments/exp1/data/investigation_20260921/learning_qc" / n for n in ["predictions.csv", "splits.csv"]]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
    protocol = {"resolution": "asv", "targets": TARGETS, "outer_folds": 3, "inner_folds": 2, "repeats": 2,
        "grouping": "BSIb→BSI, DGYb→DGY", "config": cfg.raw, "hashes": hashes,
        "models": list(MODEL_MAP.values()), "candidate_seed_aliases": MODEL_MAP,
        "selection": "both one-SE and minimum inner site-logloss; thresholds from inner OOF F2",
        "comparison": "same candidate generation, graph seeds, cohort, labels and outer/inner split rules as saved genus QC experiment",
        "path": {"lambda_s": 0, "lambda_r": .1, "lambda_f": [0, .01, .1, .3, 1, 3, 10, 30, 100]},
        "tree_paths": {d: str(ROOT / f"datasets/raw/qiimeresult/{d}/rooted-tree.nwk") for d in ["ARC", "BAC"]}}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    mp = args.output / "run_manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text())["fingerprint"] == fingerprint, "Run inputs changed; use a new output directory"
    else:
        mp.write_text(json.dumps({**protocol, "fingerprint": fingerprint, "started_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    tasks = [(target, repeat, fold) for target in TARGETS for repeat in range(2) for fold in range(3)]
    name = lambda task: "_".join(map(str, task)) + ".json"
    todo = [task for task in tasks if not (args.output / "parts" / name(task)).exists()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = {pool.submit(run_task, task): task for task in todo}
        for future in as_completed(futures):
            result = future.result()
            (args.output / "parts" / name(result["task"])).write_text(json.dumps(result, default=float))
            print("DONE", result["task"], round(result["seconds"], 1), "seconds;", len(result["rows"]), "selected fits", flush=True)
    parts = [json.loads((args.output / "parts" / name(task)).read_text()) for task in tasks]
    for key in ["rows", "predictions", "search", "failures", "splits", "coefficients_and_groups", "graph_edges", "domain_counts", "balances", "interactions", "regularization_paths", "filter_audit"]:
        records = [row for part in parts for row in part[key]]
        pd.DataFrame(records if records else [], columns=None if records else ["error"]).to_csv(args.output / (key + ".csv"), index=False)
    manifest = json.loads(mp.read_text())
    manifest.update(completed_at=datetime.now(timezone.utc).isoformat(), completed_tasks=len(parts),
        output_hashes={p.name: sha(p) for p in args.output.glob("*.csv")})
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed ASV fusion:", args.output, flush=True)


if __name__ == "__main__":
    main()
