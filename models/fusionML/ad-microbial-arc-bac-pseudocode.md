# Pseudocode: microbial fusion with ARC–BAC predictive interactions

Companion to `ad-microbial-model-design-arc-bac-interactions.md`, revised 21 September 2026. This supersedes the earlier additive-only pseudocode for implementing the interaction extension. It is an implementation specification, not executable Python or a report of fitted results. Named helper functions require real implementations. The design document defines the defaults and remains authoritative.

The current real-data task is binary classification, one chemistry-derived warning flag per run. Regression is supported only when an approved continuous target is supplied. Reference repository paths below come from the design document and must be inspected in the actual code workspace; no repository execution is claimed here.

## Step-to-code map

| Design step | Function(s) | Role |
|---|---|---|
| 9 | `load_inputs`, `make_cohort` | Read counts, trees, live chemistry, labels and physical-site grouping |
| 10 | `fit_preprocessor`, `fit_interaction_transform` | Training-only taxon filtering, log compositions, balances and products |
| 11 | `pair_statistics`, `diagnose` | Within-domain proportionality and descriptive cross-domain diagnostics |
| 12 | `build_graphs`, `mix_graphs` | Fixed graph components for each training partition |
| 13 | `fit_coefficients` | Joint constrained optimization of beta, gamma and intercept |
| 14 | `inner_search`, `nested_evaluate` | Hyperparameter selection, threshold selection and held-out-site evaluation |
| 15 | `extract_representation` | Additive groups plus interaction dependencies |
| 16 | `resolve_config`, `cache_key` | Task-specific settings, search budgets and reproducibility |
| 17 | `predict`, `save_outputs`, `main` | Frozen prediction, artifacts and CLI |
| 18 | `verify` | Scientific implementation checks |

## Model and array definitions

For each training partition with n supported samples:

| Object | Shape | Meaning |
|---|---|---|
| `L_A`, `L_B` | n × p_A, n × p_B | Separate-domain natural log compositions |
| `A_A`, `A_B` | p_A × K_A, p_B × K_B | Balance loading matrices; each column sums to zero |
| `T_A`, `T_B` | n × K_A, n × K_B | Training-centered/scaled balance values |
| `X_I` | n × M | Centered/scaled ARC–BAC products; M ≤ 4 |
| `beta_A`, `beta_B` | p_A, p_B | Taxon-level additive coefficients |
| `gamma` | M | Interaction coefficients |
| `w` | n | Equal-site sample weights, summing to 1 |

```python
eta = b0 + L_A @ beta_A + L_B @ beta_B + X_I @ gamma
constraints = [sum(beta_A) == 0, sum(beta_B) == 0]

if task == 'classification':
    prediction = sigmoid(eta)                  # probability of warning flag
else:
    prediction = y_mean + y_sd * eta           # original continuous-target units
```

Only the interaction-on variant includes `X_I @ gamma`. The additive variant fixes it to zero. `lambda_I1 = 0` does not switch interactions off. The main-effect taxon design already spans balance main effects: do not append `T_A` or `T_B` as redundant linear predictors. The beta constraints remain separate; gamma has no zero-sum constraint.

## A. Entry point and configuration — steps 16–17

```python
function main(args):
    cfg = resolve_config(args.config, args.mode)                   # [16]
    manifest = begin_run(cfg, versions, input_hashes, seed)

    if args.mode == 'predict':
        model = load_model(args.model_path)
        data = load_inputs(cfg, require_target=False)              # [9]
        return save_predictions(predict(model, data))              # [17]; NO FIT

    data, audit, blockers = load_inputs(cfg, require_target=False)  # [9]
    save_validation_report(audit, blockers)
    if args.mode == 'validate':
        return report_readiness(blockers)
    if args.mode == 'diagnose':
        return diagnose(data, cfg, label='descriptive')            # [11]

    require_resolved_training_inputs(data, cfg)
    # Classification: one configured flag per independent run/cohort.
    cohort = make_cohort(data, cfg.target)
    report_target_counts_and_distinct_positive_negative_sites(cohort)
    print_estimated_fits_graph_bootstraps_and_output_directory(cfg)
    evaluation = nested_evaluate(cohort, cfg)                     # [14]

    # Separate final deployment fitting from outer-CV evaluation.
    selection = inner_search(cohort, family='interaction_capable', cfg=cfg)
    state = prepare_partition(cohort, selection.family, cfg)
    final_model = fit_prepared(state, selection.hp, cfg)
    final_model.threshold = selection.threshold                    # classification
    save_outputs(evaluation, final_model, manifest)                # [17]

function resolve_config(path, mode):
    cfg = parse_and_validate_yaml(path)
    resolve_relative_paths_against_project_root(cfg)
    adapt_existing_repo_config_to_canonical_schema_explicitly(cfg)
    # Preserve the working live chemistry loader and sample-ID mapping.
    # Do not overwrite them with null fields from the general template.
    if cfg.task == 'classification':
        cfg.standardize_target = False
        cfg.selection_metric = 'site_balanced_log_loss'
        cfg.solver_fallback = 'SCS'
    else:
        cfg.standardize_target = True
        cfg.selection_metric = 'site_balanced_mse'
        cfg.solver_fallback = 'OSQP'
    reject_conflicting_interaction_enable_flags(cfg)
    save_effective_settings(cfg)
    # Unresolved inputs block train, but validate must still report them.
    # Smoke: explicit reduced budget, separate directory; never overwrite research.
    return cfg
```

## B. Read the real inputs and build targets — step 9

```python
function load_inputs(cfg, require_target):
    for d in ['ARC', 'BAC']:
        taxonomy[d] = inspect_and_read_silva_tsv(cfg.taxonomy[d])
        tree[d] = read_and_validate_newick(cfg.tree[d])
        counts[d] = read_qiime_tsv_preserving_actual_header(cfg.counts[d])
        # '#OTU ID' may be the header; do not discard every '#' line.
        counts[d] = infer_orientation_from_feature_ID_overlap_or_fail(
            counts[d], taxonomy[d], tree[d])
        validate_counts_nonnegative_finite_and_unique(counts[d])
        add_reversible_domain_prefix_to_feature_IDs(d)
        # metadata.tsv is taxonomy, not sample metadata.

    chemistry = read_workbook_using_explicit_yaml_cell_maps(cfg)
    # Chem_Si.Avg and Chem_Se.Avg: season row 2, measurement row 3, data row 4+.
    # Preserve substrate/effluent namespaces and numeric zeros.
    # Notes, uncached external-link formulas and missing tokens are audited as missing.
    # Preserve inferred nitrogen-season overrides as explicit assumptions.
    # Substrate values are already Q-weighted: do not weight again.
    # Match exact S/F/W/Sp tokens to 1/2/3/4.

    sample_map = use_documented_live_sample_mapping(chemistry)
    # Classification path: '{season_code}-{No.}' in the validated repo convention.
    # Keep No. strings intact; audit/exclude blank-No. MGYe2 as documented.
    # Verify actual count-table overlap; never match by row position.
    facility_map = load_and_verify_physical_facility_grouping()
    # Source record No. is not automatically an independent facility.
    manifest = align_both_assays_by_biological_sample_ID(counts, sample_map)
    data = left_join_chemistry_and_facility_map(manifest)
    reject_duplicate_or_cartesian_joins()
    if configured_target_exists:
        data.y = derive_configured_target_from_live_sources(data, cfg)
    return data, complete_source_audit, unresolved_fields

function make_cohort(data, target):
    keep = both_assays_present AND known_physical_site AND measured_target
    audit_exclusions_with_sample_site_and_reason(data, keep)
    return data[keep]

function derive_configured_target_from_live_sources(data, cfg):
    if cfg.task == 'regression':
        require_approved_column_or_formula_and_units()
        return configured_continuous_target
    return existing_verified_live_flag_loader(data, flag=cfg.target)
    # Inspect fusionML/chemistry_targets.py and code/metadata_targetwriter.py.
    # Do not use stale data/final/metadata.csv as the label source.
    # Preserve and test documented missing-trigger and zero-denominator semantics.
    # An unresolved missing label stays missing, never automatically False.
```

The five target definitions carried forward from the design are:

| Flag | Rule |
|---|---|
| Acid–base balance | pH < 6.5 or pH > 8.0 |
| Buffer capacity | TVFAs/ALK ≥ 0.30; ALK=0 and TVFAs>0 meets the trigger |
| Acid accumulation | (HPro/HAc > 1.4 and HPro > 0.1) or HPro ≥ 0.8 |
| Ammonia toxicity | NH4-N > 2.5, using the project's documented TAN convention |
| Biogas quality | CH4 < 50.0 percent |

These formulas alone do not specify every missing-value or zero-denominator case; retain the validated loader's audited policy and flag any unresolved case. Flag-defining chemistry and proxies must not become predictors in the microbiome-only comparison.

## C. Fit preprocessing on training rows — step 10

```python
function site_weights(site):
    return [1 / (number_of_sites * observations_at_site[s]) for s in site]

function fit_preprocessor(train, resolution, cfg):
    C, trees = counts_and_trees_at_matching_resolution(train, resolution)
    # Higher rank: sum RAW counts by full lineage, then use matching supplied tree.
    for d in ['ARC', 'BAC']:
        detected = (C[d] >= cfg.detection_count)                    # default 5
        ids[d] = taxa_detected_in_at_least(
            samples=max(5, ceil(0.10 * len(train))),
            sites=max(3, ceil(0.10 * distinct_sites(train))),
            detection_matrix=detected)
        require(len(ids[d]) >= 2)
    freeze(ids_and_order=ids)
    supported = positive_raw_retained_totals_in_BOTH_domains(C, ids)
    audit_training_exclusions(train[~supported])
    train = train[supported]
    require_enough_sites(train)
    w = site_weights(train.site)                                   # recompute
    state = {ids, delta: cfg.pseudocount, trees, input_contract}
    L = transform_counts(train, state).L
    if cfg.task == 'classification':
        require_both_classes_or_fail_primary_fit(train.y)
        target_state = identity_transform()
    else:
        mu = sum(w * train.y)
        sd = sqrt(sum(w * (train.y - mu)**2))
        target_state = {mu, sd, constant_target: sd == 0}
    return state, target_state, train, L, w

function transform_counts(rows, state):
    for d in ['ARC', 'BAC']:
        C = align_to_frozen_feature_order(rows[d], state.ids[d])
        # Fill omitted columns with 0 only if the source contract establishes that.
        # Otherwise fail schema validation. Ignore/audit new test-only feature mass.
        supported[d] = row_sum(C) > 0                              # before pseudocount
        Q[d] = close_rows(C[supported[d]] + state.delta)            # separately
        L[d] = log(Q[d])
        audit_pseudocount_mass_fraction()
    return aligned_L_with_joint_support_mask_and_reasons()
```

For a constant regression target, bypass coefficient fitting and record a constant model; do not divide by zero target SD. Do not standardize individual microbial log columns. Only balance and product features are standardized below. Unsupported held-out rows get missing predictions and explicit reasons; complete validation coverage is the default selection gate. For optional operational covariates, fit imputation/encoding/scaling within the same training partition, use a separate documented ridge penalty, and include a covariate-only comparator.

## D. Within-domain graphs — steps 11–12

```python
function pair_statistics(L, counts, original_sites, weight_groups, cfg):
    w = site_weights(weight_groups)
    for unordered_pair(j, k) in all_pairs_streamed_if_large:
        r = L[:,j] - L[:,k]
        v_all = weighted_population_variance(r, w)
        joint = (counts[:,j] >= 5) AND (counts[:,k] >= 5)
        eligible = sum(joint) >= 10 AND distinct(original_sites[joint]) >= 5
        if eligible:
            w_joint = site_weights(weight_groups[joint])
            v_joint = weighted_population_variance(r[joint], w_joint)
            score = max(v_all, v_joint)
        else:
            v_joint, score = NA, NA                                # no abundance edge
        emit(j, k, eligibility, v_all, v_joint, score, codetection)

function build_graphs(train, L, trees, cfg):
    for d in ['ARC', 'BAC']:
        distances = valid_patristic_distances(trees[d], retained_features[d])
        edges_P = union_of_directed_k_nearest(distances, k=5, ties='canonical_id')
        tau_P = median_positive(distances[edges_P], fallback=1)
        P[d] = normalize_edge_sum(exp(-distances[edges_P] / tau_P))
        # Each undirected edge once, no self-edges. Unmapped taxa are isolated.
        # Missing whole tree is UNAVAILABLE, different from a valid empty graph.

        full = pair_statistics(L[d], counts[d], train.site, train.site, cfg)
        candidates = mutual_k_nearest_eligible(full.score, k=5)
        selected_count, eligible_count = zeros(candidates)
        for b in range(500):
            boot = draw_whole_sites_with_replacement(train)
            # Every repeated draw is its own copy with every original visit.
            stats = pair_statistics(L[d][boot.rows], counts[d][boot.rows],
                                    boot.original_sites, boot.copy_ids, cfg)
            edges_b = mutual_k_nearest_eligible(stats.score, k=5)
            update_counts(candidates, stats.eligible, edges_b)
        support = selected_count / 500                            # ineligible draws count 0
        kept = candidates[support >= 0.80]
        tau_E = median_positive(full.score[kept], fallback=1)
        E[d] = normalize_edge_sum(support[kept] * exp(-full.score[kept] / tau_E))
        # Empty graph remains empty. Export eligibility frequency separately.
    return P, E

function mix_graphs(P, E, alpha):
    return positive_edges(alpha * P + (1-alpha) * E)               # NO renormalization
```

Use configured defaults rather than hard-coded values in executable code. Exact bootstrap neighbor selection uses all eligible pairs and a fixed training feature universe; an explicitly enabled ASV candidate-pool approximation must be labeled. No graph contains ARC–BAC fusion edges. Descriptive diagnostics include between/within-site log-ratio variance decomposition and the 0.5 versus 1.0 pseudocount sensitivity. Cross-domain balance correlations may be plotted but never gate the product predictors.

## E. Construct balances and ARC–BAC products — step 10 extension

This transformer is independent of y, beta and gamma. Fit it once per exact training partition and count-preprocessing setting; reuse it across penalty candidates.

```python
function fit_interaction_transform(train, L, trees, w, cfg):
    for d in ['ARC', 'BAC']:
        mapped = retained_taxa_with_unambiguous_tree_mapping(d)
        require_tree_or_mark_interaction_unavailable(d)
        candidates = []
        for edge in trees[d]:
            plus = retained_mapped_descendants(edge)
            minus = mapped - plus
            if empty(plus) OR empty(minus): continue
            plus, minus = canonical_orientation(plus, minus)
            # Positive side is lexicographically smaller sorted ID tuple.
            if complementary_split_already_seen(plus, minus): continue
            joint = any_raw_taxon_detected(plus, count=5) AND any_raw_taxon_detected(minus, count=5)
            if sum(joint) < 10 OR distinct(train.site[joint]) < 5: continue

            r, s = len(plus), len(minus)
            k = sqrt(r*s/(r+s))
            a = zeros(number_of_retained_taxa[d])
            a[plus], a[minus] = k/r, -k/s
            assert_close(sum(a), 0)
            z = L[d] @ a
            variance = weighted_population_variance(z, w)
            if variance <= 1e-12: continue
            candidates.append({canonical_ID, a, z, variance})

        chosen = []
        for c in sorted(candidates, keys=[descending_variance, canonical_ID]):
            if not increases_loading_matrix_rank(chosen, c, absolute_tol=1e-10): continue
            if any(abs(weighted_correlation(c.z, old.z, w)) > 0.95 for old in chosen): continue
            chosen.append(c)
            if len(chosen) == cfg.max_balances[d]: break            # default 2
        if len(chosen) == 0: return INTERACTION_UNAVAILABLE
        A[d] = column_stack(c.a for c in chosen)
        Z[d] = L[d] @ A[d]
        balance_mean[d], balance_sd[d] = weighted_column_moments(Z[d], w)
        T[d] = (Z[d] - balance_mean[d]) / balance_sd[d]

    kept_pairs, product_mean, product_sd, X = [], [], [], []
    for u, v in canonical_cartesian_pairs(T['ARC'], T['BAC']):      # at most 4
        H = T['ARC'][:,u] * T['BAC'][:,v]
        mean_H, sd_H = weighted_moments(H, w)
        if sd_H**2 <= 1e-12: continue
        x = (H - mean_H) / sd_H
        if not increases_sample_design_rank([ones, *X], x, relative_tol=1e-10): continue
        kept_pairs.append((u,v))
        product_mean.append(mean_H)
        product_sd.append(sd_H)
        X.append(x)
    if len(kept_pairs) == 0: return INTERACTION_UNAVAILABLE
    return frozen_transform(A, balance_mean, balance_sd,
                            kept_pairs, product_mean, product_sd, feature_IDs)

function transform_interactions(L, state):
    for d in ['ARC', 'BAC']:
        T[d] = (L[d] @ state.A[d] - state.balance_mean[d]) / state.balance_sd[d]
    H = column_stack(T['ARC'][:,u] * T['BAC'][:,v] for u,v in state.pairs)
    return (H - state.product_mean) / state.product_sd
```

Do not residualize products against the full taxon design: with p>n, this could remove all training products. The rank check above concerns only the intercept and other products. Polytomies require no arbitrary binary tree resolution. Balances can involve many taxa despite the small product count. For configured external balances, load audited memberships instead of generating tree splits, require every member retained, and apply the same detection/scaling checks; do not silently trim groups.

## F. Joint constrained estimator — step 13

```python
function fit_coefficients(prepared, hp, cfg):
    if cfg.task == 'regression' AND prepared.target.constant_target:
        return explicit_constant_model(prepared.target.mu)         # no interaction claim
    beta_A, beta_B = variables(p_A), variables(p_B)
    b0 = variable()
    eta = b0 + L_A @ beta_A + L_B @ beta_B
    if hp.interactions_on:
        require(prepared.interaction_state is not UNAVAILABLE)
        X_I = transform_interactions(prepared.L, prepared.interaction_state)
        gamma = variable(number_of_columns(X_I))
        eta += X_I @ gamma
        P_I = hp.lambda_I1 * norm1(gamma) + hp.lambda_I2 * sum_squares(gamma)
    else:
        gamma, P_I = empty_vector(), 0

    P_beta = hp.lambda_S * (norm1(beta_A) + norm1(beta_B))
    P_beta += hp.lambda_R * (sum_squares(beta_A) + sum_squares(beta_B))
    for d in ['ARC', 'BAC']:
        W = mix_graphs(prepared.P[d], prepared.E[d], hp.alpha[d]) if hp.lambda_F > 0 else empty_graph()
        D = sparse_edge_incidence_matrix(W.edges)                  # one +1/-1 row per edge
        P_beta += hp.lambda_F * sum(W.weights * abs(D @ beta[d]))

    if cfg.task == 'classification':
        require_both_classes(prepared.y)
        loss = sum(w * (softplus(eta) - prepared.y * eta))          # stable logistic atom
        solver_order = ['CLARABEL', 'SCS']
    else:
        y_star = prepared.target.transform(prepared.y)
        loss = sum(w * square(y_star - eta))
        solver_order = ['CLARABEL', 'OSQP']

    problem = minimize(loss + P_beta + P_I,
                       constraints=[sum(beta_A)==0, sum(beta_B)==0])
    fit = solve_with_documented_retry_policy(problem, solver_order)
    verify_status_finite_parameters_objective_and_constraints(fit, abs_tol=1e-6)
    return fit_or_explicit_failure(fit)
```

No extra n, p or 1/2 factors are introduced. W, alpha, lambdas and the balance transform are fixed during a solve. Beta, gamma and b0 are learned jointly. The model is convex for these fixed features. OSQP is not the fallback for the logistic exponential-cone formulation. The default does not impose strong heredity between interactions and main-effect coefficients.

## G. Nested model selection and thresholds — step 14

```python
function prepare_partition(train, family, cfg):
    prep, target, train_fit, L, w = fit_preprocessor(train, family.resolution, cfg)
    graphs = build_required_graph_components(train_fit, L, prep.trees, family, cfg)
    I = fit_interaction_transform(train_fit, L, prep.trees, w, cfg) if family.interaction_capable else NONE
    return {prep, target, train_fit, L, w, graphs, interaction_state:I}

function candidate_sets(cfg):
    base = deterministic_beta_alpha_grid(max_candidates=48, seed=cfg.seed)
    # Include lambda_F=0, cover allowed alpha pairs, deduplicate meaningless alpha.
    additive = [attach(base_hp, interactions_on=False) for base_hp in base]
    on = [attach(base_hp, interactions_on=True, lambda_I1=s, lambda_I2=r)
          for base_hp in base for s in [0,0.01,0.1] for r in [0.01,0.1,1]]
    return additive, additive + on                                 # up to 48 vs 480

function inner_search(train, family, cfg):
    splits = site_disjoint_folds(train, k=4, seed=cfg.seed,
                                 balance_sample_counts=True, inspect_targets=False)
    candidates = candidates_for_family(family, cfg)
    for fit_ids, valid_ids in splits:
        assert_no_shared_physical_site(fit_ids, valid_ids)
        state = prepare_partition(train[fit_ids], family, cfg)
        for hp in candidates:
            if hp.interactions_on AND state.interaction_state is UNAVAILABLE:
                record_failed_candidate(hp, 'no eligible products')
                continue
            model = fit_coefficients(state, hp, cfg)
            if model.failed:
                record_failure(hp, model.reason)
                continue
            bundle = package_model(model, state, hp)
            predictions = predict(bundle, train[valid_ids], apply_threshold=False)
            record_inner_predictions(hp, predictions)
            record_complexity(hp, extract_representation(bundle))  # [15]

    valid = candidates_with_success_in_every_fold_and_coverage_at_least(1.0)
    require_nonempty(valid)
    for hp in valid:
        oof = concatenate_inner_predictions(hp)
        site_loss[hp] = per_site_log_loss(oof) if classification else per_site_MSE(oof)
        mean_loss[hp] = mean(site_loss[hp])
    best = argmin(mean_loss)
    SE = sample_sd(site_loss[best]) / sqrt(number_of_validation_sites)
    allowed = [hp for hp in valid if mean_loss[hp] <= mean_loss[best] + SE]
    if family.interaction_capable:
        selected = min(allowed, keys=[mean_active_products, mean_required_raw_taxa,
                                      mean_active_additive_groups, mean_loss, config_order])
    else:
        selected = min(allowed, keys=[mean_active_groups, mean_active_taxa,
                                      mean_loss, config_order])
    # SE is a selection heuristic, not an independent-site confidence interval.
    threshold = select_threshold(inner_oof[selected]) if classification else NONE
    return {hp:selected, threshold, minimum_loss_hp:best, search_audit}

function select_threshold(inner_oof):
    w = site_weights(inner_oof.site)
    for threshold in [0.05, 0.10, ..., 0.95]:
        predicted_label = inner_oof.probability >= threshold
        TP, FP, FN = weighted_confusion_counts(inner_oof.y, predicted_label, w)
        F2[threshold] = 5*TP / (5*TP + 4*FN + FP) if denominator>0 else UNDEFINED
    return highest_threshold_among_defined_F2_maximizers()
    # If no threshold has a defined score, report unavailable; do not invent a optimum.
```

For classification constants, use the training-site-weighted event rate; clip probabilities to [1e-12, 1−1e-12] only for numerical metric evaluation. The reference threshold 0.5 is also evaluated. Selection of the F2 threshold follows hyperparameter selection by log loss. It never uses outer-held-out labels. A one-class held-out fold can be scored by log loss; a one-class primary training partition triggers the documented failure gate. Do not silently redraw site splits or omit failed folds. Optional constant baselines remain explicitly separate.

```python
function nested_evaluate(cohort, cfg):
    outer = site_disjoint_folds(cohort, k=6, seed=cfg.seed, inspect_targets=False)
    families = [intercept_only, genus_ridge, genus_elastic_net,
                asv_ridge, asv_elastic_net, phylogeny_only,
                abundance_only, shared_hybrid, domain_adaptive_additive,
                domain_adaptive_interaction]
    for train_ids, test_ids in outer:
        save_split_manifest(train_ids, test_ids)
        for family in families:
            if family == intercept_only:
                model = fit_site_weighted_constant_baseline(cohort[train_ids])
                if classification:
                    baseline_inner_oof = grouped_OOF_site_weighted_event_rates(cohort[train_ids])
                    model.threshold = select_threshold(baseline_inner_oof)
            else:
                selection = inner_search(cohort[train_ids], family, cfg)
                state = prepare_partition(cohort[train_ids], family, cfg)  # REBUILD
                model = package_model(fit_coefficients(state, selection.hp, cfg), state, selection.hp)
                model.threshold = selection.threshold
            predictions = predict(model, cohort[test_ids])        # untouched test sites
            save_model_and_complete_OOF_rows(family, predictions, including_failures=True)
    return evaluate_all_stored_OOF_by_target_and_family()
```

Fusion-family resolution is configured; rank-specific ridge/elastic-net baselines use their named resolutions. Missing inputs mark a family unavailable. Failures return structured records rather than terminating every unrelated target/family, and invalidate any claimed complete evaluation for the affected family. Compare classification using ROC-AUC/average precision where defined, log loss, Brier score, F2, precision/recall, confusion matrices, coverage and class/site counts. Regression uses RMSE, MAE, pooled R² and site-balanced errors. Do not average per-site AUC/R² over tiny groups. Bootstrap paired site losses using identical site draws for both models; label intervals conditional on fitted OOF models.

Reuse matching additive candidate fits, but report the interaction family's larger search budget. A selected off-candidate is an additive result, not evidence that interactions were learned. A pseudocount-1.0 sensitivity reruns the full pipeline using the same folds; never choose the pseudocount by outer-test results.

## H. Compression and dependency accounting — step 15

```python
function extract_representation(model, tolerance=1e-6):
    for d in ['ARC', 'BAC']:
        if model.lambda_F == 0:
            groups[d] = all_singletons()
        else:
            close_edges = positive_W_edges_where_abs_beta_difference_le(tolerance)
            groups[d] = connected_components_including_isolates(close_edges)
            for G in groups[d]:
                if max(beta[G])-min(beta[G]) > tolerance:
                    replace_with_singletons(G)                     # no tolerance chaining
        for G in groups[d]:
            mean_beta[G] = mean(beta[G])
            theta[G] = len(G) * mean_beta[G]
            residual[G] = beta[G] - mean_beta[G]
        # Additive score = sum_G theta[G]*mean(L[:,G]) + L[:,G]@residual[G].

    active_main_taxa = taxa_where_abs_beta_gt(tolerance)
    active_products = pairs_where_abs_gamma_gt(tolerance)
    parent_balances = unique_parents(active_products)
    required_taxa = union(active_main_taxa, all_taxa_with_nonzero_loading(parent_balances))
    return {groups, theta, residual, active_products, parent_balances, required_taxa}
```

These are reporting counts, not instructions to truncate serialized coefficients. Preserve full fitted beta/gamma and all transformations for exact predictions. A main-effect-zero taxon can be required by a product. Reconstruct the full predictor by adding the original product contribution to the compressed additive score. Do not average gamma across folds with different balance memberships. The optional b12-only comparator remains separate and off by default; no exploratory full-data block membership is inserted automatically.

## I. Frozen prediction and artifacts — steps 16–17

```python
function predict(model, rows, apply_threshold=True):
    if model.is_intercept_only_baseline:
        return baseline_prediction_for_declared_input_rows(model, rows)
    L, scorable, reasons = transform_counts(rows, model.preprocessing)
    if model.is_constant_regression_fit:
        return constant_predictions_with_support_mask(model.constant_value, scorable, reasons)
    eta = model.b0 + L.A @ model.beta_A + L.B @ model.beta_B
    if model.interactions_on:
        X_I = transform_interactions(L, model.interaction_state)
        eta += X_I @ model.gamma
    if model.task == 'classification':
        value = stable_sigmoid(eta)
        label = value >= model.threshold if apply_threshold else NONE
    else:
        value = model.target.inverse_transform(eta)
        label = NONE
    return rows_in_original_ID_order(value, label, scorable, reasons)
    # No fitting, reselection, reordering by position, or new batch scaling.

function cache_key(partition, settings):
    return hash(training_sample_IDs, target_cohort_IDs, input_hashes,
                taxonomy_and_tree_versions, count_preprocessing,
                graph_settings, interaction_settings, seed, code_version)
    # Numerical fit caches additionally include task, training-label hash,
    # penalty settings and solver options. Outcome-free caches can omit labels
    # only when the exact training rows and every transformation setting agree.

function save_outputs(evaluation, final_model, manifest):
    save_resolved_config_versions_hashes_and_run_manifest()
    save_cohort_exclusions_feature_mass_and_class_site_counts()
    save_all_inner_outer_split_manifests_and_graph_edge_tables()
    save_candidate_settings_failures_losses_coverage_and_selection()
    save_OOF_predictions_probabilities_thresholds_and_provenance()
    save_coefficients_groups_residuals_and_raw_taxon_dependencies()
    save_balance_memberships_signed_loadings_and_selection_statistics()
    save_both_stages_of_scaling_and_ordered_product_pairs()
    save_interaction_on_off_comparisons_and_conditional_effect_plots()
    serialize_final_model_separately_from_outer_fold_models()
    write_readme_and_run_report_with_real_vs_synthetic_labels()
```

Implement `validate`, `diagnose`, `train`, and `predict`. Diagnostics use count-only preprocessing when no target is needed and remain descriptive; they do not populate learned fold artifacts. Conditional effects refer to log odds for classification and must be displayed within observed balance support. Existing smoke results do not demonstrate interaction benefit or adequate power for rare flags.

## J. Scientific verification — step 18

```python
function verify():
    test_QIIME_header_orientation_taxonomy_and_rank_tree_alignment()
    test_exact_season_tokens_live_flag_formulas_and_missing_trigger_policy()
    test_preweighted_substrate_passthrough_and_duplicate_join_rejection()
    test_physical_site_disjointness_in_both_CV_loops()
    test_both_beta_sums_zero_not_just_combined_sum()
    test_positive_input_scale_invariance_of_balances_products_eta_and_probabilities()
    test_heldout_perturbation_cannot_change_training_state_or_coefficients()
    test_graph_weights_unique_edges_no_cross_domain_fusion_and_empty_graphs()
    test_bootstrap_site_copies_and_distinct_original_site_eligibility()
    test_coabsence_ineligibility_and_pre_smoothing_unsupported_rows()
    test_between_plus_within_equals_site_balanced_variance()
    test_balance_sign_deduplication_polytomies_zero_sum_and_constant_features()
    test_missing_parent_taxa_and_main_effect_zero_interaction_dependency()
    test_beta_group_residual_plus_product_reconstructs_full_predictor()
    test_interaction_off_matches_additive_for_logistic_and_squared_loss()
    test_objective_terms_solver_status_and_equality_tolerances()
    test_binary_labels_not_standardized_and_one_class_training_gate()
    test_F2_threshold_uses_inner_OOF_only()
    test_saved_model_replay_under_ID_preserving_row_column_reordering()
    run_labeled_synthetic_nested_demo_with_known_balance_product_logistic_signal()
    run_zero_interaction_synthetic_case_and_regression_demo()
    # Do not require improvement or perfect group recovery in every random draw.
```

Acceptance requires working required commands, traceable exclusions, successful scientific checks, complete declared OOF coverage or an explicit failed gate, and no hidden change of model. The real-data interaction hypothesis is assessed by the additive-versus-interaction held-out-site comparison, not by the synthetic demonstration.

## Claude Code implementation prompt

```text
Read ad-microbial-model-design-arc-bac-interactions.md and
ad-microbial-arc-bac-pseudocode.md. Implement the complete framework using
the design document as the authoritative contract and the pseudocode as the
control-flow guide. Tag major functions with their design step numbers.

Inspect the existing fusionML implementation and live warning-flag target
loader. Preserve verified target formulas, sample matching and physical-site
grouping. Add training-only tree-balance construction, balance-product scaling,
joint constrained logistic/regression fitting, nested selection, frozen
prediction and all required audits/tests. Compare the interaction-capable
family with its matched additive baseline. No all-data feature discovery may
enter validation folds.

Provide modular code, CLI commands, configuration migration, dependencies and
a README. Run scientific checks and a labeled synthetic smoke demonstration;
run real-data smoke validation only when the required files are available.
Report actual commands, failures, missing inputs and deviations. Do not claim
interaction benefits without held-out-site results.
```
