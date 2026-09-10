#!/usr/bin/env python3
"""
reduce_features_for_tree_models.py -- shrink an ASV table + rooted tree to a feature
set that Phylo-Spec / DeepPhylo can actually consume, WITHOUT destroying the tree.

Three reduction modes, in increasing order of information loss:

  --mode asv     prevalence/abundance filter only; tree is pruned to the survivors.
                 Nothing is merged. Leaf labels stay as ASV ids.

  --mode phylo   phylogenetic agglomeration: any clade whose tip-to-tip diameter is
                 <= --height is collapsed to a single tip whose abundance is the sum
                 of its members. Every merged group is a clade by construction, so
                 the reduced tree is still a valid tree with correct branch lengths.
                 This is the tip_glom / tree_glom idea, implemented on the same
                 Newick parser the rest of the pipeline uses.

  --mode genus   collapse by taxonomic label -- but only where the label is
                 monophyletic on YOUR pruned tree. A genus that is scattered across
                 the tree is split into its maximal monophyletic blocks
                 (Genus__b1, Genus__b2, ...) instead of being merged into a taxon
                 with no defensible position. Features unclassified at genus are
                 NEVER lumped together; they are carried through as their own tips,
                 because "unclassified" is not a clade.

Why this matters: an unclassified ASV still has a perfectly well-defined position in
the tree -- position comes from sequence, classification comes from a reference
database, and they are independent operations. Merging all `g__uncultured` features
into one column creates a feature whose members sit all over the tree, and there is
then no branch to attach it to. Both Phylo-Spec (topology + branch length) and
DeepPhylo (patristic distances) will silently accept the resulting file and learn
from a prior that is simply wrong.

Always emitted: a classification-rate report per rank (features and reads), a
monophyly audit of every genus, and a feature-mapping table.

Usage
-----
python reduce_features_for_tree_models.py \
    -c asv_table.csv -t rooted_tree.nwk -x taxonomy.csv -o reduced \
    --mode phylo --height 0.05 --prevalence 0.10 --min-abundance 1e-4
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_deepphylo_inputs import parse_newick, leaf_index
from build_cross_domain_graph import prune_tree

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
PREFIX = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]
PREFIX_ANY = {"d__": 0, "k__": 0, "p__": 1, "c__": 2, "o__": 3,
              "f__": 4, "g__": 5, "s__": 6}
RANK_ALIASES = {"domain": 0, "kingdom": 0, "phylum": 1, "class": 2, "order": 3,
                "family": 4, "genus": 5, "species": 6}
UNCLASSIFIED = re.compile(
    r"^\s*$|uncultured|unidentified|unassigned|metagenome|^unknown|incertae.sedis|"
    r"^bacterium$|^archaeon$|_sp\.?$|\bsp\b|^ambiguous", re.I)


# ------------------------------------------------------------------ taxonomy
def split_lineage(s):
    """Return a 7-list of rank labels, '' where absent or uninformative."""
    if not isinstance(s, str):
        return [""] * 7
    out = [""] * 7
    parts = [p.strip() for p in re.split(r"[;|]", s) if p.strip()]
    tagged = any(p[:3].lower() in PREFIX_ANY for p in parts)
    for i, p in enumerate(parts):
        if tagged:
            r = PREFIX_ANY.get(p[:3].lower())
            if r is not None:
                out[r] = p[3:].strip()
        elif i < 7:
            out[i] = p
    return ["" if UNCLASSIFIED.match(v or "") else v for v in out]


def load_taxonomy(path):
    """Accepts a long format (feature-id + one lineage string) or a wide format
    (feature-id + one column per rank). Returns {feature: 7-list}."""
    tx = pd.read_csv(path)
    fcol = tx.columns[0]
    wide = {RANK_ALIASES[c.strip().lower()]: c for c in tx.columns[1:]
            if c.strip().lower() in RANK_ALIASES}
    if len(wide) >= 3:
        out = {}
        for _, row in tx.iterrows():
            lin = [""] * 7
            for r, c in wide.items():
                v = str(row[c]) if pd.notna(row[c]) else ""
                v = v[3:] if v[:3].lower() in PREFIX_ANY else v
                lin[r] = "" if UNCLASSIFIED.match(v.strip()) else v.strip()
            out[str(row[fcol])] = lin
        return out
    tcol = next((c for c in tx.columns[1:]
                 if tx[c].astype(str).str.contains(";|__", regex=True).mean() > 0.5),
                tx.columns[1])
    return {str(r[fcol]): split_lineage(str(r[tcol])) for _, r in tx.iterrows()}


def classification_report(lin, reads):
    """Fraction of features and of total reads classified at each rank."""
    tot = reads.sum()
    rep = {}
    for r, name in enumerate(RANKS):
        ok = np.array([bool(lin[f][r]) for f in lin])
        rep[name] = {"features_pct": round(100 * ok.mean(), 1),
                     "reads_pct": round(100 * reads[ok].sum() / tot, 1) if tot else 0.0}
    return rep


# ------------------------------------------------------------------ tree ops
def postorder(root):
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(n.children)
    return order[::-1]


def tipsets(root):
    """{id(node): frozenset(tip names)} for every node."""
    ts = {}
    for n in postorder(root):
        ts[id(n)] = (frozenset([n.name]) if not n.children
                     else frozenset().union(*(ts[id(c)] for c in n.children)))
    return ts


def heights_diameters(root):
    """{id(node): (height, diameter)} -- height = max root-ward tip distance below
    the node, diameter = max tip-to-tip patristic distance within its clade."""
    hd = {}
    for n in postorder(root):
        if not n.children:
            hd[id(n)] = (0.0, 0.0)
            continue
        reach = sorted((hd[id(c)][0] + max(c.blen, 0.0) for c in n.children), reverse=True)
        h = reach[0]
        d = max([hd[id(c)][1] for c in n.children] +
                ([reach[0] + reach[1]] if len(reach) > 1 else [0.0]))
        hd[id(n)] = (h, d)
    return hd


def collapse(node, name):
    """Turn an internal node into a tip, keeping its own branch length."""
    node.children = []
    node.name = name


def to_newick(root):
    def rec(n):
        s = "(" + ",".join(rec(c) for c in n.children) + ")" if n.children else ""
        return f"{s}{n.name}:{max(n.blen, 0.0):.8f}"
    return rec(root) + ";"


def monophyly_audit(root, groups):
    """groups: {label: set(tips)}. Returns {label: n_maximal_blocks} -- 1 == monophyletic."""
    ts = tipsets(root)
    out = {}
    for lab, members in groups.items():
        blocks, stack = 0, [root]
        while stack:
            n = stack.pop()
            s = ts[id(n)]
            if not (s & members):
                continue
            if s <= members:
                blocks += 1
            else:
                stack.extend(n.children)
        out[lab] = blocks
    return out


def maximal_blocks(root, members):
    """Deepest-first list of nodes whose whole tip set lies inside `members`."""
    ts = tipsets(root)
    found, stack = [], [root]
    while stack:
        n = stack.pop()
        s = ts[id(n)]
        if not (s & members):
            continue
        if s <= members:
            found.append(n)
        else:
            stack.extend(n.children)
    return found


def phylospec_audit(root, names):
    """Checks specific to the Phylo-Spec reference implementation.

    Step 2 of PhyloSpec.forward gives every abundance column that does NOT match a
    tree tip its own 1x1 conv and concatenates it straight into the pooled feature
    vector -- it never enters conv_order, so it gets no node weight, no parent, and
    no phylogenetic context. That is the "separate node for unclassified": it is
    triggered by a NAME not matching a leaf, not by taxonomy.

    calculate_node_weights uses 1 - branch_length, so any branch longer than 1
    substitution/site produces a negative multiplier and silently flips the sign of
    that node's activations.

    match_leaf_nodes maps by substring (`if species in leaf_name`), so short feature
    ids that are prefixes of one another mis-map.
    """
    tips = set()
    blens = []
    for n in postorder(root):
        blens.append(max(n.blen, 0.0))
        if not n.children:
            tips.add(n.name)
    names = [str(x) for x in names]
    ambiguous = sorted({a for a in names for b in names if a != b and a in b})
    return {
        "features_matching_a_tip": sum(1 for f in names if f in tips),
        "features_routed_to_step2_bypass": sum(1 for f in names if f not in tips),
        "columns_named_unclassified": [f for f in names if "unclassified" in f.lower()],
        "max_branch_length": round(float(max(blens)), 4),
        "branches_ge_1_negative_node_weight": int(sum(1 for b in blens if b >= 1.0)),
        "substring_ambiguous_feature_names": ambiguous[:10],
    }


# ------------------------------------------------------------------ modes
def reduce_phylo(root, height):
    """Collapse every maximal clade with diameter <= height. Returns {new_tip: [members]}."""
    hd, ts = heights_diameters(root), tipsets(root)
    groups, stack, k = {}, [root], 0
    while stack:
        n = stack.pop()
        if not n.children:
            groups[n.name] = [n.name]
            continue
        if hd[id(n)][1] <= height:
            k += 1
            name = f"clade{k:05d}"
            groups[name] = sorted(ts[id(n)])
            collapse(n, name)
        else:
            stack.extend(n.children)
    return groups


def reduce_genus(root, lin, feats):
    """Collapse monophyletic genus blocks; keep unclassified features as themselves."""
    byg = {}
    for f in feats:
        g = lin[f][5]
        if g:
            byg.setdefault(g, set()).add(f)
    groups, split = {}, {}
    for g, members in sorted(byg.items()):
        blocks = maximal_blocks(root, members)
        if len(blocks) > 1:
            split[g] = len(blocks)
        for i, node in enumerate(blocks, 1):
            name = g if len(blocks) == 1 else f"{g}__b{i}"
            name = re.sub(r"\W+", "_", name)
            groups[name] = sorted(tipsets(root)[id(node)])
            collapse(node, name)
    for f in feats:                       # unclassified: carried through untouched
        if not lin[f][5]:
            groups[f] = [f]
    return groups, split


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--counts", required=True,
                   help="CSV: col0 sample id, feature columns, optional final label column")
    p.add_argument("-t", "--tree", required=True, help="rooted Newick")
    p.add_argument("-x", "--taxonomy", required=True,
                   help="CSV with a feature-id column and a lineage/Taxon column")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--mode", choices=["asv", "phylo", "genus"], default="phylo")
    p.add_argument("--height", type=float, default=0.05,
                   help="patristic diameter threshold for --mode phylo")
    p.add_argument("--prevalence", type=float, default=0.10,
                   help="keep features present in at least this fraction of samples")
    p.add_argument("--min-abundance", type=float, default=1e-4,
                   help="keep features with at least this mean relative abundance")
    p.add_argument("--safe-ids", action="store_true",
                   help="emit opaque F00001-style feature ids in the table and tree; the "
                        "readable name is kept in the 'label' column of the mapping file. "
                        "Use this whenever output labels are taxonomic names, because "
                        "Phylo-Spec matches columns to tips by substring.")
    p.add_argument("--protect", default="archaea",
                   help="regex matched against the full lineage; matching features get "
                        "--protect-min-abundance instead of --min-abundance. Default "
                        "'archaea' -- methanogens are a few percent of AD reads and a "
                        "bacteria-tuned filter deletes exactly the partners you care about. "
                        "Pass '' to disable.")
    p.add_argument("--protect-min-abundance", type=float, default=1e-5)
    p.add_argument("--label-col", default=None,
                   help="name of the label column; default = last column if non-numeric")
    a = p.parse_args()

    df = pd.read_csv(a.counts, index_col=0)
    lab = a.label_col
    if lab is None and not np.issubdtype(df.iloc[:, -1].dtype, np.number):
        lab = df.columns[-1]
    y = df[lab] if lab else None
    X = df.drop(columns=[lab]) if lab else df

    lineage = load_taxonomy(a.taxonomy)
    for f in X.columns:
        lineage.setdefault(str(f), [""] * 7)

    root = parse_newick(open(a.tree).read())
    intree = set(leaf_index(root))
    missing = [f for f in X.columns if f not in intree]
    if len(missing) == X.shape[1]:
        sys.exit("ERROR: no feature name matches a tip in the tree.")

    rel = X.div(X.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    prot = pd.Series(False, index=X.columns)
    if a.protect:
        rx = re.compile(a.protect, re.I)
        prot = pd.Series([bool(rx.search(";".join(lineage[str(f)]))) for f in X.columns],
                         index=X.columns)
    thr = np.where(prot, a.protect_min_abundance, a.min_abundance)
    keep = ((X > 0).mean(axis=0) >= a.prevalence) & (rel.mean(axis=0).values >= thr)
    keep &= pd.Series([f in intree for f in X.columns], index=X.columns)
    feats = list(X.columns[keep])
    if not feats:
        sys.exit("ERROR: the prevalence/abundance filter removed every feature.")

    report = {
        "input_features": int(X.shape[1]),
        "features_absent_from_tree": len(missing),
        "features_after_filter": len(feats),
        "protected_features_retained": int(prot[feats].sum()),
        "reads_retained_pct": round(100 * X[feats].values.sum() / X.values.sum(), 2),
        "classification_rate_all_features": classification_report(
            {f: lineage[str(f)] for f in X.columns}, X.sum(axis=0).values),
        "classification_rate_after_filter": classification_report(
            {f: lineage[str(f)] for f in feats}, X[feats].sum(axis=0).values),
    }

    root = prune_tree(root, set(feats))

    byg = {}
    for f in feats:
        g = lineage[str(f)][5]
        if g:
            byg.setdefault(g, set()).add(f)
    audit = monophyly_audit(root, byg)
    report["genera_on_pruned_tree"] = len(byg)
    report["genera_non_monophyletic"] = int(sum(1 for v in audit.values() if v > 1))
    report["worst_genera"] = sorted(((k, v) for k, v in audit.items() if v > 1),
                                    key=lambda kv: -kv[1])[:10]

    if a.mode == "asv":
        groups = {f: [f] for f in feats}
    elif a.mode == "phylo":
        groups = reduce_phylo(root, a.height)
    else:
        groups, split = reduce_genus(root, {str(f): lineage[str(f)] for f in feats}, feats)
        report["genera_split_into_blocks"] = split

    out_names = sorted(groups)

    # Phylo-Spec's match_leaf_nodes binds columns to tips with a SUBSTRING test
    # (`if species in leaf_name`), so a feature id that is a substring of another
    # id silently steals that tip. --safe-ids replaces every label with an opaque
    # fixed-width token; the readable name survives in the mapping file.
    if a.safe_ids:
        safe = {g: f"F{i:05d}" for i, g in enumerate(out_names)}
        for n in postorder(root):
            if not n.children and n.name in safe:
                n.name = safe[n.name]
        groups = {safe[g]: v for g, v in groups.items()}
        alias = {safe[g]: g for g in out_names}
        out_names = sorted(groups)
    else:
        alias = {g: g for g in out_names}

    red = pd.DataFrame({g: X[groups[g]].sum(axis=1) for g in out_names}, index=X.index)
    if y is not None:
        red[lab] = y
    red.to_csv(f"{a.out}_table.csv")
    open(f"{a.out}_tree.nwk", "w").write(to_newick(root) + "\n")
    pd.DataFrame([{"feature": m, "reduced_feature": g, "label": alias[g],
                   "n_members": len(groups[g])}
                  for g in out_names for m in groups[g]]).to_csv(
        f"{a.out}_mapping.csv", index=False)

    report["phylospec_audit"] = phylospec_audit(root, out_names)
    report["output_features"] = len(out_names)
    report["largest_group"] = int(max(len(v) for v in groups.values()))
    report["mode"] = a.mode
    json.dump(report, open(f"{a.out}_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
