"""Newick tree loading and patristic-distance computation.

Design-spec item 12 (phylogenetic graph):
"Compute patristic distances from the supplied domain tree. Reject negative
or nonfinite branch lengths, duplicate tip IDs and malformed mappings.
Preserve path lengths if pruning or contracting the tree. Features absent
from a partially matched tree can remain as isolated phylogenetic nodes;
export their IDs and coverage."

Patristic distance is computed from precomputed root-to-tip cumulative
branch length + a prefix-match LCA, rather than Bio.Phylo's O(n) tree.distance
per call, so it stays tractable for the hundreds of retained tips typical
after preprocessing filtering.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio import Phylo


class TreeError(ValueError):
    pass


@dataclasses.dataclass
class DomainTree:
    domain: str
    path: str
    tip_names: List[str]  # raw (non-prefixed) tip labels as in the Newick file
    _cumdist: Dict[str, float] = dataclasses.field(repr=False)
    _path_ids: Dict[str, Tuple[int, ...]] = dataclasses.field(repr=False)
    _depth_cumdist: Dict[Tuple[int, ...], float] = dataclasses.field(repr=False)
    raw_tree: object = dataclasses.field(default=None, repr=False)  # Bio.Phylo tree, for topology (clade splits)

    def coverage(self, domain_prefixed_feature_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Return (covered, uncovered) domain-prefixed feature IDs."""
        tipset = {f"{self.domain}::{t}" for t in self.tip_names}
        covered = [f for f in domain_prefixed_feature_ids if f in tipset]
        uncovered = [f for f in domain_prefixed_feature_ids if f not in tipset]
        return covered, uncovered

    def patristic_distance_matrix(self, domain_prefixed_feature_ids: List[str]) -> "np.ndarray":
        """Pairwise patristic distance among features covered by this tree.

        Only features present as tips are included; caller is responsible
        for handling uncovered features as isolated phylogenetic nodes.
        """
        raw_names = [fid.split("::", 1)[1] for fid in domain_prefixed_feature_ids]
        for n in raw_names:
            if n not in self._cumdist:
                raise TreeError(f"tip {n!r} not covered by tree; filter with coverage() first")
        n = len(raw_names)
        dist = np.zeros((n, n), dtype=float)
        paths = [self._path_ids[nm] for nm in raw_names]
        cds = [self._cumdist[nm] for nm in raw_names]
        for i in range(n):
            pi, ci = paths[i], cds[i]
            for j in range(i + 1, n):
                pj, cj = paths[j], cds[j]
                k = 0
                m = min(len(pi), len(pj))
                while k < m and pi[k] == pj[k]:
                    k += 1
                lca_cumdist = self._cumdist_at_depth(pi, k)
                d = (ci - lca_cumdist) + (cj - lca_cumdist)
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def _cumdist_at_depth(self, path_ids: Tuple[int, ...], depth: int) -> float:
        if depth == 0:
            return 0.0
        return self._depth_cumdist[path_ids[:depth]]


def load_tree(path: str, domain: str) -> DomainTree:
    tree = Phylo.read(path, "newick")
    terminals = tree.get_terminals()
    names = [t.name for t in terminals]
    if any(n is None for n in names):
        raise TreeError(f"{path}: one or more tips have no name")
    dup = pd_unique_dupes(names)
    if dup:
        raise TreeError(f"{path}: duplicate tip IDs: {dup[:10]}")

    for clade in tree.find_clades():
        bl = clade.branch_length
        if bl is not None and (not math.isfinite(bl) or bl < 0):
            raise TreeError(f"{path}: negative or nonfinite branch length at clade {clade.name!r}: {bl}")

    # Precompute, for every clade reachable from root, a unique integer id
    # per node along a depth-first walk, its cumulative distance to root,
    # and (for terminals) the full path of ids from root to that tip. This
    # turns LCA + patristic distance for any pair into an O(depth) prefix
    # comparison instead of Bio.Phylo's O(n) traversal per call.
    depth_cumdist: Dict[Tuple[int, ...], float] = {(): 0.0}
    cumdist: Dict[str, float] = {}
    path_ids: Dict[str, Tuple[int, ...]] = {}
    counter = [0]

    # Iterative DFS (not recursive): real ASV trees from FastTree/IQ-TREE
    # can be deep, unbalanced ladders that would blow Python's recursion
    # limit if walked recursively.
    stack = [(tree.root, (), 0.0)]
    while stack:
        clade, prefix_ids, prefix_dist = stack.pop()
        for child in clade.clades:
            counter[0] += 1
            cid = counter[0]
            bl = child.branch_length if child.branch_length is not None else 0.0
            new_ids = prefix_ids + (cid,)
            new_dist = prefix_dist + bl
            depth_cumdist[new_ids] = new_dist
            if child.is_terminal():
                cumdist[child.name] = new_dist
                path_ids[child.name] = new_ids
            else:
                stack.append((child, new_ids, new_dist))

    dt = DomainTree(
        domain=domain, path=path, tip_names=names,
        _cumdist=cumdist, _path_ids=path_ids, _depth_cumdist=depth_cumdist,
        raw_tree=tree,
    )
    return dt


def pd_unique_dupes(names: List[str]) -> List[str]:
    seen = set()
    dupes = []
    for n in names:
        if n in seen and n not in dupes:
            dupes.append(n)
        seen.add(n)
    return dupes
