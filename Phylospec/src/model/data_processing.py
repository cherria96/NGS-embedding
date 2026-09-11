import pickle
import tempfile
import numpy as np
from ete3 import PhyloTree
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
def load_and_preprocess_data(csv_path, tree, label_cols=None):
    data = pd.read_csv(csv_path)
    if label_cols is None:
        label_cols = [data.columns[-1]]
    n_labels = len(label_cols)
    leaf_names = [leaf.name for leaf in tree.get_terminals()]
    matched_columns = [col for col in leaf_names if col in data.columns]
    remaining_columns = [col for col in data.columns[1:-n_labels] if col not in matched_columns]
    # Reorder columns: ID | matched leaves | remaining features | label(s)
    ordered_columns = [data.columns[0]] + matched_columns + remaining_columns + label_cols
    data = data[ordered_columns]
    X = data.iloc[:, 1:-n_labels].values
    y = data.iloc[:, -1].values if n_labels == 1 else data[label_cols].values

    y_encoded, encoder = encode_labels(y)
    return X, y_encoded, encoder, data

# Encode class labels to numeric values
def encode_labels(y):
    if np.ndim(y) == 2:
        # Multi-label: columns are already binary (0/1) flags, no LabelEncoder needed
        return np.asarray(y, dtype=np.float32), None
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print(f"Class mapping: {class_mapping}")
    return y_encoded, encoder

# Match leaves in the phylogenetic tree to species in the dataset.
# Exact match only -- the old `if species in leaf_name` substring check lets a
# feature name that is a literal substring of another tip's name bind to the
# wrong tip silently (e.g. "Clostridium" into "Clostridium_sensu_stricto_1").
# Column names are expected to be the --safe-ids opaque tree-tip identifiers
# emitted by reduce_features_for_tree_models.py, so leaf name == column name
# exactly; tree_p() already prunes the tree to exactly those tips via the same
# exact `search_nodes(name=...)` match, so the two now agree by construction.
def match_leaf_nodes(tree, data):
    feature_columns = set(data.columns[1:])
    leaf_to_species = {
        leaf.name: leaf.name
        for leaf in tree.get_terminals()
        if leaf.name in feature_columns
    }
    assert len(leaf_to_species) == len(set(leaf_to_species.values())), \
        "match_leaf_nodes: duplicate leaf/column matches (should be impossible with exact matching)"
    unmatched = feature_columns - set(leaf_to_species)
    if unmatched:
        preview = sorted(unmatched)[:5]
        print(f"match_leaf_nodes: {len(unmatched)} feature column(s) have no matching tree tip "
              f"(handled by the model's unmatched-feature bypass): {preview}"
              f"{'...' if len(unmatched) > 5 else ''}")
    return leaf_to_species

# Assign unique names to internal nodes in the tree
def assign_unique_names(tree):
    internal_node_counter = 0
    for clade in tree.find_clades(order="level"):
        if not clade.is_terminal() and clade.name is None:
            clade.name = f"internal_{internal_node_counter}"
            internal_node_counter += 1
    return tree

# Traverse the tree in post-order to obtain convolutional order and relations
def get_conv_order(tree):
    nodes = []
    parents = {}
    conv_order = []
    node_relations = {}

    def postorder_traversal(node, parent, nodes, parents, conv_order):
        if node.is_terminal():
            nodes.append(node.name)
            parents[node.name] = parent
        else:
            children = []
            for child in node:
                postorder_traversal(child, node.name, nodes, parents, conv_order)
                children.append(child.name)
            nodes.append(node.name)
            parents[node.name] = parent
            # Store child-parent convolution pair
            conv_order.append((children, node.name))
            node_relations[node.name] = children

    postorder_traversal(tree.root, None, nodes, parents, conv_order)
    return nodes, parents, conv_order, node_relations

# Assign weights to nodes based on branch length (1 - length)
def calculate_node_weights(tree):
    node_weights = {}
    for clade in tree.find_clades(order="level"):
        if clade.branch_length is not None:
            node_weights[clade.name] = 1 - clade.branch_length
        else:
            node_weights[clade.name] = 1.0  # Default weight if branch length is missing
    return node_weights

# CLAUDE_CODE_TASK_cross_domain_injection.md Sec A.2: append one synthetic
# (children, parent) merge per retained cross-domain (bacterium, archaeon)
# edge directly to conv_order, instead of reweighting existing node_weights
# (the old CROSS_DOMAIN_INJECTION_BY_MODEL.md Arm 2a approach this replaces).
#
# forward() is a DAG walk, not a tree walk -- feature_map is never popped, so
# a leaf already consumed by its real parent merge is still available for a
# second, cross-domain merge. No modified Newick file is needed or wanted.
#
# Must be appended LAST: the loop in PhyloSpec.forward has no topological
# sort and trusts the given order. By the time these lines run, every leaf is
# already in feature_map (forward's Step 1) and every real internal node has
# already been merged (the preceding conv_order entries), so both children of
# every synthetic merge are guaranteed to already exist.
def inject_cross_domain_edges(conv_order, node_weights, leaf_to_species, edges):
    """edges: iterable of (bacterium, archaeon, rho) using feature-column
    (safe-id) names. Returns NEW (conv_order, node_weights) -- does not
    mutate the inputs, so the same base conv_order/node_weights can still be
    reused to build the tree-only or a differently-injected arm.

    Raises KeyError if an edge endpoint has no tree leaf -- task doc Sec 7
    forbids silently dropping one; the caller (cross_domain_edges.load_edges)
    is expected to have already filtered edges to this arm's feature set."""
    species_to_leaf = {v: k for k, v in leaf_to_species.items()}  # conv_order uses TREE names
    new_conv_order = list(conv_order)
    new_node_weights = dict(node_weights)
    for i, (bac, arc, rho) in enumerate(edges):
        lb, la = species_to_leaf.get(bac), species_to_leaf.get(arc)
        if lb is None or la is None:
            raise KeyError(f"cross-domain edge endpoint absent from leaf_to_species: {bac}, {arc}")
        name = f"syn_{i}"
        assert name not in new_node_weights
        new_conv_order.append(([lb, la], name))  # APPEND AT THE END
        new_node_weights[name] = float(abs(rho))
    return new_conv_order, new_node_weights


# Save learned node features along with tree structure and labels to a pickle file
def save_node_features_with_pickle(node_features, node_relations, node_weights, train_group, output_file):
    node_features_dict = {}
    for node, feature_list in node_features.items():
        concatenated_features = np.concatenate(feature_list, axis=0)
        node_features_dict[node] = concatenated_features
    branch_lengths = {node: weight for node, weight in node_weights.items()}

    combined_data = {
        'features': node_features_dict,
        'branch_lengths': branch_lengths,
        'relations': node_relations,
        'true_labels': train_group
    }

    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)

# Handling unclassified species through advanced taxonomy
def process_unclassified_features(tree, abundance_table, taxonomy_path, label_cols=None):
    if label_cols is None:
        label_cols = [abundance_table.columns[-1]]
    table = abundance_table.copy()
    group_cols = table[label_cols].copy()  # Save the label column(s)
    table = table.drop(columns=label_cols)
    taxonomy_table = pd.read_csv(taxonomy_path)

    unclassified_features = [col for col in abundance_table.columns if 'Unclassified' in col or 'unclassified' in col]

    if not unclassified_features:
        table[label_cols] = group_cols
        return table, tree

    for feature in unclassified_features:
        missing_species = pd.Series(dtype='str')

        taxonomic_levels = ['Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']
        for level in taxonomic_levels:
            matched_taxa = taxonomy_table.loc[taxonomy_table['Species'] == feature, level]
            if not matched_taxa.empty:
                taxon = matched_taxa.values[0]

                missing_species = taxonomy_table[
                    (taxonomy_table[level] == taxon) &
                    (~taxonomy_table['Species'].isin(abundance_table.columns)) &
                    (~taxonomy_table['Species'].str.contains('Unclassified'))
                ]['Species']

                if not missing_species.empty:
                    break

        if missing_species.empty:
            print(f"No missing species found in abundance table for feature '{feature}'.")
            continue

        unclassified_abundance = abundance_table[feature]
        average_abundance = unclassified_abundance / len(missing_species)

        for species in missing_species:
            if species in table.columns:
                table[species] += average_abundance
            else:
                table[species] = average_abundance
        table.drop(columns=[feature], inplace=True)

    table[label_cols] = group_cols
    return table, tree

def tree_p(csv_file_path, nwk_file_path, label_cols=None):
    df = pd.read_csv(csv_file_path)
    if label_cols is None:
        label_cols = [df.columns[-1]]
    feature_columns = df.columns[1:-len(label_cols)].tolist()

    tree = PhyloTree(nwk_file_path, format=1)

    existing_features = []
    for feature in feature_columns:
        nodes = tree.search_nodes(name=feature)
        if len(nodes) == 1:
            existing_features.append(nodes[0])
        elif len(nodes) > 1:

            existing_features.append(nodes[0])
        else:

            print(f"Feature not found in tree: {feature}")

    tree.prune(existing_features)
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.nwk')
    tree.write(format=1, outfile=tmp.name)
    tmp.close()
    return tmp.name
