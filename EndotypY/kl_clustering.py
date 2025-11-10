import numpy as np
import pandas as pd
from kl_clustering_analysis import tree, hierarchy_analysis
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Import the necessary functions from your library
from tree.poset_tree import PosetTree
from hierarchy_analysis.divergence_metrics import calculate_hierarchy_kl_divergence
from hierarchy_analysis.statistical_tests import (
    annotate_nodes_with_statistical_significance_tests,
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
)
from hierarchy_analysis.cluster_decomposition import ClusterDecomposer

def kl_clustering_endotypes(data: pd.DataFrame, linkage_method: str = 'complete', distance_metric: str = 'hamming', alpha: float = 0.05) -> dict:
    """
    A function to perform KL-based hierarchical clustering analysis on the provided dataset.
    Parameters:
    - data: pd.DataFrame
        The input binary dataset with samples as rows and features as columns.
    - linkage: str
        The linkage method to use for hierarchical clustering.
    - distance_metric: str
        The distance metric to use for calculating pairwise distances.
    
    """
    print("--- Starting Analysis Pipeline ---")

    # ensure data is in correct format
    data = data.astype(int)

    # 1. --- Data Preparation ---
    print(f"\nStep 1: Prepared dataset with {data.shape[0]} samples and {data.shape[1]} features.")

    # --- Execute the Core Pipeline ---
    # 2. linkage()
    Z = linkage(pdist(data.values, metric=distance_metric), method=linkage_method)
    print("\nStep 2: Created hierarchy with SciPy linkage.")

    # 3. PosetTree.from_linkage()
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    print("Step 3: Converted hierarchy to PosetTree structure.")

    # 4. calculate_hierarchy_kl_divergence()
    # Pass the NetworkX graph (the PosetTree itself), not its metadata dict
    stats_df = calculate_hierarchy_kl_divergence(tree, data)
    print("Step 4: Calculated KL-divergence for all nodes in the tree.")

    significance_level = alpha
    # 5. annotate...() functions
    stats_df = annotate_nodes_with_statistical_significance_tests(
        stats_df, data.shape[1], significance_level
    )
    stats_df = annotate_child_parent_divergence(
        tree, stats_df, data.shape[1], significance_level
    )
    # annotate_sibling_independence_cmi uses keyword-only arguments after '*'
    stats_df = annotate_sibling_independence_cmi(
        tree, stats_df, significance_level_alpha=significance_level
    )
    print("Step 5: Annotated tree with multiple statistical significance tests.")

    # 6. ClusterDecomposer.decompose_tree()
    decomposer = ClusterDecomposer(
        tree=tree,
        results_df=stats_df,
        significance_column="Are_Features_Dependent",
    )
    decomposition_results = decomposer.decompose_tree()
    print("Step 6: Decomposed the tree to extract significant clusters.")

    # --- Display Results ---
    print("\n--- Analysis Complete ---")
    num_found = decomposition_results.get("num_clusters", 0)
    print(f"\nAlgorithm found {num_found} clusters.")

    cluster_assignments = decomposition_results.get("cluster_assignments", {})
    
    predicted_labels = {}
    if cluster_assignments:
        for cluster_id, info in cluster_assignments.items():
            print(f"  - Cluster {cluster_id} (root: {info['root_node']}): {info['size']} samples")
            for leaf in info["leaves"]:
                predicted_labels[leaf] = cluster_id

    # Create a simple dict of clusters and assigned samples
    cluster_dict = {}
    for sample, cluster in predicted_labels.items():
        cluster_dict.setdefault(cluster, []).append(sample)

    return predicted_labels, cluster_assignments, cluster_dict

