import numpy as np
import pandas as pd
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


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

    # 1. ensure data is in correct format
    data = data.astype(int)

    # --- Execute the Core Pipeline ---
    # 2. linkage()
    Z = linkage(pdist(data.values, metric=distance_metric), method=linkage_method)
    print("Linkage matrix computed using method:", linkage_method, "and distance metric:", distance_metric)

    # 3. PosetTree.from_linkage()
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    # 4. tree.decompose()
    decomposition_results = tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)
    print("Tree decomposition completed with alpha_local=0.05 and sibling_alpha=0.05")

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

