import numpy as np
import pandas as pd
from kl_clustering_analysis.tree.poset_tree import PosetTree #type: ignore
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


# Create a feature binary feature matrix, where each row is a gene and each column is a GO term
def compute_feature_matrix(go_terms_dict):
    """
    Create a binary feature matrix from a dictionary of GO terms for each gene.
    
    Parameters
    ----------
    go_terms_dict : dict
        Dictionary of unique GO term IDs for each gene in the disease core module.
        
    Returns
    -------
    feature_matrix : pd.DataFrame
        Binary feature matrix where rows corresponds to gene IDs and columns represent GO term IDs.
    """

    # Get all unique GO terms
    all_go_terms = list(set([term for terms in go_terms_dict.values() for term in terms]))

    # Create an empty feature matrix
    feature_matrix = np.zeros((len(go_terms_dict), len(all_go_terms)))

    # Fill the feature matrix
    for i, gene_symbol in enumerate(go_terms_dict.keys()):
        gene_terms = go_terms_dict[gene_symbol]
        for j, term in enumerate(all_go_terms):
            if term in gene_terms:
                feature_matrix[i, j] = 1

    feature_matrix = pd.DataFrame(feature_matrix, columns=all_go_terms, index=go_terms_dict.keys()).astype(int)
    # Remove rows where all elements are 0
    zero_sum_rows = feature_matrix[feature_matrix.sum(axis=1) == 0]
    if not zero_sum_rows.empty:
        removed_samples = zero_sum_rows.index.tolist()
        print(f"Warning: Removed {len(removed_samples)} gene(s) with no associated terms: {removed_samples}")
        feature_matrix = feature_matrix[feature_matrix.sum(axis=1) > 0]

    return feature_matrix

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

