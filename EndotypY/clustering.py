import networkx as nx
import pandas as pd
import math
import multiprocessing as mp
import gprofiler as gprofiler #type: ignore
import warnings
import requests #type: ignore
import os
import json
import gseapy as gp #type: ignore
import matplotlib.pyplot as plt #type: ignore
import numpy as np

from tqdm import tqdm #type: ignore
from scipy.optimize import minimize
from scipy.stats import binom
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, ClusterWarning
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import pairwise_distances #type: ignore
from sklearn.utils import resample #type: ignore
from functools import partial

# Suppress specific warnings that are common and not actionable
warnings.filterwarnings("ignore", category=ClusterWarning, message=".*uncondensed distance matrix.*")
warnings.filterwarnings("ignore", message=".*All terms in the feature matrix.*were removed.*")


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
    return feature_matrix


# =============================================================================


def log_likelihood(p, data):
    """
    Compute the log-likelihood of a binomial distribution.
    
    Parameters
    ----------
    p : float
        Probability of success.
    data : list
        List of binary data.
    
    Returns
    -------
    log_likelihood : float
        Log-likelihood of the data.
    """
    n = len(data)
    k = np.sum(data)
    return -np.sum(binom.logpmf(data, n=1, p=p))


# =============================================================================


def filter_go_term_deviations(feature_matrix):
    """
    Filter out GO terms that deviate significantly from the mean probability of success,
    applying the False Discovery Rate (FDR) correction for multiple hypothesis testing.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Binary feature matrix where rows corresponds to gene IDs and columns represent GO term IDs.
        
    Returns
    -------
    feature_matrix : pd.DataFrame
        Filtered binary feature matrix.
    """
    data = feature_matrix.values.flatten()

    # Use the mean of the data as the initial guess for the probability of success
    initial_guess = np.mean(data)

    # Find the parameter that maximizes the log-likelihood
    result = minimize(log_likelihood, initial_guess, args=(data,), bounds=[(0, 1)])
    estimated_param = result.x[0]

    # Compute p-values for deviations
    deviations = abs(estimated_param - feature_matrix.mean())
    p_values = 2 * (1 - binom.cdf(deviations, n=1, p=estimated_param))  # Two-tailed test

    # Apply FDR correction
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    m = len(sorted_p_values)
    adjusted_p_values = np.zeros_like(sorted_p_values)

    for i, p in enumerate(sorted_p_values):
        adjusted_p_values[i] = p * m / (i + 1)

    # Map adjusted p-values back to original indices
    adjusted_p_values = adjusted_p_values[np.argsort(sorted_indices)]

    # Filter columns with significant deviations after correction
    significant_indices = np.where(adjusted_p_values < 0.05)[0]
    deviations = feature_matrix.columns[significant_indices]
    feature_matrix = feature_matrix.drop(columns=deviations)

    return feature_matrix


# =============================================================================


def get_descendants(Z, node_id, num_leaves,feature_matrix):
    """
    Extracts all descendant leaves of a given node in the dendrogram.
    
    Parameters:
    - Z: Linkage matrix
    - node_id: Index of the node (must be ≥ num_leaves since leaves are indexed from 0 to num_leaves-1)
    - num_leaves: Total number of leaves in the dendrogram

    Returns:
    - List of leaf indices that are descendants of the given node
    """
    #node = {id :node_id, value: 1}
    descendants = []
    stack = [node_id]

    while stack:
        node = stack.pop()
        if node < num_leaves:
            descendants.append(node)
        else:
            left_child = int(Z[node - num_leaves, 0])
            right_child = int(Z[node - num_leaves, 1])
            stack.extend([left_child, right_child])

    who = descendants

    return feature_matrix.index[who].tolist()


# =============================================================================


def calculate_node_number(dendro):
    """
    Calculate the number of nodes in the dendrogram.
    
    Parameters:
    - dendro: Dendrogram object
    
    Returns:
    - Number of nodes in the tree
    """
    return 2 * len(dendro['ivl']) - 1


# =============================================================================


def calculate_endotype_height_value(Z, dendro, feature_matrix):
    """
    Determine the height value to cut the dendrogram into clusters, based on the results from the Maximum Likelihood Estimation.
    
    Parameters:
    - Z: numpy.ndarray
        Linkage matrix
    - dendro: dict
        Dendrogram object (scipy.cluster.hierarchy dendrogram)
    - feature_matrix: pandas.core.frame.DataFrame
        Binary feature matrix (output of compute_feature_matrix() function)
        
    Returns:
    - height: numpy.float64
        Height value to cut the dendrogram into clusters
    """

    n_nodes = calculate_node_number(dendro)

    # Empty list to store the estimated probability of success for each node
    node_features = []

    for i in range(len(dendro['ivl']), n_nodes):
        ex = feature_matrix.loc[get_descendants(Z, i, len(feature_matrix), feature_matrix)]
        ex = filter_go_term_deviations(ex)
        if ex.empty:
            #warnings.warn(f"All terms in the feature matrix for node {i} were removed. Skipping this node.")
            node_features.append(np.nan)  # Append NaN for empty nodes to maintain the list length
            continue
        else:
            data = ex.values.flatten()
            # Use the mean of the data as the initial guess for the probability of success
            initial_guess = np.mean(data)

        # Use minimize to find the parameter that maximizes the log-likelihood
        result = minimize(log_likelihood, initial_guess, args=(data,), bounds=[(0, 1)])
        estimated_param = result.x[0]
        # Append the estimated parameter to the list
        node_features.append(estimated_param)
         
    node_features = pd.DataFrame(node_features)
    sumas = node_features.sum(axis=1).sort_values(ascending=False)
    index_node = sumas.index[0]

    # Return the top 10 height values
    top_10 = sumas[:10]

    return top_10


# =============================================================================


def endotype_dendrogram(Z, feature_matrix, height):
    """ 
    Plot the dendrogram with the clusters based on the height value computed from the Maximum Likelihood Estimation.
    
    Parameters:
    - Z: numpy.ndarray
        Linkage matrix
    - feature_matrix: pandas.core.frame.DataFrame
        Binary feature matrix (output of compute_feature_matrix() function)
    - height: numpy.float64
        Height value to cut the dendrogram into clusters (output of calculate_endotype_height_value() function)
        
    Returns:
    - cluster_assignments: dict
        Dictionary of cluster assignments for each gene in the feature matrix
    """
    # Draw the dendrogram
    plt.figure(figsize=(35, 15))
    dendro = dendrogram(Z, orientation='top', labels=feature_matrix.index, color_threshold=height)
    
    plt.xticks(fontsize=8, rotation=90)
    # Add a horizontal line at the height value
    plt.axhline(y=height, color='red', linestyle='--', label='Height value')
    plt.legend()
    plt.show()

    # Use fcluster to assign clusters based on the height threshold
    cluster_labels = fcluster(Z, t=height, criterion='distance')

    # Create a dictionary to store cluster assignments
    cluster_assignments_dict = {}
    for gene, cluster in zip(feature_matrix.index, cluster_labels):
        if cluster not in cluster_assignments_dict:
            cluster_assignments_dict[cluster] = []
        cluster_assignments_dict[cluster].append(gene)

    return cluster_assignments_dict

# =============================================================================

def perform_second_level_clustering(unassigned_genes,feature_matrix):
    unassigned_feat = feature_matrix.loc[unassigned_genes]

    distances_new = pairwise_distances(unassigned_feat, metric='hamming')
    Z_new = linkage(distances_new, method='complete')
    dendro_new = dendrogram(Z_new, orientation='right', labels=unassigned_feat.index,no_plot=True)

    new_height = calculate_endotype_height_value(Z_new, dendro_new, unassigned_feat)

    # Get the second level of clustering
    cluster_assignments_new = endotype_dendrogram(Z_new, unassigned_feat, new_height.iloc[0])
    return cluster_assignments_new

# =============================================================================
    
def recursive_endotyping(feature_matrix):
    """
    Perform recursive endotyping on the feature matrix.

    Parameters:
    - feature_matrix: pd.DataFrame
        Binary feature matrix where rows corresponds to gene IDs and columns represent GO term IDs.

    Returns:
    - cluster_assignments: dict
        Dictionary of cluster assignments for each gene in the feature matrix.
    """
    # Calculate the distance matrix
    distances = pairwise_distances(feature_matrix, metric='hamming')

    # Perform hierarchical clustering
    Z = linkage(distances, method='complete')

    # Create a dendrogram
    dendro = dendrogram(Z, orientation='right', labels=feature_matrix.index, no_plot=True)

    # Calculate the height value to cut the dendrogram into clusters
    height = calculate_endotype_height_value(Z, dendro, feature_matrix)

    # Get the cluster assignments
    cluster_assignments = endotype_dendrogram(Z, feature_matrix, height.iloc[0])

    # Check for unassigned genes
    # Get elements in cluster_assignments that are of length 1, and join them into a list
    unassigned_genes = [gene for cluster, genes in cluster_assignments.items() if len(genes) == 1 for gene in genes]
    iteration = 1

    cluster_assignments_dict = {}
    cluster_assignments_dict['It_1'] = cluster_assignments

    print(f"Iteration {iteration}: {len(unassigned_genes)} unassigned genes found.")
    print(f"Cluster numbers: {len(cluster_assignments)}")

    while len(unassigned_genes) > 1:
        # Perform second level clustering on unassigned genes
        cluster_assignments_new = perform_second_level_clustering(unassigned_genes, feature_matrix)
        unassigned_genes = [gene for cluster, genes in cluster_assignments_new.items() if len(genes) == 1 for gene in genes]
        cluster_assignments_dict[f'It_{iteration + 1}'] = cluster_assignments_new

        # If all clusters are of length 1, break the loop
        lentis = [len(genes) for genes in cluster_assignments_new.values()]
        sums = sum(lentis)
        if sums == len(cluster_assignments_new):
            print(f"Iteration {iteration}: {len(unassigned_genes)} unassigned genes found.")
            print(f"Cluster numbers: {len(cluster_assignments_new)}")
            print(f"Found no significant clustering at iteration {iteration + 1}, finishing process.")
            cluster_assignments_dict[f'It_{iteration + 1}'] = cluster_assignments_new
            break
        # Remove the 'C0' from cluster_assignments_dict
        print(f"Iteration {iteration}: {len(unassigned_genes)} unassigned genes found.")
        print(f"Cluster numbers: {len(cluster_assignments_new)}")
        iteration += 1
        # Update the cluster assignments dictionary

    # Remove clusters of length 1 from the final cluster assignments, except for the last iteration
    keys = list(cluster_assignments_dict.keys())
    for key in keys[:-1]:  # Exclude the last key
        cluster_assignments_dict[key] = {k: v for k, v in cluster_assignments_dict[key].items() if len(v) > 1}

    # Join elements of last iteration into a single list
    last_iteration = cluster_assignments_dict[keys[-1]]
    cluster_assignments_dict[keys[-1]] = {'Final_Cluster': [gene for genes in last_iteration.values() for gene in genes]}

    return cluster_assignments_dict