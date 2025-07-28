import os
import pickle
from collections import defaultdict as dd
import matplotlib.pyplot as plt

from EndotypY.rwr import rwr_from_individual_genes


def run_seed_clustering(G, 
                        seed_genes, 
                        scaling, 
                        rwr_matrix, 
                        scaling_matrix, 
                        d_ensembl_idx, 
                        d_idx_ensembl, k):
    """
    Run the seed clustering process.
    This function computes the RWR for each seed gene, clusters them based on
    their neighborhoods, and plots the results.
    It also saves the clustered seed genes to a pickle file.
    
    Parameters:
        - G: NetworkX graph representing the connected protein-protein interaction network.
        - seed_genes: List of seed genes to be clustered.
        - scaling: Scaling matrix for the RWR algorithm.
        - rwr_matrix: RWR matrix for the graph.
        - scaling_matrix: Scaling matrix for the graph.
        - d_ensembl_idx: Dictionary mapping Ensembl IDs to indices.
        - d_idx_ensembl: Dictionary mapping indices to Ensembl IDs.
    """
    
    # RUN RWR FOR EACH SEED GENE
    d_rwr_individuals = rwr_from_individual_genes(
        seed_genes = seed_genes,
        G = G,
        scaling=scaling, 
        rwr_matrix=rwr_matrix,
        scaling_matrix=scaling_matrix, 
        d_ensembl_idx=d_ensembl_idx, 
        d_idx_ensembl=d_idx_ensembl,
        k = k
        )
         
    # TEST THE CLUSTERING OVER DIFFERENT NEIGHBORHOODS
    n_cluster_size_1 = []
    n_clusters = []
    tested_neighborhoods = list(range(10, 201, 10))
    
    for k in tested_neighborhoods:

        raw_clusters, clustered_seed_genes = _cluster_seed_genes(seed_genes, d_rwr_individuals, rwr_threshold=k)
        n_clusters.append(len(clustered_seed_genes))
        sizes = [len(c) for c in clustered_seed_genes]
        n_cluster_size_1.append(sizes.count(1))
     
    # FIND THE FIRST PLATEAU
    plateau_start = _find_first_plateau(n_clusters, tested_neighborhoods, seed_genes, d_rwr_individuals)
    
    # PLOT THE RESULTS
    plt.figure(figsize=(8, 6))
    plt.scatter(tested_neighborhoods, n_cluster_size_1, 
                label="Clusters of size 1", 
                color="royalblue", s=60, alpha=0.7, edgecolors="black", marker="o")

    plt.scatter(tested_neighborhoods, n_clusters, 
                label="Total clusters", 
                color="crimson", s=60, alpha=0.7, edgecolors="black", marker="s")

    # Add horizontal line marking the plateau
    if plateau_start[1] is not None:
        plt.axhline(y=plateau_start[1], color="green", linestyle="--", linewidth=1.5, 
                    label=f"Plateau at {plateau_start[1]} clusters")

    plt.xlabel("RWR-Neighborhood Size", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Clusters", fontsize=12, fontweight="bold")
    plt.title("Cluster Count vs. Neighborhood Size", fontsize=14, fontweight="bold", pad=10)
    plt.legend(fontsize=11, loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    
    # SAVE THE CLUSTERED SEEDS
    cluster_dict = {}
    for i, cluster in enumerate(plateau_start[2], 1):
        cluster_dict[f'cluster_seed_{i}'] = cluster

    return cluster_dict


#-------------------------------------------------------------


def _cluster_seed_genes(seed_genes, d_rwr_individuals, rwr_threshold):
    
    """
    Clusters seed genes based on their neighborhoods obtained via RWR, allowing
    overlapping clusters.
    Each seed gene is assigned to a cluster based on its neighborhood, and
    overlapping clusters are merged iteratively.
    
    Parameters:
        - seed_genes (list): List of seed genes.
        - d_rwr_individuals (dict): Dictionary containing the RWR expansion
                            for each gene.
        - rwr_threshold (int): Threshold for neighborhood size to consider
                            for clustering.
    
    Returns:
        - clusters (dict): Dictionary where keys are seed genes and values
                           are lists of overlapping seed genes.
        - merged_clusters (list): List of merged clusters after iteratively 
                           merging overlapping clusters.
    """
    
    clusters = dd(list)

    # Create clusters of seed genes based on the RWR neighborhoods
    for seed in seed_genes:
        neighborhood = list(d_rwr_individuals[seed][:rwr_threshold])
        cluster = [node for node in neighborhood if node in seed_genes]
        clusters[seed] = cluster

    merged_clusters = list(clusters.values())

    # Iteratively merge overlapping clusters until no further merges are possible
    merged = True
    while merged:
        merged = False
        for i in range(len(merged_clusters)):
            for j in range(i + 1, len(merged_clusters)):
                if set(merged_clusters[i]) & set(merged_clusters[j]):
                    # Merge and break to restart the process
                    merged_clusters[i] = list(set(merged_clusters[i] + merged_clusters[j]))
                    del merged_clusters[j]
                    merged = True
                    break
            if merged:
                break

    return clusters, merged_clusters



def _find_first_plateau(cluster_counts, 
                       neighborhood_sizes, 
                       seed_genes, 
                       d_rwr_individuals, 
                       min_plateau_length=3):
    """
    Finds the first plateau in the number of clusters that extends over at least 
    3 consecutive values.

    Parameters:
    - cluster_counts (list): List of cluster counts for each neighborhood size.
    - neighborhood_sizes (list): List of neighborhood sizes corresponding to the 
                                 cluster counts.
    - seed_genes (list): List of seed genes.
    - d_rwr_individuals (dict): Dictionary containing the RWR results for each gene.
    - min_plateau_length (int): Minimum number of consecutive same values to be 
                                considered a plateau.

    Returns:
    - tuple: (start_size, plateau_value, clustered_seed_genes) where start_size 
             is the neighborhood size where the plateau starts, plateau_value is 
             the number of clusters in the plateau, and clustered_seed_genes are 
             the clusters of seed genes. Returns (None, None, seed_genes) if no 
             plateau found.
    """
    
    if len(cluster_counts) < min_plateau_length:
        return None, None
        
    for i in range(len(cluster_counts) - min_plateau_length + 1):
        # Get the value at current position
        current_value = cluster_counts[i]
        
        # Check if the next min_plateau_length values are the same
        is_plateau = all(cluster_counts[i + j] == current_value 
                        for j in range(min_plateau_length))
        
        if is_plateau:
            raw_clusters, clustered_seed_genes = _cluster_seed_genes(
                seed_genes, 
                d_rwr_individuals, 
                rwr_threshold=neighborhood_sizes[i]
                )
            
            return neighborhood_sizes[i], current_value, clustered_seed_genes
            
    return None, None, seed_genes