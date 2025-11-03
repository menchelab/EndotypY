import os
import pickle
import networkx as nx
from collections import defaultdict as dd
import matplotlib.pyplot as plt #type: ignore
from tqdm import tqdm #type: ignore
from multiprocessing import Pool, cpu_count

from EndotypY.rwr import extract_connected_module


def run_seed_clustering(G, 
                        seed_genes,
                        d_rwr_individuals, k_max=200):
    """
    Run the seed clustering process.
    This function computes the RWR for each seed gene, clusters them based on
    their neighborhoods, and plots the results.
    
    Parameters:
        - G: NetworkX graph representing the connected protein-protein interaction network.
        - seed_genes: List of seed genes to be clustered.
        - scaling: Scaling matrix for the RWR algorithm.
        - rwr_matrix: RWR matrix for the graph.
        - scaling_matrix: Scaling matrix for the graph.
        - d_idx_ensembl: Dictionary mapping indices to Ensembl IDs.
        - k_max: Maximum neighborhood size to test.
    """
    # CHECK IF THE SEED GENES ARE ALREADY CONNECTED, SKIP THE CLUSTERING IF SO
    subgraph_seeds = nx.subgraph(G, seed_genes)
    if nx.is_connected(subgraph_seeds):
       print("The seed genes are already connected.")
       cluster_dict = {}
       cluster_dict['cluster_seed_1'] = seed_genes
       return cluster_dict
         
    # TEST THE CLUSTERING OVER DIFFERENT NEIGHBORHOODS
    all_cluster_results = []
    n_clusters = []

    #tested_neighborhoods = list(range(10, min(k_max + 1, 201), 10))
    #changed this to avoid empty lists
    step = 10
    if k_max < step:
        tested_neighborhoods = [k_max]
    else:
        tested_neighborhoods = list(range(step, k_max + 1, step))
        # Ensure the final k_max is included if it's not a multiple of the step
        if k_max not in tested_neighborhoods:
            tested_neighborhoods.append(k_max)
            
    print(f"Testing neighborhood sizes in parallel on {cpu_count()} cores...")
    
    # Prepare arguments for the parallel worker function
    args = [(G, seed_genes, d_rwr_individuals, k) for k in tested_neighborhoods]

    # Run the clustering in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.starmap(_cluster_seed_genes, args), total=len(tested_neighborhoods)))

    # Unpack the results from the parallel execution
    all_cluster_results = [res[1] for res in results]
    n_clusters = [len(clusters) for clusters in all_cluster_results]
    n_cluster_size_1 = [sum(1 for cluster in clusters if len(cluster) == 1) for clusters in all_cluster_results]

    # FIND THE FIRST PLATEAU
    plateau_index = _find_first_plateau(n_clusters)
    
    # Determine the final clusters to use
    if plateau_index is not None:
        final_clusters = all_cluster_results[plateau_index]
        optimal_k = tested_neighborhoods[plateau_index]
        print(f"Optimal neighborhood size found at k={optimal_k} with {n_clusters[plateau_index]} cluster(s).")
    else:
        final_clusters = all_cluster_results[-1] # Fallback to the last result
        optimal_k = tested_neighborhoods[-1]
        print(f"No stable plateau found. Using largest tested neighborhood size k={optimal_k}.")


    # PLOT THE RESULTS
    plt.figure(figsize=(8, 6))
    plt.scatter(tested_neighborhoods, n_cluster_size_1, 
                label="Clusters of size 1", 
                color="royalblue", s=60, alpha=0.7, edgecolors="black", marker="o")

    plt.scatter(tested_neighborhoods, n_clusters, 
                label="Total clusters", 
                color="crimson", s=60, alpha=0.7, edgecolors="black", marker="s")

    # Add horizontal line marking the plateau
    if plateau_index is not None:
        plateau_value = n_clusters[plateau_index]
        plt.axhline(y=plateau_value, color="green", linestyle="--", linewidth=1.5, 
                    label=f"Plateau at {plateau_value} clusters")

    plt.xlabel("RWR-Neighborhood Size", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Clusters", fontsize=12, fontweight="bold")
    plt.title("Cluster Count vs. Neighborhood Size", fontsize=14, fontweight="bold", pad=10)
    plt.legend(fontsize=11, loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # SAVE THE CLUSTERED SEEDS
    cluster_dict = {}
    for i, cluster in enumerate(final_clusters, 1):
        cluster_dict[f'cluster_seed_{i}'] = cluster

    return cluster_dict


#-------------------------------------------------------------
def _find_first_plateau(cluster_counts, min_plateau_length=3):
    """
    Finds the index of the start of the first plateau in a list of numbers.
    A plateau is defined as a sequence of at least `min_plateau_length`
    identical numbers.

    Parameters:
    - cluster_counts (list): List of cluster counts for each neighborhood size.
    - min_plateau_length (int): Minimum number of consecutive same values to be 
                                considered a plateau.

    Returns:
    - int or None: The index where the first plateau begins, or None if no
                   plateau is found.
    """
    
    if len(cluster_counts) < min_plateau_length:
        return None
        
    for i in range(len(cluster_counts) - min_plateau_length + 1):
        # Get the value at current position
        current_value = cluster_counts[i]
        
        # Check if the next min_plateau_length values are the same
        is_plateau = all(cluster_counts[i + j] == current_value 
                        for j in range(min_plateau_length))
        
        if is_plateau:
            return i # Return the index where the plateau starts
            
    return None


def _cluster_seed_genes(G, seed_genes, d_rwr_individuals, rwr_threshold):

    #get neighborhoods for each seed gene at the given rwr_threshold (k)
    neighborhoods = {}
    for seed in seed_genes:
        if seed in d_rwr_individuals:
            # Select the RWR results for the specific, current seed.
            rwr_results_for_seed = d_rwr_individuals[seed]
            seed_neighborhoods, _ = extract_connected_module(G, [seed], rwr_results_for_seed, k=rwr_threshold)
            neighborhoods[seed] = set(seed_neighborhoods)

    # 1. Create a new graph where nodes are the seed genes.
    # This graph will represent the relationships between seeds.
    seed_graph = nx.Graph()
    seed_graph.add_nodes_from(neighborhoods.keys())

    #Get a list of the seeds to iterate over unique pairs.
    seeds_to_compare = list(neighborhoods.keys())

    # 2. Loop through all unique pairs of seeds.
    #    The outer loop goes from the first to the second-to-last seed.
    for i in range(len(seeds_to_compare)):
        # The inner loop goes from the next seed to the last one.
        # This avoids comparing a seed to itself and avoids duplicate pairs (A,B vs B,A).
        for j in range(i + 1, len(seeds_to_compare)):
            seed1 = seeds_to_compare[i]
            seed2 = seeds_to_compare[j]

            # 4. Check for overlap using the efficient .isdisjoint() method.
            #    set1.isdisjoint(set2) is False if they have at least one common element.
            if not neighborhoods[seed1].isdisjoint(neighborhoods[seed2]):
                # if there is overlap, add an edge between the two seeds in the seed graph.
                seed_graph.add_edge(seed1, seed2)
            
    # 5. Find connected components in the seed graph.
    #    Each connected component represents a cluster of overlapping seeds.
    # Convert connected components to a dictionary with cluster_seed_X format keys
    final_clusters = [list(component) for component in nx.connected_components(seed_graph)]
    n_clusters = len(final_clusters)

    return n_clusters, final_clusters




    