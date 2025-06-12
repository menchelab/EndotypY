import networkx as nx
import pandas as pd
import math
import numpy as np
from tqdm import tqdm

# RWR Ranking function (Cut version of the RWR function outputting the RWR-ranking.)
def rwr_prov(G, seed_genes, scaling, W, Dinvsqrt, d_ensembl_idx, d_idx_ensembl):
    """
    Perform Random Walk with Restart (RWR) on a given graph G using the provided seed genes, and return the ranked list of genes based on their visiting probabilities.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph object representing the network of interest. 
    seed_genes : set
        Set of seed genes to initialize the random walk.
    scaling : bool
        Whether to scale the visiting probabilities by the square root of the degree of the nodes.
    W : numpy array
        Random walk matrix, output from prepare_rwr.colwise_rnd_walk_matrix
    Dinvsqrt : numpy array
        Diagonal matrix of the inverse square root of the degree of the nodes in the graph. Output from prepare_rwr.create_scaling_matrix
    d_ensembl_idx : dict
        Dictionary mapping gene Ensembl IDs to their corresponding indices in the graph. Element [0] of the output from prepare_rwr.create_mapping_index_ensemblID
    d_idx_ensembl : dict
        Dictionary mapping indices in the graph to their corresponding gene Ensembl IDs. Element [1] of the output from prepare_rwr.create_mapping_index_ensemblID

    Returns
    -------
    rwr_rank_without_seed_genes : list
        List of genes ranked by their visiting probabilities, excluding the seed genes. The genes are sorted in descending order of their visiting probabilities.
    """

    n_nodes = G.number_of_nodes()

    p0 = np.zeros(n_nodes)

    # select only the seed genes are on the PPI network
    seed_genes_on_PPI = [gene for gene in seed_genes if gene in d_ensembl_idx.keys()]

    # initialize (with optional scaling) of the visiting probability vector
    for gene in seed_genes_on_PPI:
        if scaling == 1:
            deg = G.degree(gene)
            p0[d_ensembl_idx[gene]] = 1 * np.sqrt(deg)
        else:
            p0[d_ensembl_idx[gene]] = 1.0

    # apply the RW operator on the visiting probability vector (with optional scaling)
    if scaling == 1:
        pinf = np.array(np.dot(Dinvsqrt, np.dot(W, p0)))
    else:
        pinf = np.dot(W, p0)

    # create dictionary of gene IDs and their corresponding visiting probability in sorted order
    d_gene_pvis_sorted = {}
    for p, x in sorted(zip(pinf, range(len(pinf))), reverse=True):
        d_gene_pvis_sorted[d_idx_ensembl[x]] = p / len(seed_genes_on_PPI)

    # obtain the ranking without seed genes
    rwr_rank_without_seed_genes = [
        g for g in list(d_gene_pvis_sorted.keys()) if g not in seed_genes
    ]

    return rwr_rank_without_seed_genes

def rwr_for_gene(gene, G, scaling, W, Dinvsqrt, d_ensembl_idx, d_idx_ensembl):
    """
    Wrapper function for rwr_prov to process a single gene as seed.
    """
    return rwr_prov(G, [gene], scaling, W, Dinvsqrt, d_ensembl_idx, d_idx_ensembl)

def calculate_top_genes(G, input_gene_list, scaling, W, Dinvsqrt, d_ensembl_idx, d_idx_ensembl, neighbor_percentage=0.01):
    """
    Calculate the top 1% genes for each gene in input_gene_list using Random Walk with Restart (RWR) algorithm.
    
    Parameters
    ----------
    G : nx.Graph object
        NetworkX graph object.
    input_gene_list : list of str
        List of genes.

    scaling : bool
        Whether to scale the visiting probabilities.
    W : numpy array
        Random walk matrix.
    Dinvsqrt : numpy array
        Diagonal matrix of the inverse degree of the nodes.
    d_ensembl_idx : dict
        Dictionary mapping gene Ensembl IDs to their indices.
    d_idx_ensembl : dict
        Dictionary mapping indices to gene Ensembl IDs.
    neighbor_percentage : float
        Percentage of top genes to identify.

    Returns
    -------
    top_genes : dict
        Dictionary of the desired top % genes for each gene in input_gene_list.
    """

    # Process each gene in input_gene_list
    rwr_scores_list = []
    for gene in tqdm(input_gene_list, desc="Perfoming gene-specifc RWR..."): # tqdm is used for progress bar
        # Run rwr_for_gene for each gene
        rwr_scores = rwr_for_gene(
            gene, G=G, scaling=scaling, W=W, Dinvsqrt=Dinvsqrt, 
            d_ensembl_idx=d_ensembl_idx, d_idx_ensembl=d_idx_ensembl
        )
        rwr_scores_list.append(rwr_scores)

    # Get a dictionary of the desired top % genes for each gene in input_gene_list
    top_genes = {
        gene: rwr_scores[:math.ceil(len(rwr_scores) * neighbor_percentage)]
        for gene, rwr_scores in zip(input_gene_list, rwr_scores_list)
    }

    # Remove nan from the list
    for gene in top_genes:
        top_genes[gene] = [x for x in top_genes[gene] if isinstance(x, str)]
    
    return top_genes