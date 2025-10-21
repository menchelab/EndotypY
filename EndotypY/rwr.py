import networkx as nx
import numpy as np
from tqdm import tqdm #type: ignore


def make_p0(G, seeds, scaling:bool):
    nodelist = sorted(G.nodes())
    p0 = np.zeros(len(nodelist))

    for idx, node in enumerate(nodelist):
        if node in seeds:
            if scaling:
                deg = G.degree(node)
                p0[idx] = 1 * np.sqrt(deg)
            else:
                p0[idx] = 1
    return p0

def rwr(G, seed_genes, scaling, rwr_matrix, scaling_matrix, d_idx_ensembl):

    p0 = make_p0(G, seed_genes, scaling)

    # apply the RW operator on the visiting probability vector (with optional scaling)
    if scaling:
        pinf = np.array(np.dot(scaling_matrix, np.dot(rwr_matrix, p0)))
    else:
        pinf = np.dot(rwr_matrix, p0)

    # Map probabilities to gene names
    gene_probabilities = {d_idx_ensembl[i]: prob for i, prob in enumerate(pinf)}

    return gene_probabilities


def rwr_from_individual_genes(G, seed_genes, scaling:bool,
                              rwr_matrix, scaling_matrix, d_idx_ensembl):
    """
    Run RWR starting from every single gene in seed_genes
    and store the results.
    """
    rwr_per_gene = {}

    for gene in tqdm(seed_genes, desc="Running RWR for seed genes"):
        rwr_per_gene[gene] = rwr(G, [gene], scaling,
                                 rwr_matrix, scaling_matrix,
                                 d_idx_ensembl)
    return rwr_per_gene

def extract_connected_module(G, seed_genes, rwr_results, k:int, check_connectivity:bool=True):
    # Sort the genes by their visiting probabilities in descending order
    d_gene_pvis_sorted = dict(sorted(rwr_results.items(), key=lambda item: item[1], reverse=True))

    #obtain the ranking without seed genes
    rwr_rank_without_seed_genes = [
        g for g in list(d_gene_pvis_sorted.keys()) if g not in seed_genes]

    # form disease module
    disease_module = [g for g in seed_genes if g in G.nodes()] #seed genes in the graph
    disease_module.extend(rwr_rank_without_seed_genes[:k]) #extending disease module with the top ranked genes up to k

    if check_connectivity == False:
        # Return the largest connected component of the disease module
        subgraph = nx.subgraph(G, disease_module)
        largest_cc = max(nx.connected_components(subgraph), key=len)
        disease_module = list(largest_cc)
        subgraph = nx.subgraph(G, disease_module)

        return disease_module, subgraph

    else:
        #if the disease module is not connected, add the next ranked genes until it is connected
        i = 0
        subgraph = nx.subgraph(G, disease_module)
        
        # Add protection against infinite loop
        max_iterations = len(rwr_rank_without_seed_genes) - k
        while not nx.is_connected(subgraph) and i < max_iterations:
            if k + i < len(rwr_rank_without_seed_genes):
                disease_module.append(rwr_rank_without_seed_genes[k + i])
                subgraph = nx.subgraph(G, disease_module)
            i += 1

        return disease_module, nx.subgraph(G, disease_module)


# def extract_connected_module(G, seed_genes, rwr_results, k:int):
#     # Sort the genes by their visiting probabilities in descending order
#     d_gene_pvis_sorted = dict(sorted(rwr_results.items(), key=lambda item: item[1], reverse=True))

#     #obtain the ranking without seed genes
#     rwr_rank_without_seed_genes = [
#         g for g in list(d_gene_pvis_sorted.keys()) if g not in seed_genes]

#     # form disease module
#     disease_module = [g for g in seed_genes if g in G.nodes()] #seed genes in the graph

#         # Iteratively add top-ranked genes one by one, up to k
#     for i in range(k):
#         # Add the next best gene
#         if i < len(rwr_rank_without_seed_genes):
#             disease_module.append(rwr_rank_without_seed_genes[i])
#         else:
#             # Stop if we run out of ranked genes to add
#             break

#         subgraph = G.subgraph(disease_module)
        
#         # If connected, we have found the smallest module and can return it
#         if nx.is_connected(subgraph):
#             return list(subgraph.nodes()), subgraph
        
#         #still no one component found, return lcc
#         else:
#             largest_cc = max(nx.connected_components(subgraph), key=len)
#             return list(largest_cc), G.subgraph(largest_cc)

