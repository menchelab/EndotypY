import networkx as nx
import numpy as np



def rwr(G, seed_genes, scaling, W, Dinvsqrt, d_ensembl_idx, d_idx_ensembl, k=200):
    """
    Perform the random walk process (column-wise normalized, with scaling or not),
    find the visiting probability to each node and determine the top-k ranked genes,
    more in case the resulting module is not connected.
    The function writes the results (all ranked genes with their visiting probability)
    and outputs the connected disease module.

    Parameters:
        G:                  (networkx graph) input graph
        seed_genes:         (set) seed genes
        scaling:            (boolean) scales the visiting probabilities with the sqrt
                            of the degree of the corresponding node
        W:                  (numpy array) random walk matrix
        Dinvsqrt:           (numpy array) diagonal matrix of the inverse degree of the nodes
        d_ensembl_idx:      (dict) dictionary of gene Ensembl IDs and their corresponding index
        d_idx_ensembl:      (dict) dictionary of gene indices and their corresponding Ensembl ID
        
    Returns:
        connected_disease_module:   list of genes containing the seed genes and the top-k
                                    ranked genes that form a connected component on the
                                    interactome
    """

    n_nodes = G.number_of_nodes()

    p0 = np.zeros(n_nodes)

    # select only the seed genes are on the PPI network
    seed_genes_on_PPI = [gene for gene in seed_genes if gene in d_ensembl_idx.keys()]

    # initialize (with optional scaling) of the visiting probability vector
    for gene in seed_genes_on_PPI:
        if scaling == True:
            deg = G.degree(gene)
            p0[d_ensembl_idx[gene]] = 1 * np.sqrt(deg)
        else:
            p0[d_ensembl_idx[gene]] = 1.0

    # apply the RW operator on the visiting probability vector (with optional scaling)
    if scaling == True:
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

    # select the top X ranked genes
    disease_module = [g for g in seed_genes_on_PPI]
    disease_module.extend(rwr_rank_without_seed_genes[:k])

    # if the disease module is not connected, add the next ranked genes until it is connected
    i = 0
    subgraph = nx.subgraph(G, disease_module)
    while not nx.is_connected(subgraph):
        disease_module.append(rwr_rank_without_seed_genes[k + i])
        subgraph = nx.subgraph(G, disease_module)
        i += 1

    return disease_module, nx.subgraph(G, disease_module)