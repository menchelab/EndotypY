import networkx as nx
import numpy as np
from tqdm import tqdm #type: ignore


def make_p0(G, seeds, scaling:bool):
    """Creates the initial probability vector (p0) for the Random Walk with Restart.

    This vector represents the starting probabilities for each node in the graph.
    Nodes in the seed set are assigned an initial probability, while all other
    nodes start with a probability of 0. The assignment can be scaled by node
    degree.

    Parameters
    ----------
    G : nx.Graph
        The graph on which the random walk will be performed.
    seeds : list or set
        A collection of seed nodes that will have non-zero initial probabilities.
    scaling : bool
        If True, the initial probability for seed nodes is scaled by the
        square root of their degree. If False, it is set to 1.

    Returns
    -------
    np.ndarray
        The initial probability vector (p0), where the i-th element corresponds
        to the i-th node in the sorted list of graph nodes.
    """
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
    """Performs a Random Walk with Restart (RWR) from a given set of seed genes.

    This function calculates the steady-state probability distribution of a random
    walker that starts from the specified seed genes and randomly moves through
    the graph, with a constant probability of restarting from the seed nodes.

    Parameters
    ----------
    G : nx.Graph
        The graph on which the random walk is performed.
    seed_genes : list
        A list of seed genes from which the random walk originates.
    scaling : bool
        If True, applies degree-based scaling to the initial probabilities
        and the final probability vector.
    rwr_matrix : np.ndarray
        The precomputed Random Walk with Restart matrix.
    scaling_matrix : np.ndarray
        The precomputed diagonal matrix used for scaling, typically based on
        node degrees.
    d_idx_ensembl : dict
        A dictionary mapping the integer index of a node in the matrix to its
        corresponding gene identifier.

    Returns
    -------
    dict
        A dictionary where keys are gene identifiers and values are their
        final visiting probabilities from the RWR.
    """

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
    """Executes Random Walk with Restart (RWR) for each seed gene individually.

    This function iterates through a list of seed genes and performs a separate
    RWR for each one. It's a wrapper around the main `rwr` function, designed
    to analyze the network neighborhood and influence of each seed gene on its own.

    A progress bar is displayed to track the process.

    Parameters
    ----------
    G : nx.Graph
        The graph on which the random walks are performed.
    seed_genes : list
        A list of seed genes, where each gene will be the starting point for an
        independent RWR.
    scaling : bool
        If True, applies degree-based scaling for each RWR calculation.
    rwr_matrix : np.ndarray
        The precomputed Random Walk with Restart matrix.
    scaling_matrix : np.ndarray
        The precomputed diagonal matrix used for scaling.
    d_idx_ensembl : dict
        A dictionary mapping node indices to their corresponding gene identifiers.

    Returns
    -------
    dict
        A dictionary where each key is a seed gene. The corresponding value is
        another dictionary containing the RWR visiting probabilities from that
        seed to all other nodes in the graph.
    """
    rwr_per_gene = {}

    for gene in tqdm(seed_genes, desc="Running RWR for seed genes"):
        rwr_per_gene[gene] = rwr(G, [gene], scaling,
                                 rwr_matrix, scaling_matrix,
                                 d_idx_ensembl)
    return rwr_per_gene




def extract_connected_module(G, seed_genes:list, rwr_results, k:int):
    """Extracts a connected subgraph (module) from the graph based on RWR results.

    This function first creates a core module by combining the initial seed genes
    with the top `k` genes from the Random Walk with Restart (RWR) results.
    It then iteratively expands this module by adding more top-ranked genes until
    the resulting subgraph is fully connected.

    Parameters
    ----------
    G : nx.Graph
        The background graph from which the module is extracted.
    seed_genes : list
        The initial list of seed genes for module creation.
    rwr_results : dict
        A dictionary mapping gene identifiers to their RWR visiting probabilities.
    k : int
        The number of top-ranked (non-seed) genes to include in the initial
        module.

    Returns
    -------
    tuple[list, nx.Graph]
        A tuple containing:
        - disease_module (list): The final list of genes in the connected module.
        - subgraph (nx.Graph): The connected subgraph corresponding to the
          disease_module.
    """
    # Sort the genes by their visiting probabilities in descending order
    d_gene_pvis_sorted = dict(sorted(rwr_results.items(), key=lambda item: item[1], reverse=True))

    #obtain the ranking without seed genes
    rwr_rank_without_seed_genes = [
        g for g in list(d_gene_pvis_sorted.keys()) if g not in seed_genes]

    # form disease module
    disease_module = [g for g in seed_genes if g in G.nodes()] #seed genes in the graph
    disease_module.extend(rwr_rank_without_seed_genes[:k]) #extending disease module with the top ranked genes up to k

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


    return disease_module, subgraph