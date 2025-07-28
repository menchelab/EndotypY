import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import sys
import pickle
from tqdm import tqdm #type: ignore


def prep_rwr(G, r = 0.8):
    rwr_matrix = _colwise_rnd_walk_matrix(G, r)
    scaling_matrix = _create_scaling_matrix(G)
    d_ensembl_idx, d_idx_ensembl = _create_mapping_index_ensemblID(G)
    return rwr_matrix, scaling_matrix, d_ensembl_idx, d_idx_ensembl

#-------------------------------------------------------------


def _colwise_rnd_walk_matrix(G, r, a=1.0):
    """
    Compute the Random Walk Matrix (RWM) for a given graph G with teleportation
    probability a and damping factor r using the formula (I-r*M)^-1 where M is
    the column-wise normalized Markov matrix according to M = A D^{-1}

    Parameters:
        G: (networkx graph) input graph
        r: (float) damping factor/restart probability
        a: (float) teleportation probability

    Returns:
        W: (numpy array) RWM of the input graph G

    """

    # get the number of nodes in the graph G
    n = G.number_of_nodes()
    # get the adjacency matrix of graph G
    A = nx.adjacency_matrix(G, sorted(G.nodes()))
    A = sp.csc_matrix(A)

    # calculate the first scaling term
    factor = float((1 - a) / n)

    # prepare the second scaling term
    E = sp.csc_matrix(factor * np.ones([n, n]))
    A_tele = (a * A) + E

    # compute the column-wise normalized Markov matrix
    norm = linalg.norm(A_tele, ord=1, axis=0)
    M = A_tele / norm

    # mixture of Markov chains
    del A_tele
    del E

    U = sp.identity(n, dtype=int)
    H = (1 - r) * M
    H1 = U - H
    del U
    del M
    del H

    # compute the RWM using the formula (I-r*P)^-1
    print("Inverting the matrix takes time, please be patient...")
    W = r * np.linalg.inv(H1.toarray())

    return W


def _create_mapping_index_ensemblID(G):
    """
    Create the dictionaries to map genes' Ensembl IDs to indices and vice-versa.

    Parameter:
        G: (networkx graph) input graph

    Returns:
        d_entz_idx: dictionary Ensembl ID to index
        d_idx_entz: dictionare index to Ensembl ID
    """

    d_idx_ensembl = {}
    cc = 0
    for ensembl in sorted(G.nodes()):
        d_idx_ensembl[cc] = ensembl
        cc += 1
    d_ensembl_idx = dict((y, x) for x, y in d_idx_ensembl.items())

    return d_ensembl_idx, d_idx_ensembl


def _create_scaling_matrix(G):
    """
    Compute the diagonal matrix of the inverse degree of the nodes in graph G.

    Parameter:
        G: (networkx graph) input graph

    Returns:
        Dinvsqrt: diagonal matrix of the inverse degree of the nodes in graph G

    """

    n_nodes = G.number_of_nodes()
    Dinvsqrt = np.zeros([n_nodes, n_nodes])  # basically dimensions of your graph
    cc = 0
    for node in sorted(G.nodes()):
        kn = G.degree(node)
        Dinvsqrt[cc, cc] = np.sqrt(1.0 / kn)
        cc += 1

    return Dinvsqrt