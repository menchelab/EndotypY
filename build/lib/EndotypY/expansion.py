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
from collections import defaultdict

from EndotypY.rwr import rwr_from_individual_genes
from EndotypY.utils import convert_entrez_to_symbols



def calculate_top_genes(G, input_gene_list,
                        rwr_matrix, scaling_matrix, d_idx_ensembl,
                        neighbor_percentage=1, scaling=True):
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

    # perform gene specific RWR
    individual_rwr_results = rwr_from_individual_genes(
                            seed_genes=input_gene_list,
                            G=G,
                            scaling=scaling,
                            rwr_matrix=rwr_matrix,
                            scaling_matrix=scaling_matrix,
                            d_idx_ensembl=d_idx_ensembl)

    # Convert probability dictionaries to sorted gene lists
    individual_rwr_ranked = {}
    for gene in input_gene_list:
        if gene in individual_rwr_results:
            # Sort genes by their visiting probabilities in descending order
            sorted_genes = sorted(individual_rwr_results[gene].items(), 
                                key=lambda item: item[1], reverse=True)
            # Extract just the gene names (not the probabilities)
            individual_rwr_ranked[gene] = [gene_name for gene_name, _ in sorted_genes]
        else:
            individual_rwr_ranked[gene] = []

    # Get a dictionary of the desired top % genes for each gene in input_gene_list
    neighbor_percentage = neighbor_percentage / 100
    top_genes = {}
    for gene in input_gene_list:
        top_genes[gene] = individual_rwr_ranked[gene][:math.ceil(len(individual_rwr_ranked[gene]) * neighbor_percentage)]

    return top_genes



# =============================================================================

def get_GSEA_significant_terms(gene_list,library, sig_threshold, organism='human',background=None):
    """
    Perform Gene Set Enrichment Analysis (GSEA) using gseapy's enrichr implementarion and return significant terms.

    Parameters
    ----------
    gene_list : list
        List of genes to analyze.
    library : gene set enrichment library
        Dictionary of term - gene associations (output of gp.get_library() function).
    background : list
        Background gene list (default: None).

    Returns
    -------
    significant_terms : list
        List of significant terms from GSEA.
    """
    enr_bg = gp.enrich(gene_list=gene_list,
                       gene_sets=library,
                       background=background,
                       outdir=None
                      )
    
    # Filter results for significant terms, if any

    if enr_bg.results.empty:
        #print("No significant terms found.")
        return []
    else:
        filtered_results = enr_bg.results[enr_bg.results['Adjusted P-value'] < sig_threshold]
        significant_terms = filtered_results['Term'].tolist()
        return significant_terms


def convert_gs_lib_to_dict(go_terms_dict):
    """
    Converts a dictionary of enrichment terms from the format {term: [gene1, gene2, ...]} to
    the format {gene: [term1, term2, ...]}.

    Parameters
    ----------
    go_terms_dict : dict or list
        Dictionary where keys are enrichment term IDs and values are lists of gene symbols associated with those terms.
        Or a list from gseapy that needs to be handled differently.
    Returns
    -------
    go_terms : dict
        Dictionary where keys are gene symbols and values are lists of enrichment term IDs associated with those genes.
    """
    go_terms = defaultdict(list)
    if isinstance(go_terms_dict, dict):
        for term, genes in go_terms_dict.items():
            for gene in genes:
                go_terms[gene].append(term)
    return dict(go_terms)


# =============================================================================

def get_gene_and_neighborhood_enrichment_terms(gene,top_genes,term_library,sig_threshold):
    """
    Get the Gene Ontology (GO) terms for a given gene symbol and its neighbors.

    Parameters
    ----------
    gene : str
        Gene symbol (HGNC) of the gene.
    top_genes : dict
        Dictionary of the desired top % genes for each gene in input_gene_list (output of calculate_top_genes() function).
    term_library : dict
        Dictionary of term - gene associations (output of gp.get_library() function).
    
    Returns
    -------
    all_go_terms : list
        List of unique GO term IDs for the gene and its neighbors.
    """
    # Format term_library from gseapy output to a dictionary of gene - term associations
    term_library_ori = convert_gs_lib_to_dict(term_library)

    # extract gene to to process and convert to gene symbol if necessary
    # if gene.isdigit():
    #     gene_symbol = convert_entrez_to_symbols([gene])#[0]
    # else:
    #     gene_symbol = gene

    #extract neighborhood and convert to gene symbols if necessary
    if all(str(item).isdigit() for item in top_genes[gene]):
        neighborhood = convert_entrez_to_symbols(top_genes[gene])
    else:
        neighborhood = top_genes[gene]
        
    # Get original terms of the gene
    if len(gene) > 0:
        # If the gene symbol is found in the term library, get its terms
        if gene in term_library_ori.keys():
            ori_go_terms = term_library_ori[gene]
        else:
            ori_go_terms = []
    else:
        # If the gene symbol is not found, return an empty list
        warnings.warn(f"Gene symbol {gene} not found in term library.")
        return []
    
    # convert term_library to a list if it is not already
    if not isinstance(term_library, list):
        term_list = [term_library]

    # Run GSEA for the neighbors
    neighbor_go_terms_list = get_GSEA_significant_terms(neighborhood,
                                                        library=term_list,
                                                        sig_threshold=sig_threshold,
                                                        organism='human',
                                                        background=None)

    # Join ori_go_terms with the neighbor_go_terms_list
    neighbor_go_terms = ori_go_terms + neighbor_go_terms_list

    # Remove duplicates
    all_go_terms = list(set(neighbor_go_terms))

    return all_go_terms


# =============================================================================


def process_gene(gene, top_genes, term_library, sig_threshold):
    """
    Process a single gene to retrieve its GO terms.

    Parameters
    ----------
    gene : str
        Gene symbol to process.
    top_genes : dict
        Dictionary of top genes.
    term_library : dict
        Dictionary of term - gene associations.

    Returns
    -------
    tuple
        A tuple containing the gene symbol and its associated GO terms.
    """
    #convert gene to gene symbol if necessary
    # if gene.isdigit():
    #     gene = convert_entrez_to_symbols([gene])

    return gene, get_gene_and_neighborhood_enrichment_terms(gene, top_genes=top_genes, term_library=term_library, sig_threshold=sig_threshold)

def get_module_neighborhood_terms_dict(top_genes, term_library, sig_threshold):
    """
    Get the Gene Ontology (GO) terms for all genes in a given disease core module using parallel processing.

    Parameters
    ----------
    top_genes : dict
        Dictionary of the desired top % genes for each gene in input_gene_list (output of calculate_top_genes() function).
    term_library : dict
        Dictionary of term - gene associations (output of gp.get_library() function).
    n_cores : int, optional
        Number of cores to use for parallel processing (default: mp.cpu_count() - 2).

    Returns
    -------
    go_terms_dict : dict
        Dictionary of unique GO term IDs for each gene in the disease core module.
    """
    # Use multiprocessing to process genes in parallel
    n_cores=mp.cpu_count() - 2

    with mp.Pool(n_cores) as pool:
        results = pool.starmap(process_gene, [(gene, top_genes, term_library, sig_threshold) for gene in top_genes.keys()])

    # Combine results into a dictionary
    go_terms_dict = dict(results)
    return go_terms_dict

