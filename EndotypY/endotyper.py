from .import_export import read_network_from_file, load_seed_set_from_file
from .prepare_rwr import prep_rwr
from .rwr import rwr, extract_connected_module, rwr_from_individual_genes
from .seed_clusters import run_seed_clustering
from .expansion import calculate_top_genes, get_module_neighborhood_terms_dict
from .utils import download_enrichr_library
from .clustering import compute_feature_matrix, recursive_endotyping
from .kl_clustering import kl_clustering_endotypes
from .visualization import plot_endotype, plot_multiple_endotypes

from typing import Literal
class Endotyper:
    """
    Endotyper class for endotyping analysis.

    This class provides methods to perform endotyping analysis using a random walk approach.
    It includes methods for reading networks, preparing the random walk matrix, and performing the endotyping analysis.

    Attributes:
    ----------
    network_file : str | Path
        The path to the input network file.
    r : float
        The damping factor for the random walk.
    """

    def __init__(self):
        self.network = None
        self.seeds = None
        self.rwr_matrix = None
        self.scaling_matrix = None
        self.idx_ensembl = None
        self.disease_module = None
        self.connected_subgraph = None
        self.seed_clustering_neighborhoods = None
        self.seed_clusters = None
        self.expanded_neighborhoods = None
        self.neighborhood_annotation = None
        self.feature_matrix = None
        self.endotypes = None

    def import_network(self, network_file:str):
        """Imports a network from a file.

            Args:
                network_file (str): Path to the network file.  Supported formats are:
                -'.txt' or '.tsv' or '.csv' with two columns representing edges, tab-separated.

            Returns:
                self: The Endotyper object.

            Notes:
                - Lines that start with '#' will be ignored.
                - Self-loops are eliminated in the last filtering step
            """
        self.network = read_network_from_file(network_file)
        print("network imported successfully")
        print(f"> Network contains {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges")
        return self

    def import_seeds(self, seeds_file:str):
        """Imports seeds from a file and sets them as the seeds for the object.

        Args:
            seeds_file (str): The path to the seeds file.

        Returns:
            self: The Endotyper object.

        Notes:
            - The seeds file should contain a list of seed genes, one per line.
            - Alternative formats for the seeds file is tab separated entries on first line of file.
        """
        self.seeds = load_seed_set_from_file(seeds_file)
        print("seeds imported successfully")
        return self
        

    def prepare_rwr(self, r=0.8):
        """Prepares the Random Walk with Restart (RWR) matrix.

            This function computes the RWR matrix based on the network and restart probability,
            using the formula (I-r*M)^-1 where M is
            the column-wise normalized Markov matrix according to M = A D^{-1}.

            To provide the option of scaling the visiting probabilities, a scaling matrix is also created,
            which is the diagonal matrix of the inverse degree of the nodes in graph G.

            Args:
                r (float, optional): Damping factor/restart probability. Defaults to 0.8.

            Returns:
                self: Returns the Endotyper object with the RWR matrix, scaling matrix,
                      and index to ensembl mapping stored as attributes.
            """
        print("Preparing RWR...")

        # Add check to eliminate seeds that are not present in network
        not_present = [seed for seed in self.seeds if seed not in self.network.nodes()]
        if len(not_present) > 0:
            self.seeds = [seed for seed in self.seeds if seed in self.network.nodes()]
            print(f"Warning: The following seed genes are not present in the network and will be ignored: ")
            print(not_present)

        (self.rwr_matrix,
         self.scaling_matrix,
         self.idx_ensembl) = prep_rwr(self.network, r)
        print("RWR matrix prepared successfully")
        return self


    def explore_seed_clusters(self, scaling=True, k=200):

        """
        Run the seed clustering process.
        This function computes the RWR for each seed gene, clusters them based on
        their neighborhoods, and plots the results.

        Args:
            - k_max: Maximum neighborhood size to test.
            - scaling: Whether to apply scaling to the RWR.

        """
        # RUN RWR FOR EACH SEED GENE and save to not recompute if not needed
        
        if self.seed_clustering_neighborhoods is None:
            self.seed_clustering_neighborhoods = rwr_from_individual_genes(
                seed_genes = self.seeds,
                G = self.network,
                scaling=scaling, 
                rwr_matrix=self.rwr_matrix,
                scaling_matrix=self.scaling_matrix, 
                d_idx_ensembl=self.idx_ensembl)

        self.seed_clusters = run_seed_clustering(self.network, 
                        self.seeds,
                        self.seed_clustering_neighborhoods,
                        k_max=k)

        print(f"{len(self.seed_clusters)} Seed clusters identified")
        return self.seed_clusters


    def extract_disease_module(self, seed_cluster_id:int = None, scaling=True, k=200):
        
        if seed_cluster_id is None and self.seed_clusters is not None:
            print(f"No seed cluster ID provided, defaulting to largest seed cluster")
            seeds = max(self.seed_clusters.values(), key=len)

        elif seed_cluster_id is not None and len(self.seed_clusters[f'cluster_seed_{seed_cluster_id}']) > 0:
            print(f"Using seed cluster ID {seed_cluster_id} with {len(self.seed_clusters[f'cluster_seed_{seed_cluster_id}'])} seeds")
            seeds = self.seed_clusters[f'cluster_seed_{seed_cluster_id}']

        else:
            print(f"No seed cluster ID provided, using all seeds")
            seeds = self.seeds
    
        rwr_results = rwr(self.network, seeds, scaling,
                                       self.rwr_matrix, self.scaling_matrix,
                                       self.idx_ensembl)
        
        self.disease_module, self.connected_subgraph = extract_connected_module(self.network, seeds,
        #                                                   rwr_results, k=k, check_connectivity=True)
                                                            rwr_results, k=k)

        print(f"Connected module extracted with {self.connected_subgraph.number_of_nodes()} nodes and {self.connected_subgraph.number_of_edges()} edges")
        return self
    

    def define_local_neighborhood(self, neighbor_percentage=1, scaling=True):

        """
        Run RWR starting from every single gene in seed_genes
        and extract the top % genes from the visiting probabilities around each seed gene.

        Args:
            neighbor_percentage (int): Percentage of top genes to identify.
            scaling (bool): Whether to apply scaling to the RWR.

        """
  

        self.expanded_neighborhoods = calculate_top_genes(self.network,
                                                            self.disease_module,
                                                            self.rwr_matrix,
                                                            self.scaling_matrix,
                                                            self.idx_ensembl,
                                                            neighbor_percentage,
                                                            scaling=scaling)
        return self
    

    def annotate_local_neighborhood(self, enrichr_lib:str, organism='Human', sig_threshold=0.01):

        """ Get the Gene Ontology (GO) terms for a given gene and its RWR defined neighbors.
        This function uses the Enrichr library to perform Gene Set Enrichment Analysis (GSEA)
        and returns significant terms for the expanded neighborhood of genes (significance threshold = p-value for enrichment).

        Args:
            enrichr_lib (str): The name of the Enrichr library to use for GSEA.
            organism (str): The organism for which the GSEA is performed. Default is 'Human'.
            sig_threshold (float): The significance threshold for the GSEA results. Default is 0.01.

        """

        term_library = download_enrichr_library(enrichr_lib, organism=organism)
        top_genes = self.expanded_neighborhoods

        self.neighborhood_annotation = get_module_neighborhood_terms_dict(top_genes,
                                                                          term_library,
                                                                          sig_threshold = sig_threshold)
        return self
    
    def define_endotypes(self):

        """
        Define endotypes based on the annotated local neighborhoods.
        This function computes the feature matrix from the neighborhood annotations (binary matrix) that describes which
        enrichment terms are present for each gene based on the enrichment of the gene +local neighborhood.
        The feature matrix is a binary matrix where rows are genes and columns are enrichment terms.
        Each entry is 1 if the term is present for the gene, and 0 otherwise.

        It then performs recursive clustering to identify endotypes.

        Returns:
            self: The Endotyper object with the endotypes defined.
        """
        # make feature matrix
        self.feature_matrix = compute_feature_matrix(self.neighborhood_annotation)
        # recursive clustering
        self.endotypes = recursive_endotyping(self.feature_matrix)

        #print(f"Found {len(self.endotypes)} endotypes")
    
        return self
    
    def define_kl_endotypes(self,distance_metric: str = 'hamming', linkage_method: str = 'complete',alpha: float = 0.05):

        """
        Define endotypes based on KL divergence.
        This function computes the feature matrix from the neighborhood annotations (binary matrix) that describes which
        enrichment terms are present for each gene based on the enrichment of the gene +local neighborhood.
        The feature matrix is a binary matrix where rows are genes and columns are enrichment terms.
        Each entry is 1 if the term is present for the gene, and 0 otherwise.

        It then performs kl divergence clustering to identify endotypes.
        Returns:
            self: The Endotyper object with the endotypes defined.
        """
        # make feature matrix
        self.feature_matrix = compute_feature_matrix(self.neighborhood_annotation)
        # recursive clustering
        self.endotypes = kl_clustering_endotypes(data = self.feature_matrix,
                                                  distance_metric=distance_metric,
                                                  linkage_method=linkage_method,
                                                  alpha=alpha)

        return self

        #print(f"Found {len(self.endotypes)} endotypes")
    

    #_TYPES = Literal['degree', 'betweenness']

    def plot_endotype(self, iteration: int, cluster_id: int= None,
                        node_size: list = ['degree', 'betweenness'],
                        path_length: int = 2):
    
        """Plots the endotype network for a given iteration and cluster.
        This function generates a network plot visualizing the identified endotype,
        highlighting seed genes, endotype genes, and connecting genes within the larger network.
        Args:
            iteration (int): The iteration number of the endotyping clustering process.
            cluster_id (int, optional): The ID of the cluster to plot. If None, defaults to the first cluster. Defaults to None.
            node_size (list, optional): A list of network measures to use for node sizing.
                Defaults to ['degree', 'betweenness'].
            path_length (int, optional): The path length to consider when connecting endotype genes. Defaults to 2.
        """
    
        if cluster_id is None:
            print(f"No cluster ID provided, defaulting to first cluster")
            cluster_id = list(self.endotypes[f'It_{iteration}'].keys())[0]
        
        if iteration == len(self.endotypes):
            plot_endotype(self.endotypes[f'It_{iteration}']['Final_Cluster'],self.network,
            self.seeds,node_size=node_size,path_length=path_length)
        
        else:
            plot_endotype(self.endotypes[f'It_{iteration}'][cluster_id],self.network,
                                    self.seeds,node_size=node_size,path_length=path_length)


    def plot_multiple_endotypes(self, node_size: list = ['degree', 'betweenness'], layout: str = 'spring', path_length: int = 2):

        """Plots multiple endotypes on the network.
            This function iterates through the endotypes dictionary, combining endotypes from different iterations into a single dictionary.
            It then calls the `plot_multiple_endotypes` function to visualize these combined endotypes on the network.
            Args:
                node_size (list, optional): network measures to use for node sizing.
                Defaults to ['degree', 'betweenness'].
                layout (str, optional): The layout algorithm to use for the network plot. Defaults to 'spring'.
                path_length (int, optional): The path length to use for shortest path calculations. Defaults to 2.
            """
        
        #combine endotypes from different iterations into a single dictionary
        endotype_clustering_joined = {}
        for iteration_key in self.endotypes.keys():
            for endotype in self.endotypes[iteration_key].keys():
             endotype_clustering_joined[f"{iteration_key}_{endotype}"] = self.endotypes[iteration_key][endotype]

        plot_multiple_endotypes(endotype_clustering_joined, 
                                      self.network, self.seeds, 
                                      size_height=8, size_width=14, 
                                      node_size= node_size, 
                                      path_length=path_length,
                                      limit_lcc=True,
                                      layout=layout,
                                      #layout_seed=2025
                                      )














