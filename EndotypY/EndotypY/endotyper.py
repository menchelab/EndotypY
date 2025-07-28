from .import_export import read_network_from_file, load_seed_set_from_file
from .prepare_rwr import prep_rwr
from .rwr import rwr
from .seed_clusters import run_seed_clustering


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
    email : str, optional
        Email address for NCBI API requests (required if filtering by PubMed references).
    """

    def __init__(self):
        self.network = None
        self.seeds = None
        self.rwr_matrix = None
        self.scaling_matrix = None
        self.ensembl_idx = None
        self.idx_ensembl = None
        self.disease_module = None
        self.connected_subgraph = None
        self.seed_clusters = None

    def import_network(self, network_file:str):
        self.network = read_network_from_file(network_file)
        print("network imported successfully")
        print(f"> Network contains {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges")
        return self

    def import_seeds(self, seeds_file:str):
        self.seeds = load_seed_set_from_file(seeds_file)
        print("seeds imported successfully")
        return self
        

    def prepare_rwr(self, r=0.8):
        print("Preparing RWR...")
        (self.rwr_matrix,
         self.scaling_matrix,
         self.ensembl_idx,
         self.idx_ensembl) = prep_rwr(self.network, r)
        print("RWR matrix prepared successfully")
        return self

    # def run_rwr(self,r=0.8, k=200, scaling=True):
    #     if self.network is None or self.seeds is None:
    #         raise ValueError("Network and seeds must be imported before running RWR.")
        
    #     if self.rwr_matrix is None or self.scaling_matrix is None:
    #         self.prepare_rwr(r)
        
    #     (self.disease_module,
    #      self.connected_subgraph) = rwr(self.network, self.seeds, scaling,
    #                                    self.rwr_matrix, self.scaling_matrix,
    #                                    self.ensembl_idx, self.idx_ensembl, k)
        
    #     print(f"RWR completed. Disease module size: {len(self.disease_module)}")
    #     return self

    #--------------------------------------------------------------

    def explore_seed_clusters(self, scaling=True, k=200):
        self.seed_clusters = run_seed_clustering(self.network, 
                        self.seeds, 
                        scaling, 
                        self.rwr_matrix, 
                        self.scaling_matrix, 
                        self.ensembl_idx, self.idx_ensembl, k)

        print(f"{len(self.seed_clusters)} Seed clusters identified")
        return self.seed_clusters


    
    def extract_disease_module(self, seed_cluster_id:int = None, scaling=True, k=200):
        if self.seed_clusters is None:
            raise ValueError("Seed clusters must be identified before extracting the connected subgraph.")
        
        if seed_cluster_id is None and self.seed_clusters is not None:
            print(f"No seed cluster ID provided, defaulting to largest seed cluster")
            seeds = max(self.seed_clusters.values(), key=len)

        elif seed_cluster_id is not None and len(self.seed_clusters[f'cluster_seed_{seed_cluster_id}']) > 0:
            print(f"Using seed cluster ID {seed_cluster_id} with {len(self.seed_clusters[f'cluster_seed_{seed_cluster_id}'])} seeds")
            seeds = self.seed_clusters[f'cluster_seed_{seed_cluster_id}']

        else:
            print(f"No seed cluster ID provided, using all seeds")
            seeds = self.seeds
    
        (self.disease_module,
         self.connected_subgraph) = rwr(self.network, seeds, scaling,
                                       self.rwr_matrix, self.scaling_matrix,
                                       self.ensembl_idx, self.idx_ensembl, k)
        print(f"Connected module extracted with {self.connected_subgraph.number_of_nodes()} nodes and {self.connected_subgraph.number_of_edges()} edges")
        return self
    









