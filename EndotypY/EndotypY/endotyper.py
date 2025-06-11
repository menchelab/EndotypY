from .import_export import read_network_from_file, load_seed_set_from_file
from .utils import *
from .prepare_rwr import prepare_rwr
from .rwr import rwr

from tqdm import tqdm #type: ignore


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

    #for simplicity I do not show the option of
    # pubmed filtering here, but it can be added later
    def import_network(self, network_file:str):
        self.network = read_network_from_file(network_file)
        print("network imported successfully")

    def import_seeds(self, seeds_file:str):
        self.seeds = load_seed_set_from_file(seeds_file)
        print("seeds imported successfully")
        

    def prepare_rwr(self, r=0.8):
        print("Preparing RWR...")
        (self.rwr_matrix,
         self.scaling_matrix,
         self.ensembl_idx,
         self.idx_ensembl) = prepare_rwr(self.network, r)
        print("RWR matrix prepared successfully")

    def run_rwr(self, k=200, scaling=True):
        if self.network is None or self.seeds is None:
            raise ValueError("Network and seeds must be imported before running RWR.")
        
        (self.disease_module,
         self.connected_subgraph) = rwr(self.network, self.seeds, scaling,
                                       self.rwr_matrix, self.scaling_matrix,
                                       self.ensembl_idx, self.idx_ensembl, k)
        
        print(f"RWR completed. Disease module size: {len(self.disease_module)}")
    