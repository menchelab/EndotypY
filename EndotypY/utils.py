# Simple functions for gene ID conversion using MyGeneInfo
import mygene #type: ignore
import logging
logging.getLogger("biothings.client").setLevel(logging.ERROR) #suppress MyGeneInfo warnings
mg = mygene.MyGeneInfo()
import multiprocessing as mp
import gseapy as gp #type: ignore
import pickle
from pathlib import Path

def convert_entrez_to_symbols(gene_ids):
    results = mg.querymany(gene_ids, scopes='entrezgene', fields='symbol', species='human,', verbose=False)
    symbols = [x['symbol'] for x in results if 'symbol' in x]
    return symbols

def convert_symbols_to_entrez(gene_ids):
    results = mg.querymany(gene_ids, scopes='symbol', fields='entrezgene', species='human', verbose=False)
    symbols = [x['entrezgene'] for x in results if 'entrezgene' in x]
    return symbols


def download_enrichr_library(enrichr_lib: str, organism='Human', force_download=False):
    """Downloads and caches Enrichr library locally."""
    # Use the package installation directory
    cache_dir = Path(__file__).parent / '.enrichr_cache'
    cache_dir.mkdir(exist_ok=True)
    
    # Create filename from library name and organism
    cache_file = cache_dir / f"{enrichr_lib}_{organism}.pkl"
    
    if cache_file.exists() and not force_download:
        print(f"Loading {enrichr_lib} term library from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Downloading {enrichr_lib} term library for {organism}...")
    library = gp.get_library(name=enrichr_lib, organism=organism)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(library, f)
    
    return library