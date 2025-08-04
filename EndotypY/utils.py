# Simple functions for gene ID conversion using MyGeneInfo
import mygene #type: ignore
import logging
logging.getLogger("biothings.client").setLevel(logging.ERROR) #suppress MyGeneInfo warnings
mg = mygene.MyGeneInfo()
import multiprocessing as mp
import gseapy as gp #type: ignore

def convert_entrez_to_symbols(gene_ids):
    results = mg.querymany(gene_ids, scopes='entrezgene', fields='symbol', species='human,', verbose=False)
    symbols = [x['symbol'] for x in results if 'symbol' in x]
    return symbols

def convert_symbols_to_entrez(gene_ids):
    results = mg.querymany(gene_ids, scopes='symbol', fields='entrezgene', species='human', verbose=False)
    symbols = [x['entrezgene'] for x in results if 'entrezgene' in x]
    return symbols

# def convert_top_genes_to_symbols(top_genes):
#     with mp.Pool(mp.cpu_count() - 2) as pool:
#         results = pool.map(convert_entrez_to_symbols, top_genes.values())
#     return dict(zip(top_genes.keys(), results))


def download_enrichr_library(enrichr_lib:str, organism='Human'):
    library = gp.get_library(name=enrichr_lib,organism=organism)
    return library