import networkx as nx
from Bio import Entrez # type: ignore

def _represents_int(s):
    """Checks if a string can be converted to an integer."""
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def _has_pubmed_reference(entrez_id: int, email: str) -> bool:
    """
    Checks if a given Entrez Gene ID has at least one associated PubMed ID.

    Parameters:
    -----------
    entrez_id : int
        The Entrez Gene ID to check.
    email : str
        The user's email address (required for NCBI API requests).

    Returns:
    --------
    bool
        True if the EntrezID has at least one PubMed reference, False otherwise.
    """

    Entrez.email = email  # Set email for NCBI API requests

    try:
        # Search for PubMed articles linked to this Entrez Gene ID
        handle = Entrez.elink(dbfrom="gene", db="pubmed", id=str(entrez_id), retmode="xml")
        records = Entrez.read(handle)

        # Check if there is at least one PubMed ID associated with the gene
        for record in records:
            if 'LinkSetDb' in record and len(record['LinkSetDb']) > 0:
                return True  # At least one PubMed reference exists

        return False  # No PubMed references found

    except Exception as e:
        print(f"Warning: Failed to check PubMed for EntrezID {entrez_id}, did not filter it out - {e}")
        return True # Default to False if there is an API error