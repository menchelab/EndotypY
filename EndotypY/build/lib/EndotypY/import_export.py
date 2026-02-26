import os
from pathlib import Path
import networkx as nx
import warnings
from typing import Union

from .utils import _represents_int, _has_pubmed_reference  # Import helper functions from utils.py

def read_network_from_file(network_file: Union[str, Path], filter_pubmed: bool = False, email: str = None) -> nx.Graph:
    """
    Reads a network from an external file.

    * The edgelist must be provided as a tab-separated table. The
      first two columns of the table will be interpreted as an
      interaction gene1 <==> gene2.

    * Lines that start with '#' will be ignored.

    * The function checks that the input file ends with '.txt' or '.tsv'.

    * Each node (gene) must be represented by an integer (EntrezID). The function throws
      an error if a non-integer node is detected. Therefore, the function only allows
      protein-protein interaction networks.

    * Self-loops are eliminated in the last filtering step of the function.

    * Optionally, users can filter out genes that do not have associated PubMed references.

    Parameters:
    -----------
    network_file : str | Path
        The path to the input file. It can be provided as:
        - A string (e.g., "data/network.txt")
        - A pathlib.Path object (e.g., Path("data", "network.txt"))

    filter_pubmed : bool, optional (default=False)
        If True, removes nodes (genes) that do not have an associated PubMedID.

    email : str, optional
        Required if `filter_pubmed=True`. Used for NCBI API requests.

    Returns:
    --------
    nx.Graph
        A NetworkX graph with nodes and edges from the file.

    Raises:
    -------
    ValueError
        If the file format is not '.txt' or '.tsv'.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If any node in the file is not a valid integer.
    ValueError
        If `filter_pubmed=True` but no email is provided.

    Notes:
    ------
    - Lines starting with '#' are ignored.
    - Self-loops (edges where node1 == node2) are removed.
    - If `filter_pubmed=True`, nodes without PubMed references are removed.
    """

    # Convert the input to a Path object (handles str, Path, PosixPath)
    network_file = Path(network_file)

    if network_file.suffix not in ('.txt', '.tsv'):
        raise ValueError("Invalid file format. Expected a '.txt' or '.tsv' file.")
    if not network_file.exists():
        raise FileNotFoundError(f"File not found: {network_file}")

    G = nx.Graph()

    with network_file.open('r') as file:
        for line in file:
            # Ignore lines starting with '#'
            if line.startswith('#'):
                continue
            
            # Split line into columns
            line_data = line.strip().split('\t')
            
            if len(line_data) < 2:
                warnings.warn(f"Invalid line detected: {line}. Skipping line.")
                continue  # Skip invalid or empty lines
            
            node1, node2 = line_data[0], line_data[1]

            # Check if both nodes are valid integers
            if not (_represents_int(node1) and _represents_int(node2)):
                raise ValueError(f"Invalid node detected: '{node1}', '{node2}'. Nodes must be valid integers (EntrezIDs).")

            # Convert nodes to integers
            node1, node2 = int(node1), int(node2)

            # Add edge to the graph
            if node1 != node2:  # Avoid self-loops
                G.add_edge(node1, node2)

    # Remove any remaining self-loops (just in case)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Optional filtering: Remove nodes without PubMed references
    if filter_pubmed:
        if email is None:
            raise ValueError("An email must be provided when filter_pubmed=True (required for NCBI API requests).")
        
        nodes_to_remove = [node for node in G.nodes if not _has_pubmed_reference(node, email)]
        G.remove_nodes_from(nodes_to_remove)
        print(f"> Removed {len(nodes_to_remove)} nodes without PubMed references.")

    print("\n> Done loading network:")
    print(f"> Network contains {G.number_of_nodes()} nodes and {G.number_of_edges()} links")

    return G


## currently not used
def filter_invalid_entrez_ids(input_file: Path, output_file: Path) -> None:
    """
    Reads a protein-protein interaction file, filters out connections where at least
    one node is not a valid integer EntrezID, and saves the cleaned file to a new location.

    Parameters:
    -----------
    input_file : Path | str
        Path to the input protein interaction file.

    output_file : Path | str
        Path to save the cleaned interaction file.

    Raises:
    -------
    ValueError
        If the input file format is not '.txt' or '.tsv'.
    FileNotFoundError
        If the specified input file does not exist.

    Notes:
    ------
    - The file must be a tab-separated file.
    - Lines where at least one node is not an integer EntrezID will be removed.
    - The cleaned file is saved to `output_file`.
    """

    # Convert paths to Path objects
    input_file, output_file = Path(input_file), Path(output_file)

    # Validate input file
    if input_file.suffix not in ('.txt', '.tsv'):
        raise ValueError("Invalid file format. Expected a '.txt' or '.tsv' file.")
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    valid_lines = []
    invalid_count = 0

    with input_file.open('r') as infile:
        for line in infile:
            # Ignore comments
            if line.startswith('#'):
                valid_lines.append(line)  # Keep header/comment lines
                continue

            # Split into columns
            line_data = line.strip().split('\t')

            # Ensure there are at least two columns
            if len(line_data) < 2:
                warnings.warn(f"Skipping invalid line (not enough columns): {line.strip()}")
                invalid_count += 1
                continue

            node1, node2 = line_data[0], line_data[1]

            # Check if both nodes are valid integer EntrezIDs
            if _represents_int(node1) and _represents_int(node2):
                valid_lines.append(line)  # Keep valid interactions
            else:
                invalid_count += 1  # Count invalid interactions

    # Save cleaned data to output file
    with output_file.open('w') as outfile:
        outfile.writelines(valid_lines)

    print(f"\n> Filtering complete. Saved cleaned file to: {output_file}")
    print(f"> Removed {invalid_count} invalid interactions (non-integer EntrezIDs).")


def load_seed_set_from_file(seed_file: Union[str, Path]) -> set:
    """
    Reads a seed set from an external file.

    * The seed set must be provided as a list of integers (EntrezIDs).
    * Each line in the file must contain a single integer.
    * Lines starting with '#' will be ignored.

    Parameters:
    -----------
    seed_file : str | Path
        The path to the input file. It can be provided as:
        - A string (e.g., "data/seeds.txt")
        - A pathlib.Path object (e.g., Path("data", "seeds.txt"))

    Returns:
    --------
    list of int
        A list of unique integers (EntrezIDs) from the file.

    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If any line in the file is not a valid integer.
    """

    # Convert the input to a Path object (handles str, Path, PosixPath)
    seed_file = Path(seed_file)

    if not seed_file.exists():
        raise FileNotFoundError(f"File not found: {seed_file}")

    f = open(seed_file,'r')
    lines = f.readlines()
    f.close()
    seedset_strings = lines[0].split('\n')[0].split('\t')[1:-1]

    # Convert to integers
    if not all(_represents_int(node) for node in seedset_strings):
        raise ValueError("Invalid seed set. All lines must be valid integers (EntrezIDs).")
    
    # eliminate duplicates
    seed_set = set(map(int, seedset_strings))
    # make it into a list to return
    seed_list = list(seed_set)
    print(f"\n> Loaded {len(seed_set)} seed nodes from file: {seed_file}")

    return seed_list


