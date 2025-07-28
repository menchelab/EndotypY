import os
from pathlib import Path
import networkx as nx
import warnings
from typing import Union


def read_network_from_file(network_file: Union[str, Path]) -> nx.Graph:
    """
    Reads a network from an external file.

    * The edgelist must be provided as a tab-separated table. The
      first two columns of the table will be interpreted as an
      interaction gene1 <==> gene2.

    * Lines that start with '#' will be ignored.

    * The function checks that the input file ends with '.txt' or '.tsv' or '.csv'.

    * Self-loops are eliminated in the last filtering step of the function.

    Parameters:
    -----------
    network_file : str | Path
        The path to the input file. It can be provided as:
        - A string (e.g., "data/network.txt")
        - A pathlib.Path object (e.g., Path("data", "network.txt"))

    Returns:
    --------
    nx.Graph
        A NetworkX graph with nodes and edges from the file.

    Raises:
    -------
    ValueError
        If the file format is not '.txt' or '.tsv' or '.csv'.
    FileNotFoundError
        If the specified file does not exist.

    Notes:
    ------
    - Lines starting with '#' are ignored.
    - Self-loops (edges where node1 == node2) are removed.
    """

    # Convert the input to a Path object (handles str, Path, PosixPath)
    network_file = Path(network_file)

    if network_file.suffix not in ('.txt', '.tsv', '.csv'):
        raise ValueError("Invalid file format. Expected a '.txt' or '.tsv' or '.csv' file.")
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

            # Convert nodes to integers
            node1, node2 = int(node1), int(node2)

            # Add edge to the graph
            if node1 != node2:  # Avoid self-loops
                G.add_edge(node1, node2)

    # Remove any remaining self-loops (just in case)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def load_seed_set_from_file(seed_file) -> set:
    """
    Reads a seed set from an external file.
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
        A list of unique seeds from the file

    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist.
    """

    f = open(seed_file,'r')
    lines = f.readlines()
    f.close()
    seedset_strings = []
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            continue
        
        # Try splitting by tab, then by newline, then consider it a single seed
        split_by_tab = line.split('\t')
        if len(split_by_tab) > 1:
            seedset_strings.extend([s.strip() for s in split_by_tab[1:] if s.strip()])  # Skip the first element if it's not a seed
        else:
            split_by_newline = line.split('\n')
            if len(split_by_newline) > 1:
                seedset_strings.extend([s.strip() for s in split_by_newline if s.strip()])
            else:
                # Assume the whole line is a single seed
                seedset_strings.append(line.strip())
    
    # eliminate duplicates
    seed_set = set(seedset_strings)
    # make it into a list to return
    seed_list = list(seed_set)
    print(f"\n> Loaded {len(seed_set)} seed nodes from file: {seed_file}")

    return seed_list
