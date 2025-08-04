import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import networkx as nx  
import numpy as np 
import copy
import gseapy as gp #type: ignore




# =============================================================================

def plot_endotype(endotype, G, seed_genes,size_height=14, size_width=14,node_size='betweenness',path_length=2,endotype_color='cornflowerblue',layout_seed=2025,return_plot=True):
    """
    Draws a subgraph of the PPI network containing the endotype genes and their shortest paths.

    Parameters:
    -------
    endotype: list
        List of endotype genes to be highlighted in the graph.
    G: networkx.Graph
        The Graph object representing the reference network
    seed_genes: list
        List of seed genes used for endotype identification.
    size_height: int, optional
        Height of the figure in inches.
    size_width: int, optional
        Width of the figure in inches.
    node_size: str, optional
        Determines the centrality measure for size of the nodes in the graph. Options are 'betweenness' or 'degree'.
    path_length: int, optional
        The length of the shortest paths to consider between endotype genes. Default is 2.
    endotype_color: str, optional
        Color used to highlight the endotype genes in the graph. Default is 'orange'.
    layout_seed: int, optional
        Seed for the spring layout of the graph.
    return_plot: bool, optional
        If True, the function will not plot the graph but will still return the subgraph.
    
    Returns:
    -------
    subgraph: networkx.Graph
        The subgraph containing the endotype genes and their shortest paths.    
    """
    G_copy = G.copy()  # Create a copy of the graph to avoid modifying the original
    # Compute the edges in the shortest paths between sauna genes
    sauna = endotype
    shortest_path_edges = set()
    for i in range(len(sauna)):
        for j in range(i + 1, len(sauna)):
            if sauna[i] in G_copy and sauna[j] in G_copy:
                if nx.has_path(G_copy, source=sauna[i], target=sauna[j]):
                    path = nx.shortest_path(G_copy, source=sauna[i], target=sauna[j])
                    if len(path) - 1 <= path_length:  # Check if the path length is within the limit
                        # Add edges from the path to the set
                        shortest_path_edges.update(zip(path[:-1], path[1:]))

    # Create subgraph with only the edges in the shortest paths
    subgraph = nx.Graph()
    subgraph.add_edges_from(shortest_path_edges)
    # Remove duplicate edges

    # Limit to the largest connected component
    #largest_cc = max(nx.connected_components(subgraph), key=len)
    #subgraph = subgraph.subgraph(largest_cc)

    # Draw the subgraph with improved aesthetics
    node_colors = ['gray' if node not in sauna else endotype_color for node in subgraph.nodes()]
    node_border_colors = ['black' if node in seed_genes else 'None' for node in subgraph.nodes()]
    node_font_sizes = [12 if node in seed_genes else 10 for node in subgraph.nodes()]

    # Draw nodes based on betweenness centrality
    if node_size == 'betweenness':
        # Calculate betweenness centrality for node sizes
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        node_sizes = [600 * betweenness_centrality[node] for node in subgraph.nodes()]
    
    if node_size == 'degree':
        # Calculate degree for node sizes
        degree_centrality = dict(subgraph.degree())
        node_sizes = [600* degree_centrality[node] for node in subgraph.nodes()]

    if return_plot:
        plt.figure(figsize=(size_width, size_height))
        pos = nx.spring_layout(subgraph,seed=layout_seed)
        # Draw nodes with border colors
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, edgecolors=node_border_colors, linewidths=1.5)

        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='gray', width=1)

        # Add labels with custom font sizes
        for node, (x, y) in pos.items():
            plt.text(x, y, node, fontsize=node_font_sizes[list(subgraph.nodes()).index(node)], ha='center', va='center', color='black')

        # Add title and remove axis
        plt.title('Network of Endotype Genes', fontsize=16, fontweight='bold')
        plt.axis('off')
    
        # Add legend 
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Endotype Genes', markerfacecolor=endotype_color, markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Seed Genes', markerfacecolor='gray', markersize=10, markeredgecolor='black')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        plt.show()

    # Add node color as node attribute
    for node in subgraph.nodes():
        subgraph.nodes[node]['color'] = 'gray' if node not in sauna else endotype_color
        subgraph.nodes[node]['origin'] = f'endotype_{endotype[0]}' if node in sauna else 'Linker gene'

    return subgraph

# =============================================================================
def plot_multiple_endotypes(endotypes, G, seed_genes, size_height=14, size_width=14, node_size=100, path_length=2, layout_seed=2025, layout='spring', limit_lcc=True):
    """
    Draws multiple endotypes in a single plot.

    Parameters:
    -------
    endotypes: dict of lists
        Dictionary where keys are endotype names and values are lists of endotype genes.
    G: networkx.Graph
        The Graph object representing the reference network.
    seed_genes: list
        List of seed genes used for endotype identification.
    size_height: int, optional
        Height of the figure in inches.
    size_width: int, optional
        Width of the figure in inches.
    node_size: str or int, optional
        Determines the centrality measure for size of the nodes in the graph. Options are 'betweenness' or 'degree'. If integer, it is used as a fixed node size.
    path_length: int, optional
        The length of the shortest paths to consider between endotype genes. Default is 2.
    layout_seed: int, optional
        Seed for the spring layout of the graph.

    Returns:
    -------
    combined_subgraph: networkx.Graph
        Combined graph containing all endotypes and their shortest paths.
    """

    # Create a local copy of the graph to avoid modifying external variables
    local_G = copy.deepcopy(G)

    # Generate colors for endotypes
    endotype_colors = sns.color_palette("hsv", n_colors=len(endotypes))
    #endotype_colors = [f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})" for c in endotype_colors]

    # Combine all endotypes into a single graph
    combined_subgraph = nx.Graph()
    all_subgraphs = []
    for i, (endotype_name, endotype_genes) in enumerate(endotypes.items()):
        subgraph = plot_endotype(endotype_genes, local_G, seed_genes,
                                    path_length=path_length,
                                    endotype_color=endotype_colors[i], layout_seed=layout_seed,
                                    return_plot=False)
        all_subgraphs.append(subgraph)
        # Combine the subgraph into the main graph
        combined_subgraph = nx.compose(combined_subgraph, subgraph)
        
    for grafo in all_subgraphs:
        for node in combined_subgraph.nodes():
            if node in grafo.nodes():
                # Update node attributes for endotype and color
                if grafo.nodes[node]['color'] != 'gray':
                    combined_subgraph.nodes[node]['color'] = grafo.nodes[node]['color']
                    combined_subgraph.nodes[node]['endotype'] = grafo.nodes[node]['origin']
                else:
                    continue
            else:
                continue

    # Limit to the largest connected component if specified
    if limit_lcc:
        largest_cc = max(nx.connected_components(combined_subgraph), key=len)
        combined_subgraph = combined_subgraph.subgraph(largest_cc)

    # Draw the combined subgraph with improved aesthetics
    node_colors = [combined_subgraph.nodes[node].get('color') for node in combined_subgraph.nodes()]
    node_border_colors = ['black' if node in seed_genes else 'gray' for node in combined_subgraph.nodes()]
    node_font_sizes = [8 if node in seed_genes else 5 for node in combined_subgraph.nodes()]
    if node_size == 'betweenness':
        # Calculate betweenness centrality for node sizes
        betweenness_centrality = nx.betweenness_centrality(combined_subgraph)
        node_sizes = [600 * betweenness_centrality[node] for node in combined_subgraph.nodes()]
    if node_size == 'degree':
        # Calculate degree for node sizes
        degree_centrality = dict(combined_subgraph.degree())
        node_sizes = [10 * degree_centrality[node] for node in combined_subgraph.nodes()]
    # if node_size is integer, use it as a fixed node size
    if node_size is False:
        node_sizes = [600] * len(combined_subgraph.nodes())
    elif isinstance(node_size, int):
        # Use the provided integer as a fixed node size
        node_sizes = [node_size] * len(combined_subgraph.nodes())
    elif node_size is None:
        # Use a default fixed node size if node_size is None
        node_sizes = [600] * len(combined_subgraph.nodes())
    
    plt.figure(figsize=(size_width, size_height))
    if layout == 'spring':
        pos = nx.spring_layout(combined_subgraph, seed=layout_seed)
    elif layout == 'kk':
        pos = nx.kamada_kawai_layout(combined_subgraph)

    # Draw nodes with border colors
    nx.draw_networkx_nodes(combined_subgraph, pos, node_size=node_sizes, node_color=node_colors, edgecolors=node_border_colors, linewidths=1.5)
    # Draw edges
    nx.draw_networkx_edges(combined_subgraph, pos, alpha=0.5, edge_color='gray', width=1)
    # Add labels with custom font sizes
    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=node_font_sizes[list(combined_subgraph.nodes()).index(node)], ha='center', va='center', color='black')
    # Add title and remove axis
    plt.title('Network of Multiple Endotypes', fontsize=16, fontweight='bold')
    plt.axis('off')
    # Add legend indicating the color of each endotype
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Endotype: {name}', markerfacecolor=color, markersize=8) for name, color in zip(endotypes.keys(), endotype_colors)]
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Seed Genes', markersize=8, markeredgecolor='black'))
    plt.legend(handles=legend_elements, loc='upper right', fontsize=6)
    plt.show()

    return combined_subgraph

