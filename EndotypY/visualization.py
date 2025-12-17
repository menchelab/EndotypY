import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import networkx as nx  
import numpy as np 
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gseapy as gp #type: ignore
from .utils import download_enrichr_library




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

# =============================================================================
def plot_endotype_grid(endotypes, G, seed_genes, size_height=500, size_width=500, ncols=3, node_size='degree', 
                       path_length=2, layout_seed=2025, layout='spring', limit_lcc=True, 
                       enrichr_lib=None, organism='Human', top_terms=5, force_download=False, 
                       gsea_plot_type='dotplot'):
    """
    Draws multiple endotypes in a grid of subplots using Plotly, with optional GSEA visualization.

    Parameters:
    -------
    endotypes: dict of lists
        Dictionary where keys are endotype names and values are lists of endotype genes.
    G: networkx.Graph
        The Graph object representing the reference network.
    seed_genes: list
        List of seed genes used for endotype identification.
    size_height: int, optional
        Height of each subplot in pixels.
    size_width: int, optional
        Width of each subplot in pixels.
    ncols: int, optional
        Number of columns in the grid layout.
    node_size: str or int, optional
        Determines the centrality measure for size of the nodes in the graph. Options are 'betweenness' or 'degree'. If integer, it is used as a fixed node size.
    path_length: int, optional
        The length of the shortest paths to consider between endotype genes. Default is 2.
    layout: str, optional
        The layout algorithm to use for positioning nodes. Options are 'spring' or 'kk'
    layout_seed: int, optional
        Seed for the spring layout of the graph.
    limit_lcc: bool, optional
        If True, limits each endotype subgraph to its largest connected component.
    enrichr_lib: str, optional
        Name of the Enrichr library to use for GSEA. If None, no GSEA is performed.
    organism: str, optional
        Organism for GSEA. Default is 'Human'.
    top_terms: int, optional
        Number of top enriched terms to display in plots. Default is 5.
    force_download: bool, optional
        Force re-download of Enrichr library. Default is False.
    gsea_plot_type: str, optional
        Type of plot for GSEA results. Options: dotplot and pie. Default is 'dotplot'.

    Returns:
    -------
    fig: plotly.graph_objects.Figure
        The Plotly figure containing the grid of endotype plots.
    enrichment_results: dict, optional
        Dictionary of GSEA results for each endotype (only if enrichr_lib is provided).
    """

    num_endotypes = len(endotypes)
    
    # Determine subplot structure based on whether GSEA is enabled
    if enrichr_lib is not None:
        # Load enrichr library
        term_library = download_enrichr_library(enrichr_lib, organism=organism, force_download=force_download)
        
        # Each endotype gets 2 columns (network + GSEA plot)
        total_cols = ncols * 2
        rows = (num_endotypes + ncols - 1) // ncols
        
        # Create subplot specs with different types based on plot type
        specs = []
        if gsea_plot_type == 'pie':
            plot_spec = {'type': 'domain'}
        else:
            plot_spec = {'type': 'scatter'} if gsea_plot_type in ['dotplot', 'lollipop'] else {'type': 'bar'}
        
        for _ in range(rows):
            row_specs = []
            for _ in range(ncols):
                row_specs.extend([{'type': 'scatter'}, plot_spec])
            specs.append(row_specs)
        
        # Create subplot titles
        subplot_titles = []
        for name in endotypes.keys():
            subplot_titles.extend([f'Endotype: {name}', f'GSEA: {enrichr_lib}'])
        
        fig = make_subplots(rows=rows, cols=total_cols, 
                            specs=specs,
                            subplot_titles=subplot_titles,
                            horizontal_spacing=0.05, vertical_spacing=0.1,
                            column_widths=[0.6, 0.4] * ncols)
        
        enrichment_results = {}
    else:
        # Original layout without GSEA
        cols = ncols
        rows = (num_endotypes + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, 
                            subplot_titles=[f'Endotype: {name}' for name in endotypes.keys()],
                            horizontal_spacing=0.05, vertical_spacing=0.1)
        enrichment_results = None

    endotype_colors = sns.color_palette("hsv", n_colors=num_endotypes)
    
    for idx, (endotype_name, endotype_genes) in enumerate(endotypes.items()):
        # Calculate row and column for network plot
        if enrichr_lib is not None:
            row = idx // ncols + 1
            network_col = (idx % ncols) * 2 + 1
            barplot_col = network_col + 1
        else:
            row = idx // ncols + 1
            network_col = idx % ncols + 1
        
        # Get subgraph for this endotype
        subgraph = plot_endotype(endotype_genes, G, seed_genes,
                                  path_length=path_length,
                                  endotype_color=endotype_colors[idx],
                                  layout_seed=layout_seed,
                                  return_plot=False)
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, seed=layout_seed)
        elif layout == 'kk':
            pos = nx.kamada_kawai_layout(subgraph)
        
        # Calculate node sizes
        if node_size == 'betweenness':
            betweenness = nx.betweenness_centrality(subgraph)
            node_sizes = [30 * betweenness[node] for node in subgraph.nodes()]
        elif node_size == 'degree':
            degree = dict(subgraph.degree())
            node_sizes = [5 * degree[node] for node in subgraph.nodes()]
        elif isinstance(node_size, int):
            node_sizes = [node_size] * len(subgraph.nodes())
        else:
            node_sizes = [10] * len(subgraph.nodes())
        
        # Edge trace
        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                line=dict(width=0.5, color='gray'),
                                hoverinfo='none', showlegend=False)
        
        # Node trace
        node_x = [pos[node][0] for node in subgraph.nodes()]
        node_y = [pos[node][1] for node in subgraph.nodes()]
        
        # Determine node colors: endotype genes get the endotype color, linker genes are gray
        node_colors_list = []
        for node in subgraph.nodes():
            if node in endotype_genes:
                # Endotype genes get their specific color (converted to RGB string)
                rgb = endotype_colors[idx]
                node_colors_list.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
            else:
                # Linker genes are gray
                node_colors_list.append('gray')
        
        # Node border: black for seed genes, none (transparent) for others
        node_border_colors = ['black' if node in seed_genes else 'rgba(0,0,0,0)' for node in subgraph.nodes()]
        node_border_widths = [2 if node in seed_genes else 0 for node in subgraph.nodes()]
        
        node_text = list(subgraph.nodes())
        
        # Scale text size based on node size
        if node_size == 'betweenness' or node_size == 'degree':
            # Scale text size proportionally to node size
            text_sizes = [max(6, min(12, size/3)) for size in node_sizes]
        else:
            text_sizes = [8] * len(subgraph.nodes())
        
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                marker=dict(size=node_sizes, color=node_colors_list,
                                           line=dict(width=node_border_widths, color=node_border_colors)),
                                text=node_text, textposition='middle center',
                                textfont=dict(size=text_sizes), hoverinfo='text',
                                showlegend=False)
        
        fig.add_trace(edge_trace, row=row, col=network_col)
        fig.add_trace(node_trace, row=row, col=network_col)
        
        # Update axes for network plot
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=row, col=network_col)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=row, col=network_col)
        
        # Run GSEA and add plot if enrichr_lib is provided
        if enrichr_lib is not None:
            # Perform enrichment analysis
            enr = gp.enrich(gene_list=endotype_genes,
                           gene_sets=term_library,
                           #organism=organism,
                           outdir=None,
                           cutoff=0.05,
                           verbose=False)
            
            enrichment_results[endotype_name] = enr.results
            
            # Get top terms - check if results is a DataFrame and not empty
            if hasattr(enr.results, 'empty') and not enr.results.empty:
                top_results = enr.results.nsmallest(top_terms, 'Adjusted P-value')
                
                # Extract data
                terms = top_results['Term'].tolist()
                neg_log_pvals = -np.log10(top_results['Adjusted P-value'].values)
                gene_counts = top_results['Overlap'].apply(lambda x: int(x.split('/')[0])).values
                genes_in_terms = top_results['Genes'].tolist()  # Get the actual genes for each term
                
                # Truncate long term names
                terms_short = [term[:40] + '...' if len(term) > 40 else term for term in terms]
                
                # Get endotype color as RGB string
                endotype_rgb = f'rgb({int(endotype_colors[idx][0]*255)}, {int(endotype_colors[idx][1]*255)}, {int(endotype_colors[idx][2]*255)})'
                
                # Create appropriate plot based on gsea_plot_type                
                if gsea_plot_type == 'dotplot':
                    # Dot plot with gene count as marker size
                    gsea_trace = go.Scatter(
                        x=neg_log_pvals,
                        y=terms_short,
                        mode='markers',
                        marker=dict(
                            size=gene_counts * 3,  # Scale for visibility
                            color=endotype_rgb,
                            line=dict(width=1, color='darkgray')
                        ),
                        customdata=list(zip(gene_counts, genes_in_terms)),
                        showlegend=False,
                        hovertemplate='<b>%{y}</b><br>Adjusted p-value (-log10): %{x:.2f}<br>Gene count: %{customdata[0]}<br>Genes: %{customdata[1]}<extra></extra>'
                    )
                    fig.add_trace(gsea_trace, row=row, col=barplot_col)
                    fig.update_xaxes(title_text='Adjusted p-value (-log10)', row=row, col=barplot_col)
                    fig.update_yaxes(showticklabels=False, row=row, col=barplot_col)
                
                elif gsea_plot_type == 'pie':
                    # Pie chart
                    # Use normalized -log10 p-values as values
                    pie_trace = go.Pie(
                        labels=terms_short,
                        values=neg_log_pvals,
                        marker=dict(
                            colors=[endotype_rgb] * len(terms_short),
                            line=dict(color='white', width=2)
                        ),
                        textfont=dict(size=8),
                        customdata=list(zip(gene_counts, genes_in_terms)),
                        hovertemplate='<b>%{label}</b><br>Adjusted p-value (-log10): %{value:.2f}<br>Gene count: %{customdata[0]}<br>Genes: %{customdata[1]}<extra></extra>',
                        showlegend=False
                    )
                    fig.add_trace(pie_trace, row=row, col=barplot_col)

                else:
                    # Unsupported plot type
                    raise ValueError(f"Unsupported gsea_plot_type: {gsea_plot_type}. Supported types are 'dotplot' and 'pie'.")

                
            else:
                # No significant enrichment found
                fig.add_annotation(
                    text="No significant<br>enrichment",
                    xref=f"x{barplot_col}" if barplot_col > 1 else "x",
                    yref=f"y{barplot_col}" if barplot_col > 1 else "y",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=12),
                    row=row, col=barplot_col
                )
    
    # Update overall layout
    if enrichr_lib is not None:
        fig.update_layout(height=size_height * rows, width=size_width * ncols * 2,
                          title_text="Network of Multiple Endotypes with GSEA", showlegend=False)
        fig.show()
    else:
        fig.update_layout(height=size_height * rows, width=size_width * ncols,
                          title_text="Network of Multiple Endotypes", showlegend=False)
        fig.show()
        return None
