import io
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import networkx as nx  
import numpy as np 
import copy
import plotly.graph_objects as go #type: ignore
import gseapy as gp #type: ignore
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from itertools import combinations
import datamapplot #type: ignore
import math

# =============================================================================
# PLOT THE ENDOTYPE SUBGRAPHS AS A METAGRAPH
# ============================================================================

def plot_endotypes_metagraph(G, d_clusters, seed_genes, filter_size_endotypes=True, node_size=15):
    """
    Wrapper function to create meta positions, scale and position endotype 
    subgraphs, and visualize the final layout.
    
    Parameters:
    -----------
    G : nx.Graph
        The original full network graph
    d_clusters : dict
        A dictionary where keys are endotype identifiers and values are 
        lists of genes belonging to each endotype   
    seed_genes : list
        List of seed genes used for endotype identification.
    filter_size_endotypes : bool
        If True, only endotype subgraphs with at least one edge and more 
        than 5 nodes are considered for visualization.
    node_size : int
        Size of the nodes in the final visualization.

    """
    # create endotype subgraphs and combined graph
    d_endotypes_graphs, subG_endotypes = _create_endotype_subgraphs_and_combined(G, d_clusters, filter_size_endotypes)
    
    # compute number of inter-edges between endotypes
    d_num_interedges_endotypes = _compute_number_interedges(G, d_endotypes_graphs)
    
    # create meta positions for endotypes
    d_endotype_meta_pos = _create_meta_positions_endotypes(d_num_interedges_endotypes, k=60)
    
    # scale and position endotype subgraphs
    final_pos_unscaled = _scale_and_position_endotype_subgraphs(d_endotypes_graphs, d_endotype_meta_pos)
    
    # final global scaling to fit plotting area
    final_pos, d_endotype_meta_pos_scaled = _scale_to_fit(final_pos_unscaled, d_endotype_meta_pos)
    
    # visualize the final metagraph of endotypes
    _visualization_endotype_with_hulls(final_pos, 
                                      d_endotypes_graphs, 
                                      d_endotype_meta_pos_scaled, 
                                      d_num_interedges_endotypes, 
                                      seed_genes,
                                      node_size)

def _create_endotype_subgraphs_and_combined(G, d_clusters, filter_size_endotypes):
    
    """
    Create subgraphs for each endotype and a combined subgraph containing all 
    endotypes OR only endotypes with at least one edge and more than 5 nodes.

    Parameters:
    -----------
    G : nx.Graph
        The original full network graph
    d_clusters : dict
        A dictionary where keys are endotype identifiers and values are 
        lists of genes belonging to each endotype
    filter_size_endotypes : bool
        If True, only include endotype subgraphs with at least one edge and 
        more than 5 nodes.

    Returns:
    --------
    dict
        A dictionary where keys are endotype identifiers and values are 
        the corresponding subgraphs
    nx.Graph 
        A combined subgraph containing all endotype genes
    """
    genes_endo = []
    d_endotypes_graphs = {}
    for k,cl in d_clusters.items():
        genes = []
        for g in cl:
            genes.append(str(g))
            genes_endo.append(str(g))
            
        subG = G.subgraph(genes).copy()
        if filter_size_endotypes:
            if subG.number_of_edges() > 0 and subG.number_of_nodes() > 5:
                d_endotypes_graphs[k] = subG
        else:
            d_endotypes_graphs[k] = subG
            
    genes_endotypes = set(genes_endo)
    subG_endotypes = G.subgraph(genes_endotypes).copy()
    
    return d_endotypes_graphs, subG_endotypes


def _compute_number_interedges(G, d_subnetworks):
    """
    Compute the number of edges between each pair of subnetworks present
    in d_subnetworks.
    
    Parameters:
    -----------
    G : nx.Graph
        The original full network graph
    d_subnetworks : dict
        A dictionary where keys are subnetwork identifiers and values are 
        the corresponding subgraphs
    
    Returns:
    --------
    dict
        A dictionary mapping key tuples from d_subnetworks to the number 
        of edges between the corresponding subnetworks.
    """

    d_num_interedges = {}
    endotypes = list(d_subnetworks.items())
    pair_endotypes = combinations(endotypes, 2)

    for (k1, g1), (k2, g2) in pair_endotypes:
        nodes_g1 = set(g1.nodes())
        nodes_g2 = set(g2.nodes())
        n_edges_g1 = g1.number_of_edges()
        n_edges_g2 = g2.number_of_edges()

        union_nodes = nodes_g1 | nodes_g2
        n_edges_union = G.subgraph(list(union_nodes)).number_of_edges()
        
        n_between = n_edges_union - n_edges_g1 - n_edges_g2

        d_num_interedges[(k1, k2)] = int(n_between)

    return d_num_interedges


def _create_meta_positions_endotypes(d_num_interedges_endotypes, k=60): 
    """
    Create a meta-graph of endotypes based on inter-edge counts and 
    compute positions for each endotype meta-node using spring layout.
    
    Parameters:
    -----------
    d_num_interedges_endotypes : dict
        A dictionary mapping key tuples from endotypes to the number 
        of edges between the corresponding endotypes.
    k : float
        Optimal distance between nodes in the spring layout.
    
    Returns:
    --------
    dict
        A dictionary mapping endotype identifiers to their computed
        meta-node positions as (x, y) coordinates.
    """

    # create a meta-graph of endotypes based on inter-edge counts
    interedge_list = []
    for (k1, k2), n_edges in d_num_interedges_endotypes.items():
        if n_edges > 0:
            interedge_list.append((k1, k2, n_edges))

    G_meta_endotypes = nx.Graph()
    G_meta_endotypes.add_weighted_edges_from(interedge_list)

    # use spring layout to position meta-nodes
    pos = nx.spring_layout(G_meta_endotypes, seed=42, iterations=300, k=k)

    # store the coordinates in a dictionary
    nodes_order = list(G_meta_endotypes.nodes())
    meta_pos = np.vstack([pos[n] for n in nodes_order])
    d_endotype_meta_pos = {n: meta_pos[i] for i, n in enumerate(nodes_order)}

    return d_endotype_meta_pos


def _scale_and_position_endotype_subgraphs(d_endotypes_graphs,
                                          d_endotype_meta_pos, 
                                          meta_distance_scaling=0.12): 
    """
    Scale each individual endotype subgraph to fit within its meta-node 
    spacing.

    Parameters:
    -----------
    d_endotypes_graphs : dict
        A dictionary where keys are endotype identifiers and values are 
        the corresponding subgraphs
    d_endotype_meta_pos : dict
        A dictionary mapping endotype identifiers to their computed
        meta-node positions as (x, y) coordinates.  
    meta_distance_scaling : float
        Multiplier for cluster diameter relative to meta-node spacing and 
        min meta distance.    
        
    Returns:
    --------
    dict
        A dictionary mapping gene names to their final (x, y) positions 
        after scaling and positioning.
    """   
    # collect meta-centers and compute min inter-meta distance
    meta_coords = np.array(list(d_endotype_meta_pos.values()))

    if len(meta_coords) > 1:
        # pairwise distances
        dists = []
        for (i, a), (j, b) in combinations(enumerate(meta_coords), 2):
            dists.append(np.linalg.norm(a - b))
        min_meta_dist = max(1e-6, np.min(dists))
    else:
        min_meta_dist = 1.0
        
    # print("all pairwise meta-node distances:", dists)

    # maximum allowed diameter for any subgraph cluster (so clusters don't overlap)
    max_cluster_diam = min_meta_dist * meta_distance_scaling


    # build final positions for every gene by centering+scaling each subgraph at its meta-position
    final_pos_unscaled = {}

    # precompute node counts for scale factor (give bigger clusters more space)
    node_counts = {k: g.number_of_nodes() for k, g in d_endotypes_graphs.items()}

    for k, endograph in d_endotypes_graphs.items():
        nodes = list(endograph.nodes())
        
        # create local layout for each subgraph
        local_pos = nx.spring_layout(endograph, seed=42, iterations=100, k=0.8)
        # local_pos = nx.kamada_kawai_layout(endograph)
        
        P = np.vstack([local_pos[n] for n in nodes])
        # center local layout (but almost unnecessary)
        P_centered = P - P.mean(axis=0)

        # diameter of the layout coordinates
        current_diam = float(pdist(P_centered).max()) if len(P_centered) > 1 else 0.0
        current_diam = max(1e-6, current_diam)

        # set target diameter: scale with node count (sqrt) so bigger endotypes get more space
        size_factor = math.sqrt(max(1, node_counts.get(k, 1)))
        target_diam = max_cluster_diam * size_factor
        
        scale = target_diam / current_diam
        P_scaled = P_centered * scale 
        # translate cluster to its meta-node center (meta coords are in [-1,1])
        center = np.array(d_endotype_meta_pos.get(k, np.zeros(2)), dtype=float)
        P_translated = P_scaled + center

        for i, nname in enumerate(nodes):
            final_pos_unscaled[nname] = (float(P_translated[i, 0]), 
                                         float(P_translated[i, 1]))   

    return final_pos_unscaled


def _scale_to_fit(final_pos_unscaled, d_endotype_meta_pos, target_extent = 8.0):
    """ 
    Ensures that the whole composition (nodes + meta-nodes) is centered and 
    scaled uniformly to fill the plotting area.
    It performs a "zoom to fit" step.
    
    Parameters:
    -----------
    final_pos_unscaled : dict
        A dictionary mapping gene names to their final (x, y) positions after 
        scaling and positioning.
    d_endotype_meta_pos : dict
        A dictionary mapping endotype identifiers to their computed meta-node 
        positions as (x, y) coordinates.
    target_extent : float
        Desired extent (width/height) of the final layout. 
        How much larger you want the spread (increase to spread endotype graphs).
        
    Returns:
    --------
    dict
        A dictionary mapping gene names to their final (x, y) positions after 
        global scaling and positioning.
        If scaling is skipped, returns the input positions unchanged.
    dict
        A dictionary mapping endotype identifiers to their final meta-node 
        positions after global scaling and positioning.
        If scaling is skipped, returns the input positions unchanged.
    
    """
    # include meta positions so they are transformed the same way as the nodes
    meta_unscaled = {k: np.array(v, dtype=float) for k, v in d_endotype_meta_pos.items()}

    genes_nodes = list(final_pos_unscaled.keys())

    # stack node coords and meta coords together for a single global bbox
    coords_nodes = np.vstack([final_pos_unscaled[g] for g in genes_nodes]) if genes_nodes else np.empty((0, 2))
    coords_meta = np.vstack(list(meta_unscaled.values())) if meta_unscaled else np.empty((0, 2))

    # combine all coords to transform them together
    if coords_meta.size:
        coords_all = np.vstack([coords_nodes, coords_meta]) if coords_nodes.size else coords_meta
    else:
        coords_all = coords_nodes

    # if nothing to scale, skip
    if coords_all.size == 0:
        return final_pos_unscaled, d_endotype_meta_pos
    else:
        xmin, xmax = coords_all[:, 0].min(), coords_all[:, 0].max()
        ymin, ymax = coords_all[:, 1].min(), coords_all[:, 1].max()
        width, height = xmax - xmin, ymax - ymin
        maxdim = max(width, height, 1e-6)

        global_scale = target_extent / maxdim
        global_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])

        # transform function
        def _transform(pts):
            return (np.array(pts, dtype=float) - global_center) * global_scale

        # apply same transform to node positions and meta positions
        final_pos = {g: tuple(_transform(final_pos_unscaled[g])) for g in final_pos_unscaled}
        d_endotype_meta_pos_scaled = {k: tuple(_transform(v)) for k, v in meta_unscaled.items()}

    return final_pos, d_endotype_meta_pos_scaled
    
    
def _visualization_endotype_with_hulls(final_pos, 
                                      d_endotypes_graphs, 
                                      d_endotype_meta_pos_scaled, 
                                      d_num_interedges_endotypes, 
                                      seed_genes,
                                      node_size=15):
    """
    Visualize the endotype graphs with colored hulls and edges using the 
    datamapplot library.
    
    Parameters:
    -----------
    final_pos : dict
        A dictionary mapping gene names to their final (x, y) positions after 
        global scaling and positioning.
    d_endotypes_graphs : dict
        A dictionary where keys are endotype identifiers and values are the 
        corresponding subgraphs
    d_endotype_meta_pos_scaled : dict
        A dictionary mapping endotype identifiers to their final meta-node 
        positions after global scaling and positioning.
    d_num_interedges_endotypes : dict
        A dictionary mapping key tuples from endotypes to the number of edges
        between the corresponding endotypes.
    seed_genes : list
        List of seed genes used for endotype identification.
    node_size : int
        Size of the nodes in the plot.  
    """

    # build coords array in the expected order (genes used later)
    genes = [g for g in final_pos.keys()]
    coords = np.vstack([final_pos[g] for g in genes])

    # build list of nodes that have final positions and assign a single label per node
    gene_to_label = {}
    for k, endograph in d_endotypes_graphs.items():
        lab = str(k)
        for n in endograph.nodes():
            if n in final_pos and n not in gene_to_label:
                gene_to_label[n] = lab
    labels = np.array([gene_to_label.get(g, "unassigned") for g in genes], dtype=str)

    # create datamapplot scatter (colors assigned by label)
    fig, ax = datamapplot.create_plot(
        coords,
        labels,
        force_matplotlib=True,
        color_label_text=False,
        figsize=(14, 14),
        point_size=node_size,
        dynamic_label_size=True,
        # min_font_size=14, # does not work with version 0.2.2 but works with 0.3.0
        add_glow=False
    )

    # extract scatter and map label to color (first occurrence)
    scatter = ax.collections[0]
    colors_per_point = scatter.get_facecolors()
    unique_labels = np.unique(labels)
    label_to_color = {}
    for lab in unique_labels:
        idx = np.where(labels == lab)[0][0]
        label_to_color[lab] = colors_per_point[idx]
        
    # create coloured hull per endotype
    color_fill_alpha = 0.25

    for i, lab in enumerate(unique_labels):
        pts = coords[labels == lab]
        if pts.shape[0] < 3:
            continue
        try:
            hull = ConvexHull(pts)
            verts = pts[hull.vertices]
        except Exception:
            verts = pts

        # use the base color from label_to_color
        base_rgba = label_to_color.get(lab, None)

        centroid = verts.mean(axis=0)
        buffer_frac = 0.3  # 0.0 = no buffer, 0.1 = 10% larger
        buffered_verts = centroid + (verts - centroid) * (1.0 + buffer_frac)

        # filled coloured hull
        fill_poly = plt.Polygon(
            buffered_verts,
            closed=True,
            facecolor=(*base_rgba[:3], color_fill_alpha),
            edgecolor='none',
            zorder=1
        )
        ax.add_patch(fill_poly)

    # draw meta-edges between endotype centroids
    centroids = {}
    for k, c in d_endotype_meta_pos_scaled.items():
        centroids[str(k)] = np.array(c)

    for (u, v), w in d_num_interedges_endotypes.items():
        u_lab, v_lab = str(u), str(v)
        if u_lab in centroids and v_lab in centroids and w > 0:
            p1 = centroids[u_lab]
            p2 = centroids[v_lab]
            lw = math.sqrt(w) * 2
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color="gray", linewidth=lw, alpha=0.2, zorder=0)

    # draw intra-endotype edges on top of the scatter
    pos_coords = {g: final_pos[g] for g in genes}
    for k, endographs in d_endotypes_graphs.items():
        lab = str(k)
        color = label_to_color.get(lab, (0.7, 0.7, 0.7, 1.0))
        for u, v in endographs.edges():
            if u in pos_coords and v in pos_coords:
                p1 = pos_coords[u]
                p2 = pos_coords[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color=color, linewidth=1, alpha=0.9, zorder=2)

    # Highlight seed genes with black edge
    seed_indices = [i for i, g in enumerate(genes) if g in seed_genes]

    if seed_indices:
        seed_coords = coords[seed_indices]
        ax.scatter(
            seed_coords[:, 0],
            seed_coords[:, 1],
            s=node_size,  # Match point_size from datamapplot.create_plot
            facecolors='none', 
            edgecolors='black',
            linewidths=2, 
            zorder=2 
        )
        
    # final touches
    scatter.set_zorder(3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.set_dpi(200)

    # create legend for endotype colors and inter-endotype edges 
    handles = []
    for lab in unique_labels:
        color = label_to_color.get(lab, (0.7, 0.7, 0.7, 0.6))
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=lab))
    
    # add legend for inter-edges
    handles.append(plt.Line2D([0], [0], color='gray', linewidth=4, alpha=0.2, label='Inter-endotype edges'))
    
    # display legend
    ax.legend(handles=handles, loc='upper left', fontsize=14, framealpha=0.7)
    plt.title("Metagraph Visualization of Endotypes", fontsize=16)
    
    plt.show()
    
