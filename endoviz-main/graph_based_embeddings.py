#!/usr/bin/env python3
"""
Graph-based embedding pipeline using node2vec.
Creates node embeddings based on network structure (edges), then reduces to 2D coordinates.
Includes degree calculation, Endotype merging, and term concatenation.
Saves final result to nodes_graph_embeddings.parquet.
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("Graph-Based Embeddings Pipeline (node2vec)")
    print("=" * 60)

    # 1. Load edges and build graph
    print("\n1. Loading edges.tsv and building graph...")
    edges_df = pd.read_csv("input/edges.tsv", sep="\t")
    print(f"   Total edges: {len(edges_df):,}")

    # Clean edges: remove NaN values and empty strings
    print("   Cleaning edges (removing NaN and invalid values)...")
    initial_count = len(edges_df)

    # Convert to string and handle NaN
    edges_df["fromnodeid"] = edges_df["fromnodeid"].astype(str)
    edges_df["tonodeid"] = edges_df["tonodeid"].astype(str)

    # Remove rows with NaN (which become 'nan' string) or empty strings
    edges_df = edges_df[
        (edges_df["fromnodeid"] != "nan")
        & (edges_df["tonodeid"] != "nan")
        & (edges_df["fromnodeid"] != "")
        & (edges_df["tonodeid"] != "")
        & (edges_df["fromnodeid"].notna())
        & (edges_df["tonodeid"].notna())
    ]

    cleaned_count = len(edges_df)
    removed = initial_count - cleaned_count
    if removed > 0:
        print(f"   Removed {removed:,} invalid edges ({removed/initial_count*100:.2f}%)")
    print(f"   Valid edges: {cleaned_count:,}")

    # Create directed graph
    G = nx.DiGraph()
    G.add_edges_from(zip(edges_df["fromnodeid"], edges_df["tonodeid"]))

    print(f"   Nodes in graph: {G.number_of_nodes():,}")
    print(f"   Edges in graph: {G.number_of_edges():,}")

    # Calculate degrees before embedding
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    total_degree = {
        node: in_degree.get(node, 0) + out_degree.get(node, 0) for node in G.nodes()
    }

    print(f"   Average degree: {np.mean(list(total_degree.values())):.2f}")
    print(f"   Max degree: {max(total_degree.values())}")

    # 2. Generate node2vec embeddings
    print("\n2. Generating node2vec embeddings...")
    print("   (This may take several minutes)")

    # node2vec parameters
    dimensions = 128  # Embedding dimension
    walk_length = 30  # Length of random walks
    num_walks = 10  # Number of walks per node
    p = 1.0  # Return parameter (BFS-like)
    q = 1.0  # In-out parameter (DFS-like)
    workers = 4  # Number of parallel workers

    # Initialize node2vec
    # quiet=False enables progress reporting from gensim/Word2Vec
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        quiet=False,  # Show progress from gensim
    )

    # Fit and generate embeddings
    print("   Fitting Word2Vec model (this is the slowest part)...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Extract embeddings with progress bar
    print("   Extracting embeddings...")
    nodeids = list(G.nodes())
    embeddings = np.array(
        [model.wv[nodeid] for nodeid in tqdm(nodeids, desc="   Extracting")]
    )

    print(f"   ✓ Generated embeddings for {len(nodeids):,} nodes")
    print(f"   Embedding dimension: {embeddings.shape[1]}")

    # 3. Normalize embeddings
    print("\n3. Normalizing embeddings...")
    scaler = StandardScaler()
    embeddings_norm = scaler.fit_transform(embeddings)
    print(f"   Mean before: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
    print(
        f"   Mean after: {embeddings_norm.mean():.4f}, std: {embeddings_norm.std():.4f}"
    )

    # 4. Reduce to 2D using UMAP
    print("\n4. Reducing to 2D coordinates using UMAP...")
    print("   (This may take a few minutes)")

    n_neighbors = min(15, len(nodeids) - 1)
    min_dist = 0.1

    umap_reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        init="random",
    )

    # UMAP doesn't have built-in progress, but we can show it's running
    coords_2d = umap_reducer.fit_transform(embeddings_norm)
    print("   ✓ Reduced to 2D coordinates")

    # 5. Create base dataframe
    print("\n5. Creating base dataframe...")
    df = pd.DataFrame(
        {
            "nodeid": nodeids,
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "degree": [total_degree.get(nid, 0) for nid in nodeids],
            "in_degree": [in_degree.get(nid, 0) for nid in nodeids],
            "out_degree": [out_degree.get(nid, 0) for nid in nodeids],
        }
    )

    print(f"   ✓ Created dataframe with {len(df):,} nodes")

    # 6. Load and merge Endotype
    print("\n6. Loading and merging Endotype...")
    nodes_df = pd.read_csv("input/nodes.tsv", sep="\t")
    endotype_map = dict(zip(nodes_df["nodeid"], nodes_df["Endotype"]))

    df["Endotype"] = df["nodeid"].map(endotype_map)

    endotype_counts = df["Endotype"].value_counts().sort_index()
    print("   Endotype distribution:")
    for endotype, count in endotype_counts.items():
        endotype_str = str(endotype) if pd.notna(endotype) else "NaN"
        print(f"     Endotype {endotype_str}: {count:,} nodes")

    # 7. Load and merge terms
    print("\n7. Loading and merging terms...")
    try:
        mapping_df = pd.read_csv("input/term_node_mapping.tsv", sep="\t")
        node_terms = (
            mapping_df.groupby("nodeid")["Term"]
            .apply(lambda x: ", ".join(x.unique()))
            .to_dict()
        )
        df["terms"] = df["nodeid"].map(node_terms).fillna("")

        nodes_with_terms = (df["terms"] != "").sum()
        print(f"   Nodes with terms: {nodes_with_terms:,}")
        print(f"   Nodes without terms: {(df['terms'] == '').sum():,}")
    except FileNotFoundError:
        print("   Warning: term_node_mapping.tsv not found, skipping terms")
        df["terms"] = ""

    # 8. Reorder columns
    print("\n8. Preparing final output...")
    column_order = [
        "nodeid",
        "Endotype",
        "degree",
        "in_degree",
        "out_degree",
        "terms",
        "x",
        "y",
    ]
    output_columns = [col for col in column_order if col in df.columns]
    output_df = df[output_columns].copy()

    print(f"   Output columns: {output_columns}")

    # 9. Save to parquet
    print("\n9. Saving to nodes_graph_embeddings.parquet...")
    output_df.to_parquet(
        "output/nodes_graph_embeddings.parquet", index=False, engine="pyarrow"
    )
    print(f"   ✓ Saved {len(output_df):,} nodes to nodes_graph_embeddings.parquet")

    # 10. Print summary statistics
    print("\n10. Summary Statistics:")
    print("-" * 60)
    print(f"   Total nodes: {len(output_df):,}")
    print(f"   Nodes with degree > 0: {(output_df['degree'] > 0).sum():,}")
    print(f"   Nodes with Endotype: {output_df['Endotype'].notna().sum():,}")
    print(f"   Nodes with terms: {(output_df['terms'] != '').sum():,}")

    print("\n   Degree statistics:")
    print(f"     Mean: {output_df['degree'].mean():.2f}")
    print(f"     Median: {output_df['degree'].median():.2f}")
    print(f"     Std: {output_df['degree'].std():.2f}")
    print(f"     Min: {output_df['degree'].min()}")
    print(f"     Max: {output_df['degree'].max()}")

    print("\n   Coordinate statistics:")
    print(f"     X range: [{output_df['x'].min():.4f}, {output_df['x'].max():.4f}]")
    print(f"     Y range: [{output_df['y'].min():.4f}, {output_df['y'].max():.4f}]")
    print(f"     X std: {output_df['x'].std():.4f}")
    print(f"     Y std: {output_df['y'].std():.4f}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nOutput file:")
    print(
        "  - nodes_graph_embeddings.parquet: Graph-based embeddings with coordinates, degrees, Endotype, and terms"
    )
    print(
        "\nNote: This approach captures network structure directly, unlike term-based embeddings."
    )


if __name__ == "__main__":
    main()
