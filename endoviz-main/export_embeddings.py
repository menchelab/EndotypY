#!/usr/bin/env python3
"""
Export node embeddings, node IDs, and Endotype to a portable format.
Loads weighted mean embeddings and merges with Endotype data.
Exports to parquet format for easy transfer and reading.
"""

import pickle

import numpy as np
import pandas as pd


def main():
    print("=" * 60)
    print("Export Embeddings and Metadata")
    print("=" * 60)

    # 1. Load weighted mean embeddings
    print("\n1. Loading node_embeddings_weighted_mean.pkl...")
    with open('output/node_embeddings_weighted_mean.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)

    print(f"   ✓ Loaded embeddings for {len(embeddings_dict):,} nodes")

    # Get embedding dimension
    embedding_dim = next(iter(embeddings_dict.values())).shape[0]
    print(f"   Embedding dimension: {embedding_dim}")

    # Extract node IDs and embeddings
    nodeids = list(embeddings_dict.keys())
    embeddings_array = np.array([embeddings_dict[nodeid] for nodeid in nodeids])

    print(f"   Embeddings array shape: {embeddings_array.shape}")

    # 2. Load Endotype from nodes.tsv
    print("\n2. Loading Endotype from nodes.tsv...")
    nodes_df = pd.read_csv('input/nodes.tsv', sep='\t')
    print(f"   Nodes in nodes.tsv: {len(nodes_df):,}")

    # Create Endotype mapping
    endotype_map = dict(zip(nodes_df['nodeid'], nodes_df['Endotype']))

    # 3. Create dataframe with nodeids, embeddings, and Endotype
    print("\n3. Creating export dataframe...")

    # Create dataframe
    export_df = pd.DataFrame({
        'nodeid': nodeids,
        'Endotype': [endotype_map.get(nodeid, None) for nodeid in nodeids]
    })

    # Add embeddings as list column (parquet can handle this)
    # Convert numpy array to list of lists for parquet compatibility
    export_df['embedding'] = [emb.tolist() for emb in embeddings_array]

    print(f"   ✓ Created dataframe with {len(export_df):,} nodes")
    print(f"   Columns: {list(export_df.columns)}")

    # Count Endotype distribution
    endotype_counts = export_df['Endotype'].value_counts().sort_index()
    print("\n   Endotype distribution:")
    for endotype, count in endotype_counts.items():
        endotype_str = str(endotype) if pd.notna(endotype) else "NaN"
        print(f"     Endotype {endotype_str}: {count:,} nodes")

    # 3. Load term-node mappings and create concatenated terms
    print("\n3. Loading term_node_mapping.tsv and creating concatenated terms...")
    mapping_df = pd.read_csv('input/term_node_mapping.tsv', sep='\t')
    print(f"   Total mappings: {len(mapping_df):,}")
    
    # Remove commas from terms
    terms_before = mapping_df['Term'].str.contains(',', na=False).sum()
    mapping_df['Term'] = mapping_df['Term'].str.replace(',', '', regex=False)
    if terms_before > 0:
        print(f"   Removed commas from {terms_before:,} term occurrences")
    
    # Group terms by nodeid and concatenate with commas
    node_term_strings = mapping_df.groupby('nodeid')['Term'].apply(lambda x: ', '.join(x.unique())).to_dict()
    print(f"   Nodes with terms: {len(node_term_strings):,}")

    # Add terms column
    export_df['terms'] = [node_term_strings.get(nodeid, '') for nodeid in nodeids]
    nodes_with_terms = (export_df['terms'] != '').sum()
    print(f"   Nodes with terms: {nodes_with_terms:,}")
    print(f"   Nodes without terms: {(export_df['terms'] == '').sum():,}")

    # 4. Save to parquet
    print("\n4. Saving to export_embeddings.parquet...")
    export_df.to_parquet('output/export_embeddings.parquet', index=False, engine='pyarrow')
    print(f"   ✓ Saved {len(export_df):,} nodes to export_embeddings.parquet")

    # Also save embeddings array separately as numpy file for convenience
    print("\n5. Saving embeddings array separately...")
    np.savez('output/export_embeddings_array.npz',
             embeddings=embeddings_array,
             nodeids=np.array(nodeids, dtype=object))
    print("   ✓ Saved to export_embeddings_array.npz")

    # 6. Print summary
    print("\n6. Summary:")
    print("-" * 60)
    print(f"   Total nodes: {len(export_df):,}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Nodes with Endotype: {export_df['Endotype'].notna().sum():,}")
    print(f"   Nodes without Endotype: {export_df['Endotype'].isna().sum():,}")
    print(f"   Nodes with terms: {nodes_with_terms:,}")


if __name__ == "__main__":
    main()

