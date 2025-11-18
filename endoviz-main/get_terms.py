#!/usr/bin/env python3
"""
Filter term_node_mapping.tsv to only include node IDs that appear in edges.
Extracts unique node IDs from edges.tsv and filters term_node_mapping.tsv accordingly.
"""

import pandas as pd


def main():
    print("=" * 60)
    print("Filter Term-Node Mappings by Edge Node IDs")
    print("=" * 60)

    # Read edges.tsv
    print("\n1. Reading edges.tsv...")
    edges_df = pd.read_csv('edges.tsv', sep='\t')
    print(f"   Total edges: {len(edges_df):,}")

    # Collect unique node IDs from edges
    from_nodeids = set(edges_df['fromnodeid'].unique())
    to_nodeids = set(edges_df['tonodeid'].unique())
    all_edge_nodeids = from_nodeids | to_nodeids

    print("\n2. Extracting unique node IDs from edges:")
    print(f"   From node IDs: {len(from_nodeids):,}")
    print(f"   To node IDs: {len(to_nodeids):,}")
    print(f"   Total unique node IDs in edges: {len(all_edge_nodeids):,}")

    # Read term_node_mapping.tsv
    print("\n3. Reading term_node_mapping.tsv...")
    mapping_df = pd.read_csv('term_node_mapping.tsv', sep='\t')
    print(f"   Total mappings: {len(mapping_df):,}")
    print(f"   Unique node IDs in mapping: {mapping_df['nodeid'].nunique():,}")
    print(f"   Unique terms in mapping: {mapping_df['Term'].nunique():,}")

    # Filter mapping to only include node IDs that are in edges
    print("\n4. Filtering term_node_mapping.tsv...")
    filtered_df = mapping_df[mapping_df['nodeid'].isin(all_edge_nodeids)].copy()

    print(f"   Filtered mappings: {len(filtered_df):,}")
    print(f"   Unique node IDs in filtered mapping: {filtered_df['nodeid'].nunique():,}")
    print(f"   Unique terms in filtered mapping: {filtered_df['Term'].nunique():,}")

    # Keep only unique terms and remove nodeid column
    print("\n5. Keeping only unique terms and removing nodeid column...")
    unique_terms_df = filtered_df[['Term']].drop_duplicates().copy()
    print(f"   Unique terms after deduplication: {len(unique_terms_df):,}")

    # Calculate statistics
    original_nodeids = set(mapping_df['nodeid'].unique())
    filtered_nodeids = set(filtered_df['nodeid'].unique())

    nodes_in_both = all_edge_nodeids & original_nodeids
    nodes_in_edges_not_in_mapping = all_edge_nodeids - original_nodeids
    nodes_in_mapping_not_in_edges = original_nodeids - all_edge_nodeids

    print("\n6. Statistics:")
    print("-" * 60)
    print(f"   Node IDs in edges that are also in mapping: {len(nodes_in_both):,}")
    print(f"   Node IDs in edges but NOT in mapping: {len(nodes_in_edges_not_in_mapping):,}")
    print(f"   Node IDs in mapping but NOT in edges: {len(nodes_in_mapping_not_in_edges):,}")

    coverage = (len(nodes_in_both) / len(all_edge_nodeids)) * 100 if all_edge_nodeids else 0
    print(f"\n   Coverage: {coverage:.2f}% of edge node IDs have mappings")

    reduction = ((len(mapping_df) - len(filtered_df)) / len(mapping_df)) * 100 if len(mapping_df) > 0 else 0
    print(f"   Reduction: {reduction:.2f}% of mappings removed")

    final_reduction = ((len(filtered_df) - len(unique_terms_df)) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    print(f"   Final reduction (unique terms only): {final_reduction:.2f}% of filtered rows removed")

    # Save filtered results
    output_file = 'terms_filtered.tsv'
    print(f"\n7. Saving unique terms to {output_file}...")
    unique_terms_df.to_csv(output_file, sep='\t', index=False)
    print(f"   ✓ Saved {len(unique_terms_df):,} unique terms to {output_file}")

    print("\n" + "=" * 60)
    print("Filtering complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

