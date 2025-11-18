#!/usr/bin/env python3
"""
Sanity check script for node and edge data.
Analyzes edges.tsv, nodes.tsv, and term_node_mapping.tsv to find:
- Unique node IDs in edges
- Missing nodes (in edges but not in nodes)
- Orphaned nodes (in nodes but not in edges)
- Node IDs in edges that are missing from term_node_mapping.tsv
"""

import pandas as pd


def main():
    print("=" * 60)
    print("Sanity Check for Node and Edge Data")
    print("=" * 60)

    # Read edges.tsv
    print("\n1. Reading edges.tsv...")
    edges_df = pd.read_csv('input/edges.tsv', sep='\t')
    print(f"   Total edges: {len(edges_df):,}")

    # Collect unique node IDs from edges
    from_nodeids = set(edges_df['fromnodeid'].unique())
    to_nodeids = set(edges_df['tonodeid'].unique())
    all_edge_nodeids = from_nodeids | to_nodeids

    print("\n2. Unique node IDs in edges:")
    print(f"   From node IDs: {len(from_nodeids):,}")
    print(f"   To node IDs: {len(to_nodeids):,}")
    print(f"   Total unique node IDs in edges: {len(all_edge_nodeids):,}")

    # Read nodes.tsv
    print("\n3. Reading nodes.tsv...")
    nodes_df = pd.read_csv('input/nodes.tsv', sep='\t')
    print(f"   Total nodes: {len(nodes_df):,}")

    # Get node IDs from nodes
    node_nodeids = set(nodes_df['nodeid'].unique())
    print(f"   Unique node IDs in nodes: {len(node_nodeids):,}")

    # Compare and generate statistics
    print("\n4. Statistics:")
    print("-" * 60)

    # Nodes in edges but not in nodes
    missing_from_nodes = all_edge_nodeids - node_nodeids
    print(f"\n   Nodes in edges but NOT in nodes.tsv: {len(missing_from_nodes):,}")
    if missing_from_nodes:
        print(f"   Examples (first 10): {list(missing_from_nodes)[:10]}")

    # Nodes in nodes but not in edges (orphaned)
    orphaned_nodes = node_nodeids - all_edge_nodeids
    print(f"\n   Nodes in nodes.tsv but NOT in edges.tsv (orphaned): {len(orphaned_nodes):,}")
    if orphaned_nodes:
        print(f"   Examples (first 10): {list(orphaned_nodes)[:10]}")

    # Nodes that appear in both
    common_nodes = all_edge_nodeids & node_nodeids
    print(f"\n   Nodes present in both files: {len(common_nodes):,}")

    # Coverage statistics
    edge_coverage = (len(common_nodes) / len(all_edge_nodeids)) * 100 if all_edge_nodeids else 0
    node_coverage = (len(common_nodes) / len(node_nodeids)) * 100 if node_nodeids else 0

    print("\n   Coverage:")
    print(f"   - Edge node IDs covered by nodes.tsv: {edge_coverage:.2f}%")
    print(f"   - Node IDs that have edges: {node_coverage:.2f}%")

    # Additional statistics
    print("\n5. Additional Statistics:")
    print("-" * 60)

    # Nodes that only appear as "from" nodes
    only_from = from_nodeids - to_nodeids
    print(f"\n   Nodes that only appear as 'from' nodes: {len(only_from):,}")

    # Nodes that only appear as "to" nodes
    only_to = to_nodeids - from_nodeids
    print(f"   Nodes that only appear as 'to' nodes: {len(only_to):,}")

    # Nodes that appear in both from and to
    both_directions = from_nodeids & to_nodeids
    print(f"   Nodes that appear in both 'from' and 'to': {len(both_directions):,}")

    # Read term_node_mapping.tsv
    print("\n6. Reading term_node_mapping.tsv...")
    mapping_df = pd.read_csv('input/term_node_mapping.tsv', sep='\t')
    print(f"   Total mappings: {len(mapping_df):,}")

    # Get unique nodeid values (node IDs) from mapping
    mapping_nodeids = set(mapping_df['nodeid'].unique())
    print(f"   Unique node IDs in term_node_mapping.tsv: {len(mapping_nodeids):,}")

    # Check if all edge node IDs exist in mapping
    print("\n7. Edge Node IDs vs term_node_mapping.tsv:")
    print("-" * 60)

    # Nodes in edges but not in mapping
    missing_from_mapping = all_edge_nodeids - mapping_nodeids
    print(f"\n   Nodes in edges but NOT in term_node_mapping.tsv: {len(missing_from_mapping):,}")
    if missing_from_mapping:
        print(f"   Examples (first 10): {list(missing_from_mapping)[:10]}")

    # Nodes in mapping but not in edges
    mapping_only_nodes = mapping_nodeids - all_edge_nodeids
    print(f"\n   Nodes in term_node_mapping.tsv but NOT in edges: {len(mapping_only_nodes):,}")
    if mapping_only_nodes:
        print(f"   Examples (first 10): {list(mapping_only_nodes)[:10]}")

    # Nodes that appear in both edges and mapping
    common_with_mapping = all_edge_nodeids & mapping_nodeids
    print(f"\n   Nodes present in both edges and term_node_mapping.tsv: {len(common_with_mapping):,}")

    # Coverage statistics for mapping
    edge_mapping_coverage = (len(common_with_mapping) / len(all_edge_nodeids)) * 100 if all_edge_nodeids else 0
    mapping_edge_coverage = (len(common_with_mapping) / len(mapping_nodeids)) * 100 if mapping_nodeids else 0

    print("\n   Coverage:")
    print(f"   - Edge node IDs covered by term_node_mapping.tsv: {edge_mapping_coverage:.2f}%")
    print(f"   - Mapping node IDs that have edges: {mapping_edge_coverage:.2f}%")

    # Check for terms containing commas
    print("\n8. Checking for terms containing commas:")
    print("-" * 60)

    terms_with_commas = mapping_df[mapping_df['Term'].str.contains(',', na=False)]
    num_terms_with_commas = len(terms_with_commas)
    num_unique_terms_with_commas = terms_with_commas['Term'].nunique()

    print(f"\n   Terms containing commas: {num_terms_with_commas:,} occurrences")
    print(f"   Unique terms containing commas: {num_unique_terms_with_commas:,}")

    if num_terms_with_commas > 0:
        print("\n   ⚠️  WARNING: Found terms with commas!")
        print("   This may cause issues when concatenating terms with comma separators.")
        print("\n   Examples (first 10):")
        example_terms = terms_with_commas['Term'].unique()[:10]
        for term in example_terms:
            print(f"     - {term}")
    else:
        print("\n   ✓ No terms contain commas - safe for comma-separated concatenation")

    print("\n" + "=" * 60)
    print("Sanity check complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

