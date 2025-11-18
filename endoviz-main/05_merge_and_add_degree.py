#!/usr/bin/env python3
"""
Calculate node degrees from edges and merge with coordinates and Endotype.
Adds degree column to node_coordinates_2d.csv and merges Endotype from nodes.tsv.
Saves final result to nodes_out.tsv.
"""

from collections import Counter

import pandas as pd


def main():
    print("=" * 60)
    print("Calculate Node Degrees and Merge Data")
    print("=" * 60)

    # 1. Load edges and calculate degrees
    print("\n1. Loading edges.tsv and calculating node degrees...")
    edges_df = pd.read_csv('edges.tsv', sep='\t')
    print(f"   Total edges: {len(edges_df):,}")

    # Count in-degree (how many times node appears as tonodeid)
    in_degree = Counter(edges_df['tonodeid'])

    # Count out-degree (how many times node appears as fromnodeid)
    out_degree = Counter(edges_df['fromnodeid'])

    # Total degree = in-degree + out-degree
    total_degree = Counter()
    total_degree.update(in_degree)
    total_degree.update(out_degree)

    # Convert to DataFrame
    degree_df = pd.DataFrame([
        {'nodeid': nodeid, 'degree': degree, 'in_degree': in_degree.get(nodeid, 0), 'out_degree': out_degree.get(nodeid, 0)}
        for nodeid, degree in total_degree.items()
    ])

    print(f"   Nodes with edges: {len(degree_df):,}")
    print(f"   Average degree: {degree_df['degree'].mean():.2f}")
    print(f"   Max degree: {degree_df['degree'].max()}")
    print(f"   Min degree: {degree_df['degree'].min()}")

    # 2. Load node coordinates
    print("\n2. Loading node_coordinates_2d.csv...")
    coords_df = pd.read_csv('node_coordinates_2d.csv')
    print(f"   Nodes with coordinates: {len(coords_df):,}")

    # 3. Merge degree with coordinates
    print("\n3. Merging degree with coordinates...")
    coords_df = coords_df.merge(degree_df, on='nodeid', how='left')

    # Fill NaN degrees with 0 (nodes that have coordinates but no edges)
    coords_df['degree'] = coords_df['degree'].fillna(0).astype(int)
    coords_df['in_degree'] = coords_df['in_degree'].fillna(0).astype(int)
    coords_df['out_degree'] = coords_df['out_degree'].fillna(0).astype(int)

    nodes_with_degree = (coords_df['degree'] > 0).sum()
    print(f"   Nodes with degree > 0: {nodes_with_degree:,}")
    print(f"   Nodes with degree = 0: {(coords_df['degree'] == 0).sum():,}")

    # 4. Load nodes.tsv and extract Endotype
    print("\n4. Loading nodes.tsv and extracting Endotype...")
    nodes_df = pd.read_csv('nodes.tsv', sep='\t')
    print(f"   Nodes in nodes.tsv: {len(nodes_df):,}")

    # Create Endotype mapping
    endotype_map = dict(zip(nodes_df['nodeid'], nodes_df['Endotype']))

    # 5. Merge Endotype with coordinates
    print("\n5. Merging Endotype with coordinates...")
    coords_df['Endotype'] = coords_df['nodeid'].map(endotype_map)

    # Count Endotype distribution
    endotype_counts = coords_df['Endotype'].value_counts().sort_index()
    print("   Endotype distribution:")
    for endotype, count in endotype_counts.items():
        endotype_str = str(endotype) if pd.notna(endotype) else "NaN"
        print(f"     Endotype {endotype_str}: {count:,} nodes")

    # 6. Reorder columns for output
    print("\n6. Preparing output...")

    # Order: nodeid, Endotype, degree, in_degree, out_degree, then coordinates
    column_order = ['nodeid', 'Endotype', 'degree', 'in_degree', 'out_degree',
                   'x_mean', 'y_mean', 'x_wmean', 'y_wmean', 'x_max', 'y_max']

    # Only include columns that exist
    output_columns = [col for col in column_order if col in coords_df.columns]
    output_df = coords_df[output_columns].copy()

    print(f"   Output columns: {output_columns}")

    # 7. Save to nodes_out.tsv
    print("\n7. Saving to nodes_out.tsv...")
    output_df.to_csv('nodes_out.tsv', sep='\t', index=False)
    print(f"   ✓ Saved {len(output_df):,} nodes to nodes_out.tsv")

    # Also save as CSV for easy viewing
    output_df.to_csv('nodes_out.csv', index=False)
    print(f"   ✓ Saved {len(output_df):,} nodes to nodes_out.csv")

    # 8. Print summary statistics
    print("\n8. Summary Statistics:")
    print("-" * 60)
    print(f"   Total nodes: {len(output_df):,}")
    print(f"   Nodes with degree > 0: {(output_df['degree'] > 0).sum():,}")
    print(f"   Nodes with Endotype: {output_df['Endotype'].notna().sum():,}")
    print(f"   Nodes with both degree and Endotype: {((output_df['degree'] > 0) & output_df['Endotype'].notna()).sum():,}")

    print("\n   Degree statistics:")
    print(f"     Mean: {output_df['degree'].mean():.2f}")
    print(f"     Median: {output_df['degree'].median():.2f}")
    print(f"     Std: {output_df['degree'].std():.2f}")
    print(f"     Min: {output_df['degree'].min()}")
    print(f"     Max: {output_df['degree'].max()}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - nodes_out.tsv: Full dataset with coordinates, degrees, and Endotype")
    print("  - nodes_out.csv: Same data in CSV format")

if __name__ == "__main__":
    main()
