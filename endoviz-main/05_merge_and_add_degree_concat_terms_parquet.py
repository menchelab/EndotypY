#!/usr/bin/env python3
"""
Calculate node degrees from edges and merge with coordinates and Endotype.
Adds degree column to node_coordinates_2d.csv or node_coordinates_3d.csv and merges Endotype from nodes.tsv.
Also adds a column with all terms concatenated with commas.
Saves final result to nodes_out.parquet.
"""

import argparse
import os
from collections import Counter

import pandas as pd


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge node degrees, coordinates, and Endotype data')
    parser.add_argument('--dimensions', type=int, default=2, choices=[2, 3],
                        help='Dimension of coordinates file to load (2 or 3, default: 2)')
    args = parser.parse_args()

    n_dimensions = args.dimensions
    dim_str = f"{n_dimensions}d"

    print("=" * 60)
    print("Calculate Node Degrees and Merge Data (with Terms)")
    print("=" * 60)

    # 1. Load edges and calculate degrees
    print("\n1. Loading edges.tsv and calculating node degrees...")
    edges_df = pd.read_csv('input/edges.tsv', sep='\t')
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

    # 2. Load term-node mappings and create term strings
    print("\n2. Loading term_node_mapping.tsv and grouping terms by node...")
    mapping_df = pd.read_csv('input/term_node_mapping.tsv', sep='\t')
    print(f"   Total mappings: {len(mapping_df):,}")

    # Group terms by nodeid and concatenate with commas
    node_terms = mapping_df.groupby('nodeid')['Term'].apply(lambda x: ', '.join(x.unique())).to_dict()

    print(f"   Nodes with terms: {len(node_terms):,}")

    # Calculate statistics
    term_counts = mapping_df.groupby('nodeid')['Term'].nunique()
    print(f"   Average terms per node: {term_counts.mean():.2f}")
    print(f"   Max terms per node: {term_counts.max()}")
    print(f"   Min terms per node: {term_counts.min()}")

    # 3. Load node coordinates
    coords_file = f'output/node_coordinates_{dim_str}.csv'
    print(f"\n3. Loading {coords_file}...")

    if not os.path.exists(coords_file):
        print(f"   Error: {coords_file} not found!")
        print(f"   Please run 04_reduce.py with --dimensions {n_dimensions} first.")
        return

    coords_df = pd.read_csv(coords_file)
    print(f"   Nodes with coordinates: {len(coords_df):,}")

    # Detect actual dimension from columns (in case file exists but has different dimension)
    has_z_cols = any(col.startswith('z_') for col in coords_df.columns)
    actual_dim = 3 if has_z_cols else 2
    if actual_dim != n_dimensions:
        print(f"   Warning: File contains {actual_dim}D coordinates but {n_dimensions}D was requested.")
        print(f"   Using {actual_dim}D coordinates from file.")
        n_dimensions = actual_dim
        dim_str = f"{n_dimensions}d"

    # 4. Merge degree with coordinates
    print("\n4. Merging degree with coordinates...")
    coords_df = coords_df.merge(degree_df, on='nodeid', how='left')

    # Fill NaN degrees with 0 (nodes that have coordinates but no edges)
    coords_df['degree'] = coords_df['degree'].fillna(0).astype(int)
    coords_df['in_degree'] = coords_df['in_degree'].fillna(0).astype(int)
    coords_df['out_degree'] = coords_df['out_degree'].fillna(0).astype(int)

    nodes_with_degree = (coords_df['degree'] > 0).sum()
    print(f"   Nodes with degree > 0: {nodes_with_degree:,}")
    print(f"   Nodes with degree = 0: {(coords_df['degree'] == 0).sum():,}")

    # 5. Add terms column
    print("\n5. Adding terms column...")
    coords_df['terms'] = coords_df['nodeid'].map(node_terms)

    # Fill NaN with empty string for nodes without terms
    coords_df['terms'] = coords_df['terms'].fillna('')

    nodes_with_terms = (coords_df['terms'] != '').sum()
    print(f"   Nodes with terms: {nodes_with_terms:,}")
    print(f"   Nodes without terms: {(coords_df['terms'] == '').sum():,}")

    # Show example
    example_node = coords_df[coords_df['terms'] != ''].iloc[0]
    example_terms = example_node['terms']
    if len(example_terms) > 100:
        example_terms = example_terms[:100] + "..."
    print(f"   Example terms for {example_node['nodeid']}: {example_terms}")

    # 6. Load nodes.tsv and extract Endotype
    print("\n6. Loading nodes.tsv and extracting Endotype...")
    nodes_df = pd.read_csv('input/nodes.tsv', sep='\t')
    print(f"   Nodes in nodes.tsv: {len(nodes_df):,}")

    # Create Endotype mapping
    endotype_map = dict(zip(nodes_df['nodeid'], nodes_df['Endotype']))

    # 7. Merge Endotype with coordinates
    print("\n7. Merging Endotype with coordinates...")
    coords_df['Endotype'] = coords_df['nodeid'].map(endotype_map)

    # Count Endotype distribution
    endotype_counts = coords_df['Endotype'].value_counts().sort_index()
    print("   Endotype distribution:")
    for endotype, count in endotype_counts.items():
        endotype_str = str(endotype) if pd.notna(endotype) else "NaN"
        print(f"     Endotype {endotype_str}: {count:,} nodes")

    # 8. Reorder columns for output
    print("\n8. Preparing output...")
    
    # Order: nodeid, Endotype, degree, in_degree, out_degree, terms, then coordinates
    # Build column order dynamically based on available dimensions
    column_order = ['nodeid', 'Endotype', 'degree', 'in_degree', 'out_degree', 'terms']
    
    # Add coordinate columns: UMAP first (u = umap), then PaCMAP (p = pacmap)
    # For each method (mean, wmean, max) and each reducer (u, p)
    coord_methods = ['mean', 'wmean', 'max']
    coord_reducers = ['u', 'p']  # u = UMAP, p = PaCMAP
    
    for reducer in coord_reducers:
        for method in coord_methods:
            column_order.append(f'x_{reducer}_{method}')
            column_order.append(f'y_{reducer}_{method}')
            if n_dimensions == 3:
                column_order.append(f'z_{reducer}_{method}')
    
    # Only include columns that exist
    output_columns = [col for col in column_order if col in coords_df.columns]
    output_df = coords_df[output_columns].copy()

    print(f"   Output columns: {output_columns}")

    # 9. Save to nodes_out.parquet
    output_filename = f'output/nodes_out_{dim_str}.parquet'
    print(f"\n9. Saving to {output_filename}...")
    output_df.to_parquet(output_filename, index=False, engine='pyarrow')
    print(f"   ✓ Saved {len(output_df):,} nodes to {output_filename}")

    # 10. Print summary statistics
    print("\n10. Summary Statistics:")
    print("-" * 60)
    print(f"   Total nodes: {len(output_df):,}")
    print(f"   Nodes with degree > 0: {(output_df['degree'] > 0).sum():,}")
    print(f"   Nodes with Endotype: {output_df['Endotype'].notna().sum():,}")
    print(f"   Nodes with terms: {(output_df['terms'] != '').sum():,}")
    print(f"   Nodes with both degree and Endotype: {((output_df['degree'] > 0) & output_df['Endotype'].notna()).sum():,}")

    print("\n   Degree statistics:")
    print(f"     Mean: {output_df['degree'].mean():.2f}")
    print(f"     Median: {output_df['degree'].median():.2f}")
    print(f"     Std: {output_df['degree'].std():.2f}")
    print(f"     Min: {output_df['degree'].min()}")
    print(f"     Max: {output_df['degree'].max()}")

    print("\n   Terms statistics:")
    term_lengths = output_df[output_df['terms'] != '']['terms'].str.len()
    print(f"     Average term string length: {term_lengths.mean():.0f} characters")
    print(f"     Max term string length: {term_lengths.max():,} characters")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nOutput file:")
    print(f"  - {output_filename}: Full dataset with {n_dimensions}D coordinates, degrees, Endotype, and terms")

if __name__ == "__main__":
    main()
